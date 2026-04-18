"""
Build the elementwise max((A * B) + C, 0) (aka fused mul-add-ReLU) CoreIR graph using pycoreir, per lane.

Templated design parameters:
- unroll: number of lanes (PE groups and IO groups).
- tensor_size: total tensor size; each IO has extent tensor_size // unroll.
- mode:
    "mu_x_const_relu"             — max(MU_A * const, 0), 2 PEs per lane (mul + max).
                                    (used by zircon_dequantize_relu_fp)
    "mu_plus_vector_relu"         — max(MU_A + GLB_C, 0), 2 PEs per lane (add + max).
                                    (used by zircon_residual_relu_fp)
    "mu_x_const_plus_vector_relu" — max((MU_A * const) + GLB_C, 0), 3 PEs per lane (mul + add + max).
                                    (used by zircon_deq_ResReLU_fp)
- mul_const_val_bf16: constant mul operand (used in *_x_const_* modes).

Graph per lane i, by mode:
  mu_x_const_relu (2 PEs):
    input_A IO.out -> mul_pe.data0 ; const -> mul_pe.data1 (encoded)
    mul_pe.O0      -> max_pe.data0 ; bf16(0.0) -> max_pe.data1 (encoded)
    max_pe.O0      -> output IO.in

  mu_plus_vector_relu (2 PEs):
    input_A IO.out -> add_pe.data0 ; input_C IO.out -> add_pe.data1
    add_pe.O0      -> max_pe.data0 ; bf16(0.0) -> max_pe.data1 (encoded)
    max_pe.O0      -> output IO.in

  mu_x_const_plus_vector_relu (3 PEs):
    input_A IO.out -> mul_pe.data0 ; const -> mul_pe.data1 (encoded)
    input_C IO.out -> add_pe.data0 ; mul_pe.O0 -> add_pe.data1
    add_pe.O0      -> max_pe.data0 ; bf16(0.0) -> max_pe.data1 (encoded)
    max_pe.O0      -> output IO.in
"""

import os
import json
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
import strait.coreir_backend.utils.build_pe_inst as build_pe_inst

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_TENSOR_SIZE = 4096
DEFAULT_MUL_CONST_VAL_BF16 = 0.5
DEFAULT_MODE = "mu_x_const_plus_vector_relu"

_VALID_MODES = (
    "mu_x_const_relu",
    "mu_plus_vector_relu",
    "mu_x_const_plus_vector_relu",
)

# Logical IO names per mode. Keys are the bytes between io16[in]_ and _clkwrk_,
# which _assert_strait_names_match_halide_meta compares against design_meta_halide.json.
# input_c is None for modes that have no GLB residual input.
_MODE_NAMES = {
    "mu_x_const_relu": {
        "input_a": "mu_hw_input_stencil",
        "input_c": None,
        "output": "hw_output_stencil",
    },
    "mu_plus_vector_relu": {
        "input_a": "mu_hw_input_stencil",
        "input_c": "hw_residual_input_stencil",
        "output": "hw_output_stencil",
    },
    "mu_x_const_plus_vector_relu": {
        "input_a": "mu_hw_input_stencil",
        "input_c": "hw_residual_input_stencil",
        "output": "hw_output_stencil",
    },
}


def _mode_has_mul(mode: str) -> bool:
    return mode in ("mu_x_const_relu", "mu_x_const_plus_vector_relu")


def _mode_has_add(mode: str) -> bool:
    return mode in ("mu_plus_vector_relu", "mu_x_const_plus_vector_relu")


def _port_and_io_names(mode: str, i: int):
    """Return lane-i names as a dict with self-port and IO-instance names for each operand."""
    names = _MODE_NAMES[mode]

    def self_port(base, suffix):
        return f"{base}_clkwrk_{i}_op_hcompute_{base}_{i}_{suffix}_0"

    out = {
        "input_a_self": self_port(names["input_a"], "read"),
        "output_self": self_port(names["output"], "write"),
    }
    out["input_a_io"] = f"io16in_{out['input_a_self']}"
    out["output_io"] = f"io16_{out['output_self']}"
    if names["input_c"] is not None:
        out["input_c_self"] = self_port(names["input_c"], "read")
        out["input_c_io"] = f"io16in_{out['input_c_self']}"
    else:
        out["input_c_self"] = None
        out["input_c_io"] = None
    return out


def _elementwise_mul_add_relu_bf16_interface_type(context, unroll: int, mode: str):
    """Per-lane ports: input_A, optional input_C, output."""
    record = {}
    for i in range(unroll):
        ports = _port_and_io_names(mode, i)
        record[ports["input_a_self"]] = context.Array(16, context.BitIn())
        if ports["input_c_self"] is not None:
            record[ports["input_c_self"]] = context.Array(16, context.BitIn())
        record[ports["output_self"]] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_elementwise_mul_add_relu_bf16_graph(unroll: int, mode: str):
    """Structural construction: 2 or 3 PEs and 2 or 3 IOs per lane. Returns (context, module, instances_dict)."""
    has_mul = _mode_has_mul(mode)
    has_add = _mode_has_add(mode)

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    top = global_namespace.new_module(
        "elementwise_mul_add_relu_bf16",
        _elementwise_mul_add_relu_bf16_interface_type(context, unroll, mode),
    )
    defn = top.new_definition()
    iface = defn.interface

    io_input_a_list = []
    io_input_c_list = []
    io_output_list = []
    mul_pe_list = []
    add_pe_list = []
    max_pe_list = []

    for i in range(unroll):
        ports = _port_and_io_names(mode, i)

        input_a_io = defn.add_module_instance(ports["input_a_io"], io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(ports["output_io"], io_module, context.new_values({"mode": "out"}))
        max_pe = defn.add_module_instance(f"max_pe_{i}", pe_module)

        io_input_a_list.append(input_a_io)
        io_output_list.append(output_io)
        max_pe_list.append(max_pe)

        defn.connect(iface.select(ports["input_a_self"]), input_a_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(ports["output_self"]))

        if ports["input_c_io"] is not None:
            input_c_io = defn.add_module_instance(ports["input_c_io"], io_module, context.new_values({"mode": "in"}))
            io_input_c_list.append(input_c_io)
            defn.connect(iface.select(ports["input_c_self"]), input_c_io.select("in"))

        if has_mul:
            mul_pe = defn.add_module_instance(f"mul_pe_{i}", pe_module)
            mul_pe_list.append(mul_pe)
            defn.connect(input_a_io.select("out"), mul_pe.select("data0"))

        if has_add:
            add_pe = defn.add_module_instance(f"add_pe_{i}", pe_module)
            add_pe_list.append(add_pe)
            if has_mul:
                # (MU * const) + GLB: input_C -> add.data0; mul.O0 -> add.data1 (matches mul_add convention).
                defn.connect(input_c_io.select("out"), add_pe.select("data0"))
                defn.connect(mul_pe.select("O0"), add_pe.select("data1"))
            else:
                # MU + GLB: input_A -> add.data0; input_C -> add.data1.
                defn.connect(input_a_io.select("out"), add_pe.select("data0"))
                defn.connect(input_c_io.select("out"), add_pe.select("data1"))
            defn.connect(add_pe.select("O0"), max_pe.select("data0"))
        else:
            # No add stage: feed mul.O0 directly into max.
            defn.connect(mul_pe.select("O0"), max_pe.select("data0"))

        defn.connect(max_pe.select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "io_input_a": io_input_a_list,
        "io_output": io_output_list,
        "max_pe": max_pe_list,
    }
    if has_mul:
        instances["mul_pe"] = mul_pe_list
    if has_add:
        instances["add_pe"] = add_pe_list
        instances["io_input_c"] = io_input_c_list
    return context, top, instances


def _configure_elementwise_mul_add_relu_bf16(context, instances, unroll: int, tensor_size: int, mode: str, mul_const_val_bf16: float):
    """Wire PE instruction constants and set IO metadata per lane."""
    if unroll <= 0 or tensor_size % unroll != 0:
        raise ValueError(f"tensor_size ({tensor_size}) must be divisible by unroll ({unroll})")
    extent = tensor_size // unroll
    has_mul = _mode_has_mul(mode)
    has_add = _mode_has_add(mode)

    defn = instances["max_pe"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    if has_mul:
        mul_inst_val, mul_inst_w = build_pe_inst.pe_inst_to_bits_with_operands(
            "fp_mul", data0=("ext", None), data1=("const", build_pe_inst.bf16_bits_from_float(mul_const_val_bf16))
        )
    if has_add:
        add_inst_val, add_inst_w = build_pe_inst.pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))
    # max(x, 0.0) implements ReLU.
    max_inst_val, max_inst_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "fp_max", data0=("ext", None), data1=("const", build_pe_inst.bf16_bits_from_float(0.0))
    )

    if has_mul:
        for i, mul_pe in enumerate(instances["mul_pe"]):
            c = defn.add_generator_instance(
                f"const_inst_mul_pe_{i}",
                const_generator,
                context.new_values({"width": mul_inst_w}),
                context.new_values({"value": BitVector[mul_inst_w](mul_inst_val)}),
            )
            defn.connect(c.select("out"), mul_pe.select("inst"))
    if has_add:
        for i, add_pe in enumerate(instances["add_pe"]):
            c = defn.add_generator_instance(
                f"const_inst_add_pe_{i}",
                const_generator,
                context.new_values({"width": add_inst_w}),
                context.new_values({"value": BitVector[add_inst_w](add_inst_val)}),
            )
            defn.connect(c.select("out"), add_pe.select("inst"))
    for i, max_pe in enumerate(instances["max_pe"]):
        c = defn.add_generator_instance(
            f"const_inst_max_pe_{i}",
            const_generator,
            context.new_values({"width": max_inst_w}),
            context.new_values({"value": BitVector[max_inst_w](max_inst_val)}),
        )
        defn.connect(c.select("out"), max_pe.select("inst"))

    glb2out = json.dumps(
        {
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [extent],
            "read_data_starting_addr": [0],
            "read_data_stride": [1],
        }
    )
    in2glb = json.dumps(
        {
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [extent],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        }
    )

    for io_in in instances["io_input_a"]:
        io_in.add_metadata("glb2out_0", glb2out)
    if has_add:
        for io_in in instances["io_input_c"]:
            io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["io_output"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_mul_add_relu_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
    mode: str = DEFAULT_MODE,
    mul_const_val_bf16: float = DEFAULT_MUL_CONST_VAL_BF16,
):
    """
    Build the elementwise max((A * B) + C, 0) design.

    Args:
        unroll: Number of parallel lanes (= glb_i = glb_o).
        tensor_size: Total tensor size; each IO extent is tensor_size // unroll.
        mode: one of _VALID_MODES.
        mul_const_val_bf16: constant mul operand (used in *_x_const_* modes).

    Returns:
        (context, module)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    context, top, instances = _build_elementwise_mul_add_relu_bf16_graph(unroll, mode)
    _configure_elementwise_mul_add_relu_bf16(context, instances, unroll, tensor_size, mode, mul_const_val_bf16)
    return context, top


def emit_elementwise_mul_add_relu_bf16_design(
    unroll: int, tensor_size: int, output_path: str, mode: str = DEFAULT_MODE, mul_const_val_bf16: float = DEFAULT_MUL_CONST_VAL_BF16
):
    """Build and write the elementwise_mul_add_relu_bf16 design_top.json."""
    context, top = build_elementwise_mul_add_relu_bf16_context(unroll, tensor_size, mode, mul_const_val_bf16)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_add_relu_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_mul_add_relu_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--tensor-size", type=int, default=DEFAULT_TENSOR_SIZE)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=list(_VALID_MODES))
    parser.add_argument("--mul-const-val-bf16", type=float, default=DEFAULT_MUL_CONST_VAL_BF16)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_add_relu_bf16_design(args.unroll, args.tensor_size, args.output_path, args.mode, args.mul_const_val_bf16)
