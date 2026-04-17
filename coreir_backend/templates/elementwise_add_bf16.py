"""
Build the elementwise add CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of lanes (PEs and IO pairs).
- tensor_size: total tensor size; each IO has extent tensor_size // unroll.
- mode: "mu_plus_glb" (one MU input + one GLB input, for zircon psum reduction)
        or "glb_plus_glb" (two plain GLB inputs, for generic vector add).

Per lane i:
  input_A IO.out -> pe.data0
  input_B IO.out -> pe.data1
  pe (fp_add) O0 -> output IO.in

Logical IO names (preserved between io16[in]_ and _clkwrk, so they match Halide's design_meta_halide.json):
- "mu_plus_glb":  input_A = mu_hw_input_stencil (gets MU_ prefix via add_mu_prefix_to_io),
                  input_B = hw_partial_sum_input_stencil,
                  output  = hw_output_stencil.
- "glb_plus_glb": input_A = input_A_stencil, input_B = input_B_stencil, output = output_stencil.
"""

import os
import json
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_TENSOR_SIZE = 50176
DEFAULT_MODE = "mu_plus_glb"

# Logical IO names per mode. Keys:
#   input_a / input_b: logical name used between io16in_ and _clkwrk (MU prefix keyed off _mu_ substring).
#   output:            logical name used between io16_ and _clkwrk.
_MODE_NAMES = {
    "mu_plus_glb": {
        "input_a": "mu_hw_input_stencil",
        "input_b": "hw_partial_sum_input_stencil",
        "output":  "hw_output_stencil",
    },
    "glb_plus_glb": {
        "input_a": "input_A_stencil",
        "input_b": "input_B_stencil",
        "output":  "output_stencil",
    },
}


def _port_names(mode: str, i: int):
    """Return (input_a_self, input_b_self, output_self, input_a_io, input_b_io, output_io) for lane i."""
    names = _MODE_NAMES[mode]
    a, b, o = names["input_a"], names["input_b"], names["output"]
    input_a_self = f"{a}_clkwrk_{i}_op_hcompute_{a}_{i}_read_0"
    input_b_self = f"{b}_clkwrk_{i}_op_hcompute_{b}_{i}_read_0"
    output_self = f"{o}_clkwrk_{i}_op_hcompute_{o}_{i}_write_0"
    return (
        input_a_self,
        input_b_self,
        output_self,
        f"io16in_{input_a_self}",
        f"io16in_{input_b_self}",
        f"io16_{output_self}",
    )


def _elementwise_add_bf16_interface_type(context, unroll: int, mode: str):
    """Define the top-level interface: two inputs and one output per lane."""
    record = {}
    for i in range(unroll):
        input_a_self, input_b_self, output_self, *_ = _port_names(mode, i)
        record[input_a_self] = context.Array(16, context.BitIn())
        record[input_b_self] = context.Array(16, context.BitIn())
        record[output_self] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_elementwise_add_bf16_graph(unroll: int, mode: str):
    """Structural construction: IO instances and fp_add PE per lane. Returns (context, module, instances_dict)."""
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    add_module = global_namespace.new_module(
        "elementwise_add_bf16",
        _elementwise_add_bf16_interface_type(context, unroll, mode),
    )
    add_defn = add_module.new_definition()
    add_iface = add_defn.interface

    io_input_a_list = []
    io_input_b_list = []
    io_output_list = []
    pe_list = []

    for i in range(unroll):
        input_a_self, input_b_self, output_self, input_a_io_name, input_b_io_name, output_io_name = _port_names(mode, i)

        input_a_io = add_defn.add_module_instance(input_a_io_name, io_module, context.new_values({"mode": "in"}))
        input_b_io = add_defn.add_module_instance(input_b_io_name, io_module, context.new_values({"mode": "in"}))
        output_io = add_defn.add_module_instance(output_io_name, io_module, context.new_values({"mode": "out"}))
        pe_inst = add_defn.add_module_instance(f"add_pe_{i}", pe_module)

        io_input_a_list.append(input_a_io)
        io_input_b_list.append(input_b_io)
        io_output_list.append(output_io)
        pe_list.append(pe_inst)

        # --- Self <-> IO ---
        add_defn.connect(add_iface.select(input_a_self), input_a_io.select("in"))
        add_defn.connect(add_iface.select(input_b_self), input_b_io.select("in"))
        add_defn.connect(output_io.select("out"), add_iface.select(output_self))

        # --- Input IOs -> PE -> Output IO ---
        add_defn.connect(input_a_io.select("out"), pe_inst.select("data0"))
        add_defn.connect(input_b_io.select("out"), pe_inst.select("data1"))
        add_defn.connect(pe_inst.select("O0"), output_io.select("in"))

    add_module.definition = add_defn
    context.set_top(add_module)

    instances = {
        "io_input_a": io_input_a_list,
        "io_input_b": io_input_b_list,
        "io_output": io_output_list,
        "pe": pe_list,
    }
    return context, add_module, instances


def _configure_elementwise_add_bf16(context, instances, unroll: int, tensor_size: int):
    """Configuration step: wire fp_add PE instruction constants and set IO metadata."""
    extent = tensor_size // unroll
    add_defn = instances["pe"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    fp_add_val, fp_add_w = pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))

    for i, pe_inst in enumerate(instances["pe"]):
        pe_inst_const = add_defn.add_generator_instance(
            f"const_inst_add_pe_{i}",
            const_generator,
            context.new_values({"width": fp_add_w}),
            context.new_values({"value": BitVector[fp_add_w](fp_add_val)}),
        )
        add_defn.connect(pe_inst_const.select("out"), pe_inst.select("inst"))

    glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for io_in in instances["io_input_a"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_in in instances["io_input_b"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["io_output"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_add_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
    mode: str = DEFAULT_MODE,
):
    """
    Build the elementwise add design with templated parameters.

    Args:
        unroll: Number of parallel lanes (= glb_i = glb_o).
        tensor_size: Total tensor size.
        mode: "mu_plus_glb" (psum reduction) or "glb_plus_glb" (generic vector add).

    Returns:
        (context, add_module)
    """
    if mode not in _MODE_NAMES:
        raise ValueError(f"mode must be one of {list(_MODE_NAMES)}, got {mode!r}")

    context, add_module, instances = _build_elementwise_add_bf16_graph(unroll, mode)
    _configure_elementwise_add_bf16(context, instances, unroll, tensor_size)
    return context, add_module


def emit_elementwise_add_bf16_design(unroll: int, tensor_size: int, output_path: str, mode: str = DEFAULT_MODE):
    """Build and write the elementwise_add_bf16 design_top.json."""
    context, add_module = build_elementwise_add_bf16_context(unroll, tensor_size, mode)
    out_file = os.path.join(output_path, "design_top.json")
    add_module.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_add_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate elementwise_add_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--tensor-size", type=int, default=DEFAULT_TENSOR_SIZE)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=list(_MODE_NAMES))
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_add_bf16_design(args.unroll, args.tensor_size, args.output_path, args.mode)
