"""
Build the elementwise mul CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of lanes (PEs and IO pairs).
- vector_len: total vector size; each IO has extent vector_len // unroll.
- mode:
    "vector_x_const"  (generic-named input * constant)
    "vector_x_vector" (generic-named input_A * input_B)
    "mu_x_const"      (Halide-named MU input * constant, for zircon_dequant_fp)
- mul_const_val_bf16: constant operand for the multiply PE instruction (used in vector_x_const / mu_x_const).
"""

import os
import json
from pathlib import Path

import numpy as np

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
import strait.coreir_backend.utils.build_pe_inst as build_pe_inst

HEADERS_DIR = list(headers_pkg.__path__)[0]

# Default configuration values
DEFAULT_UNROLL = 8
DEFAULT_VECTOR_LEN = 4096
DEFAULT_MUL_CONST_VAL_BF16 = 2.0
DEFAULT_MODE = "vector_x_const"

_VALID_MODES = ("vector_x_const", "vector_x_vector", "mu_x_const")


def _lane_port_names(mode: str, i: int):
    """
    Return (input_a_self, input_b_self_or_None, output_self, input_a_io, input_b_io_or_None, output_io) for lane i.

    - mu_x_const: Halide-style logical names so design_meta_halide.json alignment holds
      (mu_hw_input_stencil -> MU_ prefix via add_mu_prefix_to_io; hw_output_stencil).
    - vector_x_const / vector_x_vector: legacy generic names (input_stencil / input_B_stencil / output_stencil).
    """
    if mode == "mu_x_const":
        a = "mu_hw_input_stencil"
        o = "hw_output_stencil"
        input_a_self = f"{a}_clkwrk_{i}_op_hcompute_{a}_{i}_read_0"
        output_self = f"{o}_clkwrk_{i}_op_hcompute_{o}_{i}_write_0"
        return input_a_self, None, output_self, f"io16in_{input_a_self}", None, f"io16_{output_self}"

    input_a_self = f"lane_{i}_in"
    output_self = f"lane_{i}_out"
    input_a_io = f"io16in_input_stencil_clkwrk_{i}_op_hcompute_stencil_{i}_read_0"
    output_io = f"io16_output_stencil_clkwrk_{i}_op_hcompute_stencil_{i}_write_0"
    if mode == "vector_x_vector":
        input_b_self = f"lane_{i}_in_B"
        input_b_io = f"io16in_input_B_stencil_clkwrk_{i}_op_hcompute_stencil_{i}_read_0"
        return input_a_self, input_b_self, output_self, input_a_io, input_b_io, output_io
    return input_a_self, None, output_self, input_a_io, None, output_io


def _elementwise_mul_bf16_interface_type(context, unroll: int, mode: str):
    """
    Define interface type for top-level elementwise_mul module.
    One input and one output port per lane; when vector_x_vector, a second input port per lane.
    """
    record = {}
    for i in range(unroll):
        input_a_self, input_b_self, output_self, *_ = _lane_port_names(mode, i)
        record[input_a_self] = context.Array(16, context.BitIn())
        if input_b_self is not None:
            record[input_b_self] = context.Array(16, context.BitIn())
        record[output_self] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_elementwise_mul_bf16_graph(unroll: int, mode: str):
    """
    Pure structural construction: create modules, instances, and connections
    for unroll lanes.

    When mode is vector_x_vector, a second set of input IOs feeds pe.data1.

    Returns (context, elementwise_mul_module, instances_dict).
    instances_dict has keys "io_in", "io_out", "pe", and optionally "io_in_B"
    """
    has_second_input = mode == "vector_x_vector"

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    elementwise_mul_bf16_module = global_namespace.new_module(
        "elementwise_mul_bf16", _elementwise_mul_bf16_interface_type(context, unroll, mode)
    )
    elementwise_mul_bf16_definition = elementwise_mul_bf16_module.new_definition()
    elementwise_mul_bf16_interface = elementwise_mul_bf16_definition.interface

    io_in_list = []
    io_in_B_list = []  # second input vector, only used when vector_x_vector
    io_out_list = []
    pe_list = []

    for i in range(unroll):
        input_a_self, input_b_self, output_self, io_in_name, io_in_B_name, io_out_name = _lane_port_names(mode, i)
        pe_name = f"pe_lane{i}"

        # Define instances
        io_in = elementwise_mul_bf16_definition.add_module_instance(
            io_in_name, io_module, context.new_values({"mode": "in"})
        )
        if has_second_input:
            io_in_B = elementwise_mul_bf16_definition.add_module_instance(
                io_in_B_name, io_module, context.new_values({"mode": "in"})
            )
            io_in_B_list.append(io_in_B)
        io_out = elementwise_mul_bf16_definition.add_module_instance(
            io_out_name, io_module, context.new_values({"mode": "out"})
        )
        pe_inst = elementwise_mul_bf16_definition.add_module_instance(pe_name, pe_module)

        io_in_list.append(io_in)
        io_out_list.append(io_out)
        pe_list.append(pe_inst)

        # Connect the instances.
        elementwise_mul_bf16_definition.connect(
            elementwise_mul_bf16_interface.select(input_a_self), io_in.select("in")
        )
        elementwise_mul_bf16_definition.connect(io_in.select("out"), pe_inst.select("data0"))
        if has_second_input:
            elementwise_mul_bf16_definition.connect(
                elementwise_mul_bf16_interface.select(input_b_self), io_in_B.select("in")
            )
            elementwise_mul_bf16_definition.connect(io_in_B.select("out"), pe_inst.select("data1"))
        elementwise_mul_bf16_definition.connect(pe_inst.select("O0"), io_out.select("in"))
        elementwise_mul_bf16_definition.connect(
            io_out.select("out"), elementwise_mul_bf16_interface.select(output_self)
        )

    elementwise_mul_bf16_module.definition = elementwise_mul_bf16_definition
    context.set_top(elementwise_mul_bf16_module)

    instances = {
        "io_in": io_in_list,
        "io_out": io_out_list,
        "pe": pe_list,
    }
    if has_second_input:
        instances["io_in_B"] = io_in_B_list
    return context, elementwise_mul_bf16_module, instances


def _configure_elementwise_mul_bf16(
    context,
    instances,
    unroll: int,
    vector_len: int,
    mode: str,
    mul_const_val_bf16: float,
):
    """Configuration step: PE instruction (vector x const, vector x vector, or mu x const) and IO metadata per lane."""
    if unroll <= 0 or vector_len % unroll != 0:
        raise ValueError(f"vector_len ({vector_len}) must be divisible by unroll ({unroll})")
    extent = vector_len // unroll
    has_second_input = mode == "vector_x_vector"

    io_in_list = instances["io_in"]
    io_out_list = instances["io_out"]
    pe_list = instances["pe"]

    if has_second_input:
        mul_bf16_inst_val, inst_width = build_pe_inst.pe_inst_to_bits_with_operands(
            "fp_mul", data0=("ext", None), data1=("ext", None)
        )
    else:
        mul_bf16_inst_val, inst_width = build_pe_inst.pe_inst_to_bits_with_operands(
            "fp_mul", data1=("const", build_pe_inst.bf16_bits_from_float(mul_const_val_bf16))
        )

    elementwise_mul_bf16_definition = pe_list[0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    for i, pe_inst in enumerate(pe_list):
        const_name = f"const_lane{i}"
        pe_inst_const = elementwise_mul_bf16_definition.add_generator_instance(
            const_name,
            const_generator,
            context.new_values({"width": inst_width}),
            context.new_values({"value": BitVector[inst_width](mul_bf16_inst_val)}),
        )
        elementwise_mul_bf16_definition.connect(pe_inst_const.select("out"), pe_inst.select("inst"))

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
    for io_in in io_in_list:
        io_in.add_metadata("glb2out_0", glb2out)
    if has_second_input:
        for io_in_B in instances["io_in_B"]:
            io_in_B.add_metadata("glb2out_0", glb2out)
    for io_out in io_out_list:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_mul_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vector_len: int = DEFAULT_VECTOR_LEN,
    mode: str = DEFAULT_MODE,
    mul_const_val_bf16: float = DEFAULT_MUL_CONST_VAL_BF16,
):
    """
    Build the elementwise_mul design with templated parameters.

    Args:
        unroll: Number of lanes (PEs and IO pairs).
        vector_len: Total vector size; each IO has extent vector_len // unroll.
        mode: one of "vector_x_const", "vector_x_vector", "mu_x_const".
        mul_const_val_bf16: Constant operand for the multiply PE instruction (used when mode is vector_x_const or mu_x_const).

    Returns:
        (context, elementwise_mul_module)
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    context, elementwise_mul_bf16_module, instances = _build_elementwise_mul_bf16_graph(unroll, mode)
    _configure_elementwise_mul_bf16(
        context, instances, unroll, vector_len, mode, mul_const_val_bf16
    )
    return context, elementwise_mul_bf16_module


def emit_elementwise_mul_bf16_design(unroll: int, tensor_size: int, output_path: str, mode: str = DEFAULT_MODE, mul_const_val_bf16: float = DEFAULT_MUL_CONST_VAL_BF16):
    """Build and write the elementwise_mul_bf16 design_top.json. Thin wrapper for hack_design_top callers."""
    context, mul_module = build_elementwise_mul_bf16_context(unroll, tensor_size, mode, mul_const_val_bf16)
    out_file = os.path.join(output_path, "design_top.json")
    mul_module.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_bf16 design_top.json to {out_file}")
    return out_file

def emit_elementwise_mul_bf16_coreir_json(
    kernel: dict,
    output_path: str,
    unroll: int = DEFAULT_UNROLL,
):
    kernel_inputs = kernel.get("inputs", {})
    assert 1 <= len(kernel_inputs) <= 2, "Elementwise mul BF16 kernel must have 1 or 2 inputs."
    vector_len = None
    mul_const_val_bf16 = None
    # Determine the mode based on whether one of the inputs is a constant.
    # Tensor constants may be absent from kernel_inputs (filtered out of scheduled_ops),
    # in which case we fall back to scanning output_path for staged constant raw files.
    mode = "vector_x_vector"
    for input_type, input in kernel_inputs.items():
        node_name = input.get("node")
        if "_tensor_constant" in node_name and input.get("shape") == [1]:
            mode = "vector_x_const"
            const_tensor_raw = os.path.join(output_path, "input_" + node_name + ".raw")
            const_tensor_npy = (np.fromfile(const_tensor_raw, dtype=np.uint16).byteswap().astype(np.uint32) << 16).view(np.float32)
            mul_const_val_bf16 = const_tensor_npy[0]
        else:
            vector_len = int(np.prod(input.get("shape")))

    if mode == "vector_x_vector" and len(kernel_inputs) == 1:
        # Constants were filtered out of scheduled_ops; look for staged constant raw files.
        const_files = sorted(Path(output_path).glob("input_*_tensor_constant*.raw"))
        if const_files:
            mode = "vector_x_const"
            const_tensor_npy = (np.fromfile(const_files[0], dtype=np.uint16).byteswap().astype(np.uint32) << 16).view(np.float32)
            mul_const_val_bf16 = const_tensor_npy[0]

    context, elementwise_mul_bf16_top = build_elementwise_mul_bf16_context(unroll, vector_len, mode, mul_const_val_bf16)
    elementwise_mul_bf16_top.save_to_file(os.path.join(output_path, "design_top.json"))
    print(f"[INFO] Wrote elementwise_mul_bf16_coreir_json to {os.path.join(output_path, 'design_top.json')}")


if __name__ == "__main__":
    context, elementwise_mul_bf16_module = build_elementwise_mul_bf16_context()
    elementwise_mul_bf16_module.save_to_file("elementwise_mul_bf16_coreir_json.json")
    print(f"[INFO] Wrote elementwise_mul_bf16_coreir_json to elementwise_mul_bf16_coreir_json.json")
