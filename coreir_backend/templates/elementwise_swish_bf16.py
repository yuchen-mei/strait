"""
Build the elementwise Swish CoreIR graph using pycoreir.

Templated design parameters:
- beta: the beta scale inside sigmoid
- unroll: number of lanes (PEs and IO pairs).
- vector_len: total vector size; each IO has extent vector_len // unroll.

Swish(x) = x * sigmoid(βx) = x / (1 + exp(-βx))
"""

import math
import os
import json
from pathlib import Path

import numpy as np

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import (
    bf16_bits_from_float,
    pe_inst_to_bits_with_operands,
)
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 8
DEFAULT_VECTOR_LEN = 4096
DEFAULT_BETA = 1.0

# Ordered list of PE operation types per lane used for instance naming and wiring
_PE_TYPES = [
    "mul_a",  # fp_mul: x * (-beta)
    "mul_b",  # fp_mul: (-beta*x) * (1/ln2) = -beta*x/ln2
    "getffrac",  # fp_getffrac: fractional part for exp ROM lookup
    "getfint",  # fp_getfint: integer part for addiexp
    "addiexp",  # fp_addiexp: reconstruct exp(-beta*x)
    "add",  # fp_add: 1 + exp(-beta*x)
    "dummy_max_nop",  # fp_max: max(1+exp(-beta*x), -inf) = 1+exp(-beta*x), feeds subexp.data1
    "getmant",  # fp_getmant: mantissa index for div ROM lookup
    "subexp",  # fp_subexp: reconstruct 1/(1+exp(-beta*x)) = sigmoid(beta*x)
    "mul_final",  # fp_mul: x * sigmoid(beta*x) = Swish(x)
]


def _compute_exp_rom_table():
    """
    Exponential ROM: maps 8-bit index to 2^(index/128) in BF16.
    Entries 0..127:   2^(i/128) for i in [0, 127]    -> values in [1.0, ~2.0)
    Entries 128..255: 2^(i/128) for i in [-128, -1]  -> values in [0.5, ~1.0)
    """
    return [bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


def _compute_div_rom_table():
    """
    Division ROM: maps 8-bit mantissa index to 1/(1 + index/128) in BF16.
    Entries 0..127:   1/(1 + i/128) for i in [0, 127]  -> values in [0.5, 1.0]
    """
    return [bf16_bits_from_float(1.0 / (1.0 + i / 128.0)) for i in range(128)]


def _compute_swish_pe_instructions(beta: float):
    """
    Compute PE instructions for all Swish pipeline stages.
    """
    neg_beta_bf16 = bf16_bits_from_float(-beta)
    ln2_inv_bf16 = bf16_bits_from_float(1.0 / math.log(2))
    one_bf16 = bf16_bits_from_float(1.0)
    neg_inf_bf16 = bf16_bits_from_float(float("-inf"))

    return {
        "mul_a": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("const", neg_beta_bf16)),
        "mul_b": pe_inst_to_bits_with_operands("fp_mul", data0=("const", ln2_inv_bf16), data1=("ext", None)),
        "getffrac": pe_inst_to_bits_with_operands("fp_getffrac", data0=("ext", None)),
        "getfint":  pe_inst_to_bits_with_operands("fp_getfint",  data0=("ext", None)),
        "addiexp": pe_inst_to_bits_with_operands("fp_addiexp", data0=("ext", None), data1=("ext", None)),
        "add": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("const", one_bf16)),
        "dummy_max_nop": pe_inst_to_bits_with_operands("fp_max", data0=("ext", None), data1=("const", neg_inf_bf16)),
        "getmant": pe_inst_to_bits_with_operands("fp_getmant", data0=("ext", None)),
        "subexp": pe_inst_to_bits_with_operands("fp_subexp", data0=("ext", None), data1=("ext", None)),
        "mul_final": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
    }


def _elementwise_swish_bf16_interface_type(context, unroll: int):
    """One IO input and one IO output port per lane."""
    record = {}
    for i in range(unroll):
        record[f"lane_{i}_in"] = context.Array(16, context.BitIn())
        record[f"lane_{i}_out"] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_elementwise_swish_bf16_graph(unroll: int, input_name: str = "input", output_name: str = "output"):
    """
    Structural construction for swish compute graph
    """
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    swish_module = global_namespace.new_module(
        "elementwise_swish_bf16",
        _elementwise_swish_bf16_interface_type(context, unroll),
    )
    swish_defn = swish_module.new_definition()
    swish_iface = swish_defn.interface

    io_in_list = []
    io_out_list = []
    pe_by_type = {k: [] for k in _PE_TYPES}
    exp_rom_list = []
    div_rom_list = []

    rom_genargs = make_mem_genargs(context)
    # Input buffer MEM: one write port (from IO), two read ports (to mul_a and mul_final).
    # Acts as a delay-matching FIFO in RV mode; RV schedule config added in _configure_.
    # The metadata overrides mode to 'lake' (UB mode) and sets is_rom=False so the backend
    # routes and configures it as a regular MEM.
    input_buf_genargs = make_mem_genargs(context)
    exp_rom_table = _compute_exp_rom_table()
    div_rom_table = _compute_div_rom_table()

    input_buf_list = []

    for i in range(unroll):
        io_in_name = f"io16in_{input_name}_stencil_clkwrk_{i}_op_hcompute_stencil_{i}_read_0"
        io_out_name = f"io16_output_{output_name}_stencil_clkwrk_{i}_op_hcompute_stencil_{i}_write_0"

        io_in = swish_defn.add_module_instance(io_in_name, io_module, context.new_values({"mode": "in"}))
        io_out = swish_defn.add_module_instance(io_out_name, io_module, context.new_values({"mode": "out"}))
        io_in_list.append(io_in)
        io_out_list.append(io_out)

        lane_pes = {}
        for pe_type in _PE_TYPES:
            pe_inst = swish_defn.add_module_instance(f"pe_{pe_type}_lane{i}", pe_module)
            lane_pes[pe_type] = pe_inst
            pe_by_type[pe_type].append(pe_inst)

        exp_rom = swish_defn.add_generator_instance(
            f"exp_rom_lane{i}",
            mem_gen,
            rom_genargs,
            context.new_values({"config": {}, "init": exp_rom_table, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("depth", json.dumps(len(exp_rom_table))),
            ("init", json.dumps(exp_rom_table)),
            ("is_rom", json.dumps(True)),
            ("mode", json.dumps("sram")),
            ("width", json.dumps(16)),
        ]:
            exp_rom.add_metadata(_key, _val)

        div_rom = swish_defn.add_generator_instance(
            f"div_rom_lane{i}",
            mem_gen,
            rom_genargs,
            context.new_values({"config": {}, "init": div_rom_table, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("depth", json.dumps(len(div_rom_table))),
            ("init", json.dumps(div_rom_table)),
            ("is_rom", json.dumps(True)),
            ("mode", json.dumps("sram")),
            ("width", json.dumps(16)),
        ]:
            div_rom.add_metadata(_key, _val)

        exp_rom_list.append(exp_rom)
        div_rom_list.append(div_rom)

        # Input buffer MEM: buffers x for dual fan-out (mul_a and mul_final).
        # RV schedule config (lake_rv_config) is added in _configure_elementwise_swish_bf16.
        input_buf_mem = swish_defn.add_generator_instance(
            f"input_buf_lane{i}",
            mem_gen,
            input_buf_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            input_buf_mem.add_metadata(_key, _val)
        input_buf_list.append(input_buf_mem)

        # io_in <-> interface; x written into input buffer MEM
        swish_defn.connect(swish_iface.select(f"lane_{i}_in"), io_in.select("in"))
        swish_defn.connect(io_in.select("out"), input_buf_mem.select("data_in_0"))
        # MEM read port 0 -> mul_a (immediate consumer); read port 1 -> mul_final (delayed consumer)
        swish_defn.connect(input_buf_mem.select("data_out_0"), lane_pes["mul_a"].select("data0"))
        swish_defn.connect(input_buf_mem.select("data_out_1"), lane_pes["mul_final"].select("data1"))

        # x * (-beta) -> mul_b -> getffrac / getfint
        swish_defn.connect(lane_pes["mul_a"].select("O0"), lane_pes["mul_b"].select("data1"))
        swish_defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getffrac"].select("data0"))
        swish_defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getfint"].select("data0"))

        # getffrac -> exp_rom -> addiexp; getfint -> addiexp
        swish_defn.connect(lane_pes["getffrac"].select("O0"), exp_rom.select("addr_in_0"))
        swish_defn.connect(exp_rom.select("data_out_0"), lane_pes["addiexp"].select("data0"))
        swish_defn.connect(lane_pes["getfint"].select("O0"), lane_pes["addiexp"].select("data1"))

        # addiexp -> add -> getmant + dummy_max_nop
        swish_defn.connect(lane_pes["addiexp"].select("O0"), lane_pes["add"].select("data0"))
        swish_defn.connect(lane_pes["add"].select("O0"), lane_pes["getmant"].select("data0"))
        swish_defn.connect(lane_pes["add"].select("O0"), lane_pes["dummy_max_nop"].select("data0"))

        # getmant -> div_rom -> subexp; dummy_max_nop -> subexp
        swish_defn.connect(lane_pes["getmant"].select("O0"), div_rom.select("addr_in_0"))
        swish_defn.connect(div_rom.select("data_out_0"), lane_pes["subexp"].select("data0"))
        swish_defn.connect(lane_pes["dummy_max_nop"].select("O0"), lane_pes["subexp"].select("data1"))

        # subexp -> mul_final (data1=x already connected above)
        swish_defn.connect(lane_pes["subexp"].select("O0"), lane_pes["mul_final"].select("data0"))

        # mul_final -> io_out <-> interface
        swish_defn.connect(lane_pes["mul_final"].select("O0"), io_out.select("in"))
        swish_defn.connect(io_out.select("out"), swish_iface.select(f"lane_{i}_out"))

    swish_module.definition = swish_defn
    context.set_top(swish_module)

    instances = {
        "io_in": io_in_list,
        "io_out": io_out_list,
        "pe": pe_by_type,
        "exp_rom": exp_rom_list,
        "div_rom": div_rom_list,
        "input_buf": input_buf_list,
    }
    return context, swish_module, instances


def _configure_elementwise_swish_bf16(context, instances, unroll: int, vector_len: int, beta: float):
    """Configuration: wire PE instruction constants and IO metadata."""
    if unroll <= 0 or vector_len % unroll != 0:
        raise ValueError(f"vector_len ({vector_len}) must be divisible by unroll ({unroll})")
    extent = vector_len // unroll

    pe_by_type = instances["pe"]
    swish_defn = pe_by_type["mul_final"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    pe_instructions = _compute_swish_pe_instructions(beta)
    for pe_type, (inst_val, inst_width) in pe_instructions.items():
        for i, pe_inst in enumerate(pe_by_type[pe_type]):
            const_inst = swish_defn.add_generator_instance(
                f"const_inst_{pe_type}_lane{i}",
                const_generator,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            swish_defn.connect(const_inst.select("out"), pe_inst.select("inst"))

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
    for io_in in instances["io_in"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["io_out"]:
        io_out.add_metadata("in2glb_0", in2glb)

    # Embed the RV schedule config directly in the coreir JSON so the lake backend
    lake_rv_cfg = json.dumps({"type": "dual_read", "input_stream_size": extent})
    for input_buf_mem in instances["input_buf"]:
        input_buf_mem.add_metadata("lake_rv_config", lake_rv_cfg)


def build_elementwise_swish_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vector_len: int = DEFAULT_VECTOR_LEN,
    beta: float = DEFAULT_BETA,
    input_name: str = "input",
    output_name: str = "output",
):
    """
    Build the elementwise Swish design.

    Args:
        unroll: Number of parallel lanes (PEs and IO pairs).
        vector_len: Total vector size; each IO has extent vector_len // unroll.
        beta: Scale factor inside sigmoid; Swish(x) = x * sigmoid(beta * x).
        input_name: Tensor name embedded in input IO instance names (must match
                    the input key in scheduled_ops.json for IO placement).
        output_name: Tensor name embedded in output IO instance names (must match
                     the output node name in scheduled_ops.json for IO placement).

    Returns:
        (context, elementwise_swish_module)
    """
    context, swish_module, instances = _build_elementwise_swish_bf16_graph(unroll, input_name, output_name)
    _configure_elementwise_swish_bf16(context, instances, unroll, vector_len, beta)
    return context, swish_module


def emit_elementwise_swish_bf16_coreir_json(
    kernel: dict,
    output_path: str,
    unroll: int = DEFAULT_UNROLL,
    beta: float = DEFAULT_BETA,
):
    kernel_inputs = kernel.get("inputs", {})
    kernel_outputs = kernel.get("outputs", {})
    assert len(kernel_inputs) == 1, "Elementwise swish BF16 kernel must have 1 input."

    # Extract tensor names from the kernel so coreir IO instance names match
    # the keys used in scheduled_ops.json, enabling IO placement by name matching.
    input_name = next(iter(kernel_inputs))
    output_name = next(iter(kernel_outputs.values())).get("node")

    for _, input_info in kernel_inputs.items():
        vector_len = int(np.prod(input_info.get("shape")))

    context, swish_top = build_elementwise_swish_bf16_context(unroll, vector_len, beta, input_name, output_name)
    swish_top.save_to_file(os.path.join(output_path, "design_top.json"))
    print(f"[INFO] Wrote elementwise_swish_bf16_coreir_json to {os.path.join(output_path, 'design_top.json')}")


if __name__ == "__main__":
    context, swish_module = build_elementwise_swish_bf16_context()
    swish_module.save_to_file("elementwise_swish_bf16_coreir.json")
    print("[INFO] Wrote elementwise_swish_bf16_coreir.json")
