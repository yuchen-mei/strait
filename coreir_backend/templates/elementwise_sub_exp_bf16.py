"""
Build the elementwise exp(input - vec_max) CoreIR graph using pycoreir.

Used by: stable_softmax_pass2_fp — pass 2 of stable softmax, which computes the
unnormalized numerator exp(x - max(x)). vec_max is broadcast across x via a
single shared IO; Halide's unrolled clkwrk IOs are consolidated into one here.

Templated design parameters:
- unroll: number of parallel input/output lanes (= glb_i).
- vec_length: inner vector length per row (= vec_width from halide args).
- num_vecs: number of rows (= vec_height from halide args).

Per lane i:
  input_host IO.out       -> sub_pe.data0
  shared vec_max IO.out   -> sub_pe.data1  (broadcast to every lane)
  sub_pe.O0               -> mul_b.data1              ; diff * (1/ln2)
  mul_b.O0                -> getffrac.data0, getfint.data0
  getffrac.O0             -> exp_rom.addr_in_0
  exp_rom.data_out_0      -> addiexp.data0
  getfint.O0              -> addiexp.data1
  addiexp.O0              -> hw_output IO.in

5 PEs + 1 exp ROM per lane. No feedback, no pond; no MEM/Pond rv_config hack
needed.
"""

import json
import math
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import (
    bf16_bits_from_float,
    pe_inst_to_bits_with_operands,
)
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_VEC_LENGTH = 128
DEFAULT_NUM_VECS = 128
TOP_MODULE = "stable_softmax_pass2_fp"

_LANE_PE_ROLES = [
    "sub",       # fp_sub(input, vec_max)
    "mul_b",     # fp_mul(diff, 1/ln2)
    "getffrac",  # fp_getffrac(z)
    "getfint",   # fp_getfint(z)
    "addiexp",   # fp_addiexp(exp_rom[frac], int) = exp(diff)
]


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _vec_max_io_name() -> str:
    # Single shared vec_max IO; use clkwrk_0 so the logical name is preserved.
    return "io16in_vec_max_host_stencil_clkwrk_0_op_hcompute_vec_max_glb_stencil_read_0"


def _vec_max_self_port() -> str:
    return "vec_max_host_stencil_clkwrk_0_op_hcompute_vec_max_glb_stencil_read_0"


def _output_io_name(i: int) -> str:
    return f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _compute_exp_rom_table():
    """
    Same ROM layout as elementwise_swish_bf16: 256 entries indexed by an 8-bit
    frac byte, mapping to 2^(i/128) for i in [0,128) and then [-128,0).
    """
    return [bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


def _compute_pe_instructions():
    ln2_inv_bf16 = bf16_bits_from_float(1.0 / math.log(2))
    return {
        "sub": pe_inst_to_bits_with_operands("fp_sub", data0=("ext", None), data1=("ext", None)),
        "mul_b": pe_inst_to_bits_with_operands("fp_mul", data0=("const", ln2_inv_bf16), data1=("ext", None)),
        "getffrac": pe_inst_to_bits_with_operands("fp_getffrac", data0=("ext", None)),
        "getfint": pe_inst_to_bits_with_operands("fp_getfint", data0=("ext", None)),
        "addiexp": pe_inst_to_bits_with_operands("fp_addiexp", data0=("ext", None), data1=("ext", None)),
    }


def _interface_type(context, unroll: int):
    record = {_vec_max_self_port(): context.Array(16, context.BitIn())}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
    if unroll < 1:
        raise ValueError(f"unroll must be >= 1, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    top = global_namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    rom_genargs = make_mem_genargs(context)
    exp_rom_table = _compute_exp_rom_table()

    vec_max_io = defn.add_module_instance(
        _vec_max_io_name(), io_module, context.new_values({"mode": "in"})
    )
    defn.connect(iface.select(_vec_max_self_port()), vec_max_io.select("in"))

    input_io_list = []
    output_io_list = []
    pe_by_role = {k: [] for k in _LANE_PE_ROLES}
    exp_rom_list = []

    for i in range(unroll):
        input_io = defn.add_module_instance(
            _input_io_name(i), io_module, context.new_values({"mode": "in"})
        )
        output_io = defn.add_module_instance(
            _output_io_name(i), io_module, context.new_values({"mode": "out"})
        )
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        input_io_list.append(input_io)
        output_io_list.append(output_io)

        lane_pes = {}
        for role in _LANE_PE_ROLES:
            pe = defn.add_module_instance(f"{role}_pe_{i}", pe_module)
            lane_pes[role] = pe
            pe_by_role[role].append(pe)

        exp_rom = defn.add_generator_instance(
            f"exp_rom_lane{i}",
            mem_gen,
            rom_genargs,
            context.new_values({"config": {}, "init": exp_rom_table, "mode": "lake"}),
        )
        for _k, _v in [
            ("config", json.dumps({})),
            ("depth", json.dumps(len(exp_rom_table))),
            ("init", json.dumps(exp_rom_table)),
            ("is_rom", json.dumps(True)),
            ("mode", json.dumps("sram")),
            ("width", json.dumps(16)),
        ]:
            exp_rom.add_metadata(_k, _v)
        exp_rom_list.append(exp_rom)

        # sub_pe = input - vec_max
        defn.connect(input_io.select("out"), lane_pes["sub"].select("data0"))
        defn.connect(vec_max_io.select("out"), lane_pes["sub"].select("data1"))

        # diff * (1/ln2)  (1/ln2 baked as data0 const on mul_b)
        defn.connect(lane_pes["sub"].select("O0"), lane_pes["mul_b"].select("data1"))

        # Split frac / int
        defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getffrac"].select("data0"))
        defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getfint"].select("data0"))

        # ROM lookup on frac; addiexp reconstructs exp from mantissa + int
        defn.connect(lane_pes["getffrac"].select("O0"), exp_rom.select("addr_in_0"))
        defn.connect(exp_rom.select("data_out_0"), lane_pes["addiexp"].select("data0"))
        defn.connect(lane_pes["getfint"].select("O0"), lane_pes["addiexp"].select("data1"))

        # exp(diff) -> output
        defn.connect(lane_pes["addiexp"].select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "vec_max_io": vec_max_io,
        "input_io": input_io_list,
        "output_io": output_io_list,
        "pe": pe_by_role,
        "exp_rom": exp_rom_list,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")

    defn = instances["output_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    pe_instrs = _compute_pe_instructions()
    for role, (inst_val, inst_width) in pe_instrs.items():
        for i, pe_inst in enumerate(instances["pe"][role]):
            c = defn.add_generator_instance(
                f"const_inst_{role}_pe_{i}",
                const_gen,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe_inst.select("inst"))

    per_lane_extent = num_vecs * vec_length // unroll

    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    # vec_max is broadcast: stream num_vecs distinct values, each held constant
    # for vec_length/unroll cycles (one per col-block of the current row).
    vec_max_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [vec_length // unroll, num_vecs],
        "read_data_starting_addr": [0],
        "read_data_stride": [0, 1],
    })
    output_in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for io_in in instances["input_io"]:
        io_in.add_metadata("glb2out_0", input_glb2out)
    instances["vec_max_io"].add_metadata("glb2out_0", vec_max_glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", output_in2glb)


def build_elementwise_sub_exp_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
):
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vec_length, num_vecs)
    return context, top


def emit_elementwise_sub_exp_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
):
    context, top = build_elementwise_sub_exp_bf16_context(unroll, vec_length, num_vecs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_sub_exp_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_sub_exp_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_sub_exp_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path)
