"""
Build the elementwise input * weight + bias CoreIR graph using pycoreir.

Used by: layer_norm_pass3_fp — pass 3 of layer norm, which applies the learned
per-channel weight and bias to normalized input.
The separate 16-lane affine pass also preserves GLB bank accessibility through
the taped-out matrix-unit region's 16 horizontal switch-box tracks.

Templated design parameters:
- unroll: number of parallel lanes (= glb_i).
- vec_length: inner vector length per row (= vec_width from halide args).
- num_vecs: number of rows (= vec_height from halide args).
- buffer_outputs: decouple standalone GLB output lanes with MEM FIFOs;
  fused LayerNorm supplies its own buffering and leaves this disabled.

Per lane i:
  input_host IO.out   -> mul_vec_pe.data0
  weight_host IO.out  -> mul_vec_pe.data1
  bias_host IO.out    -> add_vec_pe.data0
  mul_vec_pe.O0       -> add_vec_pe.data1
  add_vec_pe.O0       -> hw_output IO.in

2 PEs per lane, 3 input IOs + 1 output IO per lane.
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import (
    pe_inst_to_bits_with_operands,
)
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_VEC_LENGTH = 384
DEFAULT_NUM_VECS = 128
TOP_MODULE = "layer_norm_pass3_fp"

_LANE_PE_ROLES = [
    "mul_vec",    # fp_mul(input, weight)
    "add_vec",    # fp_add(bias, mul_vec)
]


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _weight_io_name(i: int) -> str:
    return f"io16in_weight_host_stencil_clkwrk_{i}_op_hcompute_weight_glb_stencil{_stencil_suffix(i)}_read_0"


def _weight_self_port(i: int) -> str:
    return f"weight_host_stencil_clkwrk_{i}_op_hcompute_weight_glb_stencil{_stencil_suffix(i)}_read_0"


def _bias_io_name(i: int) -> str:
    return f"io16in_bias_host_stencil_clkwrk_{i}_op_hcompute_bias_glb_stencil{_stencil_suffix(i)}_read_0"


def _bias_self_port(i: int) -> str:
    return f"bias_host_stencil_clkwrk_{i}_op_hcompute_bias_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_io_name(i: int) -> str:
    return f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _compute_pe_instructions():
    return {
        "mul_vec": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
        "add_vec": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None)),
    }


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_weight_self_port(i)] = context.Array(16, context.BitIn())
        record[_bias_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, buffer_outputs: bool):
    if unroll < 1:
        raise ValueError(f"unroll must be >= 1, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    top = global_namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    input_io_list = []
    weight_io_list = []
    bias_io_list = []
    output_io_list = []
    pe_by_role = {k: [] for k in _LANE_PE_ROLES}
    output_fifos = []
    if buffer_outputs:
        mem_gen = context.get_lib("cgralib").generators["Mem"]
        mem_genargs = make_mem_genargs(context)
        clk = defn.add_module_instance(
            "output_fifo_clk", context.get_namespace("corebit").modules["const"],
            context.new_values({"value": True}),
        )

    for i in range(unroll):
        input_io = defn.add_module_instance(_input_io_name(i), io_module, context.new_values({"mode": "in"}))
        weight_io = defn.add_module_instance(_weight_io_name(i), io_module, context.new_values({"mode": "in"}))
        bias_io = defn.add_module_instance(_bias_io_name(i), io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(_output_io_name(i), io_module, context.new_values({"mode": "out"}))
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(iface.select(_weight_self_port(i)), weight_io.select("in"))
        defn.connect(iface.select(_bias_self_port(i)), bias_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        input_io_list.append(input_io)
        weight_io_list.append(weight_io)
        bias_io_list.append(bias_io)
        output_io_list.append(output_io)

        lane_pes = {}
        for role in _LANE_PE_ROLES:
            pe = defn.add_module_instance(f"{role}_pe_{i}", pe_module)
            lane_pes[role] = pe
            pe_by_role[role].append(pe)

        # mul_vec = input * weight
        defn.connect(input_io.select("out"), lane_pes["mul_vec"].select("data0"))
        defn.connect(weight_io.select("out"), lane_pes["mul_vec"].select("data1"))
        # add_vec = bias + mul_vec  (Halide convention: data0=external add operand, data1=mul result)
        defn.connect(bias_io.select("out"), lane_pes["add_vec"].select("data0"))
        defn.connect(lane_pes["mul_vec"].select("O0"), lane_pes["add_vec"].select("data1"))
        # Standalone GLB streams need lane-skew buffering: deleting the old
        # identity prefix alone costs 185 cycles in RTL (4191 -> 4376).
        # Use SRAM FIFOs, not arithmetic padding. Fusion supplies its own
        # buffering and leaves this option disabled.
        result = lane_pes["add_vec"].select("O0")
        if buffer_outputs:
            fifo = defn.add_generator_instance(
                f"output_fifo_lane_{i}", mem_gen, mem_genargs,
                context.new_values({"config": {}, "mode": "lake"}),
            )
            for key, value in [("config", {}), ("is_rom", False),
                               ("mode", "lake"), ("width", 16)]:
                fifo.add_metadata(key, json.dumps(value))
            defn.connect(clk.select("out"), fifo.select("clk_en"))
            defn.connect(result, fifo.select("data_in_0"))
            result = fifo.select("data_out_0")
            output_fifos.append(fifo)
        defn.connect(result, output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "weight_io": weight_io_list,
        "bias_io": bias_io_list,
        "output_io": output_io_list,
        "pe": pe_by_role,
        "output_fifos": output_fifos,
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
    per_row_extent = vec_length // unroll
    for fifo in instances["output_fifos"]:
        fifo.add_metadata("lake_rv_config", json.dumps({
            "type": "fifo", "input_stream_size": per_lane_extent, "row_size": 8,
        }))

    # Input is full [vec_length, num_vecs]: flat dim=1 stream of 3072 per lane.
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    # Weight/bias are stored as [vec_length] and broadcast across num_vecs rows:
    # re-read the same per_row_extent entries each row. Net per-outer advancement
    # = per_row_extent + (1 - per_row_extent) = 1, but since each lane holds
    # starting_addr=0 constant, the stride [1, 1 - per_row_extent] rewinds the
    # data addr to 0 at the end of every row.
    broadcast_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [per_row_extent, num_vecs],
        "read_data_starting_addr": [0],
        "read_data_stride": [1, 1 - per_row_extent],
    })
    in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for io_in in instances["input_io"]:
        io_in.add_metadata("glb2out_0", input_glb2out)
    for io_in in instances["weight_io"]:
        io_in.add_metadata("glb2out_0", broadcast_glb2out)
    for io_in in instances["bias_io"]:
        io_in.add_metadata("glb2out_0", broadcast_glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_mul_add_mul_add_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    buffer_outputs: bool = False,
):
    context, top, instances = _build_graph(unroll, buffer_outputs)
    _configure(context, instances, unroll, vec_length, num_vecs)
    return context, top


def emit_elementwise_mul_add_mul_add_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
    buffer_outputs: bool = False,
):
    context, top = build_elementwise_mul_add_mul_add_bf16_context(
        unroll, vec_length, num_vecs, buffer_outputs=buffer_outputs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_add_mul_add_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_mul_add_mul_add_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_add_mul_add_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path)
