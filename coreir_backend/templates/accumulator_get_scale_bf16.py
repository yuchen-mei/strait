"""
Build a bf16 per-lane accumulate-get-scale + byte-pack CoreIR graph using
pycoreir. Used by: get_e8m0_scale_accum_gb_input — per-lane spatial reduction
over blocks of `block_size`, converted to an e8m0 shared exponent, then packed
into u16 outputs (2 scales per output lane).

Per output pixel of each lane i in [0, unroll):
    block[i, k] = abs_max over b in [0, block_size) of input_io_i.stream[k, b]
    scale_pe[i, k] = get_shared_exp(block[i, k])

Per packer p in [0, unroll/2):
    output_io_p[k] = bit8_pack(scale_pe[2p+1, k], scale_pe[2p, k])
                    // (high_byte=odd lane, low_byte=even lane)

Templated design parameters:
- unroll: number of parallel input lanes (= glb_i); must be even (>=2).
- head_dim: head dim (used for IO extent sizing).
- seq_heads_prod: seq-heads product (used for IO extent sizing).
- block_size: inner spatial reduction block length; must be even.
- top_module: name of the emitted top module.

Architecture per input lane i:
  input_io_i -> filter_mem_i  (2 read streams, alternating)
    data_out_0 -> accum_pe_0_i + accum_pond_0_i
    data_out_1 -> accum_pe_1_i + accum_pond_1_i
  -> final_reduce_pe_i (fp_abs_max across pond spills)
  -> get_shared_exp_pe_i

Per output packer p:
  scale_pe[2p+1].O0 -> pack_pe_p.data0  (high byte)
  scale_pe[2p].O0   -> pack_pe_p.data1  (low byte)
  pack_pe_p.O0      -> output_io_p.in

accum PEs require output_fifo_bypass=1 (RMW loop).

IO metadata:
- Input (per lane): 3D stream,
    extent = [block_size, head_dim/unroll, seq_heads_prod/block_size],
    read_data_stride = [head_dim/unroll, 1 - (head_dim/unroll)*(block_size - 1), 1].
- Output (per packer): 1D, extent = head_dim * (seq_heads_prod/block_size) / unroll.

Lake RV configs:
- filter_mem: get_filter_mem_two_streams(input_stream_size=block_size*num_output_pixels_per_lane).
- accum_pond: get_vec_accum_pond(num_partial_reduction=block_size/2,
                                 num_output_pixels=num_output_pixels_per_lane).
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_HEAD_DIM = 64
DEFAULT_SEQ_HEADS_PROD = 1536
DEFAULT_BLOCK_SIZE = 64
DEFAULT_TOP_MODULE = "get_e8m0_scale_accum_gb_input"


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_io_name(i: int) -> str:
    return f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
    for p in range(unroll // 2):
        record[_output_self_port(p)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, top_module: str):
    if unroll < 2 or unroll % 2 != 0:
        raise ValueError(f"unroll must be >= 2 and even, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    corebit_const = context.get_namespace("corebit").modules["const"]

    top = global_namespace.new_module(top_module, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    shared_clk = defn.add_module_instance(
        f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}),
    )

    mem_genargs = make_mem_genargs(context)
    pond_genargs = context.new_values({
        "ID": "",
        "has_stencil_valid": True,
        "num_inputs": 2,
        "num_outputs": 2,
        "width": 16,
    })

    input_ios = []
    filter_mems = []
    accum_pond_0s = []
    accum_pond_1s = []
    accum_pe_0s = []
    accum_pe_1s = []
    final_reduce_pes = []
    scale_pes = []
    pack_pes = []
    output_ios = []

    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"

        inp_io = defn.add_module_instance(_input_io_name(i), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_input_self_port(i)), inp_io.select("in"))

        filter_mem = defn.add_generator_instance(
            f"{lane_prefix}_filter_mem", mem_gen, mem_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _k, _v in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            filter_mem.add_metadata(_k, _v)
        defn.connect(inp_io.select("out"), filter_mem.select("data_in_0"))
        defn.connect(shared_clk.select("out"), filter_mem.select("clk_en"))

        accum_pond_0 = defn.add_generator_instance(
            f"{lane_prefix}_accum_pond_0", pond_gen, pond_genargs,
            context.new_values({"config": {}, "mode": "pond"}),
        )
        accum_pond_1 = defn.add_generator_instance(
            f"{lane_prefix}_accum_pond_1", pond_gen, pond_genargs,
            context.new_values({"config": {}, "mode": "pond"}),
        )
        for pond in (accum_pond_0, accum_pond_1):
            for _k, _v in [
                ("config", json.dumps({})),
                ("is_rom", json.dumps(False)),
                ("mode", json.dumps("pond")),
                ("width", json.dumps(16)),
            ]:
                pond.add_metadata(_k, _v)
            defn.connect(shared_clk.select("out"), pond.select("clk_en"))

        accum_pe_0 = defn.add_module_instance(f"{lane_prefix}_accum_pe_0", pe_module)
        accum_pe_1 = defn.add_module_instance(f"{lane_prefix}_accum_pe_1", pe_module)
        final_reduce_pe = defn.add_module_instance(f"{lane_prefix}_final_reduce_pe", pe_module)
        scale_pe = defn.add_module_instance(f"{lane_prefix}_get_shared_exp_pe", pe_module)

        defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
        defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

        defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
        defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
        defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

        defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
        defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
        defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

        defn.connect(final_reduce_pe.select("O0"), scale_pe.select("data0"))

        input_ios.append(inp_io)
        filter_mems.append(filter_mem)
        accum_pond_0s.append(accum_pond_0)
        accum_pond_1s.append(accum_pond_1)
        accum_pe_0s.append(accum_pe_0)
        accum_pe_1s.append(accum_pe_1)
        final_reduce_pes.append(final_reduce_pe)
        scale_pes.append(scale_pe)

    for p in range(unroll // 2):
        out_io = defn.add_module_instance(_output_io_name(p), io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_output_self_port(p)))

        pack_pe = defn.add_module_instance(f"{top_module}_pack_pe_{p}", pe_module)
        # Halide: bit8_pack(odd_lane, even_lane) — data0 = high byte (odd), data1 = low byte (even).
        defn.connect(scale_pes[2 * p + 1].select("O0"), pack_pe.select("data0"))
        defn.connect(scale_pes[2 * p].select("O0"), pack_pe.select("data1"))
        defn.connect(pack_pe.select("O0"), out_io.select("in"))

        pack_pes.append(pack_pe)
        output_ios.append(out_io)

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_ios,
        "filter_mem": filter_mems,
        "accum_pond_0": accum_pond_0s,
        "accum_pond_1": accum_pond_1s,
        "accum_pe_0": accum_pe_0s,
        "accum_pe_1": accum_pe_1s,
        "final_reduce_pe": final_reduce_pes,
        "scale_pe": scale_pes,
        "pack_pe": pack_pes,
        "output_io": output_ios,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, head_dim: int, seq_heads_prod: int,
               block_size: int, top_module: str):
    if block_size % 2 != 0:
        raise ValueError(f"block_size ({block_size}) must be even for two-pond accumulation")
    if head_dim % unroll != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by unroll ({unroll})")
    if seq_heads_prod % block_size != 0:
        raise ValueError(f"seq_heads_prod ({seq_heads_prod}) must be divisible by block_size ({block_size})")

    num_output_pixels_per_lane = (seq_heads_prod // block_size) * (head_dim // unroll)

    defn = instances["filter_mem"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    abs_max_val, abs_max_w = pe_inst_to_bits_with_operands(
        "fp_abs_max", data0=("ext", None), data1=("ext", None),
    )
    shared_exp_val, shared_exp_w = pe_inst_to_bits_with_operands(
        "get_shared_exp", data0=("ext", None),
    )
    pack_val, pack_w = pe_inst_to_bits_with_operands(
        "bit8_pack", data0=("ext", None), data1=("ext", None),
    )

    def _wire_const(role: str, pe_inst, inst_val: int, inst_w: int):
        c = defn.add_generator_instance(
            f"const_inst_{role}", const_gen,
            context.new_values({"width": inst_w}),
            context.new_values({"value": BitVector[inst_w](inst_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"
        _wire_const(f"{lane_prefix}_accum_pe_0", instances["accum_pe_0"][i], abs_max_val, abs_max_w)
        _wire_const(f"{lane_prefix}_accum_pe_1", instances["accum_pe_1"][i], abs_max_val, abs_max_w)
        _wire_const(f"{lane_prefix}_final_reduce_pe", instances["final_reduce_pe"][i], abs_max_val, abs_max_w)
        _wire_const(f"{lane_prefix}_get_shared_exp_pe", instances["scale_pe"][i], shared_exp_val, shared_exp_w)

    for p in range(unroll // 2):
        _wire_const(f"{top_module}_pack_pe_{p}", instances["pack_pe"][p], pack_val, pack_w)

    # IO metadata.
    read_stride_0 = head_dim // unroll
    read_stride_1 = 1 - read_stride_0 * (block_size - 1)
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1, 1],
        "dimensionality": 3,
        "extent": [block_size, head_dim // unroll, seq_heads_prod // block_size],
        "read_data_starting_addr": [0],
        "read_data_stride": [read_stride_0, read_stride_1, 1],
    })
    output_in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [num_output_pixels_per_lane],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })
    for io_in in instances["input_io"]:
        io_in.add_metadata("glb2out_0", input_glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", output_in2glb)

    # Lake RV configs.
    filter_mem_cfg = json.dumps({
        "type": "get_filter_mem_two_streams",
        "input_stream_size": block_size * num_output_pixels_per_lane,
    })
    accum_pond_cfg = json.dumps({
        "type": "get_vec_accum_pond",
        "num_partial_reduction": block_size // 2,
        "num_output_pixels": num_output_pixels_per_lane,
    })
    for i in range(unroll):
        instances["filter_mem"][i].add_metadata("lake_rv_config", filter_mem_cfg)
        instances["accum_pond_0"][i].add_metadata("lake_rv_config", accum_pond_cfg)
        instances["accum_pond_1"][i].add_metadata("lake_rv_config", accum_pond_cfg)


def _write_pe_fifos_bypass_config(output_path: str, unroll: int, top_module: str):
    bypass_cfg = {}
    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"
        bypass_cfg[f"{lane_prefix}_accum_pe_0"] = {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1}
        bypass_cfg[f"{lane_prefix}_accum_pe_1"] = {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1}
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_accumulator_get_scale_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    head_dim: int = DEFAULT_HEAD_DIM,
    seq_heads_prod: int = DEFAULT_SEQ_HEADS_PROD,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, top_module)
    _configure(context, instances, unroll, head_dim, seq_heads_prod, block_size, top_module)
    return context, top


def emit_accumulator_get_scale_bf16_design(
    unroll: int,
    head_dim: int,
    seq_heads_prod: int,
    output_path: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_accumulator_get_scale_bf16_context(
        unroll, head_dim, seq_heads_prod, block_size=block_size, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, unroll, top_module)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate accumulator_get_scale_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seq-heads-prod", type=int, default=DEFAULT_SEQ_HEADS_PROD)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_accumulator_get_scale_bf16_design(
        args.unroll, args.head_dim, args.seq_heads_prod, args.output_path,
        block_size=args.block_size, top_module=args.top_module,
    )
