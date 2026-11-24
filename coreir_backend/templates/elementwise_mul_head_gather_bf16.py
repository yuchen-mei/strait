"""
Build a Mamba2 "x times per-head scalars" CoreIR graph using pycoreir: an x-like tensor (rows x heads x head_dim) times
one or more scalars that are constant over each head's head_dim values, each scalar serialized from its GLB lanes by a
gather tree. Used by:
- mamba_x_dt_fp: X = x * dt (`mul_7`), written heads-first (`permute_1`) by the output DMA;
- mamba_x_end_fp: X_end = (x * dt) * decay_to_end (`mul_11`), position-major;
- mamba_state_decay_fp: decayed state = state * chunk decay (`mul_12` .. `mul_14`), the decay the same for every row.

One run covers `num_heads` * `slots` heads (the 8 lanes of one scalar GLB tile, `slots` words per lane per row) of
`num_vecs` rows, on the half of the channels a 16-lane pass of the `layout_lanes` (32) lane layout touches:
- x (`input`) and the output (`hw_output`) stay in the 32-lane GLB layout (lane = channel mod 32). The pass works on 16
  of those lanes, so of head h's head_dim values it sees p = 32 g + 16 q + j (lane j, g < B, pass q), and each scalar
  value is used by B = head_dim / layout_lanes beats (2 for Mamba2).
- Each scalar (`dt`, `decay`, ...) is one packed GLB tile: lane j holds head h0 + j (+ 32 s for word slot s), `slots`
  words per row (row_stride 1), the same words for every row (row_stride 0), or every row_stride-th word from a start
  word (e.g. dt in softplus's 32-lane layout: word 2 l + s, row_stride 2, start 512 c + s). Every scalar lane IO repeats each word B
  times (stride 0) and feeds a gather tree of lake `get_interleave_mem` MEMs (single reader) that take turns of B values
  (turn = B: one head's B copies). Level 1 pairs lanes (j, j + H/2) (lane slot j of both IO tiles of the GLB tile), each
  next level pairs (j, j + m/2) again, so per row the root emits the slot-0 heads h0, ..., h0 + H - 1, then the next
  slot's, each B times, in natural head order. The turn keeps each read port on one SRAM word for B values: with
  element-wise turns (a new word for every value) a lake read port gives at most 0.5 values per cycle, and the root must
  give one value per beat. A pass-through PE (`or_` with 0) sits after every tree MEM (no MEM -> MEM edges); the last
  one writes num_lanes / 2 dual-read MEMs (the bias buffers of elementwise_broadcast_add_softplus_bf16), read port k of
  buffer b feeding lane 2 b + k.
- Beat n = (row i, slot s, head h, p-group g), x's natural order: lane j multiplies x[i, 64 (h + 32 s) + 32 g + 16 q + j]
  by each scalar in turn, rounding to bf16 after each multiply.
- Lane source (a scalar's optional 4th field {"lane": j, "lanes": L, "head_stride": d}): the scalar's values already sit in
  one lane j of its packed GLB tile in head order (e.g. the chunk decays in exp_2's output: lane 31 = lane 7 of output
  GLB tile 3, word 512 c + 8 h + 7, head stride 8). All L lane IOs of the tile are instantiated with the same stream and
  only lane j is connected (the others have no sink; their ready is tied high, so they cannot stall the multi-bank DMA).
  It writes, through one register, the dual-read buffers directly (the D path of elementwise_skip_gate_bf16): no gather
  tree. One word per head (slots = 1), each held B beats, the same pattern for every row with row_stride 0.

Per lane i:
  input IO -> mul_pe.data0 ;  scalar 0 buffer i//2 port i%2 -> mul_pe.data1 ;  (mul_<scalar>_pe for each later scalar)
  last PE.O0 -> output FIFO MEM -> hw_output IO

DMA patterns (per-lane word addresses, relative strides as SKIP_GLB_DMA_STRIDE_ADJUSTMENT expects, extent-1 loops
dropped). Levels innermost first as (extent, absolute stride), H = num_heads, S = slots, B = hold, R = num_vecs:
  input  (word i H S B + B (H s + h) + g):     (H S B, 1), (R, H S B)              linear
  scalar (word start + S row_stride i + s; each held B beats): (B, 0), (S, 1), (R, S row_stride)
  scalar, lane source (word start + head_stride h + row_stride i): (B, 0), (H, head_stride), (R, row_stride)
  output position-major:                        as the input
  output heads-first (S = 1, word (h R + i) B + g): (B, 1), (H, R B), (R, B)
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.templates.elementwise_skip_gate_bf16 import relative_stream
from strait.coreir_backend.utils.build_pe_inst import (
    pe_inst_to_bits_with_operands,
)
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_LAYOUT_LANES = 32
DEFAULT_NUM_HEADS = 8
DEFAULT_HEAD_DIM = 64
DEFAULT_NUM_VECS = 256
DEFAULT_SCALARS = (("dt", 1),)   # (name, row_stride): mamba_x_dt_fp
TOP_MODULE = "mamba_x_dt_fp"
_MAX_EXTENT = 1024        # Zircon MEM extent counters
_STORE_DMA_LEVELS = 7     # garnet global_buffer_parameter store_dma_loop_level


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _lane_self_port(name: str, i: int) -> str:
    return f"{name}_host_stencil_clkwrk_{i}_op_hcompute_{name}_glb_stencil{_suffix(i)}_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_suffix(i)}_write_0"


def _scalar(sc):
    """(name, row_stride, start, lane source or None) of a scalar spec (name, row_stride[, start[, lane source]])."""
    return sc[0], sc[1], sc[2] if len(sc) > 2 else 0, sc[3] if len(sc) > 3 else None


def _scalar_lanes(sc, num_heads: int) -> int:
    lane_src = _scalar(sc)[3]
    return lane_src["lanes"] if lane_src else num_heads


def _interface_type(context, unroll: int, num_heads: int, scalars):
    record = {}
    for sc in scalars:
        for j in range(_scalar_lanes(sc, num_heads)):
            record[_lane_self_port(sc[0], j)] = context.Array(16, context.BitIn())
    for i in range(unroll):
        record[_lane_self_port("input", i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _mem(context, defn, name, mem_gen, genargs):
    mem = defn.add_generator_instance(name, mem_gen, genargs, context.new_values({"config": {}, "mode": "lake"}))
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    return mem


def _build_graph(unroll: int, num_heads: int, scalars, buffer_outputs: bool, top_module: str):
    scalar_names = [sc[0] for sc in scalars]
    if unroll < 2 or unroll % 2:
        raise ValueError(f"[ERROR] unroll must be even and >= 2, got {unroll}")
    if any(_scalar(sc)[3] is None for sc in scalars) and (num_heads < 2 or num_heads & (num_heads - 1)):
        raise ValueError(f"[ERROR] num_heads must be a power of 2 >= 2 (one gather tree leaf per head), got {num_heads}")
    for sc in scalars:
        lane_src = _scalar(sc)[3]
        if lane_src and not 0 <= lane_src["lane"] < lane_src["lanes"]:
            raise ValueError(f"[ERROR] {sc[0]}: lane {lane_src['lane']} outside the tile's {lane_src['lanes']} lanes")
    if not scalar_names or len(set(scalar_names)) != len(scalar_names) or "input" in scalar_names:
        raise ValueError(f"[ERROR] scalar names must be distinct and not 'input', got {scalar_names}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    genargs = make_mem_genargs(context)

    top = global_namespace.new_module(top_module, _interface_type(context, unroll, num_heads, scalars))
    defn = top.new_definition()
    iface = defn.interface

    scalar_io, tree_levels, buffers, pass_pes = {}, {}, {}, []
    for sc in scalars:
        name, lane_src = sc[0], _scalar(sc)[3]
        # Scalar lanes, in numeric lane order (the metadata parser packs IOs four lanes per E64 bank in this order)
        ios = []
        for j in range(_scalar_lanes(sc, num_heads)):
            io = defn.add_module_instance(f"io16in_{_lane_self_port(name, j)}", io_module, context.new_values({"mode": "in"}))
            defn.connect(iface.select(_lane_self_port(name, j)), io.select("in"))
            ios.append(io)
        scalar_io[name] = ios

        sources = [io.select("out") for io in ios]
        tree_levels[name] = []
        if lane_src:
            # Lane source: the connected lane -> one register (an IO must not write several MEMs directly) -> buffers
            reg = defn.add_generator_instance(
                f"{name}_in_reg", context.get_lib("coreir").generators["reg"], context.new_values({"width": 16}),
                context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
            )
            reg.add_metadata("extra_data", json.dumps(0))
            defn.connect(sources[lane_src["lane"]], reg.select("in"))
            sources = [reg.select("out")]

        # Gather tree: each level pairs streams (j, j + m/2) of its m inputs into get_interleave_mem MEMs; a
        # pass-through PE follows every MEM
        level = 0
        while len(sources) > 1:
            level += 1
            half = len(sources) // 2
            mems, next_sources = [], []
            for j in range(half):
                mem = _mem(context, defn, f"{name}_gather_l{level}_{j}", mem_gen, genargs)
                defn.connect(sources[j], mem.select("data_in_0"))
                defn.connect(sources[j + half], mem.select("data_in_1"))
                pe = defn.add_module_instance(f"{name}_gather_l{level}_pass_pe_{j}", pe_module)
                defn.connect(mem.select("data_out_0"), pe.select("data0"))
                pass_pes.append(pe)
                mems.append(mem)
                next_sources.append(pe.select("O0"))
            tree_levels[name].append(mems)
            sources = next_sources

        # Fan-out: one dual-read MEM per two lanes, both read ports used
        buffers[name] = []
        for b in range(unroll // 2):
            mem = _mem(context, defn, f"{name}_buffer_{b}", mem_gen, genargs)
            defn.connect(sources[0], mem.select("data_in_0"))
            buffers[name].append(mem)

    if buffer_outputs:
        clk = defn.add_module_instance(
            "output_fifo_clk", context.get_namespace("corebit").modules["const"], context.new_values({"value": True}),
        )

    input_io, output_io, mul_pes, output_fifos = [], [], [], []
    for i in range(unroll):
        io = defn.add_module_instance(f"io16in_{_lane_self_port('input', i)}", io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_lane_self_port("input", i)), io.select("in"))
        input_io.append(io)
        out_io = defn.add_module_instance(f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_output_self_port(i)))
        output_io.append(out_io)

        # One fp_mul per scalar, in order: data0 = the running product (x first), data1 = the scalar
        result = io.select("out")
        for k, name in enumerate(scalar_names):
            pe = defn.add_module_instance(f"mul_pe_{i}" if k == 0 else f"mul_{name}_pe_{i}", pe_module)
            defn.connect(result, pe.select("data0"))
            defn.connect(buffers[name][i // 2].select(f"data_out_{i % 2}"), pe.select("data1"))
            mul_pes.append(pe)
            result = pe.select("O0")

        if buffer_outputs:
            # Lane-skew buffering of the standalone GLB output lanes (elementwise_mul_add_mul_add_bf16)
            fifo = _mem(context, defn, f"output_fifo_lane_{i}", mem_gen, genargs)
            defn.connect(clk.select("out"), fifo.select("clk_en"))
            defn.connect(result, fifo.select("data_in_0"))
            result = fifo.select("data_out_0")
            output_fifos.append(fifo)
        defn.connect(result, out_io.select("in"))

    top.definition = defn
    context.set_top(top)
    instances = {
        "scalar_io": scalar_io,
        "input_io": input_io,
        "output_io": output_io,
        "tree_levels": tree_levels,
        "pass_pes": pass_pes,
        "mul_pes": mul_pes,
        "buffers": buffers,
        "output_fifos": output_fifos,
    }
    return context, top, instances


def hold(head_dim: int, layout_lanes: int) -> int:
    """Beats per scalar value: a 16-lane pass of the layout_lanes-lane layout sees head_dim / layout_lanes of each
    head's unroll-wide groups."""
    if head_dim % layout_lanes:
        raise ValueError(f"[ERROR] head_dim ({head_dim}) must be a multiple of layout_lanes ({layout_lanes})")
    return head_dim // layout_lanes


def stream_levels(unroll: int, layout_lanes: int, num_heads: int, head_dim: int, num_vecs: int, scalars=DEFAULT_SCALARS,
                  slots: int = 1, heads_first_output: bool = True):
    """The (extent, absolute stride) levels of the input, each scalar's and the output streams (module docstring)."""
    if layout_lanes % unroll:
        raise ValueError(f"[ERROR] layout_lanes ({layout_lanes}) must be a multiple of unroll ({unroll})")
    b = hold(head_dim, layout_lanes)
    if b not in (1, 2, 4):
        raise ValueError(f"[ERROR] the hold ({b}) is the gather MEMs' interleave turn: it must be 1, 2 or 4")
    if slots < 1 or (heads_first_output and slots != 1):
        raise ValueError(f"[ERROR] slots must be >= 1, and 1 for a heads-first output; got {slots}")
    if any(sc[1] < 0 for sc in scalars):
        raise ValueError(f"[ERROR] scalar row strides must be >= 0, got {scalars}")
    if any(_scalar(sc)[3] for sc in scalars) and slots != 1:
        raise ValueError(f"[ERROR] a lane-source scalar holds one word per head: slots must be 1, got {slots}")
    words = num_heads * slots * b
    if max(words, num_vecs) > _MAX_EXTENT:
        raise ValueError(f"[ERROR] a loop extent exceeds {_MAX_EXTENT}")
    levels = {"input": [(words, 1), (num_vecs, words)]}
    for sc in scalars:
        name, stride, _, lane_src = _scalar(sc)
        if lane_src:
            levels[name] = [(b, 0), (num_heads, lane_src["head_stride"]), (num_vecs, stride)]
        else:
            levels[name] = [(b, 0), (slots, 1), (num_vecs, slots * stride)]
    if heads_first_output:
        levels["output"] = [(b, 1), (num_heads, num_vecs * b), (num_vecs, b)]
    else:
        levels["output"] = levels["input"]
    return levels


def _configure(context, instances, unroll: int, layout_lanes: int, num_heads: int, head_dim: int, num_vecs: int,
               scalars, slots: int, heads_first_output: bool):
    levels = stream_levels(unroll, layout_lanes, num_heads, head_dim, num_vecs, scalars, slots, heads_first_output)
    b = hold(head_dim, layout_lanes)
    beats = num_heads * slots * b * num_vecs
    defn = instances["input_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    pe_insts = [
        (instances["mul_pes"], pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None))),
        (instances["pass_pes"], pe_inst_to_bits_with_operands("or_", data0=("ext", None), data1=("const", 0))),
    ]
    for pes, (inst_val, inst_width) in pe_insts:
        for pe in pes:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}", const_gen,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    # Tree level l merges two streams of num_vecs * slots * B * 2^(l-1) values each, in turns of B values
    for name, *_ in scalars:
        for depth, mems in enumerate(instances["tree_levels"][name]):
            size = num_vecs * slots * b * 2 ** depth
            if size % 4 or size // 4 > _MAX_EXTENT:
                raise ValueError(f"[ERROR] {name} gather level {depth + 1}: {size} values per input do not fit get_interleave_mem")
            cfg = json.dumps({"type": "get_interleave_mem", "single_input_stream_size": size, "turn": b})
            for mem in mems:
                mem.add_metadata("lake_rv_config", cfg)

    # Scalar buffers carry one value per beat; smallest 4-word multiple block that splits it into at most _MAX_EXTENT
    # blocks
    block = next((r for r in range(4, 513, 4) if beats % r == 0 and beats // r <= _MAX_EXTENT), None)
    if block is None:
        raise ValueError(f"[ERROR] {beats} values per lane do not split into <= {_MAX_EXTENT} blocks")
    buffer_cfg = json.dumps({"type": "dual_read", "input_stream_size": beats, "row_size": block})
    for name, *_ in scalars:
        for mem in instances["buffers"][name]:
            mem.add_metadata("lake_rv_config", buffer_cfg)
    for fifo in instances["output_fifos"]:
        fifo.add_metadata("lake_rv_config", json.dumps({"type": "fifo", "input_stream_size": beats, "row_size": 8}))

    input_read = json.dumps(relative_stream("read", levels["input"]))
    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", input_read)
    for sc in scalars:
        name, _, start, _ = _scalar(sc)
        scalar_read = json.dumps(relative_stream("read", levels[name], start=start))
        for io in instances["scalar_io"][name]:
            io.add_metadata("glb2out_0", scalar_read)
    output_write = json.dumps(relative_stream("write", levels["output"], max_levels=_STORE_DMA_LEVELS))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", output_write)


def build_elementwise_mul_head_gather_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    layout_lanes: int = DEFAULT_LAYOUT_LANES,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    num_vecs: int = DEFAULT_NUM_VECS,
    buffer_outputs: bool = True,
    scalars=DEFAULT_SCALARS,
    slots: int = 1,
    heads_first_output: bool = True,
    top_module: str = TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, num_heads, scalars, buffer_outputs, top_module)
    _configure(context, instances, unroll, layout_lanes, num_heads, head_dim, num_vecs, scalars, slots,
               heads_first_output)
    return context, top


def emit_elementwise_mul_head_gather_bf16_design(
    unroll: int,
    num_heads: int,
    num_vecs: int,
    output_path: str,
    layout_lanes: int = DEFAULT_LAYOUT_LANES,
    head_dim: int = DEFAULT_HEAD_DIM,
    buffer_outputs: bool = True,
    scalars=DEFAULT_SCALARS,
    slots: int = 1,
    heads_first_output: bool = True,
    top_module: str = TOP_MODULE,
):
    context, top = build_elementwise_mul_head_gather_bf16_context(
        unroll, layout_lanes, num_heads, head_dim, num_vecs, buffer_outputs, scalars, slots, heads_first_output,
        top_module)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_head_gather_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_mul_head_gather_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--layout-lanes", type=int, default=DEFAULT_LAYOUT_LANES)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_head_gather_bf16_design(args.unroll, args.num_heads, args.num_vecs, args.output_path,
                                                 args.layout_lanes, args.head_dim)
