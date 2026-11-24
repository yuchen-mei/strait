"""
Build the Mamba2 skip + gate CoreIR graph using pycoreir:

    g = (Y + D * x) * silu(z)

the fused `submodule_7` (D skip term mul_16, + permute_5(Y) add_10) and `mul_17` (gate multiply) of a Mamba2 mixer.
Used by: mamba_skip_gate_fp.

One run covers `vec_length / head_dim` heads of all `num_vecs` positions (Mamba2: 2 heads x 1024 positions = 128
columns of the [1024 x 4096] x, silu(z) and g):
- x (`input`) and silu(z) (`gate`) are position-major slabs [num_vecs x vec_length]; g (`hw_output`) is written the same
  way. Lane = column mod unroll, linear per-lane streams.
- Y (`yheads`) stays in score@x's heads-first order [chunk, head, row in chunk, p] in the GLB. Its input DMA performs
  Mamba2's permute_5 (c, h, i, p) -> (l = c * chunk_rows + i, h, p) by reading the words in x's order. p stays the
  innermost dimension, so the permute only reorders whole unroll-wide groups: no value changes lane, and all of Y's IO
  tiles step together (E64 multi-bank lockstep).
- D (`weight`, the per-head skip parameter) is stored whole in one GLB tile and read through one unpacked IO starting at
  `d_start` (the run's first head): each value is held for the head_dim / unroll beats of its head (stride 0), the run's
  heads alternate, and the pattern rewinds for every position. The IO writes, through one register, one dual-read MEM
  per two lanes (the bias buffers of elementwise_broadcast_add_softplus_bf16), and each read port feeds one lane, so
  every lane takes D when its x arrives.

Per lane i:
  input IO        -> mul_d_pe.data0 ;  D buffer i//2 port i%2 -> mul_d_pe.data1        (x * D)
  yheads IO       -> add_y_pe.data0 ;  mul_d_pe.O0             -> add_y_pe.data1        (Y + x * D)
  add_y_pe.O0     -> mul_g_pe.data0 ;  gate IO                 -> mul_g_pe.data1        (* silu(z))
  mul_g_pe.O0     -> output FIFO MEM (buffer_outputs)          -> hw_output IO
Each op rounds to bf16 once, as the graph does. 3 PEs per lane; unroll / 2 D buffers; unroll output FIFOs.

DMA patterns (per-lane word addresses, relative strides: each stride is the jump after the inner loops finish, as
SKIP_GLB_DMA_STRIDE_ADJUSTMENT expects; extent-1 loops are dropped). With Gh = head_dim / unroll, H = vec_length /
head_dim, C = num_vecs / chunk_rows, levels innermost first as (extent, absolute stride):
  input, gate, hw_output: (vec_length / unroll, 1), (num_vecs, vec_length / unroll)            linear
  yheads:                 (Gh, 1), (H, Gh * chunk_rows), (chunk_rows, Gh), (C, H * Gh * chunk_rows)
  weight (D):             (Gh, 0), (H, 1), (num_vecs, 0), starting at d_start
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
DEFAULT_VEC_LENGTH = 128
DEFAULT_NUM_VECS = 1024
DEFAULT_HEAD_DIM = 64
DEFAULT_CHUNK_ROWS = 256
TOP_MODULE = "mamba_skip_gate_fp"
_MAX_EXTENT = 1024        # Zircon MEM extent counters
_LOAD_DMA_LEVELS = 8      # garnet global_buffer_parameter load_dma_loop_level
_STORE_DMA_LEVELS = 7     # store_dma_loop_level

_LANE_INPUTS = ["input", "yheads", "gate"]
_LANE_PE_ROLES = [
    "mul_d",    # fp_mul(x, D)
    "add_y",    # fp_add(Y, mul_d)
    "mul_g",    # fp_mul(add_y, gate)
]


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _lane_self_port(name: str, i: int) -> str:
    return f"{name}_host_stencil_clkwrk_{i}_op_hcompute_{name}_glb_stencil{_suffix(i)}_read_0"


def _weight_self_port() -> str:
    return "weight_host_stencil_clkwrk_0_op_hcompute_weight_glb_stencil_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_suffix(i)}_write_0"


def _compute_pe_instructions():
    return {
        "mul_d": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
        "add_y": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None)),
        "mul_g": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
    }


def _interface_type(context, unroll: int):
    record = {_weight_self_port(): context.Array(16, context.BitIn())}
    for i in range(unroll):
        for name in _LANE_INPUTS:
            record[_lane_self_port(name, i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _mem(context, defn, name, mem_gen, genargs):
    mem = defn.add_generator_instance(name, mem_gen, genargs, context.new_values({"config": {}, "mode": "lake"}))
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    return mem


def _build_graph(unroll: int, buffer_outputs: bool):
    if unroll < 2 or unroll % 2:
        raise ValueError(f"[ERROR] unroll must be even and >= 2, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    reg_gen = context.get_lib("coreir").generators["reg"]
    genargs = make_mem_genargs(context)

    top = global_namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    # D: one IO -> one register (its only sink: the mapper does not handle an IO writing several MEMs directly) ->
    # one dual-read MEM per two lanes; read port k of buffer j feeds lane 2j + k.
    weight_io = defn.add_module_instance(f"io16in_{_weight_self_port()}", io_module, context.new_values({"mode": "in"}))
    defn.connect(iface.select(_weight_self_port()), weight_io.select("in"))
    d_reg = defn.add_generator_instance(
        "d_in_reg", reg_gen, context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    d_reg.add_metadata("extra_data", json.dumps(0))
    defn.connect(weight_io.select("out"), d_reg.select("in"))
    d_buffers = []
    for j in range(unroll // 2):
        mem = _mem(context, defn, f"d_buffer_{j}", mem_gen, genargs)
        defn.connect(d_reg.select("out"), mem.select("data_in_0"))
        d_buffers.append(mem)

    if buffer_outputs:
        clk = defn.add_module_instance(
            "output_fifo_clk", context.get_namespace("corebit").modules["const"], context.new_values({"value": True}),
        )

    lane_io = {name: [] for name in _LANE_INPUTS}
    output_io_list = []
    pe_by_role = {k: [] for k in _LANE_PE_ROLES}
    output_fifos = []
    for i in range(unroll):
        ios = {}
        for name in _LANE_INPUTS:
            io = defn.add_module_instance(f"io16in_{_lane_self_port(name, i)}", io_module, context.new_values({"mode": "in"}))
            defn.connect(iface.select(_lane_self_port(name, i)), io.select("in"))
            lane_io[name].append(io)
            ios[name] = io
        output_io = defn.add_module_instance(f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"}))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        output_io_list.append(output_io)

        pes = {}
        for role in _LANE_PE_ROLES:
            pe = defn.add_module_instance(f"{role}_pe_{i}", pe_module)
            pes[role] = pe
            pe_by_role[role].append(pe)
        # x * D
        defn.connect(ios["input"].select("out"), pes["mul_d"].select("data0"))
        defn.connect(d_buffers[i // 2].select(f"data_out_{i % 2}"), pes["mul_d"].select("data1"))
        # Y + x * D (data0 = the external add operand, data1 = the product, as in elementwise_mul_add_mul_add_bf16)
        defn.connect(ios["yheads"].select("out"), pes["add_y"].select("data0"))
        defn.connect(pes["mul_d"].select("O0"), pes["add_y"].select("data1"))
        # (Y + x * D) * silu(z)
        defn.connect(pes["add_y"].select("O0"), pes["mul_g"].select("data0"))
        defn.connect(ios["gate"].select("out"), pes["mul_g"].select("data1"))

        result = pes["mul_g"].select("O0")
        if buffer_outputs:
            # Lane-skew buffering of the standalone GLB output lanes (elementwise_mul_add_mul_add_bf16)
            fifo = _mem(context, defn, f"output_fifo_lane_{i}", mem_gen, genargs)
            defn.connect(clk.select("out"), fifo.select("clk_en"))
            defn.connect(result, fifo.select("data_in_0"))
            result = fifo.select("data_out_0")
            output_fifos.append(fifo)
        defn.connect(result, output_io.select("in"))

    top.definition = defn
    context.set_top(top)
    instances = {
        "weight_io": weight_io,
        "lane_io": lane_io,
        "output_io": output_io_list,
        "pe": pe_by_role,
        "d_buffers": d_buffers,
        "output_fifos": output_fifos,
    }
    return context, top, instances


def relative_stream(key: str, levels, start: int = 0, max_levels: int = _LOAD_DMA_LEVELS):
    """A GLB stream dict from (extent, absolute stride) levels, innermost first. Extent-1 levels are dropped (the DMA
    encodes extent - 2); the strides are made relative: the address change when a level advances after the inner levels
    finish (SKIP_GLB_DMA_STRIDE_ADJUSTMENT)."""
    kept = [(e, s) for e, s in levels if e > 1] or [(1, 1)]
    if len(kept) > max_levels:
        raise ValueError(f"[ERROR] {len(kept)} loop levels exceed the DMA's {max_levels}")
    extents, strides, inner = [], [], 0
    for e, s in kept:
        extents.append(e)
        strides.append(s - inner)
        inner += (e - 1) * s
    stream = {
        "cycle_starting_addr": [0],
        "cycle_stride": [1] * len(extents),
        "dimensionality": len(extents),
        "extent": extents,
        f"{key}_data_starting_addr": [start],
        f"{key}_data_stride": strides,
    }
    if key == "read" and start:
        # The testbench writes the input at tb_write_starting_addr, which defaults to the read start (garnet
        # tests/test_app/lib/parser.c): the data would move with the offset (rounded down to a 64-bit word) and the read
        # would miss it. Keep the tensor at the bank base; only the read starts at `start`.
        stream["tb_write_starting_addr"] = [0]
    return stream


def stream_levels(unroll: int, vec_length: int, num_vecs: int, head_dim: int, chunk_rows: int):
    """The (extent, absolute stride) levels of the linear, yheads and weight streams (see the module docstring)."""
    for name, value, div in (("vec_length", vec_length, head_dim), ("head_dim", head_dim, unroll),
                             ("num_vecs", num_vecs, chunk_rows)):
        if value % div:
            raise ValueError(f"[ERROR] {name} ({value}) must be a multiple of {div}")
    groups = vec_length // unroll
    gh, heads, chunks = head_dim // unroll, vec_length // head_dim, num_vecs // chunk_rows
    if groups > _MAX_EXTENT or num_vecs > _MAX_EXTENT or chunk_rows > _MAX_EXTENT:
        raise ValueError(f"[ERROR] a loop extent exceeds {_MAX_EXTENT}")
    return {
        "linear": [(groups, 1), (num_vecs, groups)],
        "yheads": [(gh, 1), (heads, gh * chunk_rows), (chunk_rows, gh), (chunks, heads * gh * chunk_rows)],
        "weight": [(gh, 0), (heads, 1), (num_vecs, 0)],
    }


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int, head_dim: int, chunk_rows: int,
               d_start: int):
    levels = stream_levels(unroll, vec_length, num_vecs, head_dim, chunk_rows)
    per_lane_extent = num_vecs * vec_length // unroll
    defn = instances["weight_io"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    for role, (inst_val, inst_width) in _compute_pe_instructions().items():
        for i, pe in enumerate(instances["pe"][role]):
            c = defn.add_generator_instance(
                f"const_inst_{role}_pe_{i}", const_gen,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    # D buffers: the IO repeats each D value per beat, so each read port carries the lane's full token stream.
    # Smallest 4-word multiple block that splits it into at most _MAX_EXTENT blocks.
    block = next((b for b in range(4, 513, 4) if per_lane_extent % b == 0 and per_lane_extent // b <= _MAX_EXTENT), None)
    if block is None:
        raise ValueError(f"[ERROR] {per_lane_extent} values per lane do not split into <= {_MAX_EXTENT} blocks")
    d_cfg = json.dumps({"type": "dual_read", "input_stream_size": per_lane_extent, "row_size": block})
    for mem in instances["d_buffers"]:
        mem.add_metadata("lake_rv_config", d_cfg)
    for fifo in instances["output_fifos"]:
        fifo.add_metadata("lake_rv_config", json.dumps({
            "type": "fifo", "input_stream_size": per_lane_extent, "row_size": 8,
        }))

    linear_read = json.dumps(relative_stream("read", levels["linear"]))
    for name in ("input", "gate"):
        for io in instances["lane_io"][name]:
            io.add_metadata("glb2out_0", linear_read)
    yheads_read = json.dumps(relative_stream("read", levels["yheads"]))
    for io in instances["lane_io"]["yheads"]:
        io.add_metadata("glb2out_0", yheads_read)
    instances["weight_io"].add_metadata("glb2out_0", json.dumps(relative_stream("read", levels["weight"], start=d_start)))
    linear_write = json.dumps(relative_stream("write", levels["linear"], max_levels=_STORE_DMA_LEVELS))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", linear_write)


def build_elementwise_skip_gate_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    head_dim: int = DEFAULT_HEAD_DIM,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    d_start: int = 0,
    buffer_outputs: bool = True,
):
    context, top, instances = _build_graph(unroll, buffer_outputs)
    _configure(context, instances, unroll, vec_length, num_vecs, head_dim, chunk_rows, d_start)
    return context, top


def emit_elementwise_skip_gate_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
    head_dim: int = DEFAULT_HEAD_DIM,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    d_start: int = 0,
    buffer_outputs: bool = True,
):
    context, top = build_elementwise_skip_gate_bf16_context(
        unroll, vec_length, num_vecs, head_dim, chunk_rows, d_start, buffer_outputs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_skip_gate_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_skip_gate_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--d-start", type=int, default=0)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_skip_gate_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path,
                                           args.head_dim, args.chunk_rows, args.d_start)
