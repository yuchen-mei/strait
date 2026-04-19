"""
Build the elementwise (unpack-scale + e8m0_quant + byte-pack) multi-IO CoreIR
graph using pycoreir. Used by: apply_e8m0_scale_multi_IOs.

CRITICAL: this template matches bin_gold's PE/IO naming EXACTLY so that the
alphabetically-sorted block ids (garnet/mapper/netlist_util.py CreateIDs) line
up with the old-flow placement. Diverging from these names changes thunder's
cluster partitioning (igraph seed=0) and the annealer's placement.

Per packer p in [0, unroll/2):
    suffix = "" if p == 0 else f"_{p}"         # Halide _stencil_suffix convention
    scale_io_p   -> bit8_unpack_high_p.data0   (supplies high byte = odd-lane scale)
    scale_io_p   -> bit8_unpack_low_p.data0    (supplies low  byte = even-lane scale)
    bf_act_io[2p+1] -> e8m0_quant_p_0.data0    (odd lane; first alphabetical quant)
    unpack_high_p.O0 -> e8m0_quant_p_0.data1
    bf_act_io[2p  ] -> e8m0_quant_p_1.data0    (even lane; second alphabetical quant)
    unpack_low_p.O0  -> e8m0_quant_p_1.data1
    # Halide: bit8_pack(odd, even) -> data0 = high-byte (odd), data1 = low-byte (even).
    e8m0_quant_p_0.O0 -> bit8_pack_p.data0
    e8m0_quant_p_1.O0 -> bit8_pack_p.data1
    bit8_pack_p.O0 -> output_io_p

The 5-PE group sort order within each stencil{suffix}$inner_compute$... namespace:
    bit8_pack_0  < bit8_unpack_high_0  < bit8_unpack_low_0
                 < e8m0_quant_0 (odd) < e8m0_quant_1 (even)

Alphabetical group sort (by _stencil_suffix): "" < _1 < _10 < _11 < ... < _15
                                                  < _2 < _3 < ... < _9
so block ids fill 5 consecutive p-ids per group in that suffix sort order, as
bin_gold does.

Templated design parameters:
- unroll: number of bf_act input lanes (= glb_i); must be even and >= 2.
- head_dim, seq_heads_prod: from halide_gen_args (IO extent sizing).
- block_size: inner scale block (default 64).
- top_module: emitted top module name.

IO metadata (matches the pre-strait Halide-hack exactly):
- input_bf_act (per lane): 1D, extent = head_dim * seq_heads_prod / unroll.
- input_scale (per scale lane): 3D broadcast,
    extent=[head_dim/unroll, block_size, seq_heads_prod/block_size],
    stride=[1, 1 - head_dim/unroll, 1].
- hw_output_mxint8_act (per packer): 1D, extent = head_dim * seq_heads_prod / unroll.
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits, pe_inst_to_bits_with_operands

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_HEAD_DIM = 64
DEFAULT_SEQ_HEADS_PROD = 1536
DEFAULT_BLOCK_SIZE = 64
DEFAULT_TOP_MODULE = "apply_e8m0_scale_multi_IOs"


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


# bin_gold clkwrk numbering: bf_act 0..unroll-1, scale unroll..(unroll + unroll/2 - 1),
# output (unroll + unroll/2)..(unroll + unroll - 1).
def _bf_act_clk(i: int) -> int:
    return i


def _scale_clk(j: int, unroll: int) -> int:
    return unroll + j


def _output_clk(p: int, unroll: int) -> int:
    return unroll + unroll // 2 + p


def _bf_act_io_name(i: int) -> str:
    return f"io16in_input_bf_act_host_stencil_clkwrk_{_bf_act_clk(i)}_op_hcompute_input_bf_act_glb_stencil{_stencil_suffix(i)}_read_0"


def _bf_act_self_port(i: int) -> str:
    return f"input_bf_act_host_stencil_clkwrk_{_bf_act_clk(i)}_op_hcompute_input_bf_act_glb_stencil{_stencil_suffix(i)}_read_0"


def _scale_io_name(j: int, unroll: int) -> str:
    return f"io16in_input_scale_host_stencil_clkwrk_{_scale_clk(j, unroll)}_op_hcompute_input_scale_glb_stencil{_stencil_suffix(j)}_read_0"


def _scale_self_port(j: int, unroll: int) -> str:
    return f"input_scale_host_stencil_clkwrk_{_scale_clk(j, unroll)}_op_hcompute_input_scale_glb_stencil{_stencil_suffix(j)}_read_0"


def _mxint8_io_name(p: int, unroll: int) -> str:
    return f"io16_hw_output_mxint8_act_stencil_clkwrk_{_output_clk(p, unroll)}_op_hcompute_hw_output_mxint8_act_stencil{_stencil_suffix(p)}_write_0"


def _mxint8_self_port(p: int, unroll: int) -> str:
    return f"hw_output_mxint8_act_stencil_clkwrk_{_output_clk(p, unroll)}_op_hcompute_hw_output_mxint8_act_stencil{_stencil_suffix(p)}_write_0"


def _pe_group_base(p: int) -> str:
    # Matches bin_gold: op_hcompute_output_mxint8_act_cgra_stencil{_stencil_suffix(p)}$inner_compute
    return f"op_hcompute_output_mxint8_act_cgra_stencil{_stencil_suffix(p)}$inner_compute"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_bf_act_self_port(i)] = context.Array(16, context.BitIn())
    for j in range(unroll // 2):
        record[_scale_self_port(j, unroll)] = context.Array(16, context.BitIn())
    for p in range(unroll // 2):
        record[_mxint8_self_port(p, unroll)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, top_module: str):
    if unroll < 2 or unroll % 2 != 0:
        raise ValueError(f"unroll must be even and >= 2, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    top = global_namespace.new_module(top_module, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    bf_act_ios = []
    for i in range(unroll):
        bf_io = defn.add_module_instance(_bf_act_io_name(i), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_bf_act_self_port(i)), bf_io.select("in"))
        bf_act_ios.append(bf_io)

    scale_ios = []
    for j in range(unroll // 2):
        sc_io = defn.add_module_instance(_scale_io_name(j, unroll), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_scale_self_port(j, unroll)), sc_io.select("in"))
        scale_ios.append(sc_io)

    output_ios = []
    for p in range(unroll // 2):
        out_io = defn.add_module_instance(_mxint8_io_name(p, unroll), io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_mxint8_self_port(p, unroll)))
        output_ios.append(out_io)

    # Per-packer 5-PE group. Name suffixes chosen so the intra-group alphabetical
    # sort matches bin_gold: bit8_pack < bit8_unpack_high < bit8_unpack_low
    #                       < e8m0_quant_0 (odd) < e8m0_quant_1 (even).
    pack_pes = []
    unpack_high_pes = []
    unpack_low_pes = []
    quant_odd_pes = []   # e8m0_quant_0: odd-lane (2p+1) / high-byte scale
    quant_even_pes = []  # e8m0_quant_1: even-lane (2p)  / low-byte  scale
    for p in range(unroll // 2):
        base = _pe_group_base(p)
        pack_pe = defn.add_module_instance(f"{base}$bit8_pack_0", pe_module)
        uh_pe = defn.add_module_instance(f"{base}$bit8_unpack_high_0", pe_module)
        ul_pe = defn.add_module_instance(f"{base}$bit8_unpack_low_0", pe_module)
        quant_odd = defn.add_module_instance(f"{base}$e8m0_quant_0", pe_module)
        quant_even = defn.add_module_instance(f"{base}$e8m0_quant_1", pe_module)

        # Scale path for this group.
        # CRITICAL: Lassen's bit8_unpack_low rewrite rule binds its external input
        # to data1 (not data0); data0 is in CONST mode holding an inline constant.
        # bit8_unpack_high is the opposite: external input on data0.
        # Wiring the scale to the wrong port reads the inline const instead of the
        # real scale and corrupts the low byte silently at runtime.
        defn.connect(scale_ios[p].select("out"), uh_pe.select("data0"))
        defn.connect(scale_ios[p].select("out"), ul_pe.select("data1"))

        # Activation path: odd lane (2p+1) + high byte.
        defn.connect(bf_act_ios[2 * p + 1].select("out"), quant_odd.select("data0"))
        defn.connect(uh_pe.select("O0"), quant_odd.select("data1"))

        # Activation path: even lane (2p) + low byte.
        defn.connect(bf_act_ios[2 * p].select("out"), quant_even.select("data0"))
        defn.connect(ul_pe.select("O0"), quant_even.select("data1"))

        # Pack: bit8_pack(odd_quant, even_quant) -> data0 = high byte, data1 = low byte.
        defn.connect(quant_odd.select("O0"), pack_pe.select("data0"))
        defn.connect(quant_even.select("O0"), pack_pe.select("data1"))
        defn.connect(pack_pe.select("O0"), output_ios[p].select("in"))

        pack_pes.append(pack_pe)
        unpack_high_pes.append(uh_pe)
        unpack_low_pes.append(ul_pe)
        quant_odd_pes.append(quant_odd)
        quant_even_pes.append(quant_even)

    top.definition = defn
    context.set_top(top)

    instances = {
        "bf_act_io": bf_act_ios,
        "scale_io": scale_ios,
        "output_io": output_ios,
        "pack_pe": pack_pes,
        "unpack_high_pe": unpack_high_pes,
        "unpack_low_pe": unpack_low_pes,
        "quant_odd_pe": quant_odd_pes,
        "quant_even_pe": quant_even_pes,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, head_dim: int, seq_heads_prod: int,
               block_size: int, top_module: str):
    if head_dim % unroll != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by unroll ({unroll})")
    if seq_heads_prod % block_size != 0:
        raise ValueError(f"seq_heads_prod ({seq_heads_prod}) must be divisible by block_size ({block_size})")

    defn = instances["pack_pe"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    quant_val, quant_w = pe_inst_to_bits_with_operands(
        "e8m0_quant", data0=("ext", None), data1=("ext", None),
    )
    pack_val, pack_w = pe_inst_to_bits_with_operands(
        "bit8_pack", data0=("ext", None), data1=("ext", None),
    )
    # bit8_unpack_{high,low}: use the rewrite rule's default instruction bits.
    # The rule already encodes rega=2 (external-operand mode). Passing
    # data0=("ext", None) to pe_inst_to_bits_with_operands overwrites rega with
    # Mode_t.BYPASS, which ends up with a different bit pattern than the
    # Halide-emitted bin_gold instruction and silently corrupts the unpacked
    # byte. Use pe_inst_to_bits (default) to match bin_gold byte-for-byte.
    uh_val, uh_w = pe_inst_to_bits("bit8_unpack_high")
    ul_val, ul_w = pe_inst_to_bits("bit8_unpack_low")

    def _wire_const(role: str, pe_inst, inst_val: int, inst_w: int):
        c = defn.add_generator_instance(
            f"const_inst_{role}", const_gen,
            context.new_values({"width": inst_w}),
            context.new_values({"value": BitVector[inst_w](inst_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    for p in range(unroll // 2):
        base = _pe_group_base(p)
        _wire_const(f"{base}$bit8_pack_0", instances["pack_pe"][p], pack_val, pack_w)
        _wire_const(f"{base}$bit8_unpack_high_0", instances["unpack_high_pe"][p], uh_val, uh_w)
        _wire_const(f"{base}$bit8_unpack_low_0", instances["unpack_low_pe"][p], ul_val, ul_w)
        _wire_const(f"{base}$e8m0_quant_0", instances["quant_odd_pe"][p], quant_val, quant_w)
        _wire_const(f"{base}$e8m0_quant_1", instances["quant_even_pe"][p], quant_val, quant_w)

    # IO metadata.
    bf_act_extent = head_dim * seq_heads_prod // unroll
    bf_act_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [bf_act_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    scale_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1, 1],
        "dimensionality": 3,
        "extent": [head_dim // unroll, block_size, seq_heads_prod // block_size],
        "read_data_starting_addr": [0],
        "read_data_stride": [1, 1 - (head_dim // unroll), 1],
    })
    mxint8_in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [bf_act_extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for bf_io in instances["bf_act_io"]:
        bf_io.add_metadata("glb2out_0", bf_act_glb2out)
    for sc_io in instances["scale_io"]:
        sc_io.add_metadata("glb2out_0", scale_glb2out)
    for mx_io in instances["output_io"]:
        mx_io.add_metadata("in2glb_0", mxint8_in2glb)


def build_elementwise_unpack_quant_pack_multi_IOs_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    head_dim: int = DEFAULT_HEAD_DIM,
    seq_heads_prod: int = DEFAULT_SEQ_HEADS_PROD,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, top_module)
    _configure(context, instances, unroll, head_dim, seq_heads_prod, block_size, top_module)
    return context, top


def emit_elementwise_unpack_quant_pack_multi_IOs_bf16_design(
    unroll: int,
    head_dim: int,
    seq_heads_prod: int,
    output_path: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_elementwise_unpack_quant_pack_multi_IOs_bf16_context(
        unroll, head_dim, seq_heads_prod, block_size=block_size, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_unpack_quant_pack_multi_IOs_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seq-heads-prod", type=int, default=DEFAULT_SEQ_HEADS_PROD)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_unpack_quant_pack_multi_IOs_bf16_design(
        args.unroll, args.head_dim, args.seq_heads_prod, args.output_path,
        block_size=args.block_size, top_module=args.top_module,
    )
