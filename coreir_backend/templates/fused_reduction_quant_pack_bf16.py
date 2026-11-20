"""Build one-pass MX (MXINT8, E8M0 scale) quantization of a GLB tensor for Zircon.

The kernel reads a BF16 [num_pixels, num_channels] tensor once and writes both
outputs of the existing two-pass flow (get_e8m0_scale_tree_gb_input then
apply_e8m0_scale_single_IO), in the same GLB layouts:

- hw_output_mxint8_act: INT8 values, two per 16-bit word (odd channel in the
  high byte).
- hw_scale_packed_output: E8M0 block scales, two per 16-bit word, alternating
  between two output IOs (GLB bank toggle).

An MX block is 64 consecutive channels of one pixel. Each cycle the
``unroll`` = 32 input lanes hold 32 consecutive channels (one pass), so a block
is an even pass followed by an odd pass.

Scale path, one scale per block:
    lane register -> Pond FIFO (-> nop PE)  -> elastic tree inputs
    fp_abs_max tree over the lanes          -> |max| of one pass
    MEM splits even and odd passes          -> pass pairs of one block
    fp_abs_max(even pass, odd pass)         -> |max| of the block
    get_shared_exp                          -> E8M0 scale (biased exponent - 6)
The scale reaches the quantizers through ``BRANCH`` pass-through PEs, so no single
output drives every quantizer (the extra stage also evens out the broadcast:
a direct 66-way fan-out ran slower).

Every interconnect hop is a 2-entry register FIFO, so rigid paths from one
input GLB packet (the GLB hands the 8 lanes of two IO tiles over together)
that meet again in the tree run at full rate only if their routes match
exactly. Every lane therefore enters the tree through a Pond FIFO in the tile
of the PE that reads it: lane 2j into data0 of first-stage PE j, lane 2j + 1
through a nop PE (or_ with 0) into its data1. A lane register is the IO's
only sink and writes both the lane buffer and the Pond: a net that ends in a
register must not also pass that register's switch-box output on to another
sink, or the output's mux would have to select both. A Pond cannot hold back
its writer (its input ready is tied high), so it must never fill; here the
tree's output goes to a MEM and the GLB packets start together and stream at
one pass per cycle, so a Pond only holds the latency gap of its lane's path.

Each lane's values wait in a MEM while the scale of their block is computed.
The MEM splits the lane's stream: read port 0 returns the even passes and read
port 1 the odd passes. Each lane has one ``e8m0_quant`` PE per parity, and each
of them uses one scale per block, so the scale never has to be repeated.
``bit8_pack`` pairs lanes into output words, separately for both parities.

The outputs use 8 E64 IO tiles: the 16 even-pass words go to the first four
and the 16 odd-pass words to the last four. The host spreads a tensor's 64-bit
words over its IO tiles in turn, so with 8 tiles the four words of an even pass
land in tiles 0-3 and those of the next (odd) pass in tiles 4-7, both at the
block index: the GLB holds the tensor in its original order.

The block scales feed the scale packing of apply_e8m0_scale_single_IO:
``bit8_pack`` of a scale and its one-token-delayed copy; the two scale IOs
take the words for blocks (1, 0), (5, 4), ... and (3, 2), (7, 6), ...

At unroll 32 the kernel uses 31 + 16 + 1 + 1 + 8 + 64 + 32 + 1 = 154 PEs (tree,
lane nops, block max, scale, branches, quantizers, packers, scale pack),
33 MEMs (32 lane buffers and the pass split), 32 Ponds and 43 registers.

Templated design parameters:
- unroll: input lanes; 32, so that two passes form one 64-channel block.
- num_pixels: rows (vec_height).
- num_channels: channels per row (vec_width), a multiple of 128 so each row
  has an even number of blocks to pack.
"""

import json
import math
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_NUM_PIXELS = 128
DEFAULT_NUM_CHANNELS = 256
DEFAULT_NUM_PIPELINE_FIFOS = 10
DEFAULT_TOP_MODULE = "fused_e8m0_quant_single_IO"
BLOCK_SIZE = 64
NUM_SCALE_IOS = 2
BRANCH = 8  # quantizers fed by one scale branch PE
MAX_EXTENT = 1024  # Zircon MEM extent counters are 11-bit signed
PARITIES = ("even", "odd")


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _bf_act_self_port(i: int) -> str:
    return f"input_bf_act_host_stencil_clkwrk_{i}_op_hcompute_input_bf_act_glb_stencil{_suffix(i)}_read_0"


def _mxint8_self_port(i: int) -> str:
    return f"hw_output_mxint8_act_stencil_clkwrk_{i}_op_hcompute_hw_output_mxint8_act_stencil{_suffix(i)}_write_0"


def _scale_packed_self_port(i: int) -> str:
    # Halide emits write_0 for idx 0 and write_1 for idx 1 under bank-toggle mode.
    return f"hw_scale_packed_output_stencil_clkwrk_{i}_op_hcompute_hw_scale_packed_output_stencil_{i}_write_{i}"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_bf_act_self_port(i)] = context.Array(16, context.BitIn())
    for p in range(unroll):
        record[_mxint8_self_port(p)] = context.Array(16, context.Bit())
    for i in range(NUM_SCALE_IOS):
        record[_scale_packed_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _add_mem(defn, context, mem_gen, mem_genargs, clk_en, name):
    mem = defn.add_generator_instance(
        name, mem_gen, mem_genargs, context.new_values({"config": {}, "mode": "lake"})
    )
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    defn.connect(clk_en.select("out"), mem.select("clk_en"))
    return mem


def _add_reg(defn, context, reg_gen, name, extra_data):
    """A coreir.reg that becomes an interconnect FIFO; extra_data=1 starts it with one zero token."""
    reg = defn.add_generator_instance(
        name,
        reg_gen,
        context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    reg.add_metadata("extra_data", json.dumps(extra_data))
    return reg


def _build_graph(unroll: int, num_pipeline_fifos: int, top_module: str):
    if unroll * 2 != BLOCK_SIZE:
        raise ValueError(f"[ERROR] unroll must be {BLOCK_SIZE // 2} (two passes per MX block), got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    namespace = context.global_namespace
    pe_module = namespace.modules["PE"]
    io_module = namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    reg_gen = context.get_lib("coreir").generators["reg"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    corebit_const = context.get_namespace("corebit").modules["const"]
    mem_genargs = make_mem_genargs(context)

    top = namespace.new_module(top_module, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface
    clk_en = defn.add_module_instance(
        f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True})
    )
    pes = {"fp_abs_max": [], "get_shared_exp": [], "or_": [], "e8m0_quant": [], "bit8_pack": []}

    def pe(name, op):
        inst = defn.add_module_instance(f"{top_module}_{name}", pe_module)
        pes[op].append(inst)
        return inst

    # Inputs: each lane feeds the abs-max tree and its own buffer MEM.
    input_ios, lane_buffers, lane_regs = [], [], []
    for i in range(unroll):
        io = defn.add_module_instance(
            f"io16in_{_bf_act_self_port(i)}", io_module, context.new_values({"mode": "in"})
        )
        defn.connect(iface.select(_bf_act_self_port(i)), io.select("in"))
        buf = _add_mem(defn, context, mem_gen, mem_genargs, clk_en, f"{top_module}_lane_buffer_{i}")
        # The lane register is the IO's only sink and feeds both the lane buffer
        # and the lane's Pond FIFO. A net that ends in a register must not also
        # pass that register's switch-box output on to another sink: that
        # output's mux would have to select the register and the bypass at once.
        lane_reg = _add_reg(defn, context, reg_gen, f"{top_module}_lane_reg_{i}", 0)
        defn.connect(io.select("out"), lane_reg.select("in"))
        defn.connect(lane_reg.select("out"), buf.select("data_in_0"))
        input_ios.append(io)
        lane_buffers.append(buf)
        lane_regs.append(lane_reg.select("out"))

    fifo_ponds = []

    def pond_fifo(src, t):
        """Elastic FIFO from a register output src into data0 of PE t, in the Pond of t's tile.

        Writing the Pond from a register keeps the mapper from packing it with a
        PE that drives it (inside a tile a PE output reaches the Pond without the
        fan-out handshake). A Pond cannot hold back its writer, so it must never
        fill: here it only holds how long its tokens wait for the other inputs
        of the tree.
        """
        pond = defn.add_generator_instance(
            f"{t.name}_in_pond",
            pond_gen,
            context.new_values({"ID": "", "has_stencil_valid": True, "num_inputs": 2, "num_outputs": 2, "width": 16}),
            context.new_values({"config": {}, "mode": "pond"}),
        )
        for key, value in [("config", {}), ("is_rom", False), ("mode", "pond"), ("width", 16)]:
            pond.add_metadata(key, json.dumps(value))
        defn.connect(clk_en.select("out"), pond.select("clk_en"))
        defn.connect(src, pond.select("data_in_pond_0"))
        defn.connect(pond.select("data_out_pond_0"), t.select("data0"))
        fifo_ponds.append(pond)

    # |max| of one pass over all lanes (fp_abs_max is exact and commutative).
    # Every interconnect hop is a 2-entry register FIFO, so two rigid paths
    # from one input GLB packet (the GLB hands its 8 lanes over together) that
    # meet again run at full rate only if their routes match exactly. Every
    # lane therefore enters the tree through a Pond FIFO: lane 2j into data0
    # of first-stage PE j, lane 2j + 1 through a nop PE (or_ with 0) into its
    # data1. The tree then has no rigid paths that meet again, its output goes
    # to a MEM, and the GLB packets start together and stream at one pass per
    # cycle, so each Pond holds only the latency gap of its lane's path.
    level = []
    for j in range(unroll // 2):
        t = pe(f"tree_stage1_{j}_pe", "fp_abs_max")
        pond_fifo(lane_regs[2 * j], t)
        nop = pe(f"lane_nop_{2 * j + 1}_pe", "or_")
        pond_fifo(lane_regs[2 * j + 1], nop)
        defn.connect(nop.select("O0"), t.select("data1"))
        level.append(t.select("O0"))
    for stage in range(2, int(math.log2(unroll)) + 1):
        nxt = []
        for j in range(len(level) // 2):
            t = pe(f"tree_stage{stage}_{j}_pe", "fp_abs_max")
            defn.connect(level[2 * j], t.select("data0"))
            defn.connect(level[2 * j + 1], t.select("data1"))
            nxt.append(t.select("O0"))
        level = nxt
    pass_max = level[0]

    # |max| of a block: a MEM splits the pass maxima into the even passes (read
    # port 0) and the odd passes (read port 1), and one fp_abs_max joins them,
    # so the stream has one block maximum, and then one scale, per block. The
    # two read ports run independently, so the two branches need not match.
    pass_split = _add_mem(defn, context, mem_gen, mem_genargs, clk_en, f"{top_module}_pass_split_mem")
    defn.connect(pass_max, pass_split.select("data_in_0"))
    block_max = pe("block_max_pe", "fp_abs_max")
    defn.connect(pass_split.select("data_out_0"), block_max.select("data0"))
    defn.connect(pass_split.select("data_out_1"), block_max.select("data1"))
    scale_pe = pe("get_shared_exp_pe", "get_shared_exp")
    defn.connect(block_max.select("O0"), scale_pe.select("data0"))
    block_scale = scale_pe.select("O0")

    # Quantize: per lane one PE per pass parity, all fed with the block scale
    # through branch PEs (or_ with 0 passes the scale through unchanged).
    branches = []
    for b in range(2 * unroll // BRANCH):
        br = pe(f"scale_branch_pe_{b}", "or_")
        defn.connect(block_scale, br.select("data0"))
        branches.append(br.select("O0"))
    quant = {parity: [] for parity in PARITIES}
    for i in range(unroll):
        for k, parity in enumerate(PARITIES):
            q = pe(f"e8m0_quant_{parity}_pe_{i}", "e8m0_quant")
            defn.connect(lane_buffers[i].select(f"data_out_{k}"), q.select("data0"))
            defn.connect(branches[(k * unroll + i) // BRANCH], q.select("data1"))
            quant[parity].append(q)

    # Pack lane pairs; output lanes 0-15 carry even passes, 16-31 odd passes.
    output_ios = []
    for k, parity in enumerate(PARITIES):
        for p in range(unroll // 2):
            lane = k * (unroll // 2) + p
            io = defn.add_module_instance(
                f"io16_{_mxint8_self_port(lane)}", io_module, context.new_values({"mode": "out"})
            )
            defn.connect(io.select("out"), iface.select(_mxint8_self_port(lane)))
            pk = pe(f"output_pack_{parity}_pe_{p}", "bit8_pack")
            # bit8_pack(odd lane, even lane): data0 is the high byte.
            defn.connect(quant[parity][2 * p + 1].select("O0"), pk.select("data0"))
            defn.connect(quant[parity][2 * p].select("O0"), pk.select("data1"))
            defn.connect(pk.select("O0"), io.select("in"))
            output_ios.append(io)

    # Scale packing, as in apply_e8m0_scale_single_IO: the "a_" prefix keeps this
    # PE first in name order (netlist id p0), and its direct edge to scale IO 1
    # lets the manual placer find it.
    scale_pack = pe("a_scale_pack_pe", "bit8_pack")
    scale_shift = _add_reg(defn, context, reg_gen, f"{top_module}_scale_shift_fifo", 1)
    defn.connect(block_scale, scale_pack.select("data0"))
    defn.connect(block_scale, scale_shift.select("in"))
    defn.connect(scale_shift.select("out"), scale_pack.select("data1"))
    scale_ios = []
    for i in range(NUM_SCALE_IOS):
        io = defn.add_module_instance(
            f"io16_{_scale_packed_self_port(i)}", io_module, context.new_values({"mode": "out"})
        )
        defn.connect(io.select("out"), iface.select(_scale_packed_self_port(i)))
        scale_ios.append(io)
    upstream = scale_pack.select("O0")
    for f in range(num_pipeline_fifos):
        reg = _add_reg(defn, context, reg_gen, f"{top_module}_pipeline_fifo_scale_packed_output_{f}", 0)
        defn.connect(upstream, reg.select("in"))
        upstream = reg.select("out")
    defn.connect(upstream, scale_ios[0].select("in"))
    defn.connect(scale_pack.select("O0"), scale_ios[1].select("in"))

    top.definition = defn
    context.set_top(top)
    instances = {
        "input_io": input_ios,
        "output_io": output_ios,
        "scale_io": scale_ios,
        "lane_buffer": lane_buffers,
        "pass_split": pass_split,
        "fifo_ponds": fifo_ponds,
        "pe": pes,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, num_pixels: int, num_channels: int):
    """Attach PE instructions, GLB streams and MEM schedules. Mutates the instances."""
    if num_channels % (2 * BLOCK_SIZE) != 0:
        raise ValueError(
            f"[ERROR] num_channels ({num_channels}) must be a multiple of {2 * BLOCK_SIZE} "
            f"so each row has an even number of MX blocks"
        )
    passes = num_pixels * num_channels // unroll
    num_scales = passes // 2
    defn = instances["input_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]
    operands = {
        "get_shared_exp": {"data0": ("ext", None)},
        "or_": {"data0": ("ext", None), "data1": ("const", 0)},
    }
    for op, pe_list in instances["pe"].items():
        inst_val, inst_w = pe_inst_to_bits_with_operands(
            op, **operands.get(op, {"data0": ("ext", None), "data1": ("ext", None)})
        )
        for pe in pe_list:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}",
                const_gen,
                context.new_values({"width": inst_w}),
                context.new_values({"value": BitVector[inst_w](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    def stream(key, extent, start=0, stride=1):
        return json.dumps(
            {
                "cycle_starting_addr": [start],
                "cycle_stride": [stride],
                "dimensionality": 1,
                "extent": [extent],
                f"{key}_data_starting_addr": [0],
                f"{key}_data_stride": [1],
            }
        )

    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", stream("read", passes))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", stream("write", num_scales))
    # The pack PE sees (scale of block t, scale of block t - 1); word t = 1, 5, 9, ...
    # goes to scale IO 0 and t = 3, 7, 11, ... to scale IO 1.
    for idx, io in enumerate(instances["scale_io"]):
        io.add_metadata(
            "in2glb_0", stream("write", num_scales // 2 // NUM_SCALE_IOS, 1 + idx * 2, NUM_SCALE_IOS * 2)
        )

    # Lane buffer: even passes on read port 0, odd passes on read port 1, in
    # blocks of 8 passes so the block counters fit the extent limit.
    row = next(r for r in range(8, MAX_EXTENT + 1, 2) if passes % r == 0 and passes // r <= MAX_EXTENT)
    lane_cfg = json.dumps({"type": "get_filter_mem_two_streams", "input_stream_size": passes, "row_size": row})
    for buf in instances["lane_buffer"]:
        buf.add_metadata("lake_rv_config", lane_cfg)
    # Pass maxima: even passes on read port 0, odd passes on read port 1, in
    # rows of 4 passes (one SRAM word), so a row is read only after the
    # following row was written and no word is still in the write aggregator.
    instances["pass_split"].add_metadata(
        "lake_rv_config", json.dumps({"type": "get_filter_mem_two_streams", "input_stream_size": passes, "row_size": 4})
    )
    pond_cfg = json.dumps({"type": "get_fifo_pond", "input_stream_size": passes, "read_port": 2})
    for pond in instances["fifo_ponds"]:
        pond.add_metadata("lake_rv_config", pond_cfg)


def emit_fused_reduction_quant_pack_bf16_design(
    unroll: int,
    num_pixels: int,
    num_channels: int,
    output_path: str,
    num_pipeline_fifos: int = DEFAULT_NUM_PIPELINE_FIFOS,
    top_module: str = DEFAULT_TOP_MODULE,
):
    """Write design_top.json for one one-pass MX quantization kernel into output_path."""
    context, top, instances = _build_graph(unroll, num_pipeline_fifos, top_module)
    _configure(context, instances, unroll, num_pixels, num_channels)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote fused_reduction_quant_pack_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one-pass MX quantization design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--num_pixels", type=int, default=DEFAULT_NUM_PIXELS)
    parser.add_argument("--num_channels", type=int, default=DEFAULT_NUM_CHANNELS)
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    emit_fused_reduction_quant_pack_bf16_design(args.unroll, args.num_pixels, args.num_channels, args.output_path)
