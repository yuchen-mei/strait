"""
Build the Mamba2 fused decay chain CoreIR graph using pycoreir:

    decay[s, i, j] = exp(cumsum_i(a[s, i] * strict[i, j])) * lower[i, j]

the fused `mul_8` (a times the strict-lower mask), `cumsum_1` (down the rows i), `exp` and `mul_9` (times the lower
mask) of a Mamba2 mixer, per chunk and head a 256 x 256 matrix. Used by: mamba_decay_chain_fp.

One run covers num_mats matrices (heads) of scan_rows rows, on the columns of one 16-lane pass of the 32-lane column
layout (half q of every 32 columns: column 32 m + 16 q + j for lane j, m < groups):
- a (`a`) is one packed GLB tile of `mul_6`'s 16-lane layout (lane = head mod 16, word 4 l + head div 16): lane jl of
  the tile holds heads jl + 8 t + 16 s at word 4 l + s, so one lane carries the run's num_mats heads at every position in
  order. All a_lanes lane IOs of the tile are instantiated with the same stream, only lane `a_lane` is connected (its
  ready decides; the others have no sink and their ready is tied high, so they cannot stall the multi-bank DMA). The
  lane holds a[s, i] for the `groups` beats of row i (stride 0) and writes, through one register, one dual-read MEM per
  two lanes (the D path of elementwise_skip_gate_bf16), read port k of buffer b feeding lane 2 b + k.
- The masks (`mask_strict`, `mask_lower`) are [scan_rows x groups * unroll] (the pass's columns), re-read per matrix.
- The output (`hw_output`) is [num_mats * scan_rows x groups * unroll], position order (matrix, row, group, lane).

Per lane i (beat = matrix s, row i, column group m):
  a buffer i//2 port i%2 -> mul8_pe.data0 ;  mask_strict IO -> mul8_pe.data1       x = a * strict (exact: * 0 or * 1)
  Pond port 0 -> sum_pe.data0 ; mul8_pe.O0 -> sum_pe.data1 ; sum_pe.O0 -> Pond      S = S_up + x (col_prefix_sum_bf16)
  Pond port 1 -> mulb_pe (bf16(1 / ln2) * S) -> clamp_pe (max(z, clamp)) -> z buffer (dual-read MEM)
  z buffer port 0 -> getffrac_pe -> exp ROM ;  ROM, z buffer port 1 -> getfint_pe -> addiexp_pe   2^z ~ exp(S)
  addiexp_pe.O0 -> mul9_pe.data0 ;  mask_lower IO -> mul9_pe.data1                 * lower
  mul9_pe.O0 -> output FIFO MEM -> hw_output IO
The column scan is col_prefix_sum_bf16's recurrence: the sum PE and the Pond of its tile (lake get_col_cumsum_pond, a
ring of the groups running sums; cleared before each matrix), the sum PE's output FIFO bypassed. fp_addiexp adds the
integer to the 8-bit exponent field mod 256, so z < -127 would wrap to a huge value: the clamp (-127) makes those
exactly 0 (exponent field 127 - 127, ROM[0] = 1.0 has mantissa 0). 8 PEs, 1 Pond, 1 ROM, 1 z buffer and 1 output
FIFO per lane, unroll / 2 a buffers.

The z buffer: z feeds two branches that meet again at addiexp (getffrac -> ROM -> data0, getfint -> data1). Fed
directly, the fork sends only when both branches accept, so the short branch's FIFOs fill while the ROM branch is in
flight and a lane moved about 0.5 values per cycle (RTL waveform of the first k0 runs). The Pond's port 1 must keep up
with its loop: its input ready is tied high (an RTL bug), so once port 1 is 30 tokens behind the writes the Pond drops
sums and the scan hangs; that hang was the half-rate chain (paced further by the lockstep mask_lower lanes). As in
elementwise_broadcast_add_softplus_bf16, z is written to a dual-read MEM that is its only consumer; each read port
feeds one branch and runs ahead as far as its branch allows.

DMA patterns (per-lane word addresses, relative strides as SKIP_GLB_DMA_STRIDE_ADJUSTMENT expects, extent-1 loops
dropped). Levels innermost first as (extent, absolute stride), G = groups, R = scan_rows, M = num_mats:
  a:           (G, 0), (R, a_row_words), (M, 1), from a_start (the chunk's first word, tb_write_starting_addr [0])
  masks:       (G, 1), (R, G), (M, 0)
  hw_output:   (G * R * M, 1)                                                     linear
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.templates.col_prefix_sum_bf16 import _shape as col_scan_shape
from strait.coreir_backend.templates.elementwise_broadcast_add_softplus_bf16 import bf16_bits_rne, exp_rom_table
from strait.coreir_backend.templates.elementwise_skip_gate_bf16 import relative_stream
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_GROUPS = 8
DEFAULT_SCAN_ROWS = 256
DEFAULT_NUM_MATS = 4
DEFAULT_A_LANES = 8
DEFAULT_A_ROW_WORDS = 4
DEFAULT_CLAMP = -127.0
TOP_MODULE = "mamba_decay_chain_fp"
_MAX_EXTENT = 1024        # Zircon MEM extent counters
_STORE_DMA_LEVELS = 7     # garnet global_buffer_parameter store_dma_loop_level

_MASKS = ["mask_strict", "mask_lower"]
_LANE_PE_ROLES = [
    "mul8",      # fp_mul(a, mask_strict)
    "sum",       # fp_add(S_up, x): the column scan with the Pond
    "mulb",      # fp_mul(1/ln2, S)
    "clamp",     # fp_max(z, clamp)
    "getffrac",  # fp_getffrac(z)
    "getfint",   # fp_getfint(z)
    "addiexp",   # fp_addiexp(exp_rom[frac], int) = 2^z
    "mul9",      # fp_mul(exp, mask_lower)
]


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _lane_self_port(name: str, i: int) -> str:
    return f"{name}_host_stencil_clkwrk_{i}_op_hcompute_{name}_glb_stencil{_suffix(i)}_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_suffix(i)}_write_0"


def _pe_name(role: str, i: int) -> str:
    # The mapper's Pond packing ILP matches names as substrings: no Pond or PE name may be part of another's
    # (col_prefix_sum_bf16), so every lane name starts with "lane<i>_" and ends with its role.
    # Both Pond outputs feed PEs (port 0 the sum PE, port 1 mulb), and the ILP (garnet mapper/netlist_util.py
    # PackRegsIntoPonds.find_packing) may give the in-tile port 0 to either; it gave it to mulb, and the port swap
    # that follows fails on an RV Pond config (KeyError 'in2regfile_0'). A PE with "bias" in its name is never packed
    # (the ILP's rule 1), so the sum PE keeps port 0 and the scan loop stays in the tile.
    return f"lane{i}_{role}_bias_pe" if role == "mulb" else f"lane{i}_{role}_pe"


def _interface_type(context, unroll: int, a_lanes: int):
    record = {}
    for j in range(a_lanes):
        record[_lane_self_port("a", j)] = context.Array(16, context.BitIn())
    for i in range(unroll):
        for name in _MASKS:
            record[_lane_self_port(name, i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _mem(context, defn, name, mem_gen, genargs):
    mem = defn.add_generator_instance(name, mem_gen, genargs, context.new_values({"config": {}, "mode": "lake"}))
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    return mem


def _rom(context, defn, name, table, mem_gen, genargs):
    rom = defn.add_generator_instance(
        name, mem_gen, genargs, context.new_values({"config": {}, "init": table, "mode": "lake"}),
    )
    for key, value in [("config", {}), ("depth", len(table)), ("init", table), ("is_rom", True), ("mode", "sram"),
                       ("width", 16)]:
        rom.add_metadata(key, json.dumps(value))
    return rom


def _build_graph(unroll: int, a_lanes: int, a_lane: int, top_module: str):
    if unroll < 2 or unroll % 2:
        raise ValueError(f"[ERROR] unroll must be even and >= 2, got {unroll}")
    if not 0 <= a_lane < a_lanes:
        raise ValueError(f"[ERROR] a_lane {a_lane} outside the {a_lanes} lanes of the a GLB tile")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    reg_gen = context.get_lib("coreir").generators["reg"]
    corebit_const = context.get_namespace("corebit").modules["const"]
    genargs = make_mem_genargs(context)
    table = exp_rom_table()

    top = global_namespace.new_module(top_module, _interface_type(context, unroll, a_lanes))
    defn = top.new_definition()
    iface = defn.interface

    # a: all lane IOs of its GLB tile, in numeric lane order (the metadata parser packs IOs four lanes per E64 bank in
    # this order); only lane a_lane drives the register -> one dual-read MEM per two lanes
    a_io = []
    for j in range(a_lanes):
        io = defn.add_module_instance(f"io16in_{_lane_self_port('a', j)}", io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_lane_self_port("a", j)), io.select("in"))
        a_io.append(io)
    a_reg = defn.add_generator_instance(
        "a_in_reg", reg_gen, context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    a_reg.add_metadata("extra_data", json.dumps(0))
    defn.connect(a_io[a_lane].select("out"), a_reg.select("in"))
    a_buffers = []
    for b in range(unroll // 2):
        mem = _mem(context, defn, f"a_buffer_{b}", mem_gen, genargs)
        defn.connect(a_reg.select("out"), mem.select("data_in_0"))
        a_buffers.append(mem)

    pond_clk = defn.add_module_instance(f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}))
    fifo_clk = defn.add_module_instance("output_fifo_clk", corebit_const, context.new_values({"value": True}))

    mask_io = {name: [] for name in _MASKS}
    output_io, pondl, roms, fifos, z_buffers = [], [], [], [], []
    pe_by_role = {role: [] for role in _LANE_PE_ROLES}
    for i in range(unroll):
        ios = {}
        for name in _MASKS:
            io = defn.add_module_instance(f"io16in_{_lane_self_port(name, i)}", io_module, context.new_values({"mode": "in"}))
            defn.connect(iface.select(_lane_self_port(name, i)), io.select("in"))
            mask_io[name].append(io)
            ios[name] = io
        out_io = defn.add_module_instance(f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_output_self_port(i)))
        output_io.append(out_io)

        pes = {}
        for role in _LANE_PE_ROLES:
            pe = defn.add_module_instance(_pe_name(role, i), pe_module)
            pes[role] = pe
            pe_by_role[role].append(pe)
        pond = defn.add_generator_instance(
            f"lane{i}_col_sum_pond", pond_gen,
            context.new_values({"ID": "", "has_stencil_valid": True, "num_inputs": 2, "num_outputs": 2, "width": 16}),
            context.new_values({"config": {}, "mode": "pond"}),
        )
        for key, value in [("config", {}), ("is_rom", False), ("mode", "pond"), ("width", 16)]:
            pond.add_metadata(key, json.dumps(value))
        defn.connect(pond_clk.select("out"), pond.select("clk_en"))
        pondl.append(pond)
        rom = _rom(context, defn, f"lane{i}_exp_rom", table, mem_gen, genargs)
        roms.append(rom)
        z_buffer = _mem(context, defn, f"lane{i}_z_buffer", mem_gen, genargs)
        z_buffers.append(z_buffer)

        # x = a * strict
        defn.connect(a_buffers[i // 2].select(f"data_out_{i % 2}"), pes["mul8"].select("data0"))
        defn.connect(ios["mask_strict"].select("out"), pes["mul8"].select("data1"))
        # S = S_up + x: the sum one row up from Pond port 0; the new sum back into the Pond; port 1 sends it on
        defn.connect(pond.select("data_out_pond_0"), pes["sum"].select("data0"))
        defn.connect(pes["mul8"].select("O0"), pes["sum"].select("data1"))
        defn.connect(pes["sum"].select("O0"), pond.select("data_in_pond_0"))
        # z = max(S / ln2, clamp)  (1/ln2 is mulb's data0 constant)
        defn.connect(pond.select("data_out_pond_1"), pes["mulb"].select("data1"))
        defn.connect(pes["mulb"].select("O0"), pes["clamp"].select("data0"))
        # 2^z: ROM[frac(z)] with the exponent raised by int(z); z through the fork buffer (one read port per branch)
        defn.connect(pes["clamp"].select("O0"), z_buffer.select("data_in_0"))
        defn.connect(z_buffer.select("data_out_0"), pes["getffrac"].select("data0"))
        defn.connect(z_buffer.select("data_out_1"), pes["getfint"].select("data0"))
        defn.connect(pes["getffrac"].select("O0"), rom.select("addr_in_0"))
        defn.connect(rom.select("data_out_0"), pes["addiexp"].select("data0"))
        defn.connect(pes["getfint"].select("O0"), pes["addiexp"].select("data1"))
        # * lower
        defn.connect(pes["addiexp"].select("O0"), pes["mul9"].select("data0"))
        defn.connect(ios["mask_lower"].select("out"), pes["mul9"].select("data1"))
        # Lane-skew buffering of the standalone GLB output lanes (elementwise_mul_add_mul_add_bf16)
        fifo = _mem(context, defn, f"output_fifo_lane_{i}", mem_gen, genargs)
        defn.connect(fifo_clk.select("out"), fifo.select("clk_en"))
        defn.connect(pes["mul9"].select("O0"), fifo.select("data_in_0"))
        defn.connect(fifo.select("data_out_0"), out_io.select("in"))
        fifos.append(fifo)

    names = [pe.name for pes in pe_by_role.values() for pe in pes] + [p.name for p in pondl]
    clash = [(x, y) for x in names for y in names if x != y and x in y]
    assert not clash, f"[ERROR] PE / Pond names inside others (Pond packing): {clash[:4]}"

    top.definition = defn
    context.set_top(top)
    instances = {
        "a_io": a_io,
        "mask_io": mask_io,
        "output_io": output_io,
        "a_buffers": a_buffers,
        "pe": pe_by_role,
        "pond": pondl,
        "rom": roms,
        "output_fifos": fifos,
        "z_buffers": z_buffers,
    }
    return context, top, instances


def stream_levels(groups: int, scan_rows: int, num_mats: int, a_row_words: int = DEFAULT_A_ROW_WORDS):
    """The (extent, absolute stride) levels of the a, mask and output streams (module docstring)."""
    if max(groups, scan_rows, num_mats) > _MAX_EXTENT:
        raise ValueError(f"[ERROR] a loop extent exceeds {_MAX_EXTENT}")
    if a_row_words < num_mats:
        raise ValueError(f"[ERROR] the a tile's {a_row_words} words per row cannot hold {num_mats} matrices")
    return {
        "a": [(groups, 0), (scan_rows, a_row_words), (num_mats, 1)],
        "mask": [(groups, 1), (scan_rows, groups), (num_mats, 0)],
        "output": [(groups * scan_rows * num_mats, 1)],
    }


def _configure(context, instances, unroll: int, groups: int, scan_rows: int, num_mats: int, a_row_words: int,
               a_start: int, clamp: float):
    # The Pond's limits (ring depth, chunked counting of long scans) are col_prefix_sum_bf16's
    col_scan_shape(unroll, groups * unroll, num_mats * scan_rows, scan_rows)
    levels = stream_levels(groups, scan_rows, num_mats, a_row_words)
    beats = groups * scan_rows * num_mats
    if beats > 8 * _MAX_EXTENT:
        raise ValueError(f"[ERROR] {beats} beats per lane exceed the linear output stream's {8 * _MAX_EXTENT}")
    defn = instances["output_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    insts = {
        "mul8": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
        "sum": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None)),
        "mulb": pe_inst_to_bits_with_operands("fp_mul", data0=("const", bf16_bits_rne(1.0 / 0.6931471805599453)),
                                              data1=("ext", None)),
        "clamp": pe_inst_to_bits_with_operands("fp_max", data0=("ext", None), data1=("const", bf16_bits_rne(clamp))),
        "getffrac": pe_inst_to_bits_with_operands("fp_getffrac", data0=("ext", None)),
        "getfint": pe_inst_to_bits_with_operands("fp_getfint", data0=("ext", None)),
        "addiexp": pe_inst_to_bits_with_operands("fp_addiexp", data0=("ext", None), data1=("ext", None)),
        "mul9": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
    }
    for role, (inst_val, inst_width) in insts.items():
        for pe in instances["pe"][role]:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}", const_gen,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    pond_cfg = json.dumps({"type": "get_col_cumsum_pond", "num_groups": groups, "num_rows": scan_rows,
                           "num_mats": num_mats})
    for pond in instances["pond"]:
        pond.add_metadata("lake_rv_config", pond_cfg)

    # a buffers carry one value per beat; smallest 4-word multiple block that splits it into at most _MAX_EXTENT blocks
    block = next((r for r in range(4, 513, 4) if beats % r == 0 and beats // r <= _MAX_EXTENT), None)
    if block is None:
        raise ValueError(f"[ERROR] {beats} values per lane do not split into <= {_MAX_EXTENT} blocks")
    buffer_cfg = json.dumps({"type": "dual_read", "input_stream_size": beats, "row_size": block})
    for mem in instances["a_buffers"] + instances["z_buffers"]:
        mem.add_metadata("lake_rv_config", buffer_cfg)
    for fifo in instances["output_fifos"]:
        fifo.add_metadata("lake_rv_config", json.dumps({"type": "fifo", "input_stream_size": beats, "row_size": 8}))

    a_read = json.dumps(relative_stream("read", levels["a"], start=a_start))
    for io in instances["a_io"]:
        io.add_metadata("glb2out_0", a_read)
    mask_read = json.dumps(relative_stream("read", levels["mask"]))
    for name in _MASKS:
        for io in instances["mask_io"][name]:
            io.add_metadata("glb2out_0", mask_read)
    output_write = json.dumps(relative_stream("write", levels["output"], max_levels=_STORE_DMA_LEVELS))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", output_write)


def _write_pe_fifos_bypass_config(output_path: str, unroll: int):
    """Bypass the output FIFO of every sum PE: its only sink is the Pond of its tile (col_prefix_sum_bf16)."""
    cfg = {_pe_name("sum", i): {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1} for i in range(unroll)}
    with open(os.path.join(output_path, "PE_fifos_bypass_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def build_masked_col_cumsum_exp_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    groups: int = DEFAULT_GROUPS,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    num_mats: int = DEFAULT_NUM_MATS,
    a_lanes: int = DEFAULT_A_LANES,
    a_lane: int = 0,
    a_row_words: int = DEFAULT_A_ROW_WORDS,
    a_start: int = 0,
    clamp: float = DEFAULT_CLAMP,
    top_module: str = TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, a_lanes, a_lane, top_module)
    _configure(context, instances, unroll, groups, scan_rows, num_mats, a_row_words, a_start, clamp)
    return context, top


def emit_masked_col_cumsum_exp_bf16_design(
    unroll: int,
    groups: int,
    scan_rows: int,
    num_mats: int,
    output_path: str,
    a_lanes: int = DEFAULT_A_LANES,
    a_lane: int = 0,
    a_row_words: int = DEFAULT_A_ROW_WORDS,
    a_start: int = 0,
    clamp: float = DEFAULT_CLAMP,
    top_module: str = TOP_MODULE,
):
    """Write design_top.json and PE_fifos_bypass_config.json for one decay chain run."""
    context, top = build_masked_col_cumsum_exp_bf16_context(
        unroll, groups, scan_rows, num_mats, a_lanes, a_lane, a_row_words, a_start, clamp, top_module)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote masked_col_cumsum_exp_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, unroll)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate masked_col_cumsum_exp_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--scan-rows", type=int, default=DEFAULT_SCAN_ROWS)
    parser.add_argument("--num-mats", type=int, default=DEFAULT_NUM_MATS)
    parser.add_argument("--a-lane", type=int, default=0)
    parser.add_argument("--a-start", type=int, default=0)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_masked_col_cumsum_exp_bf16_design(args.unroll, args.groups, args.scan_rows, args.num_mats, args.output_path,
                                           a_lane=args.a_lane, a_start=args.a_start)
