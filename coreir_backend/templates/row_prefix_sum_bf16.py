"""Build the row-wise BF16 prefix sum (cumsum along the last axis) CoreIR graph for Zircon.

    output[r, c] = input[r, 0] + input[r, 1] + ... + input[r, c]

Templated design parameters:
- unroll: number of parallel lanes; a power of two of at least 4.
- vec_length: row length; G = vec_length // unroll column groups per row,
  2 <= G <= MAX_GROUPS.
- num_vecs: number of rows, 2 to 1024.

Each cycle the lanes hold one group of ``unroll`` consecutive columns of one
row; the groups of a row stream in order and the rows follow each other, so
the tensor keeps its row-major order in GLB. A Brent-Kung network adds the
group's inclusive prefix sums in place (2 * unroll - 2 - log2(unroll) fp_add
PEs); its last lane is the group total T. The carry of group g, the totals
of the earlier groups of the same row, is added to every lane.

Every interconnect hop is a 2-entry register FIFO, so two branches of one
value that meet again run at full rate only if their latencies match; the
short branch of an unbalanced fork needs an elastic FIFO. A down-sweep add
joins a completed prefix with a partial sum 1 to 7 adds shallower, so the
shallow operand waits in a FIFO: a MEM FIFO for a gap of MEM_FIFO_DEPTH_GAP
adds or more, else a Pond FIFO in the add's tile (``get_fifo_pond``).

Carry: shift Pond d (d = 1 .. G - 1, ``get_row_shift_pond``) returns
T[g - d] of the same row, or 0 for g < d, and a chain of adds sums the
shifted streams, carry = ((T[g-1] + T[g-2]) + T[g-3]) + ... Nothing loops,
so a carry leaves every cycle. Each lane's local prefix waits in a MEM FIFO
for the carry of its group; the carry reaches each carry add through a Pond
FIFO in its tile, so the 8 lanes of one output GLB packet do not wait for
the longest carry route.

Two Zircon rules shape the FIFOs:
- Inside a tile, a PE output reaches the tile's Pond without the fan-out
  handshake: a Pond packed with the PE that drives it stores a value again
  whenever another reader stalls. Every Pond FIFO therefore sits in the
  tile of the PE that reads it and is written through a register.
- A Pond's input ready is tied high: a Pond write the schedule holds back is
  lost. A Pond FIFO must never fill, and a shift Pond must never wait for
  its reader, so each shift Pond feeds a nop PE (fp_max with -inf) that
  writes into a MEM FIFO, and the carry adds join the MEM outputs.

At unroll 32 and G = 4 the kernel has 94 PEs, 49 MEMs, 47 Ponds and 47
registers.
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
DEFAULT_VEC_LENGTH = 128
DEFAULT_NUM_VECS = 512
TOP_MODULE = "row_prefix_sum_fp"
MAX_GROUPS = 8  # a shift Pond keeps rows of G groups in 2G - 1 rounded up to a power of 2 of its 32 words
BF16_NEG_INF = 0xFF80
MAX_EXTENT = 1024  # Zircon MEM and Pond extent counters are 11-bit signed
# A down-sweep add whose operands lie this many or more adds apart in depth
# buffers its shallow operand in a MEM FIFO; closer ones use a Pond FIFO.
MEM_FIFO_DEPTH_GAP = 4


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _add_mem(defn, context, mem_gen, mem_genargs, name):
    mem = defn.add_generator_instance(
        name, mem_gen, mem_genargs, context.new_values({"config": {}, "mode": "lake"})
    )
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    return mem


def _add_pond(defn, context, pond_gen, name):
    pond = defn.add_generator_instance(
        name,
        pond_gen,
        context.new_values(
            {"ID": "", "has_stencil_valid": True, "num_inputs": 2, "num_outputs": 2, "width": 16}
        ),
        context.new_values({"config": {}, "mode": "pond"}),
    )
    for key, value in [("config", {}), ("is_rom", False), ("mode", "pond"), ("width", 16)]:
        pond.add_metadata(key, json.dumps(value))
    return pond


def _brent_kung_scan(defn, pe_module, lane_outs, prefix, mem_fifo, pond_fifo):
    """Wire an inclusive Brent-Kung prefix network over ``lane_outs``.

    ``lane_outs`` are the per-lane coreir output selects of the group's
    values. Returns (lane prefix selects, list of the fp_add PE instances).
    Lane i of the result is the sum of lanes 0..i; the last lane is the group
    total. Every PE adds its left neighbour's partial sum to its own.

    An up-sweep add joins two operands of the same depth. A down-sweep add
    joins a completed prefix with a partial sum up to 7 adds shallower, so
    the shallow operand waits in a FIFO: ``mem_fifo(name, src, dst)`` for a
    gap of at least MEM_FIFO_DEPTH_GAP adds, else ``pond_fifo(name, src, pe)``,
    which feeds data0 of the PE from the Pond of its tile.
    """
    n = len(lane_outs)
    values = list(lane_outs)
    depth = [0] * n
    pes = []
    levels = int(math.log2(n))
    # Up-sweep: lane (2^(d+1) - 1 + k * 2^(d+1)) accumulates the lane 2^d below.
    for d in range(levels):
        step = 1 << (d + 1)
        for i in range(step - 1, n, step):
            pe = defn.add_module_instance(f"{prefix}_up{d}_{i}_pe", pe_module)
            defn.connect(values[i - step // 2], pe.select("data0"))
            defn.connect(values[i], pe.select("data1"))
            values[i] = pe.select("O0")
            depth[i] = max(depth[i - step // 2], depth[i]) + 1
            pes.append(pe)
    # Down-sweep: the lanes between the up-sweep lanes receive the nearest
    # completed prefix to their left.
    for d in range(levels - 2, -1, -1):
        step = 1 << (d + 1)
        for i in range(step - 1 + step // 2, n, step):
            name = f"{prefix}_down{d}_{i}"
            pe = defn.add_module_instance(f"{name}_pe", pe_module)
            deep, shallow = values[i - step // 2], values[i]
            gap = depth[i - step // 2] - depth[i]
            assert gap >= 1
            # fp_add is commutative, so the operands may take either input
            if gap >= MEM_FIFO_DEPTH_GAP:
                defn.connect(deep, pe.select("data0"))
                mem_fifo(f"{name}_fifo_mem", shallow, pe.select("data1"))
            else:
                defn.connect(deep, pe.select("data1"))
                pond_fifo(f"{name}_fifo_pond", shallow, pe)
            values[i] = pe.select("O0")
            depth[i] = depth[i - step // 2] + 1
            pes.append(pe)
    return values, pes


def _build_graph(unroll: int, groups: int):
    if unroll < 4 or (unroll & (unroll - 1)) != 0:
        raise ValueError(f"[ERROR] unroll must be a power of two >= 4, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    namespace = context.global_namespace
    pe_module = namespace.modules["PE"]
    io_module = namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    reg_gen = context.get_lib("coreir").generators["reg"]
    corebit_const = context.get_namespace("corebit").modules["const"]
    mem_genargs = make_mem_genargs(context)

    top = namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface
    clk_en = defn.add_module_instance(
        f"{TOP_MODULE}_clk_en_const", corebit_const, context.new_values({"value": True})
    )

    input_ios, output_ios = [], []
    for i in range(unroll):
        io_in = defn.add_module_instance(
            f"io16in_{_input_self_port(i)}", io_module, context.new_values({"mode": "in"})
        )
        io_out = defn.add_module_instance(
            f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"})
        )
        defn.connect(iface.select(_input_self_port(i)), io_in.select("in"))
        defn.connect(io_out.select("out"), iface.select(_output_self_port(i)))
        input_ios.append(io_in)
        output_ios.append(io_out)

    fifo_mems, fifo_ponds, shift_ponds = [], [], []

    def mem_fifo(name, src, dst):
        mem = _add_mem(defn, context, mem_gen, mem_genargs, name)
        defn.connect(src, mem.select("data_in_0"))
        defn.connect(mem.select("data_out_0"), dst)
        fifo_mems.append(mem)

    def pond_fifo(name, src, pe, ponds=fifo_ponds):
        """Elastic FIFO in the Pond of pe's tile, feeding pe's data0.

        The Pond is written through the interconnect: inside a tile, a PE
        output reaches the tile's Pond without the fan-out handshake, so a
        Pond packed with the PE that drives src would store a value again
        whenever another reader of src stalls. The register between src and
        the Pond keeps the mapper from packing it there.
        """
        pond = _add_pond(defn, context, pond_gen, name)
        reg = defn.add_generator_instance(f"{name}_in_reg", reg_gen, context.new_values({"width": 16}),
                                          context.new_values({"clk_posedge": True, "init": BitVector[16](0)}))
        defn.connect(src, reg.select("in"))
        defn.connect(reg.select("out"), pond.select("data_in_pond_0"))
        defn.connect(pond.select("data_out_pond_0"), pe.select("data0"))
        ponds.append(pond)

    # Group-local inclusive prefix sums; the last lane is the group total.
    local_prefix, scan_pes = _brent_kung_scan(
        defn, pe_module, [io.select("out") for io in input_ios], f"{TOP_MODULE}_scan", mem_fifo, pond_fifo
    )

    # The carry of group g of a row is T[g-1] + ... + T[0], the totals of the
    # earlier groups of the row. Shift Pond d returns T[g-d] of the same row,
    # or 0 for g < d (``get_row_shift_pond``), through a nop PE (fp_max with
    # -inf) in its tile, and a chain of adds sums the shifted streams:
    # carry = ((T[g-1] + T[g-2]) + T[g-3]) + ... Nothing loops, so a carry
    # leaves every cycle. A Pond cannot stall its writer (its input ready is
    # tied high), so a shift Pond must never wait for its reader; each nop PE
    # therefore writes into a MEM FIFO, and the adds join the MEM outputs.
    total = local_prefix[unroll - 1]
    streams, nop_pes, sum_pes = [], [], []
    for d in range(1, groups):
        nop = defn.add_module_instance(f"{TOP_MODULE}_carry_shift{d}_pe", pe_module)
        pond_fifo(f"{TOP_MODULE}_carry_shift{d}_pond", total, nop, shift_ponds)
        nop_pes.append(nop)
        if groups == 2:
            streams.append(nop.select("O0"))  # feeds only Pond FIFOs, which never stall
        else:
            mem = _add_mem(defn, context, mem_gen, mem_genargs, f"{TOP_MODULE}_carry_shift{d}_fifo_mem")
            defn.connect(nop.select("O0"), mem.select("data_in_0"))
            streams.append(mem.select("data_out_0"))
            fifo_mems.append(mem)
    carry = streams[0]
    for d, stream in enumerate(streams[1:], start=2):
        pe = defn.add_module_instance(f"{TOP_MODULE}_carry_sum{d}_pe", pe_module)
        defn.connect(stream, pe.select("data0"))
        defn.connect(carry, pe.select("data1"))
        sum_pes.append(pe)
        carry = pe.select("O0")

    # Each lane's local prefix waits in a MEM FIFO for the carry of its group,
    # then the carry PE adds the row carry to it. The carry reaches each carry
    # PE through a Pond FIFO in its tile: the carry fans out to all lanes and
    # the 8 lanes of one GLB packet are written together, so without the
    # FIFOs the shortest carry route would hold back the longest.
    lane_fifos, carry_add_pes = [], []
    for i in range(unroll):
        fifo = _add_mem(defn, context, mem_gen, mem_genargs, f"{TOP_MODULE}_lane_fifo_{i}")
        defn.connect(local_prefix[i], fifo.select("data_in_0"))
        pe = defn.add_module_instance(f"{TOP_MODULE}_carry_add_{i}_pe", pe_module)
        pond_fifo(f"{TOP_MODULE}_carry_fifo_pond_{i}", carry, pe)
        defn.connect(fifo.select("data_out_0"), pe.select("data1"))
        defn.connect(pe.select("O0"), output_ios[i].select("in"))
        lane_fifos.append(fifo)
        carry_add_pes.append(pe)

    for inst in (*lane_fifos, *fifo_mems, *fifo_ponds, *shift_ponds):
        defn.connect(clk_en.select("out"), inst.select("clk_en"))

    top.definition = defn
    context.set_top(top)
    instances = {
        "input_io": input_ios,
        "output_io": output_ios,
        "add_pes": scan_pes + sum_pes + carry_add_pes,
        "nop_pes": nop_pes,
        "lane_fifos": lane_fifos + fifo_mems,
        "fifo_ponds": fifo_ponds,
        "shift_ponds": shift_ponds,
    }
    return context, top, instances


def _groups_per_row(unroll: int, vec_length: int, num_vecs: int) -> int:
    groups = vec_length // unroll
    if vec_length % unroll or not 2 <= groups <= MAX_GROUPS:
        raise ValueError(
            f"[ERROR] row prefix sum needs rows of 2 to {MAX_GROUPS} whole lane groups; "
            f"got vec_length {vec_length}, unroll {unroll}"
        )
    if not 2 <= num_vecs <= MAX_EXTENT:
        raise ValueError(f"[ERROR] row prefix sum supports 2 to {MAX_EXTENT} rows, got {num_vecs}")
    return groups


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int):
    """Attach PE instructions, GLB streams and the MEM/Pond schedules. Mutates the instances."""
    groups = _groups_per_row(unroll, vec_length, num_vecs)
    defn = instances["output_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    insts = [
        (instances["add_pes"], pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))),
        (instances["nop_pes"], pe_inst_to_bits_with_operands("fp_max", data0=("ext", None), data1=("const", BF16_NEG_INF))),
    ]
    for pes, (val, width) in insts:
        for pe in pes:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}",
                const_gen,
                context.new_values({"width": width}),
                context.new_values({"value": BitVector[width](val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    # Per-lane GLB order: one address per group, rows back to back.
    total_groups = groups * num_vecs

    def stream(key):
        return json.dumps(
            {
                "cycle_starting_addr": [0],
                "cycle_stride": [1],
                "dimensionality": 1,
                "extent": [total_groups],
                f"{key}_data_starting_addr": [0],
                f"{key}_data_stride": [1],
            }
        )

    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", stream("read"))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", stream("write"))

    # The MEM FIFOs see one token per group and use one row of groups as
    # their block, so their block counters stay within the 11-bit extent
    # limit and protect unread words when the SRAM wraps.
    fifo_cfg = json.dumps({"type": "fifo", "input_stream_size": total_groups, "row_size": groups})
    for fifo in instances["lane_fifos"]:
        fifo.add_metadata("lake_rv_config", fifo_cfg)
    pond_fifo_cfg = json.dumps({"type": "get_fifo_pond", "input_stream_size": total_groups, "read_port": 2})
    for pond in instances["fifo_ponds"]:
        pond.add_metadata("lake_rv_config", pond_fifo_cfg)
    for d, pond in enumerate(instances["shift_ponds"], start=1):
        pond.add_metadata(
            "lake_rv_config",
            json.dumps({"type": "get_row_shift_pond", "num_groups": groups, "num_rows": num_vecs, "shift": d}),
        )


def _write_pe_fifos_bypass_config(output_path: str):
    """Write an empty PE_fifos_bypass_config.json (no PE needs it), replacing any older one."""
    path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(path, "w") as f:
        json.dump({}, f)


def emit_row_prefix_sum_bf16_design(unroll: int, vec_length: int, num_vecs: int, output_path: str):
    """Write design_top.json (and an empty PE_fifos_bypass_config.json) for one row-prefix-sum kernel."""
    context, top, instances = _build_graph(unroll, _groups_per_row(unroll, vec_length, num_vecs))
    _configure(context, instances, unroll, vec_length, num_vecs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote row_prefix_sum_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate row prefix sum design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec_length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num_vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    emit_row_prefix_sum_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path)
