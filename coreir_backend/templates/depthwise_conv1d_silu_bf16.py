"""Build the Zircon four-tap causal depthwise Conv1D, optionally fused with SiLU.

    y[t, c] = bias[c] + w[c, 0] x[t-3, c] + w[c, 1] x[t-2, c] + w[c, 2] x[t-1, c] + w[c, 3] x[t, c]
    out[t, c] = y[t, c] * sigmoid(y[t, c])      (with SiLU)

x is the [length, channels] activation of one L2 tile (the Mamba conv1d input
in NLC order), w the [channels, 1, 4] weights and bias the [channels] bias.
Values before t = 0 are zero (PyTorch padding 3, first ``length`` outputs).

One kernel call handles ``unroll`` channels, one channel per lane. The tile's
GLB layout is position-major, so each cycle the lanes receive one position of
all their channels and every lane streams the whole sequence of its channel.
A longer channel dimension is split into L2 tiles of ``unroll`` channels.

Every interconnect hop of a route is a 2-entry register FIFO, so a value
that feeds two branches which meet again runs at full rate only if both
branches have the same latency; otherwise the short branch fills up and the
fork runs at about 2 L_short / (L_long + L_short). Each fork below therefore
gets an elastic buffer: a dual-read MEM at the fork, whose two read ports
run independently, or a FIFO on the short branch.

Per lane (transposed FIR form):
- x is written to two dual-read MEMs, its only consumers. Their four read
  ports feed the four fp_mul taps, so each tap has its own x stream. (Fed
  straight from the IO, the taps' routes differ by several hops and the
  slowest lane ran at about 75%.) The MEMs also absorb the latency
  differences between the 8 lanes, which share their IO packets.
- Product k is delayed by 3 - k register FIFOs that each start with one zero:
  p_k(t) = w[c, k] x[t - (3 - k)], and the zeros are the values before t = 0.
- Four fp_add PEs sum a balanced tree, y = ((p3 + p2) + (p1 + p0)) + bias;
  multiply and add round separately.
- SiLU uses the exp and reciprocal MEM ROMs of the Zircon swish template:
      n = y * (-1/ln2); e = 2^n = exp(-y)           (fp_getffrac -> exp ROM, fp_getfint, fp_addiexp)
      d = e + 1; 1/d from the mantissa ROM         (fp_getmant -> div ROM, fp_subexp)
      out = y * (1/d)
  y, n and d each feed a long branch directly and a short branch through a
  FIFO, so no buffer sits on the critical path: y reaches the final multiply
  through a MEM FIFO (that branch waits for the whole SiLU chain); n reaches
  fp_getfint, and d reaches fp_subexp through a nop PE, through a Pond FIFO
  (``get_fifo_pond``) in the tile of the reading PE. The Pond is written
  through the interconnect: inside a tile, a PE output reaches the tile's
  Pond without the fan-out handshake, so a Pond packed with the PE that
  writes it stores a value again whenever the other reader stalls. A
  register between the source and the Pond keeps the mapper from packing it
  there.

Weights and bias come from GLB once and are then replayed inside the tile.
A channel's four weights are one 64-bit GLB word, so the four lanes of one
E64 weight IO are the four taps of one channel; each bias IO carries the
64-bit word of four consecutive channels' bias. Each IO reads its word
``_CONST_READS`` times, and every tap multiply (and the bias add) keeps the
value in the Pond of its own tile (``get_repeat_pond``), which returns it
once per position. Streaming the weights from GLB at every position instead
ties the taps of two channels together through the GLB packet.

Resources at unroll 8 with SiLU: 136 PEs, 40 MEMs (two x buffers, a y FIFO
and two ROMs per lane), 56 Ponds (weights, bias and two FIFOs per lane),
64 registers and 14 E64 IO tiles: 2 input, 8 weight, 2 bias, 2 output.

The SiLU exp ROM covers |y| up to about 88; larger magnitudes overflow the
exponent arithmetic.
"""

import json
import math
import os
import struct
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 8
DEFAULT_LENGTH = 1024
TOP_MODULE = "depthwise_conv1d_silu_fp"
TAPS = 4
LANES_PER_IO = 4  # E64: one IO tile carries four 16-bit lanes
MAX_EXTENT = 1024  # Zircon MEM extent counters are 11-bit signed
_CONST_READS = 2  # GLB reads of each weight/bias word; GLB loop extents must be at least 2


def bf16_bits_rne(x: float) -> int:
    """BF16 bit pattern of x, rounded to nearest even."""
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF


def exp_rom_table():
    """256 entries indexed by the fp_getffrac byte: 2^(i/128), i in [0, 128) then [-128, 0)."""
    return [bf16_bits_rne(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_rne(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


def div_rom_table():
    """128 entries indexed by the fp_getmant mantissa: 1 / (1 + i/128)."""
    return [bf16_bits_rne(1.0 / (1.0 + i / 128.0)) for i in range(128)]


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _port(tensor: str, i: int, write: bool = False) -> str:
    kind = "write" if write else "read"
    stage = tensor if write else tensor.replace("_host", "_glb")
    return f"{tensor}_stencil_clkwrk_{i}_op_hcompute_{stage}_stencil{_suffix(i)}_{kind}_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_port("input_host", i)] = context.Array(16, context.BitIn())
        record[_port("hw_output", i, write=True)] = context.Array(16, context.Bit())
    for i in range(unroll * TAPS):
        record[_port("weight_host", i)] = context.Array(16, context.BitIn())
    for i in range(unroll):
        record[_port("bias_host", i)] = context.Array(16, context.BitIn())
    return context.Record(record)


def _add_mem(defn, context, mem_gen, genargs, clk_en, name, rom_table=None):
    mem = defn.add_generator_instance(
        name,
        mem_gen,
        genargs,
        context.new_values(
            {"config": {}, "init": rom_table, "mode": "lake"}
            if rom_table is not None
            else {"config": {}, "mode": "lake"}
        ),
    )
    if rom_table is not None:
        meta = [("config", {}), ("depth", len(rom_table)), ("init", rom_table), ("is_rom", True),
                ("mode", "sram"), ("width", 16)]
    else:
        meta = [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]
        defn.connect(clk_en.select("out"), mem.select("clk_en"))
    for key, value in meta:
        mem.add_metadata(key, json.dumps(value))
    return mem


def _add_delay(defn, context, reg_gen, name):
    """Register FIFO that starts with one zero: its output stream is the input delayed by one."""
    reg = defn.add_generator_instance(
        name,
        reg_gen,
        context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    reg.add_metadata("extra_data", json.dumps(1))
    return reg


# PE roles in one lane and their instructions: (op, operands); "ext" marks a
# wired operand, a float a BF16 constant.
_CONV_OPS = {f"tap{k}_mul": ("fp_mul", ("ext", "ext")) for k in range(TAPS)}
_CONV_OPS.update({f"acc{k}_add": ("fp_add", ("ext", "ext")) for k in range(TAPS)})
_SILU_OPS = {
    "silu_neg_log2": ("fp_mul", ("ext", -1.0 / math.log(2.0))),
    "silu_getffrac": ("fp_getffrac", ("ext",)),
    "silu_getfint": ("fp_getfint", ("ext",)),
    "silu_addiexp": ("fp_addiexp", ("ext", "ext")),
    "silu_add_one": ("fp_add", ("ext", 1.0)),
    "silu_exp_nop": ("fp_max", ("ext", float("-inf"))),
    "silu_getmant": ("fp_getmant", ("ext",)),
    "silu_subexp": ("fp_subexp", ("ext", "ext")),
    "silu_mul_final": ("fp_mul", ("ext", "ext")),
}


def _build_graph(unroll: int, fuse_silu: bool):
    if unroll % LANES_PER_IO:
        raise ValueError(f"[ERROR] unroll must be a multiple of {LANES_PER_IO}, got {unroll}")
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
    genargs = make_mem_genargs(context)

    top = namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface
    clk_en = defn.add_module_instance(f"{TOP_MODULE}_clk_en_const", corebit_const, context.new_values({"value": True}))

    def io(name, mode):
        inst = defn.add_module_instance(("io16in_" if mode == "in" else "io16_") + name, io_module,
                                        context.new_values({"mode": mode}))
        if mode == "in":
            defn.connect(iface.select(name), inst.select("in"))
        else:
            defn.connect(inst.select("out"), iface.select(name))
        return inst

    instances = {"input_io": [], "weight_io": [], "bias_io": [], "output_io": [], "x_buffer": [], "y_fifo": [],
                 "pond": [], "fifo_pond": [], "pe": {}}

    def _add_pond(name):
        pond = defn.add_generator_instance(
            name,
            pond_gen,
            context.new_values({"ID": "", "has_stencil_valid": True, "num_inputs": 2, "num_outputs": 2, "width": 16}),
            context.new_values({"config": {}, "mode": "pond"}),
        )
        for key, value in [("config", {}), ("is_rom", False), ("mode", "pond"), ("width", 16)]:
            pond.add_metadata(key, json.dumps(value))
        defn.connect(clk_en.select("out"), pond.select("clk_en"))
        return pond

    def repeat_pond(name, src, pe_port):
        """Pond that stores the value from src and feeds it to pe_port once per position."""
        pond = _add_pond(name)
        defn.connect(src, pond.select("data_in_pond_0"))
        defn.connect(pond.select("data_out_pond_0"), pe_port)
        instances["pond"].append(pond)

    def fifo_pond(name, src, pe_port):
        """Elastic FIFO Pond in the tile of the PE that reads it (pe_port must be data0).

        The Pond is written through the interconnect: a register between src and
        the Pond keeps the mapper from packing it with the PE that drives src.
        Inside a tile, a PE output reaches its Pond without the fan-out
        handshake, so a Pond packed with the source PE would store a value
        again whenever another reader of that output stalls.
        """
        pond = _add_pond(name)
        reg = defn.add_generator_instance(f"{name}_in_reg", reg_gen, context.new_values({"width": 16}),
                                          context.new_values({"clk_posedge": True, "init": BitVector[16](0)}))
        defn.connect(src, reg.select("in"))
        defn.connect(reg.select("out"), pond.select("data_in_pond_0"))
        defn.connect(pond.select("data_out_pond_0"), pe_port)
        instances["fifo_pond"].append(pond)
    ops = dict(_CONV_OPS, **(_SILU_OPS if fuse_silu else {}))
    for role in ops:
        instances["pe"][role] = []
    for i in range(unroll * TAPS):
        instances["weight_io"].append(io(_port("weight_host", i), "in"))
    for i in range(unroll):
        instances["bias_io"].append(io(_port("bias_host", i), "in"))

    for lane in range(unroll):
        x_io = io(_port("input_host", lane), "in")
        out_io = io(_port("hw_output", lane, write=True), "out")
        instances["input_io"].append(x_io)
        instances["output_io"].append(out_io)
        pe = {}
        for role in ops:
            pe[role] = defn.add_module_instance(f"{role}_pe_{lane}", pe_module)
            instances["pe"][role].append(pe[role])

        # Tap k multiplies x[t] by w[c, k]; weight lane 4c + k is tap k of channel c.
        # The weight enters through the tile's Pond on data0, the PE input the
        # Pond of its tile reaches (fp_mul and fp_add are commutative).
        # x goes to two dual-read MEMs; their four read ports feed the four taps.
        x_buffers = [_add_mem(defn, context, mem_gen, genargs, clk_en, f"x_buffer{b}_lane{lane}") for b in range(2)]
        instances["x_buffer"] += x_buffers
        for buf in x_buffers:
            defn.connect(x_io.select("out"), buf.select("data_in_0"))
        products = {}
        for k in range(TAPS):
            repeat_pond(f"weight_pond_tap{k}_lane{lane}", instances["weight_io"][TAPS * lane + k].select("out"),
                        pe[f"tap{k}_mul"].select("data0"))
            # taps 3, 2 read x_buffer0 and taps 1, 0 read x_buffer1
            buf = x_buffers[0 if k >= 2 else 1]
            defn.connect(buf.select(f"data_out_{(k + 1) % 2}"), pe[f"tap{k}_mul"].select("data1"))
            # Delay product k by 3 - k positions: p_k(t) = w[c, k] x[t - (3 - k)].
            value = pe[f"tap{k}_mul"].select("O0")
            for d in range(1, TAPS - k):
                reg = _add_delay(defn, context, reg_gen, f"p{k}_delay{d}_lane{lane}")
                defn.connect(value, reg.select("in"))
                value = reg.select("out")
            products[f"tap{k}_mul"] = value
        # y = ((p3 + p2) + (p1 + p0)) + bias
        tree = [
            ("acc0_add", products["tap3_mul"], products["tap2_mul"]),
            ("acc1_add", products["tap1_mul"], products["tap0_mul"]),
            ("acc2_add", pe["acc0_add"].select("O0"), pe["acc1_add"].select("O0")),
        ]
        for dst, a, b in tree:
            defn.connect(a, pe[dst].select("data0"))
            defn.connect(b, pe[dst].select("data1"))
        repeat_pond(f"bias_pond_lane{lane}", instances["bias_io"][lane].select("out"), pe["acc3_add"].select("data0"))
        defn.connect(pe["acc2_add"].select("O0"), pe["acc3_add"].select("data1"))
        y = pe["acc3_add"].select("O0")

        if not fuse_silu:
            defn.connect(y, out_io.select("in"))
            continue

        # y, n and d each feed a long branch directly and a short branch
        # through an elastic FIFO: a MEM for y, whose short branch waits for
        # the whole SiLU chain, and a Pond in the reading PE's tile for n and d.
        y_fifo = _add_mem(defn, context, mem_gen, genargs, clk_en, f"y_fifo_lane{lane}")
        instances["y_fifo"].append(y_fifo)
        exp_rom = _add_mem(defn, context, mem_gen, genargs, clk_en, f"exp_rom_lane{lane}", exp_rom_table())
        div_rom = _add_mem(defn, context, mem_gen, genargs, clk_en, f"div_rom_lane{lane}", div_rom_table())
        fifo_pond(f"n_fifo_pond_lane{lane}", pe["silu_neg_log2"].select("O0"), pe["silu_getfint"].select("data0"))
        fifo_pond(f"d_fifo_pond_lane{lane}", pe["silu_add_one"].select("O0"), pe["silu_exp_nop"].select("data0"))
        wires = [
            (y, pe["silu_neg_log2"].select("data0")),
            (y, y_fifo.select("data_in_0")),
            (pe["silu_neg_log2"].select("O0"), pe["silu_getffrac"].select("data0")),
            (pe["silu_getffrac"].select("O0"), exp_rom.select("addr_in_0")),
            (exp_rom.select("data_out_0"), pe["silu_addiexp"].select("data0")),
            (pe["silu_getfint"].select("O0"), pe["silu_addiexp"].select("data1")),
            (pe["silu_addiexp"].select("O0"), pe["silu_add_one"].select("data0")),
            (pe["silu_add_one"].select("O0"), pe["silu_getmant"].select("data0")),
            (pe["silu_getmant"].select("O0"), div_rom.select("addr_in_0")),
            (div_rom.select("data_out_0"), pe["silu_subexp"].select("data0")),
            (pe["silu_exp_nop"].select("O0"), pe["silu_subexp"].select("data1")),
            (pe["silu_subexp"].select("O0"), pe["silu_mul_final"].select("data0")),
            (y_fifo.select("data_out_0"), pe["silu_mul_final"].select("data1")),
            (pe["silu_mul_final"].select("O0"), out_io.select("in")),
        ]
        for src, dst in wires:
            defn.connect(src, dst)

    top.definition = defn
    context.set_top(top)
    instances["ops"] = ops
    return context, top, instances


def _block_size_for(stream: int):
    for b in range(4, 513, 4):
        if stream % b == 0 and stream // b <= MAX_EXTENT:
            return b
    raise ValueError(f"[ERROR] a stream of {stream} words cannot be split into at most {MAX_EXTENT} blocks")


def _configure(context, instances, length: int):
    defn = instances["input_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]
    for role, (op, operands) in instances["ops"].items():
        kwargs = {}
        for port, operand in zip(("data0", "data1", "data2"), operands):
            kwargs[port] = ("ext", None) if operand == "ext" else ("const", bf16_bits_rne(operand))
        inst_val, inst_w = pe_inst_to_bits_with_operands(op, **kwargs)
        for pe in instances["pe"][role]:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}",
                const_gen,
                context.new_values({"width": inst_w}),
                context.new_values({"value": BitVector[inst_w](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    def stream(key, stride, extent=length):
        return json.dumps(
            {
                "cycle_starting_addr": [0],
                "cycle_stride": [1],
                "dimensionality": 1,
                "extent": [extent],
                f"{key}_data_starting_addr": [0],
                f"{key}_data_stride": [stride],
            }
        )

    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", stream("read", 1))
    # Weights and bias: the IO reads its one GLB word _CONST_READS times; the
    # Ponds replay it for every position.
    for io in instances["weight_io"] + instances["bias_io"]:
        io.add_metadata("glb2out_0", stream("read", 0, _CONST_READS))
    pond_cfg = json.dumps({"type": "get_repeat_pond", "num_writes": _CONST_READS, "num_reads": length})
    for pond in instances["pond"]:
        pond.add_metadata("lake_rv_config", pond_cfg)
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", stream("write", 1))
    block = _block_size_for(length)
    for mem in instances["x_buffer"]:
        mem.add_metadata("lake_rv_config", json.dumps({"type": "dual_read", "input_stream_size": length, "row_size": block}))
    for mem in instances["y_fifo"]:
        mem.add_metadata("lake_rv_config", json.dumps({"type": "fifo", "input_stream_size": length, "row_size": block}))
    for pond in instances["fifo_pond"]:
        pond.add_metadata("lake_rv_config", json.dumps({"type": "get_fifo_pond", "input_stream_size": length, "read_port": 2}))


def emit_depthwise_conv1d_silu_bf16_design(unroll: int, length: int, output_path: str, fuse_silu: bool = True):
    """Write design_top.json for one depthwise Conv1D (+ SiLU) kernel of ``unroll`` channels."""
    context, top, instances = _build_graph(unroll, fuse_silu)
    _configure(context, instances, length)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote depthwise_conv1d_silu_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate depthwise Conv1D (+ SiLU) design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--no_silu", action="store_true")
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    emit_depthwise_conv1d_silu_bf16_design(args.unroll, args.length, args.output_path, not args.no_silu)
