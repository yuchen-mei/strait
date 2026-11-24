"""Build the bf16 per-row bias add followed by softplus CoreIR graph for Zircon.

    output[r, c] = softplus(input[r, c] + bias[r]),  softplus(s) = ln(1 + exp(s))

With op="add" the kernel is the bias add alone, output[r, c] = input[r, c] + bias[r] (one fp_add per lane, same
IOs, bias broadcast and streams; used as the carry add of a split-row cumsum).

``input`` is the [num_vecs, vec_length] activation. Each lane streams every
``unroll``-th value of a row, so a lane sees ``vec_length / unroll`` values of
row 0, then of row 1, and so on. ``bias`` holds one value per row and is read
by a single IO that repeats the row's value for each of those beats. The IO
writes it, through one register, into one dual-read MEM per two lanes (the
bias buffers), and each read port feeds one lane's add. The activation lanes arrive together (one GLB
packet). Without the buffers the bias would reach the adds through one rigid
fan-out, and two forks that meet at every lane run at full rate only if each
lane's two routes have the same latency. With them, each lane's add takes its
bias whenever its activation arrives. In the Mamba dt path, rows are heads, columns are sequence
positions and the bias is the per-head dt_bias.

The Zircon PE has no exp or ln instruction, so both come from MEM ROMs, as in
the swish and softmax templates. Per lane:

    s = input + bias              (fp_add)
    u = s | 0x8000 = -|s|         (or_)
    z = u * (1 / ln2)             (fp_mul)
    e = 2^z = exp(u)              (fp_getffrac -> exp ROM, fp_getfint, fp_addiexp)
    v = e + 1                     (fp_add), 1 < v <= 2
    w = v - 1                     (fp_sub), exact for v in [1, 2]
    d = e - w                     (fp_sub), the part of e lost when rounding v
    l = ln(v)                     (and_ 0x00ff -> ln ROM)
    t = max(s, 0)                 (fp_max)
    out = t + (l + d)             (two fp_add)

softplus(s) = max(s, 0) + ln(1 + exp(-|s|)). Computing 1 + e directly loses
the small values of e for negative s (1 + e rounds to a multiple of 2^-7), so
d adds that part back: l + d is ln(1 + e) corrected to first order.

The ln ROM is addressed by the low byte of v. A value in [1, 2) has biased
exponent 127, so its low byte is 0x80 | mantissa and entries 128..255 hold
ln(1 + m / 128). v = 2.0 has low byte 0, so entry 0 holds ln(2). Entries
1..127 are never read.

s, z, e and v each feed two branches that meet again later (for example e
goes straight to d and also through v and w). When one PE output feeds two
PEs directly, it can only send a value when both accept it, so the short
branch's 2-entry input FIFO limits the lane to about 2 values per latency of
the long branch (one value per 8 cycles was measured). Each of these four
values is therefore written to a dual-read MEM that is its only consumer;
read port 0 feeds one branch and read port 1 the other, and each port runs
ahead as far as its branch allows. Both reads trail the write by up to two
4-word or 8-word blocks, until the block is in SRAM.

Each lane uses 13 PEs and 6 MEMs (two ROMs and four fork buffers), plus half
a bias buffer. With 8 lanes the kernel uses 104 PEs and 52 MEMs; 16 lanes
would need 208 PEs, more than the 192 PE tiles of the Zircon fabric.

Templated design parameters:
- unroll: number of lanes; vec_length must be divisible by it.
- vec_length: values per row.
- num_vecs: number of rows (= bias length).
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
DEFAULT_VEC_LENGTH = 1024
DEFAULT_NUM_VECS = 64
TOP_MODULE = "add_softplus_fp"

# The fork buffers count their stream in blocks of a few words; the block
# count must fit the 11-bit signed MEM extent counter.
_MAX_EXTENT = 1024

# Per-lane PE chain in dataflow order: (role, op, operand sources).
# "in" is the activation IO, "bias" the lane's bias buffer read port, "rom:x" the data
# output of ROM x, "b:0" / "b:1" the two read ports of the fork buffer of value
# b and other strings the output of an earlier PE of the lane. A tuple ("const", value) is a constant operand:
# a float becomes a BF16 constant, an int is used as raw 16-bit data.
_PE_CHAIN = (
    ("bias_add", "fp_add", ("in", "bias")),
    ("neg_abs", "or_", ("s:0", ("const", 0x8000))),
    ("to_log2", "fp_mul", ("neg_abs", ("const", 1.0 / math.log(2.0)))),
    ("getffrac", "fp_getffrac", ("z:0",)),
    ("getfint", "fp_getfint", ("z:1",)),
    ("addiexp", "fp_addiexp", ("rom:exp", "getfint")),
    ("add_one", "fp_add", ("e:0", ("const", 1.0))),
    ("sub_one", "fp_sub", ("v:1", ("const", 1.0))),
    ("residual", "fp_sub", ("e:1", "sub_one")),
    ("low_byte", "and_", ("v:0", ("const", 0x00FF))),
    ("ln1p", "fp_add", ("rom:ln", "residual")),
    ("relu", "fp_max", ("s:1", ("const", 0.0))),
    ("softplus", "fp_add", ("relu", "ln1p")),
)
# op="add": the bias add alone, out = input + bias[row] (e.g. the carry of a split-row cumsum). Same IOs, bias
# broadcast and streams; no ROMs or fork buffers.
_ADD_CHAIN = (("bias_add", "fp_add", ("in", "bias")),)
_CHAINS = {"softplus": _PE_CHAIN, "add": _ADD_CHAIN}
_OUTPUT_ROLE = {"softplus": "softplus", "add": "bias_add"}
# ROM name -> PE whose output is the ROM address.
_ROM_ADDRESS = {"exp": "getffrac", "ln": "low_byte"}
# Fork buffer name -> PE whose output it holds.
_FORK_SOURCE = {"s": "bias_add", "z": "to_log2", "e": "addiexp", "v": "add_one"}


def bf16_bits_rne(x: float) -> int:
    """BF16 bit pattern of x, rounded to nearest even.

    The ROM tables and constants use this instead of truncation so each entry
    is the closest BF16 value.
    """
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16) & 0xFFFF


def exp_rom_table():
    """256 entries indexed by the fp_getffrac byte: 2^(i/128), i in [0, 128) then [-128, 0)."""
    return [bf16_bits_rne(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_rne(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


# BF16 -0.0. The flow drops every 0 word from a ROM init list (remove_rom_zeros
# in hack_design_top.py), which would shift the later entries to lower
# addresses, so the ROM tables use -0.0 wherever they need a zero or a filler.
_NEG_ZERO = 0x8000


def ln_rom_table():
    """256 entries indexed by the low byte of v in (1, 2]: entry 0 is ln(2), 128 + m is ln(1 + m/128).

    ln(1) is stored as -0.0 and the unused entries 1..127 hold -0.0, so the
    table has no 0 word.
    """
    table = [_NEG_ZERO] * 256
    table[0] = bf16_bits_rne(math.log(2.0))
    for m in range(1, 128):
        table[128 + m] = bf16_bits_rne(math.log1p(m / 128.0))
    return table


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_suffix(i)}_read_0"


def _bias_self_port() -> str:
    return "bias_host_stencil_clkwrk_0_op_hcompute_bias_glb_stencil_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_suffix(i)}_write_0"


def _interface_type(context, unroll: int):
    record = {_bias_self_port(): context.Array(16, context.BitIn())}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _add_rom(context, defn, name, table, mem_gen, genargs):
    rom = defn.add_generator_instance(
        name,
        mem_gen,
        genargs,
        context.new_values({"config": {}, "init": table, "mode": "lake"}),
    )
    for key, value in [
        ("config", {}),
        ("depth", len(table)),
        ("init", table),
        ("is_rom", True),
        ("mode", "sram"),
        ("width", 16),
    ]:
        rom.add_metadata(key, json.dumps(value))
    return rom


def _build_graph(unroll: int, op: str = "softplus"):
    if op not in _CHAINS:
        raise ValueError(f"[ERROR] op must be one of {sorted(_CHAINS)}, got {op!r}")
    chain = _CHAINS[op]
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    namespace = context.global_namespace
    pe_module = namespace.modules["PE"]
    io_module = namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    top = namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface
    genargs = make_mem_genargs(context)
    tables = {"exp": exp_rom_table(), "ln": ln_rom_table()}

    bias_io = defn.add_module_instance(
        f"io16in_{_bias_self_port()}", io_module, context.new_values({"mode": "in"})
    )
    defn.connect(iface.select(_bias_self_port()), bias_io.select("in"))

    instances = {
        "bias_io": bias_io,
        "input_io": [],
        "output_io": [],
        "fork_buffer": [],
        "pe": {role: [] for role, _, _ in chain},
        "chain": chain,
    }

    def fork_buffer(name):
        mem = defn.add_generator_instance(
            name, mem_gen, genargs, context.new_values({"config": {}, "mode": "lake"})
        )
        for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
            mem.add_metadata(key, json.dumps(value))
        instances["fork_buffer"].append(mem)
        return mem

    # One bias buffer per two lanes; read port k feeds lane 2j + k. The IO
    # writes them through one register, which is its only sink: the mapper
    # does not handle an IO that writes several MEMs directly, and a register
    # sink must not also pass its own switch-box output on to other sinks.
    if unroll % 2:
        raise ValueError(f"[ERROR] unroll ({unroll}) must be even")
    reg_gen = context.get_lib("coreir").generators["reg"]
    bias_reg = defn.add_generator_instance(
        "bias_in_reg",
        reg_gen,
        context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    bias_reg.add_metadata("extra_data", json.dumps(0))
    defn.connect(bias_io.select("out"), bias_reg.select("in"))
    bias_buffers = []
    for j in range(unroll // 2):
        mem = fork_buffer(f"bias_buffer_{j}")
        defn.connect(bias_reg.select("out"), mem.select("data_in_0"))
        bias_buffers.append(mem)
    for i in range(unroll):
        input_io = defn.add_module_instance(
            f"io16in_{_input_self_port(i)}", io_module, context.new_values({"mode": "in"})
        )
        output_io = defn.add_module_instance(
            f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"})
        )
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        instances["input_io"].append(input_io)
        instances["output_io"].append(output_io)

        roms = {
            name: _add_rom(context, defn, f"{name}_rom_lane{i}", table, mem_gen, genargs)
            for name, table in tables.items()
        } if op == "softplus" else {}
        sources = {"in": input_io.select("out"), "bias": bias_buffers[i // 2].select(f"data_out_{i % 2}")}
        forks = {}
        for name in (_FORK_SOURCE if op == "softplus" else {}):
            mem = fork_buffer(f"{name}_buffer_lane{i}")
            forks[name] = mem
            sources[f"{name}:0"] = mem.select("data_out_0")
            sources[f"{name}:1"] = mem.select("data_out_1")
        for name, rom in roms.items():
            sources[f"rom:{name}"] = rom.select("data_out_0")
        pes = {}
        for role, _, operands in chain:
            pe = defn.add_module_instance(f"{role}_pe_{i}", pe_module)
            for port, operand in zip(("data0", "data1", "data2"), operands):
                if isinstance(operand, str):
                    defn.connect(sources[operand], pe.select(port))
            sources[role] = pe.select("O0")
            pes[role] = pe
            instances["pe"][role].append(pe)

        for name, rom in roms.items():
            defn.connect(sources[_ROM_ADDRESS[name]], rom.select("addr_in_0"))
        for name, src in (_FORK_SOURCE.items() if op == "softplus" else ()):
            defn.connect(sources[src], forks[name].select("data_in_0"))
        defn.connect(sources[_OUTPUT_ROLE[op]], output_io.select("in"))

    top.definition = defn
    context.set_top(top)
    return context, top, instances


def _pe_instruction(op, operands):
    kwargs = {}
    for port, operand in zip(("data0", "data1", "data2"), operands):
        if isinstance(operand, tuple):
            value = operand[1]
            bits = value if isinstance(value, int) else bf16_bits_rne(value)
            kwargs[port] = ("const", bits)
        else:
            kwargs[port] = ("ext", None)
    return pe_inst_to_bits_with_operands(op, **kwargs)


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int):
    """Attach PE instructions, the fork buffer schedules and the GLB streams. Mutates the instances."""
    if vec_length % unroll != 0:
        raise ValueError(
            f"[ERROR] vec_length ({vec_length}) must be divisible by unroll ({unroll})"
        )
    per_row_extent = vec_length // unroll
    per_lane_extent = per_row_extent * num_vecs
    # Smallest 4-word multiple block that divides the stream into at most
    # _MAX_EXTENT blocks.
    block = next(
        (b for b in range(4, 513, 4) if per_lane_extent % b == 0 and per_lane_extent // b <= _MAX_EXTENT),
        None,
    )
    if block is None:
        raise ValueError(
            f"[ERROR] values per lane ({per_lane_extent}) must split into at most "
            f"{_MAX_EXTENT} blocks of a multiple of 4 words"
        )

    defn = instances["bias_io"].module_def
    const_gen = context.get_lib("coreir").generators["const"]
    for role, op, operands in instances["chain"]:
        inst_val, inst_w = _pe_instruction(op, operands)
        for pe in instances["pe"][role]:
            c = defn.add_generator_instance(
                f"const_inst_{pe.name}",
                const_gen,
                context.new_values({"width": inst_w}),
                context.new_values({"value": BitVector[inst_w](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    fork_cfg = json.dumps({"type": "dual_read", "input_stream_size": per_lane_extent, "row_size": block})
    for mem in instances["fork_buffer"]:
        mem.add_metadata("lake_rv_config", fork_cfg)

    def stream(extents, strides, key):
        return json.dumps(
            {
                "cycle_starting_addr": [0],
                "cycle_stride": [1] * len(extents),
                "dimensionality": len(extents),
                "extent": extents,
                f"{key}_data_starting_addr": [0],
                f"{key}_data_stride": strides,
            }
        )

    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", stream([per_lane_extent], [1], "read"))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", stream([per_lane_extent], [1], "write"))
    # The row's bias is read again for each beat of the row (stride 0), then
    # the stream moves to the next row's value.
    if per_row_extent == 1:
        bias = stream([num_vecs], [1], "read")
    elif num_vecs == 1:
        bias = stream([per_row_extent], [0], "read")
    else:
        bias = stream([per_row_extent, num_vecs], [0, 1], "read")
    instances["bias_io"].add_metadata("glb2out_0", bias)


def build_elementwise_broadcast_add_softplus_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    op: str = "softplus",
):
    context, top, instances = _build_graph(unroll, op)
    _configure(context, instances, unroll, vec_length, num_vecs)
    return context, top


def emit_elementwise_broadcast_add_softplus_bf16_design(
    unroll: int, vec_length: int, num_vecs: int, output_path: str, op: str = "softplus"
):
    """Write design_top.json for one bias-add-softplus kernel (op="add": the bias add alone) into output_path."""
    context, top = build_elementwise_broadcast_add_softplus_bf16_context(
        unroll, vec_length, num_vecs, op
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_broadcast_add_softplus_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate bias add + softplus design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec_length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num_vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--op", type=str, default="softplus", choices=sorted(_CHAINS))
    args = parser.parse_args()
    emit_elementwise_broadcast_add_softplus_bf16_design(
        args.unroll, args.vec_length, args.num_vecs, args.output_path, args.op
    )
