"""Build the one-pass MX quantization along dim -2 (get scale + apply scale) CoreIR graph for Zircon.

The input is a BF16 tensor of seq_heads_prod rows and head_dim columns, stored
row-major as an activation (32 lanes in 4 GLB tiles). Each MX block is 64
consecutive rows of one column (``block_size``), as in the two-pass flow
``get_e8m0_scale_accum_gb_input`` + ``apply_e8m0_scale_multi_IOs``:

    scale[s, d] = get_shared_exp(max over r of |x[64 s + r, d]|)
    q[r, d]     = e8m0_quant(x[r, d], scale[r / 64, d])

The kernel writes both outputs of the two-pass flow in the same GLB layouts:

- ``hw_output_mxint8_act``: INT8 values, two adjacent columns per 16-bit word
  (odd column in the high byte), 16 lanes in 2 GLB tiles;
- ``hw_scale_packed_output``: scales, two adjacent columns per 16-bit word,
  16 lanes in 2 GLB tiles.

Lane j holds column 32 g + j for column groups g = 0 .. G - 1 (G = head_dim /
unroll). Its GLB address generator reads two groups at a time, interleaved
(row r of groups 2h and 2h + 1, then row r + 1, ... for the 64 rows of the
block), so the block maximum is a recurrence at distance 2:

    m[r, s] = abs_max(m[r - 1, s], x[r, s]),   m[-1, s] = 0    (s = 0, 1)

For G = 2 this is simply the row-major order. Per lane:

- **Block |max|**: one ``fp_abs_max`` PE and the Pond of its tile
  (``get_block_absmax_pond``). The Pond is a ring of running maxima; read
  port 0 gives the PE the maximum one row up, the clear zeroes the ring
  before each block, and read port 1 reads the two block maxima once the
  block is written.
- **Scale**: ``get_shared_exp`` turns the two block maxima into the two
  block scales, at two values per block. A register is its only sink and
  feeds the scale output and the replay Pond (a net that ends in a register
  must not also pass on to another sink).
- **Scale output**: ``bit8_pack`` joins the scales of lanes 2k and 2k + 1
  into scale output lane k.
- **Scale replay**: the replay Pond (``get_block_replay_pond``) replays the
  two scales once per row of the block, so its PE (``or_`` with 0, a pass
  through) sends one scale per token, in the token order of the data. A Pond
  output reaches only data0 of its tile's PE, while ``e8m0_quant`` takes the
  scale on data1, hence the pass-through PE.
- **Data**: the input also goes to a MEM FIFO, which holds each block while
  its scale is computed (``fifo``).
- **Quantize**: one ``e8m0_quant`` PE per lane (data from the MEM FIFO, scale
  from the replay). ``bit8_pack`` joins lanes 2k and 2k + 1 into data output
  lane k.

Every path that carries one value per cycle is a chain from an elastic source
(a MEM or Pond read port) to an output, with no fan-out. Every interconnect
hop is a 2-entry FIFO, so a full-rate value that fed two joins would need the
same latency difference to both joins in every lane of an output packet.
The first version fed the quantizer and the scale output from one full-rate
scale stream and ran at 64%. The fan-out of ``get_shared_exp`` carries two
values per block, where such a mismatch does not cost throughput.

The mapper's Pond packing ILP can pack the ring Pond with any PE that its
read port 1 reaches; the ``get_shared_exp`` PE is named ``..._biased_exp_pe``
(the E8M0 scale is a biased exponent), and a PE whose name contains "bias" is
never a packing target, so the ring stays with its ``fp_abs_max`` PE.

Templated design parameters:
- unroll: input lanes (32 for an activation in 4 GLB tiles).
- head_dim: columns; a multiple of 2 * unroll.
- seq_heads_prod: rows; a multiple of block_size.
- block_size: rows per MX block (64).
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
DEFAULT_TOP_MODULE = "fused_e8m0_quant_multi_IOs"
INTERLEAVE = 2  # column groups per lane interleaved in the stream (the recurrence distance)
MEM_WORDS = 2048
MAX_EXTENT = 1024  # Zircon MEM and Pond extent counters are 11-bit signed


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _bf_act_self_port(i: int) -> str:
    return f"input_bf_act_host_stencil_clkwrk_{i}_op_hcompute_input_bf_act_glb_stencil{_suffix(i)}_read_0"


def _mxint8_self_port(i: int) -> str:
    return f"hw_output_mxint8_act_stencil_clkwrk_{i}_op_hcompute_hw_output_mxint8_act_stencil{_suffix(i)}_write_0"


def _scale_self_port(i: int) -> str:
    return f"hw_scale_packed_output_stencil_clkwrk_{i}_op_hcompute_hw_scale_packed_output_stencil{_suffix(i)}_write_0"


def _absmax_pe_name(i: int) -> str:
    # The mapper's Pond packing ILP matches names as substrings, so the lane
    # index comes first: no Pond or PE name may be part of another lane's name.
    return f"lane{i}_absmax_pe"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_bf_act_self_port(i)] = context.Array(16, context.BitIn())
    for p in range(unroll // 2):
        record[_mxint8_self_port(p)] = context.Array(16, context.Bit())
    for p in range(unroll // 2):
        record[_scale_self_port(p)] = context.Array(16, context.Bit())
    return context.Record(record)


def _shape(unroll: int, head_dim: int, seq_heads_prod: int, block_size: int):
    """Column groups per lane, MX blocks per column, and Pond blocks (pairs of groups) per lane."""
    if unroll % 2 != 0:
        raise ValueError(f"[ERROR] unroll must be even, got {unroll}")
    if head_dim % (INTERLEAVE * unroll) != 0:
        raise ValueError(f"[ERROR] head_dim ({head_dim}) must be a multiple of {INTERLEAVE * unroll}")
    if seq_heads_prod % block_size != 0:
        raise ValueError(f"[ERROR] seq_heads_prod ({seq_heads_prod}) must be a multiple of block_size ({block_size})")
    groups = head_dim // unroll
    blocks = seq_heads_prod // block_size
    pond_blocks = blocks * groups // INTERLEAVE
    if not 2 <= pond_blocks <= MAX_EXTENT:
        raise ValueError(f"[ERROR] {pond_blocks} blocks per lane must lie in [2, {MAX_EXTENT}]")
    if INTERLEAVE * block_size > MAX_EXTENT:
        raise ValueError(f"[ERROR] block_size ({block_size}) is too large for the Pond extent counters")
    return groups, blocks, pond_blocks


def _add_pond(defn, context, pond_gen, clk_en, name):
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


def _add_mem(defn, context, mem_gen, mem_genargs, clk_en, name):
    mem = defn.add_generator_instance(name, mem_gen, mem_genargs, context.new_values({"config": {}, "mode": "lake"}))
    for key, value in [("config", {}), ("is_rom", False), ("mode", "lake"), ("width", 16)]:
        mem.add_metadata(key, json.dumps(value))
    defn.connect(clk_en.select("out"), mem.select("clk_en"))
    return mem


def _add_reg(defn, context, reg_gen, name):
    """A coreir.reg, which becomes an interconnect FIFO."""
    reg = defn.add_generator_instance(
        name, reg_gen,
        context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    reg.add_metadata("extra_data", json.dumps(0))
    return reg


def _build_graph(unroll: int, top_module: str):
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
    clk_en = defn.add_module_instance(f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}))

    instances = {
        "input_io": [], "mxint8_io": [], "scale_io": [], "absmax_pond": [], "replay_pond": [], "data_fifo": [],
        "pe": {"fp_abs_max": [], "get_shared_exp": [], "or_": [], "e8m0_quant": [], "bit8_pack": []},
    }

    def pe(name, op):
        inst = defn.add_module_instance(name, pe_module)
        instances["pe"][op].append(inst)
        return inst

    block_scales, quant_pes = [], []
    for i in range(unroll):
        io = defn.add_module_instance(f"io16in_{_bf_act_self_port(i)}", io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_bf_act_self_port(i)), io.select("in"))
        x = io.select("out")

        # Block |max|: the ring Pond and its PE. The PE output feeds only its
        # own Pond, and its output FIFO is bypassed (PE_fifos_bypass_config).
        absmax_pe = pe(_absmax_pe_name(i), "fp_abs_max")
        absmax_pond = _add_pond(defn, context, pond_gen, clk_en, f"lane{i}_absmax_pond")
        defn.connect(absmax_pond.select("data_out_pond_0"), absmax_pe.select("data0"))
        defn.connect(x, absmax_pe.select("data1"))
        defn.connect(absmax_pe.select("O0"), absmax_pond.select("data_in_pond_0"))

        # Block scales, two per block: read port 1 of the ring Pond gives the
        # block maxima. "bias" in the name keeps the Pond packing ILP from
        # packing the ring Pond here instead of with its fp_abs_max PE.
        exp_pe = pe(f"lane{i}_biased_exp_pe", "get_shared_exp")
        defn.connect(absmax_pond.select("data_out_pond_1"), exp_pe.select("data0"))
        scale_reg = _add_reg(defn, context, reg_gen, f"lane{i}_scale_reg")
        defn.connect(exp_pe.select("O0"), scale_reg.select("in"))

        # Scale replay: the Pond of a pass-through PE returns the two scales
        # once per row of the block. Writing it from a register keeps the
        # mapper from packing it with the PE that drives it.
        scale_pe = pe(f"lane{i}_scale_pe", "or_")
        replay_pond = _add_pond(defn, context, pond_gen, clk_en, f"lane{i}_scale_pond")
        defn.connect(scale_reg.select("out"), replay_pond.select("data_in_pond_0"))
        defn.connect(replay_pond.select("data_out_pond_0"), scale_pe.select("data0"))

        # Data waits in a MEM FIFO, the input's other sink, for its scale.
        data_fifo = _add_mem(defn, context, mem_gen, mem_genargs, clk_en, f"lane{i}_data_fifo")
        defn.connect(x, data_fifo.select("data_in_0"))
        quant_pe = pe(f"lane{i}_quant_pe", "e8m0_quant")
        defn.connect(data_fifo.select("data_out_0"), quant_pe.select("data0"))
        defn.connect(scale_pe.select("O0"), quant_pe.select("data1"))

        instances["input_io"].append(io)
        instances["absmax_pond"].append(absmax_pond)
        instances["replay_pond"].append(replay_pond)
        instances["data_fifo"].append(data_fifo)
        block_scales.append(scale_reg)
        quant_pes.append(quant_pe)

    # Output lane k packs input lanes 2k and 2k + 1 (columns 32 g + 2k and
    # 32 g + 2k + 1). bit8_pack(odd lane, even lane): data0 is the high byte.
    for k in range(unroll // 2):
        for kind, srcs, port_fn, key in (
            ("data", quant_pes, _mxint8_self_port, "mxint8_io"),
            ("scale", block_scales, _scale_self_port, "scale_io"),
        ):
            io = defn.add_module_instance(f"io16_{port_fn(k)}", io_module, context.new_values({"mode": "out"}))
            defn.connect(io.select("out"), iface.select(port_fn(k)))
            pk = pe(f"pair{k}_{kind}_pack_pe", "bit8_pack")
            out = "O0" if kind == "data" else "out"
            defn.connect(srcs[2 * k + 1].select(out), pk.select("data0"))
            defn.connect(srcs[2 * k].select(out), pk.select("data1"))
            defn.connect(pk.select("O0"), io.select("in"))
            instances[key].append(io)

    top.definition = defn
    context.set_top(top)
    return context, top, instances


def _glb_stream(key: str, levels):
    """GLB address-generator metadata from (extent, address step) levels, innermost first.

    The step of a level is the address change when that level advances and
    all inner levels restart. Levels of extent 1 are dropped and a level
    whose step equals its inner level's step is merged into it.
    """
    dims = []
    for extent, step in levels:
        if extent == 1:
            continue
        if dims and dims[-1][1] == step:
            dims[-1] = (dims[-1][0] * extent, step)
        else:
            dims.append((extent, step))
    return json.dumps(
        {
            "cycle_starting_addr": [0],
            "cycle_stride": [1] * len(dims),
            "dimensionality": len(dims),
            "extent": [e for e, _ in dims],
            f"{key}_data_starting_addr": [0],
            f"{key}_data_stride": [s for _, s in dims],
        }
    )


def _configure(context, instances, unroll: int, head_dim: int, seq_heads_prod: int, block_size: int):
    groups, blocks, pond_blocks = _shape(unroll, head_dim, seq_heads_prod, block_size)
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
                f"const_inst_{pe.name}", const_gen,
                context.new_values({"width": inst_w}), context.new_values({"value": BitVector[inst_w](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    # Word address of lane-local token (s, r, h, b) = s + G r + 2 h + 64 G b:
    # groups 2h + s of row r of block b. The output lanes have the same
    # layout (head_dim / 2 words per row over unroll / 2 lanes).
    i, g = INTERLEAVE, groups
    data_levels = [(i, 1), (block_size, g - i + 1), (g // i, 1 - g * (block_size - 1)), (blocks, 1)]
    # Scale word of (s, h, b) = s + 2 h + G b: the scales leave in GLB order.
    scale_levels = [(i, 1), (g // i, 1), (blocks, 1)]
    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", _glb_stream("read", data_levels))
    for io in instances["mxint8_io"]:
        io.add_metadata("in2glb_0", _glb_stream("write", data_levels))
    for io in instances["scale_io"]:
        io.add_metadata("in2glb_0", _glb_stream("write", scale_levels))

    ring_cfg = json.dumps(
        {"type": "get_block_absmax_pond", "num_groups": i, "block_rows": block_size, "num_blocks": pond_blocks}
    )
    replay_cfg = json.dumps(
        {"type": "get_block_replay_pond", "num_groups": i, "block_rows": block_size, "num_blocks": pond_blocks}
    )
    for pond in instances["absmax_pond"]:
        pond.add_metadata("lake_rv_config", ring_cfg)
    for pond in instances["replay_pond"]:
        pond.add_metadata("lake_rv_config", replay_cfg)

    # The MEM FIFO holds a block (2 * 64 tokens) plus the scale latency. It
    # releases a row of row_size words once the writer is two rows ahead.
    tokens = groups * seq_heads_prod
    row = next(
        (r for r in (32, 64, 16, 128, 8, 256) if tokens % r == 0 and tokens // r <= MAX_EXTENT),
        None,
    )
    if row is None:
        raise ValueError(f"[ERROR] no MEM FIFO row size fits {tokens} tokens per lane")
    fifo_cfg = json.dumps({"type": "fifo", "input_stream_size": tokens, "row_size": row})
    for mem in instances["data_fifo"]:
        mem.add_metadata("lake_rv_config", fifo_cfg)


def _write_pe_fifos_bypass_config(output_path: str, unroll: int):
    """Bypass the output FIFO of every block |max| PE: its only sink is the Pond of its tile."""
    cfg = {_absmax_pe_name(i): {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1} for i in range(unroll)}
    path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {path}")


def emit_fused_accumulator_quant_pack_bf16_design(
    unroll: int,
    head_dim: int,
    seq_heads_prod: int,
    output_path: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    """Write design_top.json and PE_fifos_bypass_config.json for one dim -2 MX quantization kernel."""
    context, top, instances = _build_graph(unroll, top_module)
    _configure(context, instances, unroll, head_dim, seq_heads_prod, block_size)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote fused_accumulator_quant_pack_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, unroll)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one-pass dim -2 MX quantization design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--head_dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--seq_heads_prod", type=int, default=DEFAULT_SEQ_HEADS_PROD)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    emit_fused_accumulator_quant_pack_bf16_design(
        args.unroll, args.head_dim, args.seq_heads_prod, args.output_path, block_size=args.block_size
    )
