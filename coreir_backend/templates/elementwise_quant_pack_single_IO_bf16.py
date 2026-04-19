"""
Build the elementwise (e8m0_quant + byte-pack) + scale-packing CoreIR graph
using pycoreir. Used by: apply_e8m0_scale_single_IO.

For each activation lane i in [0, unroll):
  mxint8[i, y] = e8m0_quant(input_bf_act_io[i].stream, input_scale_io.stream)

For each packer p in [0, unroll/2):
  output_mxint8_act_io[p] = bit8_pack(mxint8[2p+1, y], mxint8[2p, y])
    // data0 = high byte (odd lane), data1 = low byte (even lane)

Scale packing path (a single scale IO broadcasts to all e8m0_quant PEs and also
feeds a bit8_pack PE that pairs the scale with a 1-cycle-delayed copy of itself
to form packed scale words):

  scale_io.out                             -> scale_pack_pe.data0
  scale_io.out -> shift_fifo.in -> shift_fifo.out -> scale_pack_pe.data1
  scale_pack_pe.O0 -> scale_output_io_1.in                            (direct)
  scale_pack_pe.O0 -> pipeline_fifo_0.in -> ... -> pipeline_fifo_{N-1}.out
                   -> scale_output_io_0.in

The two scale output IOs share a single packed scale stream but are sampled by
the IO-level cycle stride pattern into an alternating two-bank toggle.

NOTE on manual placement:
  garnet/mapper/netlist_graph.py manually_place_apply_scale_single_IO places
  every PE that has `io16_hw_scale_packed_output_stencil_clkwrk_1_...` in its
  sinks. The scale_pack_pe has a DIRECT `.O0` connection to scale_output_io_1,
  so it will be the unique matching PE and get placed near the first scale
  output IO. Do not remove that direct edge.

Templated design parameters:
- unroll: number of parallel bf_act input lanes (= glb_i); must be even >= 2.
- num_pixels: vec_height.
- num_channels: vec_width (to derive num_blocks = num_channels / (unroll*2)).
- num_pipeline_fifos: length of the pipeline-fifo chain feeding scale_output_io_0.
- top_module: name of the emitted top module.

IO metadata:
- input_bf_act_host_stencil (per lane): 1D, extent = num_pixels * num_channels / unroll.
- input_scale_host_stencil: 2D broadcast, extent=[2, num_pixels*num_blocks], stride=[0,1].
- hw_output_mxint8_act_stencil (per packer): 1D, extent = num_pixels*num_channels/unroll.
- hw_scale_packed_output_stencil (two IOs): 1D, cycle_starting_addr=[2+idx*4],
  cycle_stride=[num_scale_ios*4=8], extent = num_pixels * num_blocks / (2 * num_scale_ios).
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_NUM_PIXELS = 128
DEFAULT_NUM_CHANNELS = 256
DEFAULT_NUM_PIPELINE_FIFOS = 10
DEFAULT_TOP_MODULE = "apply_e8m0_scale_single_IO"

NUM_SCALE_IOS = 2


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _bf_act_io_name(i: int) -> str:
    return f"io16in_input_bf_act_host_stencil_clkwrk_{i}_op_hcompute_input_bf_act_glb_stencil{_stencil_suffix(i)}_read_0"


def _bf_act_self_port(i: int) -> str:
    return f"input_bf_act_host_stencil_clkwrk_{i}_op_hcompute_input_bf_act_glb_stencil{_stencil_suffix(i)}_read_0"


def _scale_io_name() -> str:
    return "io16in_input_scale_host_stencil_clkwrk_0_op_hcompute_input_scale_glb_stencil_read_0"


def _scale_self_port() -> str:
    return "input_scale_host_stencil_clkwrk_0_op_hcompute_input_scale_glb_stencil_read_0"


def _mxint8_io_name(i: int) -> str:
    return f"io16_hw_output_mxint8_act_stencil_clkwrk_{i}_op_hcompute_hw_output_mxint8_act_stencil{_stencil_suffix(i)}_write_0"


def _mxint8_self_port(i: int) -> str:
    return f"hw_output_mxint8_act_stencil_clkwrk_{i}_op_hcompute_hw_output_mxint8_act_stencil{_stencil_suffix(i)}_write_0"


def _scale_packed_io_name(i: int) -> str:
    # Halide emits write_0 for idx 0 and write_1 for idx 1 under bank-toggle mode.
    return f"io16_hw_scale_packed_output_stencil_clkwrk_{i}_op_hcompute_hw_scale_packed_output_stencil_{i}_write_{i}"


def _scale_packed_self_port(i: int) -> str:
    return f"hw_scale_packed_output_stencil_clkwrk_{i}_op_hcompute_hw_scale_packed_output_stencil_{i}_write_{i}"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_bf_act_self_port(i)] = context.Array(16, context.BitIn())
    record[_scale_self_port()] = context.Array(16, context.BitIn())
    for p in range(unroll // 2):
        record[_mxint8_self_port(p)] = context.Array(16, context.Bit())
    for i in range(NUM_SCALE_IOS):
        record[_scale_packed_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, num_pipeline_fifos: int, top_module: str):
    if unroll < 2 or unroll % 2 != 0:
        raise ValueError(f"unroll must be even and >= 2, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    reg_gen = context.get_lib("coreir").generators["reg"]

    top = global_namespace.new_module(top_module, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    scale_io = defn.add_module_instance(_scale_io_name(), io_module, context.new_values({"mode": "in"}))
    defn.connect(iface.select(_scale_self_port()), scale_io.select("in"))

    bf_act_ios = []
    e8m0_quant_pes = []
    for i in range(unroll):
        bf_io = defn.add_module_instance(_bf_act_io_name(i), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_bf_act_self_port(i)), bf_io.select("in"))
        quant_pe = defn.add_module_instance(f"{top_module}_e8m0_quant_pe_{i}", pe_module)
        defn.connect(bf_io.select("out"), quant_pe.select("data0"))
        defn.connect(scale_io.select("out"), quant_pe.select("data1"))
        bf_act_ios.append(bf_io)
        e8m0_quant_pes.append(quant_pe)

    mxint8_output_ios = []
    output_pack_pes = []
    for p in range(unroll // 2):
        out_io = defn.add_module_instance(_mxint8_io_name(p), io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_mxint8_self_port(p)))
        pack_pe = defn.add_module_instance(f"{top_module}_output_pack_pe_{p}", pe_module)
        # Halide: bit8_pack(odd_lane, even_lane) -> data0=high (odd), data1=low (even).
        defn.connect(e8m0_quant_pes[2 * p + 1].select("O0"), pack_pe.select("data0"))
        defn.connect(e8m0_quant_pes[2 * p].select("O0"), pack_pe.select("data1"))
        defn.connect(pack_pe.select("O0"), out_io.select("in"))
        mxint8_output_ios.append(out_io)
        output_pack_pes.append(pack_pe)

    # Scale packing path.
    # The instance suffix "a_scale_pack_pe" (leading 'a_' < "e8m0_..." and "output_...")
    # makes this instance alphabetically first among PEs. garnet/mapper/netlist_util.py
    # CreateIDs assigns block ids p0, p1, ... in sorted-name order; bin_gold had the
    # scale packing PE at p0. thunder partitions the netlist with an igraph seed
    # (cgra_pnr/thunder/src/graph.cc partition_netlist, seed=0), so a different
    # p-id shifts cluster assignment and lets the annealer collide another PE at
    # (12, 2) despite MANUAL_PLACER pinning scale_pack_pe. Matching bin_gold's p0
    # avoids that collision.
    scale_pack_pe = defn.add_module_instance(f"{top_module}_a_scale_pack_pe", pe_module)
    defn.connect(scale_io.select("out"), scale_pack_pe.select("data0"))

    scale_shift_fifo = defn.add_generator_instance(
        f"{top_module}_scale_shift_fifo", reg_gen,
        context.new_values({"width": 16}),
        context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
    )
    scale_shift_fifo.add_metadata("extra_data", json.dumps(1))
    defn.connect(scale_io.select("out"), scale_shift_fifo.select("in"))
    defn.connect(scale_shift_fifo.select("out"), scale_pack_pe.select("data1"))

    # Two scale output IOs. idx=0 takes the pipeline-fifo chain, idx=1 is direct.
    # The manual placer detects the scale_pack_pe via its direct `.O0 -> scale_output_io_1.in`
    # edge; keep that direct connection.
    scale_output_ios = []
    pipeline_fifos = []
    for i in range(NUM_SCALE_IOS):
        out_io = defn.add_module_instance(_scale_packed_io_name(i), io_module, context.new_values({"mode": "out"}))
        defn.connect(out_io.select("out"), iface.select(_scale_packed_self_port(i)))
        scale_output_ios.append(out_io)

    if num_pipeline_fifos > 0:
        upstream_sel = scale_pack_pe.select("O0")
        for f in range(num_pipeline_fifos):
            fifo = defn.add_generator_instance(
                f"{top_module}_pipeline_fifo_scale_packed_output_{f}", reg_gen,
                context.new_values({"width": 16}),
                context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
            )
            fifo.add_metadata("extra_data", json.dumps(0))
            defn.connect(upstream_sel, fifo.select("in"))
            upstream_sel = fifo.select("out")
            pipeline_fifos.append(fifo)
        defn.connect(upstream_sel, scale_output_ios[0].select("in"))
    else:
        defn.connect(scale_pack_pe.select("O0"), scale_output_ios[0].select("in"))

    defn.connect(scale_pack_pe.select("O0"), scale_output_ios[1].select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "scale_io": scale_io,
        "bf_act_io": bf_act_ios,
        "e8m0_quant_pe": e8m0_quant_pes,
        "output_pack_pe": output_pack_pes,
        "mxint8_output_io": mxint8_output_ios,
        "scale_pack_pe": scale_pack_pe,
        "scale_shift_fifo": scale_shift_fifo,
        "pipeline_fifos": pipeline_fifos,
        "scale_output_io": scale_output_ios,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, num_pixels: int, num_channels: int,
               top_module: str):
    if num_channels % (unroll * 2) != 0:
        raise ValueError(
            f"num_channels ({num_channels}) must be divisible by (unroll*2) = {unroll * 2}"
        )
    num_blocks = num_channels // (unroll * 2)
    if num_blocks < 1:
        raise ValueError(f"num_blocks ({num_blocks}) must be >= 1")

    defn = instances["scale_io"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    quant_val, quant_w = pe_inst_to_bits_with_operands(
        "e8m0_quant", data0=("ext", None), data1=("ext", None),
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

    for i, pe_inst in enumerate(instances["e8m0_quant_pe"]):
        _wire_const(f"{top_module}_e8m0_quant_pe_{i}", pe_inst, quant_val, quant_w)
    for p, pe_inst in enumerate(instances["output_pack_pe"]):
        _wire_const(f"{top_module}_output_pack_pe_{p}", pe_inst, pack_val, pack_w)
    _wire_const(f"{top_module}_a_scale_pack_pe", instances["scale_pack_pe"], pack_val, pack_w)

    # IO metadata.
    bf_act_extent = num_pixels * num_channels // unroll
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
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [2, num_pixels * num_blocks],
        "read_data_starting_addr": [0],
        "read_data_stride": [0, 1],
    })
    mxint8_in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [bf_act_extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })
    extent_per_scale_io = num_pixels * num_blocks // 2 // NUM_SCALE_IOS

    for bf_io in instances["bf_act_io"]:
        bf_io.add_metadata("glb2out_0", bf_act_glb2out)
    instances["scale_io"].add_metadata("glb2out_0", scale_glb2out)
    for mx_io in instances["mxint8_output_io"]:
        mx_io.add_metadata("in2glb_0", mxint8_in2glb)
    for idx, sc_io in enumerate(instances["scale_output_io"]):
        sc_io.add_metadata("in2glb_0", json.dumps({
            "cycle_starting_addr": [2 + idx * 4],
            "cycle_stride": [NUM_SCALE_IOS * 4],
            "dimensionality": 1,
            "extent": [extent_per_scale_io],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        }))


def build_elementwise_quant_pack_single_IO_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    num_pixels: int = DEFAULT_NUM_PIXELS,
    num_channels: int = DEFAULT_NUM_CHANNELS,
    num_pipeline_fifos: int = DEFAULT_NUM_PIPELINE_FIFOS,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, num_pipeline_fifos, top_module)
    _configure(context, instances, unroll, num_pixels, num_channels, top_module)
    return context, top


def emit_elementwise_quant_pack_single_IO_bf16_design(
    unroll: int,
    num_pixels: int,
    num_channels: int,
    output_path: str,
    num_pipeline_fifos: int = DEFAULT_NUM_PIPELINE_FIFOS,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_elementwise_quant_pack_single_IO_bf16_context(
        unroll, num_pixels, num_channels,
        num_pipeline_fifos=num_pipeline_fifos, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_quant_pack_single_IO_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--num-pixels", type=int, default=DEFAULT_NUM_PIXELS)
    parser.add_argument("--num-channels", type=int, default=DEFAULT_NUM_CHANNELS)
    parser.add_argument("--num-pipeline-fifos", type=int, default=DEFAULT_NUM_PIPELINE_FIFOS)
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_quant_pack_single_IO_bf16_design(
        args.unroll, args.num_pixels, args.num_channels, args.output_path,
        num_pipeline_fifos=args.num_pipeline_fifos, top_module=args.top_module,
    )
