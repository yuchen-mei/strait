"""
Build the bf16 get-e8m0-scale tree CoreIR graph using pycoreir.

Two input modes:

- mode="mu" (get_e8m0_scale_tree_mu_input):
    MU inputs feed a line-buffer MEM per lane; the MEM's two read ports provide
    the "direct" (non-delayed) and "delayed by one pass" streams that drive two
    parallel fp_abs_max trees. Also emits a per-lane unquantized passthrough
    output. A mem_scale_filter downsamples the scale stream 1:2 on-chip.

- mode="gb" (get_e8m0_scale_tree_gb_input):
    GLB inputs drive tree_glb directly and also feed a 1-cycle shift_fifo that
    drives tree_fifo. No unquantized output and no mem_scale_filter; scale is
    downsampled 1:2 via the output IO's cycle stride.

For each row y in [0, num_vecs) and each of total_channels/unroll "passes":
  direct[lane,y]  = input stream at lane (current pass)
  delayed[lane,y] = input stream at lane one pass earlier
  tree_direct_top[y]  = abs_max over lanes of direct
  tree_delayed_top[y] = abs_max over lanes of delayed
  combined_max[y]     = abs_max(tree_direct_top[y], tree_delayed_top[y])
  scale_pe[y]         = get_shared_exp(combined_max[y])
  # mu mode:  mem_scale_filter retains every other scale
  # gb mode:  scale_output IO samples every other cycle (stride=2, start=1)

Templated design parameters:
- unroll: number of parallel input lanes (must be a power of 2).
- num_vecs: rows per pass (= vec_height).
- total_channels: total channels across all passes (= vec_width).
- mode: "mu" or "gb".
- top_module: name of the emitted top module.

IO metadata:
- mu mode:
  - mu_input_io (per lane): 1D, extent = num_vecs.
  - unquantized_output_io (per lane): 1D, extent = num_vecs * (total_channels/unroll).
  - scale_output_io: 1D, extent = num_vecs * (total_channels/(2*unroll)),
    cycle_starting_addr=[0], cycle_stride=[1].
- gb mode:
  - input_io (per lane): 1D, extent = num_vecs * (total_channels/unroll).
  - scale_output_io: 1D, extent = num_vecs * (total_channels/(2*unroll)),
    cycle_starting_addr=[1], cycle_stride=[2].

Lake RV configs (mu mode only):
- mem_mu2tree: get_single_mem_line_buffer(buffer_size=num_vecs, num_lines=total_channels/unroll).
- mem_scale_filter: get_filter_scale_mem(img_size=num_vecs, total_channels, mu_OC=unroll, packed=False).
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
DEFAULT_NUM_VECS = 196
DEFAULT_TOTAL_CHANNELS = 256
DEFAULT_MODE = "mu"
DEFAULT_TOP_MODULE = "get_e8m0_scale_tree_mu_input"

_VALID_MODES = ("mu", "gb")


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


# --- mu mode IO names ---------------------------------------------------------

def _mu_input_io_name(i: int) -> str:
    # Naming contains "_mu_" so strait's add_mu_prefix_to_io flips this to a MU IO.
    return f"io16in_mu_input_host_stencil_clkwrk_{i}_op_hcompute_mu_input_io_stencil{_stencil_suffix(i)}_read_0"


def _mu_input_self_port(i: int) -> str:
    return f"mu_input_host_stencil_clkwrk_{i}_op_hcompute_mu_input_io_stencil{_stencil_suffix(i)}_read_0"


def _unquantized_io_name(i: int) -> str:
    return f"io16_unquantized_output_stencil_clkwrk_{i}_op_hcompute_unquantized_output_stencil_{i}_write_0"


def _unquantized_self_port(i: int) -> str:
    return f"unquantized_output_stencil_clkwrk_{i}_op_hcompute_unquantized_output_stencil_{i}_write_0"


# --- gb mode IO names ---------------------------------------------------------

def _gb_input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_gb_stencil{_stencil_suffix(i)}_read_0"


def _gb_input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_gb_stencil{_stencil_suffix(i)}_read_0"


# --- shared output IO names ---------------------------------------------------

def _scale_output_io_name() -> str:
    return "io16_hw_scale_output_stencil_clkwrk_0_op_hcompute_hw_scale_output_stencil_0_write_0"


def _scale_output_self_port() -> str:
    return "hw_scale_output_stencil_clkwrk_0_op_hcompute_hw_scale_output_stencil_0_write_0"


def _interface_type(context, unroll: int, mode: str):
    record = {}
    if mode == "mu":
        for i in range(unroll):
            record[_mu_input_self_port(i)] = context.Array(16, context.BitIn())
            record[_unquantized_self_port(i)] = context.Array(16, context.Bit())
    else:  # gb
        for i in range(unroll):
            record[_gb_input_self_port(i)] = context.Array(16, context.BitIn())
    record[_scale_output_self_port()] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, mode: str, top_module: str):
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if unroll < 2 or (unroll & (unroll - 1)) != 0:
        raise ValueError(f"unroll must be a power of 2 and >= 2, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    reg_gen = context.get_lib("coreir").generators["reg"]
    corebit_const = context.get_namespace("corebit").modules["const"]

    top = global_namespace.new_module(top_module, _interface_type(context, unroll, mode))
    defn = top.new_definition()
    iface = defn.interface

    shared_clk = defn.add_module_instance(
        f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}),
    )

    mem_genargs = make_mem_genargs(context)

    input_ios = []
    unquantized_ios = []
    delay_units = []   # mem_mu2tree (mu) or shift_fifo (gb) per lane
    direct_streams = []  # per-lane selectable for tree_direct stage 1
    delayed_streams = []  # per-lane selectable for tree_delayed stage 1

    for i in range(unroll):
        if mode == "mu":
            inp_io = defn.add_module_instance(_mu_input_io_name(i), io_module, context.new_values({"mode": "in"}))
            defn.connect(iface.select(_mu_input_self_port(i)), inp_io.select("in"))

            unq_io = defn.add_module_instance(_unquantized_io_name(i), io_module, context.new_values({"mode": "out"}))
            defn.connect(unq_io.select("out"), iface.select(_unquantized_self_port(i)))
            defn.connect(inp_io.select("out"), unq_io.select("in"))
            unquantized_ios.append(unq_io)

            mem = defn.add_generator_instance(
                f"mem_mu2tree_{i}", mem_gen, mem_genargs,
                context.new_values({"config": {}, "mode": "lake"}),
            )
            for _k, _v in [
                ("config", json.dumps({})),
                ("is_rom", json.dumps(False)),
                ("mode", json.dumps("lake")),
                ("width", json.dumps(16)),
            ]:
                mem.add_metadata(_k, _v)
            defn.connect(inp_io.select("out"), mem.select("data_in_0"))
            defn.connect(shared_clk.select("out"), mem.select("clk_en"))
            delay_units.append(mem)

            direct_streams.append(mem.select("data_out_1"))
            delayed_streams.append(mem.select("data_out_0"))
        else:  # gb
            inp_io = defn.add_module_instance(_gb_input_io_name(i), io_module, context.new_values({"mode": "in"}))
            defn.connect(iface.select(_gb_input_self_port(i)), inp_io.select("in"))

            fifo = defn.add_generator_instance(
                f"shift_fifo_glb2tree_{i}", reg_gen,
                context.new_values({"width": 16}),
                context.new_values({"clk_posedge": True, "init": BitVector[16](0)}),
            )
            fifo.add_metadata("extra_data", json.dumps(1))
            defn.connect(inp_io.select("out"), fifo.select("in"))
            delay_units.append(fifo)

            direct_streams.append(inp_io.select("out"))
            delayed_streams.append(fifo.select("out"))

        input_ios.append(inp_io)

    # Two parallel trees.
    tree_stages = int(math.log2(unroll))

    def build_tree(tree_label: str, lane_outs):
        stages = []
        prev_outs = list(lane_outs)
        for s in range(1, tree_stages + 1):
            this_stage = []
            next_outs = []
            for j in range(len(prev_outs) // 2):
                pe = defn.add_module_instance(f"{tree_label}_stage{s}_pe_{j}", pe_module)
                defn.connect(prev_outs[2 * j], pe.select("data0"))
                defn.connect(prev_outs[2 * j + 1], pe.select("data1"))
                this_stage.append(pe)
                next_outs.append(pe.select("O0"))
            stages.append(this_stage)
            prev_outs = next_outs
        return stages, prev_outs[0]

    direct_label = "tree_mu" if mode == "mu" else "tree_glb"
    delayed_label = "tree_mem" if mode == "mu" else "tree_fifo"
    tree_direct_stages, tree_direct_top = build_tree(direct_label, direct_streams)
    tree_delayed_stages, tree_delayed_top = build_tree(delayed_label, delayed_streams)

    combined_max_pe = defn.add_module_instance(f"{top_module}_combined_max_pe", pe_module)
    defn.connect(tree_direct_top, combined_max_pe.select("data0"))
    defn.connect(tree_delayed_top, combined_max_pe.select("data1"))

    scale_pe = defn.add_module_instance(f"{top_module}_get_shared_exp_pe", pe_module)
    defn.connect(combined_max_pe.select("O0"), scale_pe.select("data0"))

    scale_output_io = defn.add_module_instance(
        _scale_output_io_name(), io_module, context.new_values({"mode": "out"}),
    )
    defn.connect(scale_output_io.select("out"), iface.select(_scale_output_self_port()))

    mem_scale_filter = None
    if mode == "mu":
        mem_scale_filter = defn.add_generator_instance(
            "mem_scale_filter", mem_gen, mem_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _k, _v in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            mem_scale_filter.add_metadata(_k, _v)
        defn.connect(shared_clk.select("out"), mem_scale_filter.select("clk_en"))
        defn.connect(scale_pe.select("O0"), mem_scale_filter.select("data_in_0"))
        defn.connect(mem_scale_filter.select("data_out_0"), scale_output_io.select("in"))
    else:
        defn.connect(scale_pe.select("O0"), scale_output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_ios,
        "unquantized_io": unquantized_ios,
        "delay_units": delay_units,
        "tree_direct_stages": tree_direct_stages,
        "tree_delayed_stages": tree_delayed_stages,
        "direct_label": direct_label,
        "delayed_label": delayed_label,
        "combined_max_pe": combined_max_pe,
        "scale_pe": scale_pe,
        "mem_scale_filter": mem_scale_filter,
        "scale_output_io": scale_output_io,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, num_vecs: int, total_channels: int,
               mode: str, top_module: str):
    if total_channels % unroll != 0:
        raise ValueError(f"total_channels ({total_channels}) must be divisible by unroll ({unroll})")
    num_lines = total_channels // unroll
    if num_lines % 2 != 0:
        raise ValueError(f"total_channels/unroll ({num_lines}) must be even for block-of-2 scale filtering")

    defn = instances["scale_output_io"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    abs_max_val, abs_max_w = pe_inst_to_bits_with_operands(
        "fp_abs_max", data0=("ext", None), data1=("ext", None),
    )
    shared_exp_val, shared_exp_w = pe_inst_to_bits_with_operands(
        "get_shared_exp", data0=("ext", None),
    )

    # All tree + combined_max PEs run fp_abs_max; scale_pe runs get_shared_exp.
    abs_max_pes = []
    for label, stages in [
        (instances["direct_label"], instances["tree_direct_stages"]),
        (instances["delayed_label"], instances["tree_delayed_stages"]),
    ]:
        for s_idx, stage in enumerate(stages):
            for j, pe in enumerate(stage):
                abs_max_pes.append((pe, f"{label}_stage{s_idx + 1}_pe_{j}"))
    abs_max_pes.append((instances["combined_max_pe"], f"{top_module}_combined_max_pe"))

    for pe_inst, name in abs_max_pes:
        c = defn.add_generator_instance(
            f"const_inst_{name}", const_gen,
            context.new_values({"width": abs_max_w}),
            context.new_values({"value": BitVector[abs_max_w](abs_max_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    c = defn.add_generator_instance(
        f"const_inst_{top_module}_get_shared_exp_pe", const_gen,
        context.new_values({"width": shared_exp_w}),
        context.new_values({"value": BitVector[shared_exp_w](shared_exp_val)}),
    )
    defn.connect(c.select("out"), instances["scale_pe"].select("inst"))

    # IO metadata.
    per_input_extent = num_vecs if mode == "mu" else num_vecs * num_lines
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_input_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    scale_extent = num_vecs * (num_lines // 2)
    if mode == "mu":
        scale_in2glb = json.dumps({
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [scale_extent],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        })
    else:
        scale_in2glb = json.dumps({
            "cycle_starting_addr": [1],
            "cycle_stride": [2],
            "dimensionality": 1,
            "extent": [scale_extent],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        })

    for inp_io in instances["input_io"]:
        inp_io.add_metadata("glb2out_0", input_glb2out)
    instances["scale_output_io"].add_metadata("in2glb_0", scale_in2glb)

    if mode == "mu":
        unquant_in2glb = json.dumps({
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [num_vecs * num_lines],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        })
        for unq_io in instances["unquantized_io"]:
            unq_io.add_metadata("in2glb_0", unquant_in2glb)

        mem_line_buffer_cfg = json.dumps({
            "type": "get_single_mem_line_buffer",
            "buffer_size": num_vecs,
            "num_lines": num_lines,
        })
        for mem in instances["delay_units"]:
            mem.add_metadata("lake_rv_config", mem_line_buffer_cfg)

        filter_scale_cfg = json.dumps({
            "type": "get_filter_scale_mem",
            "img_size": num_vecs,
            "total_channels": total_channels,
            "mu_OC": unroll,
            "packed": False,
        })
        instances["mem_scale_filter"].add_metadata("lake_rv_config", filter_scale_cfg)


def build_reduction_get_scale_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    num_vecs: int = DEFAULT_NUM_VECS,
    total_channels: int = DEFAULT_TOTAL_CHANNELS,
    mode: str = DEFAULT_MODE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, mode, top_module)
    _configure(context, instances, unroll, num_vecs, total_channels, mode, top_module)
    return context, top


def emit_reduction_get_scale_bf16_design(
    unroll: int,
    num_vecs: int,
    total_channels: int,
    output_path: str,
    mode: str = DEFAULT_MODE,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_reduction_get_scale_bf16_context(
        unroll, num_vecs, total_channels, mode=mode, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reduction_get_scale_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--total-channels", type=int, default=DEFAULT_TOTAL_CHANNELS)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=list(_VALID_MODES))
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_reduction_get_scale_bf16_design(
        args.unroll, args.num_vecs, args.total_channels, args.output_path,
        mode=args.mode, top_module=args.top_module,
    )
