"""
Build the bf16 max-reduction CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of parallel input lanes (= glb_i, must be a power of 2).
- vec_length: length of one inner vector (= vec_width from halide args).
- num_vecs: number of vectors to reduce (= vec_height from halide args).

Architecture (per row):
  [unroll input IOs] -> balanced max-tree of (unroll - 1) fp_max PEs
      -> filter_mem MEM (2 read streams, alternating)
      -> accum_pe_0 + accum_pond_0  (half of the partial maxes)
      -> accum_pe_1 + accum_pond_1  (other half)
      -> final_reduce_pe (fp_max across the two pond spills)
      -> output IO

`num_partial_reduction = vec_length // unroll` partial maxes are produced per
output pixel. num_partial_reduction must be even so the two ponds each absorb
num_partial_reduction // 2 partials.

The two accum PEs require `output_fifo_bypass=1` so the RMW-style accumulation
loop does not stall on the tile-level output FIFO. This template emits
`PE_fifos_bypass_config.json` alongside `design_top.json`.
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
DEFAULT_NUM_VECS = 128
TOP_MODULE = "stable_softmax_pass1_fp"


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_io_name() -> str:
    return "io16_hw_output_stencil_op_hcompute_hw_output_stencil_write_0"


def _output_self_port() -> str:
    return "hw_output_stencil_op_hcompute_hw_output_stencil_write_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
    record[_output_self_port()] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
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
    pond_gen = context.get_lib("cgralib").generators["Pond"]

    top = global_namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    input_io_list = []
    for i in range(unroll):
        io = defn.add_module_instance(_input_io_name(i), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_input_self_port(i)), io.select("in"))
        input_io_list.append(io)

    output_io = defn.add_module_instance(_output_io_name(), io_module, context.new_values({"mode": "out"}))
    defn.connect(output_io.select("out"), iface.select(_output_self_port()))

    # Max-reduction tree: stage 1 takes the `unroll` input IO lanes;
    # each later stage halves the count until a single PE remains.
    tree_stages = int(math.log2(unroll))
    tree_pes_by_stage = []
    prev_outs = [io.select("out") for io in input_io_list]
    for stage in range(1, tree_stages + 1):
        this_stage = []
        num_pairs = len(prev_outs) // 2
        next_outs = []
        for j in range(num_pairs):
            pe = defn.add_module_instance(f"tree_stage{stage}_pe_{j}", pe_module)
            defn.connect(prev_outs[2 * j], pe.select("data0"))
            defn.connect(prev_outs[2 * j + 1], pe.select("data1"))
            this_stage.append(pe)
            next_outs.append(pe.select("O0"))
        tree_pes_by_stage.append(this_stage)
        prev_outs = next_outs
    assert len(prev_outs) == 1, "Tree should collapse to a single output"
    tree_final_out = prev_outs[0]

    # filter_mem MEM: 1 write port, 2 read ports that alternate (two streams).
    mem_genargs = make_mem_genargs(context)
    filter_mem = defn.add_generator_instance(
        f"{TOP_MODULE}_filter_mem",
        mem_gen,
        mem_genargs,
        context.new_values({"config": {}, "mode": "lake"}),
    )
    for _k, _v in [
        ("config", json.dumps({})),
        ("is_rom", json.dumps(False)),
        ("mode", json.dumps("lake")),
        ("width", json.dumps(16)),
    ]:
        filter_mem.add_metadata(_k, _v)

    defn.connect(tree_final_out, filter_mem.select("data_in_0"))

    # Two accumulator PEs + ponds and the final reducer.
    pond_genargs = context.new_values({
        "ID": "",
        "has_stencil_valid": True,
        "num_inputs": 2,
        "num_outputs": 2,
        "width": 16,
    })
    accum_pond_0 = defn.add_generator_instance(
        f"{TOP_MODULE}_accum_pond_0",
        pond_gen,
        pond_genargs,
        context.new_values({"config": {}, "mode": "pond"}),
    )
    accum_pond_1 = defn.add_generator_instance(
        f"{TOP_MODULE}_accum_pond_1",
        pond_gen,
        pond_genargs,
        context.new_values({"config": {}, "mode": "pond"}),
    )
    for pond in (accum_pond_0, accum_pond_1):
        for _k, _v in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("pond")),
            ("width", json.dumps(16)),
        ]:
            pond.add_metadata(_k, _v)

    accum_pe_0 = defn.add_module_instance(f"{TOP_MODULE}_accum_pe_0", pe_module)
    accum_pe_1 = defn.add_module_instance(f"{TOP_MODULE}_accum_pe_1", pe_module)
    final_reduce_pe = defn.add_module_instance(f"{TOP_MODULE}_final_reduce_pe", pe_module)

    # Shared clock enable const (drives filter_mem and both ponds).
    corebit_const = context.get_namespace("corebit").modules["const"]
    shared_clk = defn.add_module_instance(
        f"{TOP_MODULE}_clk_en_const",
        corebit_const,
        context.new_values({"value": True}),
    )
    defn.connect(shared_clk.select("out"), filter_mem.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_0.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_1.select("clk_en"))

    # filter_mem -> accum PEs (data1); ponds feed back on data0 + spill to final_reduce.
    defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
    defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

    defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
    defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
    defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

    defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
    defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
    defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

    defn.connect(final_reduce_pe.select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "output_io": output_io,
        "tree_pes_by_stage": tree_pes_by_stage,
        "filter_mem": filter_mem,
        "accum_pond_0": accum_pond_0,
        "accum_pond_1": accum_pond_1,
        "accum_pe_0": accum_pe_0,
        "accum_pe_1": accum_pe_1,
        "final_reduce_pe": final_reduce_pe,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")
    num_partial_reduction = vec_length // unroll
    if num_partial_reduction % 2 != 0:
        raise ValueError(f"num_partial_reduction ({num_partial_reduction}) must be even for two-pond accumulation")

    defn = instances["filter_mem"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    # All PEs run fp_max with both operands from the external interconnect.
    fp_max_val, fp_max_w = pe_inst_to_bits_with_operands(
        "fp_max", data0=("ext", None), data1=("ext", None)
    )

    all_pes = []
    for stage_idx, stage in enumerate(instances["tree_pes_by_stage"]):
        for j, pe in enumerate(stage):
            all_pes.append((pe, f"tree_stage{stage_idx + 1}_pe_{j}"))
    all_pes.append((instances["accum_pe_0"], f"{TOP_MODULE}_accum_pe_0"))
    all_pes.append((instances["accum_pe_1"], f"{TOP_MODULE}_accum_pe_1"))
    all_pes.append((instances["final_reduce_pe"], f"{TOP_MODULE}_final_reduce_pe"))

    for pe_inst, name in all_pes:
        c = defn.add_generator_instance(
            f"const_inst_{name}",
            const_gen,
            context.new_values({"width": fp_max_w}),
            context.new_values({"value": BitVector[fp_max_w](fp_max_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    # IO metadata: each input lane streams num_vecs * vec_length / unroll per schedule,
    # the single output lane emits num_vecs values (one per input row).
    per_lane_extent = num_vecs * vec_length // unroll
    glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [num_vecs],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })
    for io_in in instances["input_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    instances["output_io"].add_metadata("in2glb_0", in2glb)

    # MEM + Pond lake_rv_config metadata so spec.py can configure them directly.
    filter_mem_cfg = json.dumps({
        "type": "get_filter_mem_two_streams",
        "input_stream_size": num_partial_reduction * num_vecs,
    })
    instances["filter_mem"].add_metadata("lake_rv_config", filter_mem_cfg)

    accum_pond_cfg = json.dumps({
        "type": "get_vec_accum_pond",
        "num_partial_reduction": num_partial_reduction // 2,
        "num_output_pixels": num_vecs,
    })
    instances["accum_pond_0"].add_metadata("lake_rv_config", accum_pond_cfg)
    instances["accum_pond_1"].add_metadata("lake_rv_config", accum_pond_cfg)


def _write_pe_fifos_bypass_config(output_path: str):
    """
    The two accumulator PEs need their tile-level output FIFOs bypassed so the
    read-modify-write accumulation loop does not stall. Write the config file
    alongside design_top.json.
    """
    bypass_cfg = {
        f"{TOP_MODULE}_accum_pe_0": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
        f"{TOP_MODULE}_accum_pe_1": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
    }
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_reduction_max_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
):
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vec_length, num_vecs)
    return context, top


def emit_reduction_max_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
):
    context, top = build_reduction_max_bf16_context(unroll, vec_length, num_vecs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote reduction_max_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reduction_max_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_reduction_max_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path)
