"""
Build the bf16 sum-reduction + scalar-multiply + broadcast elementwise-add CoreIR
graph using pycoreir.

Computes input - mean per row.
Output = input(x, y) + sum_x(input(x, y)) * (-1 / vec_width).
The scalar multiply by -1/vec_width produces the negated row mean, and the
per-lane fp_add subtracts it from every input element.

Per row:
    1. balanced fp_add tree over unroll lanes + filter_mem + 2 accum ponds/PEs
       + final_reduce_pe                            ->  sum(y)
    2. scalar_mul_pe: fp_mul(sum(y), -1/vec_width_const)   ->  -mean(y)
    3. broadcast MEM holds the per-row -mean; each lane reads the same scalar
       vec_length/unroll times while the input MEM spills the buffered row out
       on its second read port.
    4. per-lane elementwise fp_add: tile_input_mem.data_out_1 + (-mean)
       -> output IO.

Parameters:
    unroll: glb_i (tree width / parallel lanes, must be power of 2).
    vec_length: vec_width (row length, used for the -1/vec_width constant).
    num_vecs: vec_height (number of rows per invocation).

The 2 accum_pe PEs need output_fifo_bypass=1; the template writes
PE_fifos_bypass_config.json next to design_top.json.
"""

import json
import math
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import (
    bf16_bits_from_float,
    pe_inst_to_bits_with_operands,
)
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_VEC_LENGTH = 768
DEFAULT_NUM_VECS = 128
TOP_MODULE = "layer_norm_pass1_fp"


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_io_name(i: int) -> str:
    return f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
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

    mem_genargs = make_mem_genargs(context)
    pond_genargs = context.new_values({
        "ID": "",
        "has_stencil_valid": True,
        "num_inputs": 2,
        "num_outputs": 2,
        "width": 16,
    })

    input_io_list = []
    output_io_list = []
    tile_input_mem_list = []
    elementwise_add_pe_list = []

    for i in range(unroll):
        input_io = defn.add_module_instance(
            _input_io_name(i), io_module, context.new_values({"mode": "in"})
        )
        output_io = defn.add_module_instance(
            _output_io_name(i), io_module, context.new_values({"mode": "out"})
        )
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        input_io_list.append(input_io)
        output_io_list.append(output_io)

        tile_input_mem = defn.add_generator_instance(
            f"tile_input_lane_{i}",
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
            tile_input_mem.add_metadata(_k, _v)
        tile_input_mem_list.append(tile_input_mem)

        ew_add_pe = defn.add_module_instance(f"elementwise_add_pe_{i}", pe_module)
        elementwise_add_pe_list.append(ew_add_pe)

        # input IO -> tile_input_mem.data_in_0
        defn.connect(input_io.select("out"), tile_input_mem.select("data_in_0"))
        # data_out_1 (delayed read) -> elementwise add.data0
        defn.connect(tile_input_mem.select("data_out_1"), ew_add_pe.select("data0"))
        # elementwise add.O0 -> output IO
        defn.connect(ew_add_pe.select("O0"), output_io.select("in"))

    # Reduction tree: stage 1 consumes tile_input_mem.data_out_0 from every lane.
    tree_stages = int(math.log2(unroll))
    tree_pes_by_stage = []
    prev_outs = [mem.select("data_out_0") for mem in tile_input_mem_list]
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

    # filter_mem (same as pass3): write once, dual-read alternating.
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

    accum_pond_0 = defn.add_generator_instance(
        f"{TOP_MODULE}_accum_pond_0", pond_gen, pond_genargs,
        context.new_values({"config": {}, "mode": "pond"}),
    )
    accum_pond_1 = defn.add_generator_instance(
        f"{TOP_MODULE}_accum_pond_1", pond_gen, pond_genargs,
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

    corebit_const = context.get_namespace("corebit").modules["const"]
    shared_clk = defn.add_module_instance(
        f"{TOP_MODULE}_clk_en_const", corebit_const, context.new_values({"value": True}),
    )
    defn.connect(shared_clk.select("out"), filter_mem.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_0.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_1.select("clk_en"))
    for tim in tile_input_mem_list:
        defn.connect(shared_clk.select("out"), tim.select("clk_en"))

    defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
    defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

    defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
    defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
    defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

    defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
    defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
    defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

    # Scalar multiply: sum * (-1 / vec_length) = -mean(y).
    scalar_mul_pe = defn.add_module_instance(f"{TOP_MODULE}_scalar_mul_pe", pe_module)

    broadcast_mem = defn.add_generator_instance(
        f"{TOP_MODULE}_broadcast_mem",
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
        broadcast_mem.add_metadata(_k, _v)
    defn.connect(shared_clk.select("out"), broadcast_mem.select("clk_en"))

    defn.connect(final_reduce_pe.select("O0"), scalar_mul_pe.select("data0"))
    defn.connect(scalar_mul_pe.select("O0"), broadcast_mem.select("data_in_0"))

    # broadcast MEM output fans out to every elementwise add PE's data1.
    for ew_pe in elementwise_add_pe_list:
        defn.connect(broadcast_mem.select("data_out_0"), ew_pe.select("data1"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "output_io": output_io_list,
        "tile_input_mem": tile_input_mem_list,
        "elementwise_add_pe": elementwise_add_pe_list,
        "tree_pes_by_stage": tree_pes_by_stage,
        "filter_mem": filter_mem,
        "accum_pond_0": accum_pond_0,
        "accum_pond_1": accum_pond_1,
        "accum_pe_0": accum_pe_0,
        "accum_pe_1": accum_pe_1,
        "final_reduce_pe": final_reduce_pe,
        "scalar_mul_pe": scalar_mul_pe,
        "broadcast_mem": broadcast_mem,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")
    num_partial_reduction = vec_length // unroll
    if num_partial_reduction % 2 != 0:
        raise ValueError(f"num_partial_reduction ({num_partial_reduction}) must be even for dual-pond accum")
    per_lane_extent = vec_length * num_vecs // unroll

    defn = instances["filter_mem"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    fp_add_val, fp_add_w = pe_inst_to_bits_with_operands(
        "fp_add", data0=("ext", None), data1=("ext", None)
    )
    # Scalar multiply: fp_mul(sum, -1/vec_length_const) == -mean.
    neg_inv_vec_length = bf16_bits_from_float(-1.0 / float(vec_length))
    scalar_mul_val, scalar_mul_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("const", neg_inv_vec_length)
    )

    pe_inst_list = []
    for stage_idx, stage in enumerate(instances["tree_pes_by_stage"]):
        for j, pe in enumerate(stage):
            pe_inst_list.append((pe, f"tree_stage{stage_idx + 1}_pe_{j}", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["accum_pe_0"], f"{TOP_MODULE}_accum_pe_0", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["accum_pe_1"], f"{TOP_MODULE}_accum_pe_1", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["final_reduce_pe"], f"{TOP_MODULE}_final_reduce_pe", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["scalar_mul_pe"], f"{TOP_MODULE}_scalar_mul_pe", scalar_mul_val, scalar_mul_w))
    for i, pe in enumerate(instances["elementwise_add_pe"]):
        pe_inst_list.append((pe, f"elementwise_add_pe_{i}", fp_add_val, fp_add_w))

    for pe, name, inst_val, inst_w in pe_inst_list:
        c = defn.add_generator_instance(
            f"const_inst_{name}",
            const_gen,
            context.new_values({"width": inst_w}),
            context.new_values({"value": BitVector[inst_w](inst_val)}),
        )
        defn.connect(c.select("out"), pe.select("inst"))

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
        "extent": [per_lane_extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })
    for io_in in instances["input_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", in2glb)

    tile_input_cfg = json.dumps({
        "type": "dual_read",
        "input_stream_size": per_lane_extent,
    })
    for tim in instances["tile_input_mem"]:
        tim.add_metadata("lake_rv_config", tile_input_cfg)

    filter_cfg = json.dumps({
        "type": "get_filter_mem_two_streams",
        "input_stream_size": num_partial_reduction * num_vecs,
    })
    instances["filter_mem"].add_metadata("lake_rv_config", filter_cfg)

    accum_pond_cfg = json.dumps({
        "type": "get_vec_accum_pond",
        "num_partial_reduction": num_partial_reduction // 2,
        "num_output_pixels": num_vecs,
    })
    instances["accum_pond_0"].add_metadata("lake_rv_config", accum_pond_cfg)
    instances["accum_pond_1"].add_metadata("lake_rv_config", accum_pond_cfg)

    broadcast_cfg = json.dumps({
        "type": "get_broadcast_mem",
        "input_stream_size": num_vecs,
        "replicate_factor": num_partial_reduction,
        "raw_scalar": 4,
    })
    instances["broadcast_mem"].add_metadata("lake_rv_config", broadcast_cfg)


def _write_pe_fifos_bypass_config(output_path: str):
    bypass_cfg = {
        f"{TOP_MODULE}_accum_pe_0": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
        f"{TOP_MODULE}_accum_pe_1": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
    }
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_reduction_sum_mul_elementwise_sub_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
):
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vec_length, num_vecs)
    return context, top


def emit_reduction_sum_mul_elementwise_sub_bf16_design(
    unroll: int, vec_length: int, num_vecs: int, output_path: str,
):
    context, top = build_reduction_sum_mul_elementwise_sub_bf16_context(unroll, vec_length, num_vecs)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote reduction_sum_mul_elementwise_sub_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reduction_sum_mul_elementwise_sub_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_reduction_sum_mul_elementwise_sub_bf16_design(
        args.unroll, args.vec_length, args.num_vecs, args.output_path
    )
