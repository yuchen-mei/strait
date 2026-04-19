"""
Build a bf16 GEMV (matrix-vector multiply + optional bias) CoreIR graph using
pycoreir. Computes per output row y:

    output[y] = sum_x( matrix[x, y] * vector[x] )   [ + bias[y] if has_bias ]

Two modes:
- has_bias=False: pure GEMV (matrix-vector multiply with row reduction).
- has_bias=True:  GEMV + bias add per output row.

Templated design parameters:
- unroll: number of parallel input lanes (= glb_i, must be a power of 2).
- vec_length: inner reduction length per row (matrix_width).
- num_vecs: number of output rows (matrix_height).
- has_bias: if True, adds bias input IO and a bias_add PE after final reduce.
- top_module: name of the emitted top module (e.g. "fully_connected_layer_fp").

Architecture (per row):
  matrix lane i IO.out -> mul_pe_i.data0
  vector lane i IO.out -> mul_pe_i.data1
  all mul_pe.O0  -> balanced fp_add tree of (unroll - 1) PEs
      -> filter_mem MEM (2 read streams, alternating)
      -> accum_pe_0 + accum_pond_0  (half of the partial sums)
      -> accum_pe_1 + accum_pond_1  (other half)
      -> final_reduce_pe (fp_add across the two pond spills)
      [-> bias_add_pe.data1;  bias IO.out -> bias_add_pe.data0]
      -> output IO

`num_partial_reduction = vec_length // unroll` partial sums per output row.
num_partial_reduction must be even so each pond absorbs half.

The two accum PEs require `output_fifo_bypass=1` so the RMW-style accumulation
loop does not stall on the tile-level output FIFO. This template emits
`PE_fifos_bypass_config.json` alongside `design_top.json`.

IO metadata:
- Matrix (per lane): 1D, extent = vec_length * num_vecs / unroll (flat stream).
- Vector (per lane): 2D broadcast, extent = [vec_length / unroll, num_vecs],
  read_data_stride = [1, 0] so the same inner vector is re-read each row.
- Bias: 1D, extent = num_vecs (one element per output row).
- Output: 1D, extent = num_vecs.
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
DEFAULT_VEC_LENGTH = 512
DEFAULT_NUM_VECS = 500
DEFAULT_HAS_BIAS = True
DEFAULT_TOP_MODULE = "fully_connected_layer_fp"


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _matrix_io_name(i: int) -> str:
    return f"io16in_matrix_host_stencil_clkwrk_{i}_op_hcompute_matrix_glb_stencil{_stencil_suffix(i)}_read_0"


def _matrix_self_port(i: int) -> str:
    return f"matrix_host_stencil_clkwrk_{i}_op_hcompute_matrix_glb_stencil{_stencil_suffix(i)}_read_0"


def _vector_io_name(i: int) -> str:
    return f"io16in_vector_host_stencil_clkwrk_{i}_op_hcompute_vector_glb_stencil{_stencil_suffix(i)}_read_0"


def _vector_self_port(i: int) -> str:
    return f"vector_host_stencil_clkwrk_{i}_op_hcompute_vector_glb_stencil{_stencil_suffix(i)}_read_0"


def _bias_io_name() -> str:
    return "io16in_bias_host_stencil_clkwrk_0_op_hcompute_bias_glb_stencil_read_0"


def _bias_self_port() -> str:
    return "bias_host_stencil_clkwrk_0_op_hcompute_bias_glb_stencil_read_0"


def _output_io_name() -> str:
    return "io16_hw_output_stencil_op_hcompute_hw_output_stencil_write_0"


def _output_self_port() -> str:
    return "hw_output_stencil_op_hcompute_hw_output_stencil_write_0"


def _interface_type(context, unroll: int, has_bias: bool):
    record = {}
    for i in range(unroll):
        record[_matrix_self_port(i)] = context.Array(16, context.BitIn())
        record[_vector_self_port(i)] = context.Array(16, context.BitIn())
    if has_bias:
        record[_bias_self_port()] = context.Array(16, context.BitIn())
    record[_output_self_port()] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, has_bias: bool, top_module: str):
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

    top = global_namespace.new_module(top_module, _interface_type(context, unroll, has_bias))
    defn = top.new_definition()
    iface = defn.interface

    matrix_io_list = []
    vector_io_list = []
    mul_pe_list = []
    for i in range(unroll):
        m_io = defn.add_module_instance(_matrix_io_name(i), io_module, context.new_values({"mode": "in"}))
        v_io = defn.add_module_instance(_vector_io_name(i), io_module, context.new_values({"mode": "in"}))
        mul_pe = defn.add_module_instance(f"mul_pe_{i}", pe_module)
        defn.connect(iface.select(_matrix_self_port(i)), m_io.select("in"))
        defn.connect(iface.select(_vector_self_port(i)), v_io.select("in"))
        defn.connect(m_io.select("out"), mul_pe.select("data0"))
        defn.connect(v_io.select("out"), mul_pe.select("data1"))
        matrix_io_list.append(m_io)
        vector_io_list.append(v_io)
        mul_pe_list.append(mul_pe)

    output_io = defn.add_module_instance(_output_io_name(), io_module, context.new_values({"mode": "out"}))
    defn.connect(output_io.select("out"), iface.select(_output_self_port()))

    if has_bias:
        bias_io = defn.add_module_instance(_bias_io_name(), io_module, context.new_values({"mode": "in"}))
        defn.connect(iface.select(_bias_self_port()), bias_io.select("in"))
    else:
        bias_io = None

    # Sum-reduction tree over the per-lane products.
    tree_stages = int(math.log2(unroll))
    tree_pes_by_stage = []
    prev_outs = [pe.select("O0") for pe in mul_pe_list]
    for stage in range(1, tree_stages + 1):
        this_stage = []
        next_outs = []
        for j in range(len(prev_outs) // 2):
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
        f"{top_module}_filter_mem",
        mem_gen, mem_genargs,
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
        f"{top_module}_accum_pond_0", pond_gen, pond_genargs,
        context.new_values({"config": {}, "mode": "pond"}),
    )
    accum_pond_1 = defn.add_generator_instance(
        f"{top_module}_accum_pond_1", pond_gen, pond_genargs,
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

    accum_pe_0 = defn.add_module_instance(f"{top_module}_accum_pe_0", pe_module)
    accum_pe_1 = defn.add_module_instance(f"{top_module}_accum_pe_1", pe_module)
    final_reduce_pe = defn.add_module_instance(f"{top_module}_final_reduce_pe", pe_module)

    corebit_const = context.get_namespace("corebit").modules["const"]
    shared_clk = defn.add_module_instance(
        f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}),
    )
    defn.connect(shared_clk.select("out"), filter_mem.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_0.select("clk_en"))
    defn.connect(shared_clk.select("out"), accum_pond_1.select("clk_en"))

    defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
    defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

    defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
    defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
    defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

    defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
    defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
    defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

    if has_bias:
        bias_add_pe = defn.add_module_instance(f"{top_module}_bias_add_pe", pe_module)
        defn.connect(bias_io.select("out"), bias_add_pe.select("data0"))
        defn.connect(final_reduce_pe.select("O0"), bias_add_pe.select("data1"))
        defn.connect(bias_add_pe.select("O0"), output_io.select("in"))
    else:
        bias_add_pe = None
        defn.connect(final_reduce_pe.select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "matrix_io": matrix_io_list,
        "vector_io": vector_io_list,
        "mul_pe": mul_pe_list,
        "bias_io": bias_io,
        "output_io": output_io,
        "tree_pes_by_stage": tree_pes_by_stage,
        "filter_mem": filter_mem,
        "accum_pond_0": accum_pond_0,
        "accum_pond_1": accum_pond_1,
        "accum_pe_0": accum_pe_0,
        "accum_pe_1": accum_pe_1,
        "final_reduce_pe": final_reduce_pe,
        "bias_add_pe": bias_add_pe,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int,
               has_bias: bool, top_module: str):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")
    num_partial_reduction = vec_length // unroll
    if num_partial_reduction % 2 != 0:
        raise ValueError(f"num_partial_reduction ({num_partial_reduction}) must be even for two-pond accumulation")

    defn = instances["filter_mem"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    mul_val, mul_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("ext", None)
    )
    add_val, add_w = pe_inst_to_bits_with_operands(
        "fp_add", data0=("ext", None), data1=("ext", None)
    )

    # Per-lane mul_pe instruction.
    for i, mul_pe in enumerate(instances["mul_pe"]):
        c = defn.add_generator_instance(
            f"const_inst_mul_pe_{i}", const_gen,
            context.new_values({"width": mul_w}),
            context.new_values({"value": BitVector[mul_w](mul_val)}),
        )
        defn.connect(c.select("out"), mul_pe.select("inst"))

    # Tree + accum + final (+ bias_add) PEs all run fp_add.
    add_pes = []
    for stage_idx, stage in enumerate(instances["tree_pes_by_stage"]):
        for j, pe in enumerate(stage):
            add_pes.append((pe, f"tree_stage{stage_idx + 1}_pe_{j}"))
    add_pes.append((instances["accum_pe_0"], f"{top_module}_accum_pe_0"))
    add_pes.append((instances["accum_pe_1"], f"{top_module}_accum_pe_1"))
    add_pes.append((instances["final_reduce_pe"], f"{top_module}_final_reduce_pe"))
    if has_bias:
        add_pes.append((instances["bias_add_pe"], f"{top_module}_bias_add_pe"))

    for pe_inst, name in add_pes:
        c = defn.add_generator_instance(
            f"const_inst_{name}", const_gen,
            context.new_values({"width": add_w}),
            context.new_values({"value": BitVector[add_w](add_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    # IO metadata.
    per_matrix_extent = num_vecs * vec_length // unroll
    per_row_extent = vec_length // unroll

    matrix_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_matrix_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    vector_glb2out = json.dumps({
        "cycle_starting_addr": [0, 0],
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [per_row_extent, num_vecs],
        "read_data_starting_addr": [0, 0],
        "read_data_stride": [1, 0],
    })
    bias_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [num_vecs],
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

    for io_in in instances["matrix_io"]:
        io_in.add_metadata("glb2out_0", matrix_glb2out)
    for io_in in instances["vector_io"]:
        io_in.add_metadata("glb2out_0", vector_glb2out)
    if has_bias:
        instances["bias_io"].add_metadata("glb2out_0", bias_glb2out)
    instances["output_io"].add_metadata("in2glb_0", in2glb)

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


def _write_pe_fifos_bypass_config(output_path: str, top_module: str):
    bypass_cfg = {
        f"{top_module}_accum_pe_0": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
        f"{top_module}_accum_pe_1": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
    }
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_reduction_gemv_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    has_bias: bool = DEFAULT_HAS_BIAS,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, has_bias, top_module)
    _configure(context, instances, unroll, vec_length, num_vecs, has_bias, top_module)
    return context, top


def emit_reduction_gemv_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
    has_bias: bool = DEFAULT_HAS_BIAS,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_reduction_gemv_bf16_context(
        unroll, vec_length, num_vecs, has_bias=has_bias, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, top_module)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate reduction_gemv_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--has-bias", action="store_true", default=DEFAULT_HAS_BIAS)
    parser.add_argument("--no-bias", dest="has_bias", action="store_false")
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_reduction_gemv_bf16_design(
        args.unroll, args.vec_length, args.num_vecs, args.output_path,
        has_bias=args.has_bias, top_module=args.top_module,
    )
