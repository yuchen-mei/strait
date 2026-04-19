"""
Build a bf16 per-lane spatial-average accumulator CoreIR graph using pycoreir.
Used by: avgpool_layer_fp — per-channel global spatial average.

For each of `unroll` independent output lanes, compute:

    output[c] = scale * sum over (x, y) in [0, vec_length) of input[c, x, y]

where `scale` is a bf16 constant (typically 1 / (in_img - pad)^2). Each lane
independently handles `num_vecs` channels (one per output pixel), so there is
NO tree reduction across lanes — each lane owns its own filter_mem + two
accumulator ponds + final reducer.

Templated design parameters:
- unroll: number of parallel output lanes (= glb_i = glb_o).
- vec_length: spatial reduction length per lane per channel (in_img * in_img);
  must be even so the two ponds each absorb vec_length / 2 partials.
- num_vecs: output pixels per lane (= n_ic / glb_i).
- scale_bf16_val: per-element constant multiplier (Python float; encoded as bf16).
- top_module: name of the emitted top module (e.g. "avgpool_layer_fp").

Architecture (per lane, independent):
  input IO.out -> mul_pe.data0  ; data1 = const(scale)
  mul_pe.O0    -> filter_mem.data_in_0 (2 read streams, alternating)
               -> accum_pe_0 + accum_pond_0  (half of the partial sums)
               -> accum_pe_1 + accum_pond_1  (other half)
               -> final_reduce_pe (fp_add across the two pond spills)
               -> output IO

Both accum PEs per lane require `output_fifo_bypass=1` to avoid RMW stalls on
the tile-level output FIFO. This template emits `PE_fifos_bypass_config.json`
alongside `design_top.json`.

IO metadata (per lane):
- Input: 2D stream, extent = [vec_length, num_vecs],
  cycle_stride = [1, vec_length], read_data_stride = [num_vecs, 1].
- Output: 1D stream, extent = [num_vecs], stride = [1].
"""

import json
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
DEFAULT_VEC_LENGTH = 64
DEFAULT_NUM_VECS = 16
DEFAULT_SCALE_BF16_VAL = 1.0 / 64.0
DEFAULT_TOP_MODULE = "avgpool_layer_fp"


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


def _build_graph(unroll: int, top_module: str):
    if unroll < 1:
        raise ValueError(f"unroll must be >= 1, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    corebit_const = context.get_namespace("corebit").modules["const"]

    top = global_namespace.new_module(top_module, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    # One shared clk const for all ponds/mems in the design.
    shared_clk = defn.add_module_instance(
        f"{top_module}_clk_en_const", corebit_const, context.new_values({"value": True}),
    )

    pond_genargs = context.new_values({
        "ID": "",
        "has_stencil_valid": True,
        "num_inputs": 2,
        "num_outputs": 2,
        "width": 16,
    })
    mem_genargs = make_mem_genargs(context)

    input_io_list = []
    output_io_list = []
    mul_pe_list = []
    filter_mem_list = []
    accum_pond_0_list = []
    accum_pond_1_list = []
    accum_pe_0_list = []
    accum_pe_1_list = []
    final_reduce_pe_list = []

    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"

        input_io = defn.add_module_instance(_input_io_name(i), io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(_output_io_name(i), io_module, context.new_values({"mode": "out"}))
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))

        mul_pe = defn.add_module_instance(f"{lane_prefix}_mul_pe", pe_module)
        defn.connect(input_io.select("out"), mul_pe.select("data0"))

        filter_mem = defn.add_generator_instance(
            f"{lane_prefix}_filter_mem", mem_gen, mem_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _k, _v in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            filter_mem.add_metadata(_k, _v)
        defn.connect(mul_pe.select("O0"), filter_mem.select("data_in_0"))
        defn.connect(shared_clk.select("out"), filter_mem.select("clk_en"))

        accum_pond_0 = defn.add_generator_instance(
            f"{lane_prefix}_accum_pond_0", pond_gen, pond_genargs,
            context.new_values({"config": {}, "mode": "pond"}),
        )
        accum_pond_1 = defn.add_generator_instance(
            f"{lane_prefix}_accum_pond_1", pond_gen, pond_genargs,
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
            defn.connect(shared_clk.select("out"), pond.select("clk_en"))

        accum_pe_0 = defn.add_module_instance(f"{lane_prefix}_accum_pe_0", pe_module)
        accum_pe_1 = defn.add_module_instance(f"{lane_prefix}_accum_pe_1", pe_module)
        final_reduce_pe = defn.add_module_instance(f"{lane_prefix}_final_reduce_pe", pe_module)

        defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
        defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

        defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
        defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
        defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

        defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
        defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
        defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

        defn.connect(final_reduce_pe.select("O0"), output_io.select("in"))

        input_io_list.append(input_io)
        output_io_list.append(output_io)
        mul_pe_list.append(mul_pe)
        filter_mem_list.append(filter_mem)
        accum_pond_0_list.append(accum_pond_0)
        accum_pond_1_list.append(accum_pond_1)
        accum_pe_0_list.append(accum_pe_0)
        accum_pe_1_list.append(accum_pe_1)
        final_reduce_pe_list.append(final_reduce_pe)

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "output_io": output_io_list,
        "mul_pe": mul_pe_list,
        "filter_mem": filter_mem_list,
        "accum_pond_0": accum_pond_0_list,
        "accum_pond_1": accum_pond_1_list,
        "accum_pe_0": accum_pe_0_list,
        "accum_pe_1": accum_pe_1_list,
        "final_reduce_pe": final_reduce_pe_list,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int,
               scale_bf16_val: float, top_module: str):
    if vec_length % 2 != 0:
        raise ValueError(f"vec_length ({vec_length}) must be even for two-pond accumulation")

    defn = instances["filter_mem"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    scale_bits = bf16_bits_from_float(scale_bf16_val)
    mul_val, mul_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("const", scale_bits),
    )
    add_val, add_w = pe_inst_to_bits_with_operands(
        "fp_add", data0=("ext", None), data1=("ext", None),
    )

    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"
        c = defn.add_generator_instance(
            f"const_inst_{lane_prefix}_mul_pe", const_gen,
            context.new_values({"width": mul_w}),
            context.new_values({"value": BitVector[mul_w](mul_val)}),
        )
        defn.connect(c.select("out"), instances["mul_pe"][i].select("inst"))

        for role, pe_inst in [
            ("accum_pe_0", instances["accum_pe_0"][i]),
            ("accum_pe_1", instances["accum_pe_1"][i]),
            ("final_reduce_pe", instances["final_reduce_pe"][i]),
        ]:
            c = defn.add_generator_instance(
                f"const_inst_{lane_prefix}_{role}", const_gen,
                context.new_values({"width": add_w}),
                context.new_values({"value": BitVector[add_w](add_val)}),
            )
            defn.connect(c.select("out"), pe_inst.select("inst"))

    # IO metadata.
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, vec_length],
        "dimensionality": 2,
        "extent": [vec_length, num_vecs],
        "read_data_starting_addr": [0],
        "read_data_stride": [num_vecs, 1],
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
        io_in.add_metadata("glb2out_0", input_glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", in2glb)

    # Lake RV configs per lane.
    filter_mem_cfg = json.dumps({
        "type": "get_filter_mem_two_streams",
        "input_stream_size": vec_length * num_vecs,
    })
    accum_pond_cfg = json.dumps({
        "type": "get_vec_accum_pond",
        "num_partial_reduction": vec_length // 2,
        "num_output_pixels": num_vecs,
    })
    for i in range(unroll):
        instances["filter_mem"][i].add_metadata("lake_rv_config", filter_mem_cfg)
        instances["accum_pond_0"][i].add_metadata("lake_rv_config", accum_pond_cfg)
        instances["accum_pond_1"][i].add_metadata("lake_rv_config", accum_pond_cfg)


def _write_pe_fifos_bypass_config(output_path: str, unroll: int, top_module: str):
    bypass_cfg = {}
    for i in range(unroll):
        lane_prefix = f"{top_module}_lane_{i}"
        bypass_cfg[f"{lane_prefix}_accum_pe_0"] = {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1}
        bypass_cfg[f"{lane_prefix}_accum_pe_1"] = {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1}
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_accumulator_avg_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    scale_bf16_val: float = DEFAULT_SCALE_BF16_VAL,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top, instances = _build_graph(unroll, top_module)
    _configure(context, instances, unroll, vec_length, num_vecs, scale_bf16_val, top_module)
    return context, top


def emit_accumulator_avg_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
    scale_bf16_val: float = DEFAULT_SCALE_BF16_VAL,
    top_module: str = DEFAULT_TOP_MODULE,
):
    context, top = build_accumulator_avg_bf16_context(
        unroll, vec_length, num_vecs, scale_bf16_val=scale_bf16_val, top_module=top_module,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, unroll, top_module)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate accumulator_avg_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--scale-bf16-val", type=float, default=DEFAULT_SCALE_BF16_VAL)
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_accumulator_avg_bf16_design(
        args.unroll, args.vec_length, args.num_vecs, args.output_path,
        scale_bf16_val=args.scale_bf16_val, top_module=args.top_module,
    )
