"""
Build the RoPE (Rotary Position Embedding) CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of lanes (= glb_i = glb_o from halide_gen_args).
- vector_len: total vector size; each IO has extent vector_len // unroll.

For each unrolled lane i:
  output_lower = input_lower * cos - input_upper * sin  -> output_buf MEM.data_in_0
  output_upper = input_upper * cos + input_lower * sin  -> output_buf MEM.data_in_1
  output_buf MEM.data_out_0 -> output IO

Architecture per lane:
  - input IO  -> input_buf MEM.data_in_0
  - sin IO    -> sin_buf MEM.data_in_0
  - cos IO    -> cos_buf MEM.data_in_0
  - input_buf MEM.data_out_0 (lower) and .data_out_1 (upper) feed 4 fp_mul PEs
  - cos_buf MEM.data_out_0 (lower cos) feeds mul_lower_cos, .data_out_1 (upper cos) feeds mul_upper_cos
  - sin_buf MEM.data_out_0 (lower sin) feeds mul_lower_sin, .data_out_1 (upper sin) feeds mul_upper_sin
  - 1 fp_sub PE:  input_lower*cos_lower - input_upper*sin_lower  -> output_buf MEM.data_in_0
  - 1 fp_add PE:  input_upper*cos_upper + input_lower*sin_upper  -> output_buf MEM.data_in_1
  - output_buf MEM.data_out_0 -> output IO
"""

import os
import json
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_HEAD_DIM_HALF = 32
DEFAULT_SEQ_LEN = 512
DEFAULT_N_HEADS = 32


def _rope_bf16_interface_type(context, unroll: int):
    """
    Define the top-level interface: per lane, one input IO, one sin IO,
    one cos IO, and one output IO.
    """
    record = {}
    for i in range(unroll):
        record[f"cos_host_stencil_clkwrk_{i}_op_hcompute_cos_glb_stencil_{i}_read_0"] = context.Array(16, context.BitIn())
        record[f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil_{i}_read_0"] = context.Array(16, context.BitIn())
        record[f"sin_host_stencil_clkwrk_{i}_op_hcompute_sin_glb_stencil_{i}_read_0"] = context.Array(16, context.BitIn())
        record[f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_rope_bf16_graph(unroll: int):
    """
    Structural construction: create modules, instances, and connections
    for the RoPE compute graph with `unroll` lanes.

    Returns (context, rope_module, instances_dict).
    """
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    rope_module = global_namespace.new_module(
        "rope_fp",
        _rope_bf16_interface_type(context, unroll),
    )
    rope_defn = rope_module.new_definition()
    rope_iface = rope_defn.interface

    mem_genargs = make_mem_genargs(context)

    # Collect per-lane instances for configuration step
    io_cos_list = []
    io_input_list = []
    io_sin_list = []
    io_output_list = []
    input_buf_list = []
    sin_buf_list = []
    cos_buf_list = []
    output_buf_list = []
    pe_mul_lower_cos_list = []
    pe_mul_upper_sin_list = []
    pe_mul_upper_cos_list = []
    pe_mul_lower_sin_list = []
    pe_sub_list = []
    pe_add_list = []

    for i in range(unroll):
        # ── IO and self-port names ────────────────────────────────────
        cos_io_name = f"io16in_cos_host_stencil_clkwrk_{i}_op_hcompute_cos_glb_stencil_{i}_read_0"
        input_io_name = f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil_{i}_read_0"
        sin_io_name = f"io16in_sin_host_stencil_clkwrk_{i}_op_hcompute_sin_glb_stencil_{i}_read_0"
        output_io_name = f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"

        cos_self = f"cos_host_stencil_clkwrk_{i}_op_hcompute_cos_glb_stencil_{i}_read_0"
        input_self = f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil_{i}_read_0"
        sin_self = f"sin_host_stencil_clkwrk_{i}_op_hcompute_sin_glb_stencil_{i}_read_0"
        output_self = f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"

        # ── Create IO instances ──────────────────────────────────────
        cos_io = rope_defn.add_module_instance(cos_io_name, io_module, context.new_values({"mode": "in"}))
        input_io = rope_defn.add_module_instance(input_io_name, io_module, context.new_values({"mode": "in"}))
        sin_io = rope_defn.add_module_instance(sin_io_name, io_module, context.new_values({"mode": "in"}))
        output_io = rope_defn.add_module_instance(output_io_name, io_module, context.new_values({"mode": "out"}))

        io_cos_list.append(cos_io)
        io_input_list.append(input_io)
        io_sin_list.append(sin_io)
        io_output_list.append(output_io)

        # ── Create MEM instances ─────────────────────────────────────
        mem_modargs = context.new_values({"config": {}, "mode": "lake"})
        input_buf = rope_defn.add_generator_instance(f"input_buf_mem_{i}", mem_gen, mem_genargs, mem_modargs)
        sin_buf = rope_defn.add_generator_instance(f"sin_buf_mem_{i}", mem_gen, mem_genargs, mem_modargs)
        cos_buf = rope_defn.add_generator_instance(f"cos_buf_mem_{i}", mem_gen, mem_genargs, mem_modargs)
        output_buf = rope_defn.add_generator_instance(f"output_buf_mem_{i}", mem_gen, mem_genargs, mem_modargs)

        # MEM metadata – shared base, then per-type placeholders
        for mem_inst in [input_buf, sin_buf, cos_buf, output_buf]:
            for _key, _val in [
                ("config", json.dumps({})),
                ("is_rom", json.dumps(False)),
                ("mode", json.dumps("lake")),
                ("width", json.dumps(16)),
            ]:
                mem_inst.add_metadata(_key, _val)


        input_buf_list.append(input_buf)
        sin_buf_list.append(sin_buf)
        cos_buf_list.append(cos_buf)
        output_buf_list.append(output_buf)

        # ── Create PE instances ──────────────────────────────────────
        pe_mul_lower_cos = rope_defn.add_module_instance(f"mul_lower_cos_pe_{i}", pe_module)
        pe_mul_upper_sin = rope_defn.add_module_instance(f"mul_upper_sin_pe_{i}", pe_module)
        pe_mul_upper_cos = rope_defn.add_module_instance(f"mul_upper_cos_pe_{i}", pe_module)
        pe_mul_lower_sin = rope_defn.add_module_instance(f"mul_lower_sin_pe_{i}", pe_module)
        pe_sub = rope_defn.add_module_instance(f"sub_pe_{i}", pe_module)
        pe_add = rope_defn.add_module_instance(f"add_pe_{i}", pe_module)

        pe_mul_lower_cos_list.append(pe_mul_lower_cos)
        pe_mul_upper_sin_list.append(pe_mul_upper_sin)
        pe_mul_upper_cos_list.append(pe_mul_upper_cos)
        pe_mul_lower_sin_list.append(pe_mul_lower_sin)
        pe_sub_list.append(pe_sub)
        pe_add_list.append(pe_add)

        # ══════════════════════════════════════════════════════════════
        # Connections
        # ══════════════════════════════════════════════════════════════

        # --- Self <-> IO ---
        rope_defn.connect(rope_iface.select(cos_self), cos_io.select("in"))
        rope_defn.connect(rope_iface.select(input_self), input_io.select("in"))
        rope_defn.connect(rope_iface.select(sin_self), sin_io.select("in"))
        rope_defn.connect(output_io.select("out"), rope_iface.select(output_self))

        # --- IO -> MEM (data_in_0) ---
        rope_defn.connect(input_io.select("out"), input_buf.select("data_in_0"))
        rope_defn.connect(sin_io.select("out"), sin_buf.select("data_in_0"))
        rope_defn.connect(cos_io.select("out"), cos_buf.select("data_in_0"))

        # --- input_buf MEM -> mul PEs ---
        # data_out_0 = lower half, data_out_1 = upper half
        rope_defn.connect(input_buf.select("data_out_0"), pe_mul_lower_cos.select("data0"))
        rope_defn.connect(input_buf.select("data_out_0"), pe_mul_lower_sin.select("data0"))
        rope_defn.connect(input_buf.select("data_out_1"), pe_mul_upper_sin.select("data0"))
        rope_defn.connect(input_buf.select("data_out_1"), pe_mul_upper_cos.select("data0"))

        # --- cos_buf MEM -> mul PEs ---
        # data_out_0 = lower cos, data_out_1 = upper cos
        rope_defn.connect(cos_buf.select("data_out_0"), pe_mul_lower_cos.select("data1"))
        rope_defn.connect(cos_buf.select("data_out_1"), pe_mul_upper_cos.select("data1"))

        # --- sin_buf MEM -> mul PEs ---
        # data_out_0 = lower sin, data_out_1 = upper sin
        rope_defn.connect(sin_buf.select("data_out_0"), pe_mul_lower_sin.select("data1"))
        rope_defn.connect(sin_buf.select("data_out_1"), pe_mul_upper_sin.select("data1"))

        # --- mul PEs -> sub/add PEs ---
        # output_lower = input_lower * cos_lower - input_upper * sin_lower
        rope_defn.connect(pe_mul_lower_cos.select("O0"), pe_sub.select("data0"))
        rope_defn.connect(pe_mul_upper_sin.select("O0"), pe_sub.select("data1"))
        # output_upper = input_upper * cos_upper + input_lower * sin_upper
        rope_defn.connect(pe_mul_upper_cos.select("O0"), pe_add.select("data0"))
        rope_defn.connect(pe_mul_lower_sin.select("O0"), pe_add.select("data1"))

        # --- sub/add PEs -> output_buf MEM ---
        rope_defn.connect(pe_sub.select("O0"), output_buf.select("data_in_0"))
        rope_defn.connect(pe_add.select("O0"), output_buf.select("data_in_1"))

        # --- output_buf MEM -> output IO ---
        rope_defn.connect(output_buf.select("data_out_0"), output_io.select("in"))

    rope_module.definition = rope_defn
    context.set_top(rope_module)

    instances = {
        "io_cos": io_cos_list,
        "io_input": io_input_list,
        "io_sin": io_sin_list,
        "io_output": io_output_list,
        "input_buf": input_buf_list,
        "sin_buf": sin_buf_list,
        "cos_buf": cos_buf_list,
        "output_buf": output_buf_list,
        "pe_mul_lower_cos": pe_mul_lower_cos_list,
        "pe_mul_upper_sin": pe_mul_upper_sin_list,
        "pe_mul_upper_cos": pe_mul_upper_cos_list,
        "pe_mul_lower_sin": pe_mul_lower_sin_list,
        "pe_sub": pe_sub_list,
        "pe_add": pe_add_list,
    }
    return context, rope_module, instances


def _configure_rope_bf16(context, instances, unroll: int, head_dim_half: int, seq_len: int, n_heads: int):
    """
    Configuration step: wire PE instruction constants and set IO metadata.
    """

    # Get the definition from any existing instance
    rope_defn = instances["pe_sub"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    # ── PE instructions ──────────────────────────────────────────────
    fp_mul_val, fp_mul_w = pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None))
    fp_sub_val, fp_sub_w = pe_inst_to_bits_with_operands("fp_sub", data0=("ext", None), data1=("ext", None))
    fp_add_val, fp_add_w = pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))

    pe_inst_map = {
        "pe_mul_lower_cos": (fp_mul_val, fp_mul_w),
        "pe_mul_upper_sin": (fp_mul_val, fp_mul_w),
        "pe_mul_upper_cos": (fp_mul_val, fp_mul_w),
        "pe_mul_lower_sin": (fp_mul_val, fp_mul_w),
        "pe_sub": (fp_sub_val, fp_sub_w),
        "pe_add": (fp_add_val, fp_add_w),
    }

    for pe_key, (inst_val, inst_width) in pe_inst_map.items():
        for i, pe_inst in enumerate(instances[pe_key]):
            pe_inst_const = rope_defn.add_generator_instance(f"const_inst_{pe_key}_lane{i}", const_generator, context.new_values({"width": inst_width}), context.new_values({"value": BitVector[inst_width](inst_val)}))
            rope_defn.connect(pe_inst_const.select("out"), pe_inst.select("inst"))

    extent = head_dim_half * seq_len * n_heads // unroll
    # ── IO metadata ──────────────────────────────────────────────────
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    sin_cos_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [head_dim_half * seq_len // unroll, n_heads],
        "read_data_starting_addr": [0],
        "read_data_stride": [1, 1 - head_dim_half * seq_len // unroll],
    })
    in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for io_in in instances["io_cos"]:
        io_in.add_metadata("glb2out_0", sin_cos_glb2out)
    for io_in in instances["io_input"]:
        io_in.add_metadata("glb2out_0", input_glb2out)
    for io_in in instances["io_sin"]:
        io_in.add_metadata("glb2out_0", sin_cos_glb2out)
    for io_out in instances["io_output"]:
        io_out.add_metadata("in2glb_0", in2glb)

    # ── MEM lake_rv_config metadata ──────────────────────────────────
    # input_buf: 1 write, 2 reads (serial-in parallel-out deinterleave into lower/upper)
    for input_buf in instances["input_buf"]:
        input_buf.add_metadata("lake_rv_config", json.dumps({"type": "get_filter_mem_two_streams", "input_stream_size": extent}))
    # sin_buf: 1 write, 2 reads (serial-in parallel-out deinterleave into lower/upper sin)
    for sin_buf in instances["sin_buf"]:
        sin_buf.add_metadata("lake_rv_config", json.dumps({"type": "get_filter_mem_two_streams", "input_stream_size": extent}))
    # cos_buf: 1 write, 2 reads (serial-in parallel-out deinterleave into lower/upper cos)
    for cos_buf in instances["cos_buf"]:
        cos_buf.add_metadata("lake_rv_config", json.dumps({"type": "get_filter_mem_two_streams", "input_stream_size": extent}))
    # output_buf: 2 writes (parallel-in serial-out interleave from sub/add), 1 read
    for output_buf in instances["output_buf"]:
        output_buf.add_metadata("lake_rv_config", json.dumps({"type": "get_interleave_mem", "single_input_stream_size": extent // 2}))


def build_rope_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    head_dim_half: int = DEFAULT_HEAD_DIM_HALF,
    seq_len: int = DEFAULT_SEQ_LEN,
    n_heads: int = DEFAULT_N_HEADS,
):
    """
    Build the RoPE design with templated parameters.

    Args:
        unroll: Number of parallel lanes (= glb_i = glb_o).
        head_dim_half: Half the head dimension.
        seq_len: Sequence length.
        n_heads: Number of attention heads.

    Returns:
        (context, rope_module)
    """
    context, rope_module, instances = _build_rope_bf16_graph(unroll)
    _configure_rope_bf16(context, instances, unroll, head_dim_half, seq_len, n_heads)
    return context, rope_module


def emit_rope_bf16_design(unroll: int, head_dim_half: int, seq_len: int, n_heads: int, output_path: str):
    """Build and write the rope_fp design_top.json."""
    context, rope_module = build_rope_bf16_context(unroll, head_dim_half, seq_len, n_heads)
    out_file = os.path.join(output_path, "design_top.json")
    rope_module.save_to_file(out_file)
    print(f"[INFO] Wrote rope_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate RoPE CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--head-dim-half", type=int, default=DEFAULT_HEAD_DIM_HALF)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_rope_bf16_design(args.unroll, args.head_dim_half, args.seq_len, args.n_heads, args.output_path)
