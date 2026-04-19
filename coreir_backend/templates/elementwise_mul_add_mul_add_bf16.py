"""
Build the elementwise ((input * (1/gamma)) + (-beta/gamma)) * weight + bias
CoreIR graph using pycoreir.

Used by: layer_norm_pass3_fp — pass 3 of layer norm, which applies the affine
rescale ((input - beta) * (1/gamma)) * weight + bias. Algebraically rewritten as
(input * (1/gamma) + (-beta/gamma)) * weight + bias to give a straight 4-PE
per-lane chain: mul_const -> add_const -> mul_vec -> add_vec.

Templated design parameters:
- unroll: number of parallel lanes (= glb_i).
- vec_length: inner vector length per row (= vec_width from halide args).
- num_vecs: number of rows (= vec_height from halide args).
- gamma: affine scale (default 1.2 from Halide hardcoded value).
- beta: affine offset (default -0.35 from Halide hardcoded value).

Per lane i:
  input_host IO.out   -> mul_const_pe.data0  ; data1 = const(1/gamma)
  mul_const_pe.O0     -> add_const_pe.data0  ; data1 = const(-beta/gamma)
  add_const_pe.O0     -> mul_vec_pe.data0
  weight_host IO.out  -> mul_vec_pe.data1
  bias_host IO.out    -> add_vec_pe.data0
  mul_vec_pe.O0       -> add_vec_pe.data1
  add_vec_pe.O0       -> hw_output IO.in

4 PEs per lane, 3 input IOs + 1 output IO per lane. Straight pipe, all lanes
identical depth — no path balancing required.
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

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 16
DEFAULT_VEC_LENGTH = 384
DEFAULT_NUM_VECS = 128
DEFAULT_GAMMA = 1.2
DEFAULT_BETA = -0.35
TOP_MODULE = "layer_norm_pass3_fp"

_LANE_PE_ROLES = [
    "mul_const",  # fp_mul(input, 1/gamma)
    "add_const",  # fp_add(mul_const, -beta/gamma)
    "mul_vec",    # fp_mul(add_const, weight)
    "add_vec",    # fp_add(bias, mul_vec)
]


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_io_name(i: int) -> str:
    return f"io16in_input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _weight_io_name(i: int) -> str:
    return f"io16in_weight_host_stencil_clkwrk_{i}_op_hcompute_weight_glb_stencil{_stencil_suffix(i)}_read_0"


def _weight_self_port(i: int) -> str:
    return f"weight_host_stencil_clkwrk_{i}_op_hcompute_weight_glb_stencil{_stencil_suffix(i)}_read_0"


def _bias_io_name(i: int) -> str:
    return f"io16in_bias_host_stencil_clkwrk_{i}_op_hcompute_bias_glb_stencil{_stencil_suffix(i)}_read_0"


def _bias_self_port(i: int) -> str:
    return f"bias_host_stencil_clkwrk_{i}_op_hcompute_bias_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_io_name(i: int) -> str:
    return f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _compute_pe_instructions(gamma: float, beta: float):
    inv_gamma_bf16 = bf16_bits_from_float(1.0 / gamma)
    neg_beta_over_gamma_bf16 = bf16_bits_from_float(-beta / gamma)
    return {
        "mul_const": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("const", inv_gamma_bf16)),
        "add_const": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("const", neg_beta_over_gamma_bf16)),
        "mul_vec": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
        "add_vec": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None)),
    }


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_weight_self_port(i)] = context.Array(16, context.BitIn())
        record[_bias_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
    if unroll < 1:
        raise ValueError(f"unroll must be >= 1, got {unroll}")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    top = global_namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    input_io_list = []
    weight_io_list = []
    bias_io_list = []
    output_io_list = []
    pe_by_role = {k: [] for k in _LANE_PE_ROLES}

    for i in range(unroll):
        input_io = defn.add_module_instance(_input_io_name(i), io_module, context.new_values({"mode": "in"}))
        weight_io = defn.add_module_instance(_weight_io_name(i), io_module, context.new_values({"mode": "in"}))
        bias_io = defn.add_module_instance(_bias_io_name(i), io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(_output_io_name(i), io_module, context.new_values({"mode": "out"}))
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(iface.select(_weight_self_port(i)), weight_io.select("in"))
        defn.connect(iface.select(_bias_self_port(i)), bias_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))
        input_io_list.append(input_io)
        weight_io_list.append(weight_io)
        bias_io_list.append(bias_io)
        output_io_list.append(output_io)

        lane_pes = {}
        for role in _LANE_PE_ROLES:
            pe = defn.add_module_instance(f"{role}_pe_{i}", pe_module)
            lane_pes[role] = pe
            pe_by_role[role].append(pe)

        # mul_const = input * (1/gamma)
        defn.connect(input_io.select("out"), lane_pes["mul_const"].select("data0"))
        # add_const = mul_const + (-beta/gamma)
        defn.connect(lane_pes["mul_const"].select("O0"), lane_pes["add_const"].select("data0"))
        # mul_vec = add_const * weight
        defn.connect(lane_pes["add_const"].select("O0"), lane_pes["mul_vec"].select("data0"))
        defn.connect(weight_io.select("out"), lane_pes["mul_vec"].select("data1"))
        # add_vec = bias + mul_vec  (Halide convention: data0=external add operand, data1=mul result)
        defn.connect(bias_io.select("out"), lane_pes["add_vec"].select("data0"))
        defn.connect(lane_pes["mul_vec"].select("O0"), lane_pes["add_vec"].select("data1"))
        # add_vec -> output
        defn.connect(lane_pes["add_vec"].select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "weight_io": weight_io_list,
        "bias_io": bias_io_list,
        "output_io": output_io_list,
        "pe": pe_by_role,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int, gamma: float, beta: float):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")

    defn = instances["output_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    pe_instrs = _compute_pe_instructions(gamma, beta)
    for role, (inst_val, inst_width) in pe_instrs.items():
        for i, pe_inst in enumerate(instances["pe"][role]):
            c = defn.add_generator_instance(
                f"const_inst_{role}_pe_{i}",
                const_gen,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe_inst.select("inst"))

    per_lane_extent = num_vecs * vec_length // unroll
    per_row_extent = vec_length // unroll

    # Input is full [vec_length, num_vecs]: flat dim=1 stream of 3072 per lane.
    input_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [per_lane_extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    # Weight/bias are stored as [vec_length] and broadcast across num_vecs rows:
    # re-read the same per_row_extent entries each row. Net per-outer advancement
    # = per_row_extent + (1 - per_row_extent) = 1, but since each lane holds
    # starting_addr=0 constant, the stride [1, 1 - per_row_extent] rewinds the
    # data addr to 0 at the end of every row.
    broadcast_glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1, 1],
        "dimensionality": 2,
        "extent": [per_row_extent, num_vecs],
        "read_data_starting_addr": [0],
        "read_data_stride": [1, 1 - per_row_extent],
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
        io_in.add_metadata("glb2out_0", input_glb2out)
    for io_in in instances["weight_io"]:
        io_in.add_metadata("glb2out_0", broadcast_glb2out)
    for io_in in instances["bias_io"]:
        io_in.add_metadata("glb2out_0", broadcast_glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_mul_add_mul_add_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    gamma: float = DEFAULT_GAMMA,
    beta: float = DEFAULT_BETA,
):
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vec_length, num_vecs, gamma, beta)
    return context, top


def emit_elementwise_mul_add_mul_add_bf16_design(
    unroll: int,
    vec_length: int,
    num_vecs: int,
    output_path: str,
    gamma: float = DEFAULT_GAMMA,
    beta: float = DEFAULT_BETA,
):
    context, top = build_elementwise_mul_add_mul_add_bf16_context(unroll, vec_length, num_vecs, gamma, beta)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_add_mul_add_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_mul_add_mul_add_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_add_mul_add_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.output_path, args.gamma, args.beta)
