"""
Build the half-add-swish + half-nop CoreIR graph using pycoreir, per lane.

Used by: zircon_add_gelu_pass1_mu_input_fp (aka add_gelu_pass1_mu_input_fp).

Templated design parameters:
- unroll: total number of lanes (= mu_i from halide_gen_args). Must be even.
- vector_len: total tensor size (vec_width * vec_height); each IO extent = vector_len // unroll.
- beta: swish beta (default 1.702).

Lane partition:
- upper half (lanes 0 .. unroll/2 - 1):
    mu_input IO + input_psum0 IO -> fp_add -> MEM buf (dual read) -> swish pipeline -> hw_add_gelu_upper_output IO
- lower half (lanes unroll/2 .. unroll - 1):
    mu_input IO -> dummy_max_nop PE (fp_max(x, -inf)) -> hw_psum1_lower_output IO

IO naming mirrors the gold clkwrk layout for mu_i=unroll:
  input_psum0:             clkwrk = i                    (i in [0, unroll/2))
  mu_input:                clkwrk = unroll + i           (i in [0, unroll))
  hw_add_gelu_upper_output: clkwrk = 2*unroll + i        (i in [0, unroll/2))
  hw_psum1_lower_output:    clkwrk = 2*unroll + i        (i in [unroll/2, unroll))
"""

import math
import os
import json
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
DEFAULT_VECTOR_LEN = 3072 * 64
DEFAULT_BETA = 1.702

# Swish pipeline roles (per upper-half lane). The outer 'add' (mu+psum0) is named separately.
_SWISH_PE_ROLES = [
    "mul_a",           # fp_mul: x * (-beta)
    "mul_b",           # fp_mul: (-beta*x) * (1/ln2)
    "getffrac",        # fp_getffrac
    "getfint",         # fp_getfint
    "addiexp",         # fp_addiexp
    "add",             # fp_add: 1 + exp(-beta*x)
    "dummy_max_nop",   # fp_max(x, -inf): path-balance nop before subexp
    "getmant",         # fp_getmant
    "subexp",          # fp_subexp
    "mul_final",       # fp_mul: x * sigmoid(beta*x)
]


def _stencil_suffix(idx: int) -> str:
    return "" if idx == 0 else f"_{idx}"


def _mu_input_io_name(unroll: int, i: int) -> str:
    return (
        f"io16in_mu_input_host_stencil_clkwrk_{unroll + i}"
        f"_op_hcompute_mu_input_glb_stencil{_stencil_suffix(i)}_read_0"
    )


def _psum0_io_name(i: int) -> str:
    return (
        f"io16in_input_psum0_host_stencil_clkwrk_{i}"
        f"_op_hcompute_input_psum0_glb_stencil{_stencil_suffix(i)}_read_0"
    )


def _upper_out_io_name(unroll: int, i: int) -> str:
    return (
        f"io16_hw_add_gelu_upper_output_stencil_clkwrk_{2 * unroll + i}"
        f"_op_hcompute_hw_add_gelu_upper_output_stencil{_stencil_suffix(i)}_write_0"
    )


def _lower_out_io_name(unroll: int, i: int) -> str:
    return (
        f"io16_hw_psum1_lower_output_stencil_clkwrk_{2 * unroll + i}"
        f"_op_hcompute_hw_psum1_lower_output_stencil_{i}_write_0"
    )


def _mu_input_self_port(unroll: int, i: int) -> str:
    return (
        f"mu_input_host_stencil_clkwrk_{unroll + i}"
        f"_op_hcompute_mu_input_glb_stencil{_stencil_suffix(i)}_read_0"
    )


def _psum0_self_port(i: int) -> str:
    return (
        f"input_psum0_host_stencil_clkwrk_{i}"
        f"_op_hcompute_input_psum0_glb_stencil{_stencil_suffix(i)}_read_0"
    )


def _upper_out_self_port(unroll: int, i: int) -> str:
    return (
        f"hw_add_gelu_upper_output_stencil_clkwrk_{2 * unroll + i}"
        f"_op_hcompute_hw_add_gelu_upper_output_stencil{_stencil_suffix(i)}_write_0"
    )


def _lower_out_self_port(unroll: int, i: int) -> str:
    return (
        f"hw_psum1_lower_output_stencil_clkwrk_{2 * unroll + i}"
        f"_op_hcompute_hw_psum1_lower_output_stencil_{i}_write_0"
    )


def _compute_exp_rom_table():
    return [bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


def _compute_div_rom_table():
    return [bf16_bits_from_float(1.0 / (1.0 + i / 128.0)) for i in range(128)]


_BF16_NEG_MAX = 0xFF7F  # largest-magnitude finite negative bf16 (sign=1, exp=0xFE, mantissa=0x7F).
                         # Matches gold's fp_max nop constant; avoids bf16 +/-inf which the PE fp_max
                         # does not treat as a proper identity.


def _compute_swish_pe_instructions(beta: float):
    neg_beta_bf16 = bf16_bits_from_float(-beta)
    ln2_inv_bf16 = bf16_bits_from_float(1.0 / math.log(2))
    one_bf16 = bf16_bits_from_float(1.0)

    return {
        "mul_a": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("const", neg_beta_bf16)),
        "mul_b": pe_inst_to_bits_with_operands("fp_mul", data0=("const", ln2_inv_bf16), data1=("ext", None)),
        "getffrac": pe_inst_to_bits_with_operands("fp_getffrac", data0=("ext", None)),
        "getfint":  pe_inst_to_bits_with_operands("fp_getfint",  data0=("ext", None)),
        "addiexp": pe_inst_to_bits_with_operands("fp_addiexp", data0=("ext", None), data1=("ext", None)),
        "add": pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("const", one_bf16)),
        "dummy_max_nop": pe_inst_to_bits_with_operands("fp_max", data0=("ext", None), data1=("const", _BF16_NEG_MAX)),
        "getmant": pe_inst_to_bits_with_operands("fp_getmant", data0=("ext", None)),
        "subexp": pe_inst_to_bits_with_operands("fp_subexp", data0=("ext", None), data1=("ext", None)),
        "mul_final": pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None)),
    }


def _interface_type(context, unroll: int):
    if unroll % 2 != 0:
        raise ValueError(f"unroll must be even, got {unroll}")
    half = unroll // 2
    record = {}
    for i in range(half):
        record[_mu_input_self_port(unroll, i)] = context.Array(16, context.BitIn())
        record[_psum0_self_port(i)] = context.Array(16, context.BitIn())
        record[_upper_out_self_port(unroll, i)] = context.Array(16, context.Bit())
    for i in range(half, unroll):
        record[_mu_input_self_port(unroll, i)] = context.Array(16, context.BitIn())
        record[_lower_out_self_port(unroll, i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
    half = unroll // 2

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    top = global_namespace.new_module(
        "add_gelu_pass1_mu_input_fp",
        _interface_type(context, unroll),
    )
    defn = top.new_definition()
    iface = defn.interface

    mu_io_list = []                              # length = unroll
    psum0_io_list = []                           # length = half (upper only)
    upper_out_io_list = []                       # length = half
    lower_out_io_list = []                       # length = half

    outer_add_pe_list = []                       # length = half
    swish_pe_by_role = {k: [] for k in _SWISH_PE_ROLES}  # each length = half
    exp_rom_list = []                            # length = half
    div_rom_list = []                            # length = half
    input_buf_list = []                          # length = half
    lower_nop_pe_list = []                       # length = half

    rom_genargs = make_mem_genargs(context)
    input_buf_genargs = make_mem_genargs(context)
    exp_rom_table = _compute_exp_rom_table()
    div_rom_table = _compute_div_rom_table()

    # ─── Upper half: add + swish pipeline ───────────────────────────────
    for i in range(half):
        mu_io = defn.add_module_instance(_mu_input_io_name(unroll, i), io_module, context.new_values({"mode": "in"}))
        psum0_io = defn.add_module_instance(_psum0_io_name(i), io_module, context.new_values({"mode": "in"}))
        out_io = defn.add_module_instance(_upper_out_io_name(unroll, i), io_module, context.new_values({"mode": "out"}))
        mu_io_list.append(mu_io)
        psum0_io_list.append(psum0_io)
        upper_out_io_list.append(out_io)

        defn.connect(iface.select(_mu_input_self_port(unroll, i)), mu_io.select("in"))
        defn.connect(iface.select(_psum0_self_port(i)), psum0_io.select("in"))
        defn.connect(out_io.select("out"), iface.select(_upper_out_self_port(unroll, i)))

        outer_add_pe = defn.add_module_instance(f"outer_add_pe_{i}", pe_module)
        outer_add_pe_list.append(outer_add_pe)

        lane_pes = {}
        for role in _SWISH_PE_ROLES:
            pe_inst = defn.add_module_instance(f"swish_{role}_pe_{i}", pe_module)
            lane_pes[role] = pe_inst
            swish_pe_by_role[role].append(pe_inst)

        exp_rom = defn.add_generator_instance(
            f"exp_rom_lane{i}",
            mem_gen,
            rom_genargs,
            context.new_values({"config": {}, "init": exp_rom_table, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("depth", json.dumps(len(exp_rom_table))),
            ("init", json.dumps(exp_rom_table)),
            ("is_rom", json.dumps(True)),
            ("mode", json.dumps("sram")),
            ("width", json.dumps(16)),
        ]:
            exp_rom.add_metadata(_key, _val)
        exp_rom_list.append(exp_rom)

        div_rom = defn.add_generator_instance(
            f"div_rom_lane{i}",
            mem_gen,
            rom_genargs,
            context.new_values({"config": {}, "init": div_rom_table, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("depth", json.dumps(len(div_rom_table))),
            ("init", json.dumps(div_rom_table)),
            ("is_rom", json.dumps(True)),
            ("mode", json.dumps("sram")),
            ("width", json.dumps(16)),
        ]:
            div_rom.add_metadata(_key, _val)
        div_rom_list.append(div_rom)

        input_buf_mem = defn.add_generator_instance(
            f"input_buf_lane{i}",
            mem_gen,
            input_buf_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            input_buf_mem.add_metadata(_key, _val)
        input_buf_list.append(input_buf_mem)

        # Outer add: mu_input + psum0 -> input_buf MEM
        defn.connect(mu_io.select("out"), outer_add_pe.select("data0"))
        defn.connect(psum0_io.select("out"), outer_add_pe.select("data1"))
        defn.connect(outer_add_pe.select("O0"), input_buf_mem.select("data_in_0"))

        # MEM dual-read: port 0 -> mul_a (immediate), port 1 -> mul_final (delayed)
        defn.connect(input_buf_mem.select("data_out_0"), lane_pes["mul_a"].select("data0"))
        defn.connect(input_buf_mem.select("data_out_1"), lane_pes["mul_final"].select("data1"))

        # Swish compute graph (matches elementwise_swish_bf16)
        defn.connect(lane_pes["mul_a"].select("O0"), lane_pes["mul_b"].select("data1"))
        defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getffrac"].select("data0"))
        defn.connect(lane_pes["mul_b"].select("O0"), lane_pes["getfint"].select("data0"))

        defn.connect(lane_pes["getffrac"].select("O0"), exp_rom.select("addr_in_0"))
        defn.connect(exp_rom.select("data_out_0"), lane_pes["addiexp"].select("data0"))
        defn.connect(lane_pes["getfint"].select("O0"), lane_pes["addiexp"].select("data1"))

        defn.connect(lane_pes["addiexp"].select("O0"), lane_pes["add"].select("data0"))
        defn.connect(lane_pes["add"].select("O0"), lane_pes["getmant"].select("data0"))
        defn.connect(lane_pes["add"].select("O0"), lane_pes["dummy_max_nop"].select("data0"))

        defn.connect(lane_pes["getmant"].select("O0"), div_rom.select("addr_in_0"))
        defn.connect(div_rom.select("data_out_0"), lane_pes["subexp"].select("data0"))
        defn.connect(lane_pes["dummy_max_nop"].select("O0"), lane_pes["subexp"].select("data1"))

        defn.connect(lane_pes["subexp"].select("O0"), lane_pes["mul_final"].select("data0"))
        defn.connect(lane_pes["mul_final"].select("O0"), out_io.select("in"))

    # ─── Lower half: dummy_max_nop passthrough ──────────────────────────
    for i in range(half, unroll):
        mu_io = defn.add_module_instance(_mu_input_io_name(unroll, i), io_module, context.new_values({"mode": "in"}))
        out_io = defn.add_module_instance(_lower_out_io_name(unroll, i), io_module, context.new_values({"mode": "out"}))
        mu_io_list.append(mu_io)
        lower_out_io_list.append(out_io)

        defn.connect(iface.select(_mu_input_self_port(unroll, i)), mu_io.select("in"))
        defn.connect(out_io.select("out"), iface.select(_lower_out_self_port(unroll, i)))

        nop_pe = defn.add_module_instance(f"lower_nop_pe_{i}", pe_module)
        lower_nop_pe_list.append(nop_pe)

        defn.connect(mu_io.select("out"), nop_pe.select("data0"))
        defn.connect(nop_pe.select("O0"), out_io.select("in"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "mu_io": mu_io_list,
        "psum0_io": psum0_io_list,
        "upper_out_io": upper_out_io_list,
        "lower_out_io": lower_out_io_list,
        "outer_add_pe": outer_add_pe_list,
        "swish_pe": swish_pe_by_role,
        "exp_rom": exp_rom_list,
        "div_rom": div_rom_list,
        "input_buf": input_buf_list,
        "lower_nop_pe": lower_nop_pe_list,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vector_len: int, beta: float):
    if unroll <= 0 or vector_len % unroll != 0:
        raise ValueError(f"vector_len ({vector_len}) must be divisible by unroll ({unroll})")
    extent = vector_len // unroll
    half = unroll // 2

    defn = instances["swish_pe"]["mul_final"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    # Outer add PE instruction
    add_inst_val, add_inst_w = pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))
    for i, pe_inst in enumerate(instances["outer_add_pe"]):
        c = defn.add_generator_instance(
            f"const_inst_outer_add_pe_{i}",
            const_generator,
            context.new_values({"width": add_inst_w}),
            context.new_values({"value": BitVector[add_inst_w](add_inst_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    # Swish PE instructions
    swish_instr = _compute_swish_pe_instructions(beta)
    for role, (inst_val, inst_width) in swish_instr.items():
        for i, pe_inst in enumerate(instances["swish_pe"][role]):
            c = defn.add_generator_instance(
                f"const_inst_swish_{role}_pe_{i}",
                const_generator,
                context.new_values({"width": inst_width}),
                context.new_values({"value": BitVector[inst_width](inst_val)}),
            )
            defn.connect(c.select("out"), pe_inst.select("inst"))

    # Lower-half dummy_max_nop PE: fp_max(ext, bf16_neg_max) — passthrough.
    nop_inst_val, nop_inst_w = pe_inst_to_bits_with_operands(
        "fp_max", data0=("ext", None), data1=("const", _BF16_NEG_MAX)
    )
    for i, pe_inst in enumerate(instances["lower_nop_pe"]):
        c = defn.add_generator_instance(
            f"const_inst_lower_nop_pe_{half + i}",
            const_generator,
            context.new_values({"width": nop_inst_w}),
            context.new_values({"value": BitVector[nop_inst_w](nop_inst_val)}),
        )
        defn.connect(c.select("out"), pe_inst.select("inst"))

    # IO metadata
    glb2out = json.dumps(
        {
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [extent],
            "read_data_starting_addr": [0],
            "read_data_stride": [1],
        }
    )
    in2glb = json.dumps(
        {
            "cycle_starting_addr": [0],
            "cycle_stride": [1],
            "dimensionality": 1,
            "extent": [extent],
            "write_data_starting_addr": [0],
            "write_data_stride": [1],
        }
    )
    for io_in in instances["mu_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_in in instances["psum0_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["upper_out_io"]:
        io_out.add_metadata("in2glb_0", in2glb)
    for io_out in instances["lower_out_io"]:
        io_out.add_metadata("in2glb_0", in2glb)

    # MEM dual-read RV schedule
    lake_rv_cfg = json.dumps({"type": "dual_read", "input_stream_size": extent})
    for input_buf_mem in instances["input_buf"]:
        input_buf_mem.add_metadata("lake_rv_config", lake_rv_cfg)


def build_elementwise_half_add_swish_half_nop_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vector_len: int = DEFAULT_VECTOR_LEN,
    beta: float = DEFAULT_BETA,
):
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vector_len, beta)
    return context, top


def emit_elementwise_half_add_swish_half_nop_bf16_design(
    unroll: int, vector_len: int, output_path: str, beta: float = DEFAULT_BETA
):
    context, top = build_elementwise_half_add_swish_half_nop_bf16_context(unroll, vector_len, beta)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_half_add_swish_half_nop_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate elementwise_half_add_swish_half_nop_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vector-len", type=int, default=DEFAULT_VECTOR_LEN)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_half_add_swish_half_nop_bf16_design(args.unroll, args.vector_len, args.output_path, args.beta)
