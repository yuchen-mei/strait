"""
Build the bf16 sum-of-squares + rsqrt + broadcast elementwise mul-add CoreIR
graph using pycoreir.

Input is x - mean from pass1; output is (x - mean) * (sqrt(N) * gamma) / sqrt(sum_x((x-mean)^2)) + beta.

Per row:
    1. per-lane pre_square_pe: fp_mul(input, input) -> x^2
    2. balanced fp_add tree over unroll lanes + filter_mem + 2 accum ponds/PEs
       + final_reduce_pe                                  ->  sum_x(x^2)
    3. scalar pipeline (sqrt(N)*gamma / sqrt(sum)):
         getmant_log + ln_rom                   ->  log(mantissa)
         cnvexp2f + scalar_mul_a(ln2)           ->  exponent * ln(2)
         scalar_fp_add                          ->  log(sum)
         scalar_mul_b(const=0.5)                ->  0.5 * log(sum)
         scalar_mul_c(const=1/ln2)              ->  0.5 * log2(sum) = log2(sqrt(sum))
         fp_getffrac + exp_rom + fp_getfint + addiexp
                                                ->  sqrt(sum)
         dummy_max_nop_pe (fp_max(x, -max)) for path balance between addiexp
                                                    and subexp.data1
         getmant_div + div_rom + subexp         ->  1/sqrt(sum)
         scalar_mul_d(const=sqrt(vec_width)*gamma)
                                                ->  rstd = sqrt(N)*gamma / sqrt(sum)
    4. broadcast MEM holds the per-row rstd.
    5. per-lane elementwise fp_mul(input, rstd) -> elementwise fp_add(..., const=beta)
       -> output IO.

Parameters:
    unroll: glb_i (tree width / parallel lanes, must be power of 2).
    vec_length: vec_width (row length; used for sqrt(vec_width)*gamma const).
    num_vecs: vec_height (number of rows per invocation).
    gamma, beta: affine parameters. Defaults match the Halide hardcoded
                 gamma=1.2, beta=-0.35.
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
DEFAULT_GAMMA = 1.2
DEFAULT_BETA = -0.35
DEFAULT_TOP_MODULE = "layer_norm_pass2_fp"
DEFAULT_HAS_BETA = True

# bf16 for largest-magnitude finite negative. Matches gold dummy_max_nop const;
# avoids bf16 +/-inf which fp_max does not treat as a proper identity.
_BF16_NEG_MAX = 0xFF7F


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


def _ln_rom_table():
    """128-entry bf16 ROM: ln(1 + i/128) for i in [0, 128)."""
    # mantissa lookup for log(mant). Input mantissa is in [1, 2), encoded as
    # 1 + i/128 where i is the 7 bits of mantissa.
    return [bf16_bits_from_float(math.log(1.0 + i / 128.0)) for i in range(128)]


def _exp_rom_table():
    """256-entry bf16 ROM: 2^(i/128). Matches layout used by swish template."""
    return [bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(128)] + [
        bf16_bits_from_float(2.0 ** (i / 128.0)) for i in range(-128, 0)
    ]


def _div_rom_table():
    """128-entry bf16 reciprocal ROM: div_rom[i] = 1 / (1 + i/128)."""
    return [bf16_bits_from_float(1.0 / (1.0 + i / 128.0)) for i in range(128)]


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int, top_module: str = DEFAULT_TOP_MODULE, has_beta: bool = DEFAULT_HAS_BETA):
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

    top = global_namespace.new_module(top_module, _interface_type(context, unroll))
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
    pre_square_pe_list = []
    elementwise_mul_pe_list = []
    elementwise_add_pe_list = []
    output_dummy_max_nop_pe_list = []

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
            mem_gen, mem_genargs,
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

        # Per-lane x^2 (both PE operands come from tile_input data_out_0).
        pre_square_pe = defn.add_module_instance(f"pre_square_pe_{i}", pe_module)
        pre_square_pe_list.append(pre_square_pe)

        # Per-lane elementwise: fp_mul(input, rstd) -> (optional) fp_add(..., beta_const).
        ew_mul_pe = defn.add_module_instance(f"elementwise_mul_pe_{i}", pe_module)
        elementwise_mul_pe_list.append(ew_mul_pe)
        if has_beta:
            ew_add_pe = defn.add_module_instance(f"elementwise_add_pe_{i}", pe_module)
            elementwise_add_pe_list.append(ew_add_pe)

        # input IO -> tile_input_mem.data_in_0
        defn.connect(input_io.select("out"), tile_input_mem.select("data_in_0"))
        # tile_input.data_out_0 -> pre_square_pe.data0 and data1 (x * x)
        defn.connect(tile_input_mem.select("data_out_0"), pre_square_pe.select("data0"))
        defn.connect(tile_input_mem.select("data_out_0"), pre_square_pe.select("data1"))
        # tile_input.data_out_1 -> elementwise_mul_pe.data0 (delayed for scaling)
        defn.connect(tile_input_mem.select("data_out_1"), ew_mul_pe.select("data0"))
        if has_beta:
            # ew_mul.O0 -> ew_add.data0 ; ew_add.O0 -> output IO
            defn.connect(ew_mul_pe.select("O0"), ew_add_pe.select("data0"))
            defn.connect(ew_add_pe.select("O0"), output_io.select("in"))
        else:
            # No beta: insert per-lane output_dummy_max_nop_pe between ew_mul.O0
            # and output IO to match gold's timing padding. Named by lane index
            # `i` (= output lane under identity topology); the remap aligns
            # gold's counter-indexed dummies to these by tracing gold's dummy
            # -> output_io connection.
            out_dummy_pe = defn.add_module_instance(
                f"{top_module}_output_dummy_max_nop_pe_{i}", pe_module
            )
            output_dummy_max_nop_pe_list.append(out_dummy_pe)
            defn.connect(ew_mul_pe.select("O0"), out_dummy_pe.select("data0"))
            defn.connect(out_dummy_pe.select("O0"), output_io.select("in"))

    # Reduction tree: stage 1 consumes pre_square_pe.O0 from every lane.
    tree_stages = int(math.log2(unroll))
    tree_pes_by_stage = []
    prev_outs = [pe.select("O0") for pe in pre_square_pe_list]
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

    # filter_mem: write once, dual-read alternating (same as pass3/pass1).
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

    def make_const_clk_en(const_name_suffix):
        return defn.add_module_instance(
            f"{top_module}_{const_name_suffix}_clk_en_const",
            corebit_const, context.new_values({"value": True}),
        )

    # Gold wires clk_en with per-instance local consts (not one shared fanout),
    # so P&R can place each const adjacent to its consumer — producing short,
    # local wires. Mirror gold's grouping:
    #   - one const per tile_input_lane_i (32 total)
    #   - one const for broadcast_mem
    #   - one SHARED const for filter_mem + accum_pond_0 + accum_pond_1 (these
    #     3 MEMs/Ponds sit close in placement; short 3-way local fanout is fine)
    # Large shared fanouts (observed on initial port with a single shared const
    # feeding ALL 36 MEMs/Ponds) inflate routing-register count and runtime
    # cycle count.
    shared_filter_accum_const = make_const_clk_en("filter_accum")
    defn.connect(shared_filter_accum_const.select("out"), filter_mem.select("clk_en"))
    defn.connect(shared_filter_accum_const.select("out"), accum_pond_0.select("clk_en"))
    defn.connect(shared_filter_accum_const.select("out"), accum_pond_1.select("clk_en"))
    for i, tim in enumerate(tile_input_mem_list):
        c = make_const_clk_en(f"tile_input_lane_{i}")
        defn.connect(c.select("out"), tim.select("clk_en"))

    defn.connect(filter_mem.select("data_out_0"), accum_pe_0.select("data1"))
    defn.connect(filter_mem.select("data_out_1"), accum_pe_1.select("data1"))

    defn.connect(accum_pond_0.select("data_out_pond_0"), accum_pe_0.select("data0"))
    defn.connect(accum_pe_0.select("O0"), accum_pond_0.select("data_in_pond_0"))
    defn.connect(accum_pond_0.select("data_out_pond_1"), final_reduce_pe.select("data0"))

    defn.connect(accum_pond_1.select("data_out_pond_0"), accum_pe_1.select("data0"))
    defn.connect(accum_pe_1.select("O0"), accum_pond_1.select("data_in_pond_0"))
    defn.connect(accum_pond_1.select("data_out_pond_1"), final_reduce_pe.select("data1"))

    # Scalar pipeline PEs
    getmant_log_pe = defn.add_module_instance(f"{top_module}_getmant_log_pe", pe_module)
    cnvexp2f_pe = defn.add_module_instance(f"{top_module}_cnvexp2f_pe", pe_module)
    scalar_mul_a_pe = defn.add_module_instance(f"{top_module}_scalar_mul_a_pe", pe_module)
    scalar_fp_add_pe = defn.add_module_instance(f"{top_module}_scalar_fp_add_pe", pe_module)
    scalar_mul_b_pe = defn.add_module_instance(f"{top_module}_scalar_mul_b_pe", pe_module)
    scalar_mul_c_pe = defn.add_module_instance(f"{top_module}_scalar_mul_c_pe", pe_module)
    getffrac_pe = defn.add_module_instance(f"{top_module}_getffrac_pe", pe_module)
    getfint_pe = defn.add_module_instance(f"{top_module}_getfint_pe", pe_module)
    addiexp_pe = defn.add_module_instance(f"{top_module}_addiexp_pe", pe_module)
    dummy_max_nop_pe = defn.add_module_instance(f"{top_module}_dummy_max_nop_pe", pe_module)
    getmant_div_pe = defn.add_module_instance(f"{top_module}_getmant_div_pe", pe_module)
    subexp_pe = defn.add_module_instance(f"{top_module}_subexp_pe", pe_module)
    scalar_mul_d_pe = defn.add_module_instance(f"{top_module}_scalar_mul_d_pe", pe_module)

    # ROMs
    ln_rom_table = _ln_rom_table()
    exp_rom_table = _exp_rom_table()
    div_rom_table = _div_rom_table()

    ln_rom = defn.add_generator_instance(
        f"{top_module}_ln_rom",
        mem_gen, mem_genargs,
        context.new_values({"config": {}, "init": ln_rom_table, "mode": "lake"}),
    )
    for _k, _v in [
        ("config", json.dumps({})),
        ("depth", json.dumps(len(ln_rom_table))),
        ("init", json.dumps(ln_rom_table)),
        ("is_rom", json.dumps(True)),
        ("mode", json.dumps("sram")),
        ("width", json.dumps(16)),
    ]:
        ln_rom.add_metadata(_k, _v)

    exp_rom = defn.add_generator_instance(
        f"{top_module}_exp_rom",
        mem_gen, mem_genargs,
        context.new_values({"config": {}, "init": exp_rom_table, "mode": "lake"}),
    )
    for _k, _v in [
        ("config", json.dumps({})),
        ("depth", json.dumps(len(exp_rom_table))),
        ("init", json.dumps(exp_rom_table)),
        ("is_rom", json.dumps(True)),
        ("mode", json.dumps("sram")),
        ("width", json.dumps(16)),
    ]:
        exp_rom.add_metadata(_k, _v)

    div_rom = defn.add_generator_instance(
        f"{top_module}_div_rom",
        mem_gen, mem_genargs,
        context.new_values({"config": {}, "init": div_rom_table, "mode": "lake"}),
    )
    for _k, _v in [
        ("config", json.dumps({})),
        ("depth", json.dumps(len(div_rom_table))),
        ("init", json.dumps(div_rom_table)),
        ("is_rom", json.dumps(True)),
        ("mode", json.dumps("sram")),
        ("width", json.dumps(16)),
    ]:
        div_rom.add_metadata(_k, _v)

    # Broadcast MEM (holds per-row rstd = sqrt(N)*gamma / sqrt(sum)).
    broadcast_mem = defn.add_generator_instance(
        f"{top_module}_broadcast_mem",
        mem_gen, mem_genargs,
        context.new_values({"config": {}, "mode": "lake"}),
    )
    for _k, _v in [
        ("config", json.dumps({})),
        ("is_rom", json.dumps(False)),
        ("mode", json.dumps("lake")),
        ("width", json.dumps(16)),
    ]:
        broadcast_mem.add_metadata(_k, _v)
    broadcast_clk_en_const = make_const_clk_en("broadcast_mem")
    defn.connect(broadcast_clk_en_const.select("out"), broadcast_mem.select("clk_en"))

    # Scalar pipeline wiring
    # final_reduce.O0 = sum(x^2) fans out to both getmant_log and cnvexp2f
    defn.connect(final_reduce_pe.select("O0"), getmant_log_pe.select("data0"))
    defn.connect(final_reduce_pe.select("O0"), cnvexp2f_pe.select("data0"))

    # log(sum) = ln_rom[mantissa(sum)] + cnvexp2f(sum) * ln(2)
    defn.connect(getmant_log_pe.select("O0"), ln_rom.select("addr_in_0"))
    defn.connect(cnvexp2f_pe.select("O0"), scalar_mul_a_pe.select("data0"))
    defn.connect(scalar_mul_a_pe.select("O0"), scalar_fp_add_pe.select("data0"))
    defn.connect(ln_rom.select("data_out_0"), scalar_fp_add_pe.select("data1"))

    # 0.5 * log(sum) * (1/ln2) = log2(sqrt(sum))
    defn.connect(scalar_fp_add_pe.select("O0"), scalar_mul_b_pe.select("data0"))
    defn.connect(scalar_mul_b_pe.select("O0"), scalar_mul_c_pe.select("data1"))

    # getffrac + getfint; exp_rom[frac]; addiexp reconstructs sqrt(sum)
    defn.connect(scalar_mul_c_pe.select("O0"), getffrac_pe.select("data0"))
    defn.connect(scalar_mul_c_pe.select("O0"), getfint_pe.select("data0"))
    defn.connect(getffrac_pe.select("O0"), exp_rom.select("addr_in_0"))
    defn.connect(exp_rom.select("data_out_0"), addiexp_pe.select("data0"))
    defn.connect(getfint_pe.select("O0"), addiexp_pe.select("data1"))

    # addiexp.O0 = sqrt(sum)
    # Fan out to (a) dummy_max_nop -> subexp.data1, and (b) getmant_div -> div_rom -> subexp.data0.
    defn.connect(addiexp_pe.select("O0"), dummy_max_nop_pe.select("data0"))
    defn.connect(addiexp_pe.select("O0"), getmant_div_pe.select("data0"))

    defn.connect(getmant_div_pe.select("O0"), div_rom.select("addr_in_0"))
    defn.connect(div_rom.select("data_out_0"), subexp_pe.select("data0"))
    defn.connect(dummy_max_nop_pe.select("O0"), subexp_pe.select("data1"))

    # subexp.O0 = 1/sqrt(sum); scale by sqrt(N)*gamma; write to broadcast MEM.
    defn.connect(subexp_pe.select("O0"), scalar_mul_d_pe.select("data0"))
    defn.connect(scalar_mul_d_pe.select("O0"), broadcast_mem.select("data_in_0"))

    # broadcast MEM fans out to every lane's elementwise_mul.data1.
    for ew_mul_pe in elementwise_mul_pe_list:
        defn.connect(broadcast_mem.select("data_out_0"), ew_mul_pe.select("data1"))

    top.definition = defn
    context.set_top(top)

    instances = {
        "input_io": input_io_list,
        "output_io": output_io_list,
        "tile_input_mem": tile_input_mem_list,
        "pre_square_pe": pre_square_pe_list,
        "elementwise_mul_pe": elementwise_mul_pe_list,
        "elementwise_add_pe": elementwise_add_pe_list,
        "output_dummy_max_nop_pe": output_dummy_max_nop_pe_list,
        "tree_pes_by_stage": tree_pes_by_stage,
        "filter_mem": filter_mem,
        "accum_pond_0": accum_pond_0,
        "accum_pond_1": accum_pond_1,
        "accum_pe_0": accum_pe_0,
        "accum_pe_1": accum_pe_1,
        "final_reduce_pe": final_reduce_pe,
        "getmant_log_pe": getmant_log_pe,
        "cnvexp2f_pe": cnvexp2f_pe,
        "scalar_mul_a_pe": scalar_mul_a_pe,
        "scalar_fp_add_pe": scalar_fp_add_pe,
        "scalar_mul_b_pe": scalar_mul_b_pe,
        "scalar_mul_c_pe": scalar_mul_c_pe,
        "getffrac_pe": getffrac_pe,
        "getfint_pe": getfint_pe,
        "addiexp_pe": addiexp_pe,
        "dummy_max_nop_pe": dummy_max_nop_pe,
        "getmant_div_pe": getmant_div_pe,
        "subexp_pe": subexp_pe,
        "scalar_mul_d_pe": scalar_mul_d_pe,
        "ln_rom": ln_rom,
        "exp_rom": exp_rom,
        "div_rom": div_rom,
        "broadcast_mem": broadcast_mem,
    }
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int,
               gamma: float, beta: float,
               top_module: str = DEFAULT_TOP_MODULE, has_beta: bool = DEFAULT_HAS_BETA):
    if vec_length % unroll != 0:
        raise ValueError(f"vec_length ({vec_length}) must be divisible by unroll ({unroll})")
    num_partial_reduction = vec_length // unroll
    if num_partial_reduction % 2 != 0:
        raise ValueError(f"num_partial_reduction ({num_partial_reduction}) must be even for dual-pond accum")
    per_lane_extent = vec_length * num_vecs // unroll

    defn = instances["filter_mem"].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    # Instructions
    fp_add_val, fp_add_w = pe_inst_to_bits_with_operands(
        "fp_add", data0=("ext", None), data1=("ext", None)
    )
    fp_mul_val, fp_mul_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("ext", None)
    )
    # Scalar const-bearing instructions
    ln2_bf16 = bf16_bits_from_float(math.log(2.0))
    half_bf16 = bf16_bits_from_float(0.5)
    inv_ln2_bf16 = bf16_bits_from_float(1.0 / math.log(2.0))
    sqrt_n_gamma_bf16 = bf16_bits_from_float(math.sqrt(float(vec_length)) * gamma)
    beta_bf16 = bf16_bits_from_float(beta)

    scalar_mul_a_val, scalar_mul_a_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("const", ln2_bf16)
    )
    scalar_mul_b_val, scalar_mul_b_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("const", half_bf16)
    )
    scalar_mul_c_val, scalar_mul_c_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("const", inv_ln2_bf16), data1=("ext", None)
    )
    scalar_mul_d_val, scalar_mul_d_w = pe_inst_to_bits_with_operands(
        "fp_mul", data0=("ext", None), data1=("const", sqrt_n_gamma_bf16)
    )
    elementwise_add_val, elementwise_add_w = pe_inst_to_bits_with_operands(
        "fp_add", data0=("ext", None), data1=("const", beta_bf16)
    )
    getmant_val, getmant_w = pe_inst_to_bits_with_operands(
        "fp_getmant", data0=("ext", None), data1=("const", 0)
    )
    cnvexp2f_val, cnvexp2f_w = pe_inst_to_bits_with_operands(
        "fp_cnvexp2f", data0=("ext", None)
    )
    getffrac_val, getffrac_w = pe_inst_to_bits_with_operands(
        "fp_getffrac", data0=("ext", None)
    )
    getfint_val, getfint_w = pe_inst_to_bits_with_operands(
        "fp_getfint", data0=("ext", None)
    )
    addiexp_val, addiexp_w = pe_inst_to_bits_with_operands(
        "fp_addiexp", data0=("ext", None), data1=("ext", None)
    )
    subexp_val, subexp_w = pe_inst_to_bits_with_operands(
        "fp_subexp", data0=("ext", None), data1=("ext", None)
    )
    dummy_max_val, dummy_max_w = pe_inst_to_bits_with_operands(
        "fp_max", data0=("ext", None), data1=("const", _BF16_NEG_MAX)
    )

    pe_inst_list = []
    for i, pe in enumerate(instances["pre_square_pe"]):
        pe_inst_list.append((pe, f"pre_square_pe_{i}", fp_mul_val, fp_mul_w))
    for stage_idx, stage in enumerate(instances["tree_pes_by_stage"]):
        for j, pe in enumerate(stage):
            pe_inst_list.append((pe, f"tree_stage{stage_idx + 1}_pe_{j}", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["accum_pe_0"], f"{top_module}_accum_pe_0", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["accum_pe_1"], f"{top_module}_accum_pe_1", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["final_reduce_pe"], f"{top_module}_final_reduce_pe", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["getmant_log_pe"], f"{top_module}_getmant_log_pe", getmant_val, getmant_w))
    pe_inst_list.append((instances["cnvexp2f_pe"], f"{top_module}_cnvexp2f_pe", cnvexp2f_val, cnvexp2f_w))
    pe_inst_list.append((instances["scalar_mul_a_pe"], f"{top_module}_scalar_mul_a_pe", scalar_mul_a_val, scalar_mul_a_w))
    pe_inst_list.append((instances["scalar_fp_add_pe"], f"{top_module}_scalar_fp_add_pe", fp_add_val, fp_add_w))
    pe_inst_list.append((instances["scalar_mul_b_pe"], f"{top_module}_scalar_mul_b_pe", scalar_mul_b_val, scalar_mul_b_w))
    pe_inst_list.append((instances["scalar_mul_c_pe"], f"{top_module}_scalar_mul_c_pe", scalar_mul_c_val, scalar_mul_c_w))
    pe_inst_list.append((instances["getffrac_pe"], f"{top_module}_getffrac_pe", getffrac_val, getffrac_w))
    pe_inst_list.append((instances["getfint_pe"], f"{top_module}_getfint_pe", getfint_val, getfint_w))
    pe_inst_list.append((instances["addiexp_pe"], f"{top_module}_addiexp_pe", addiexp_val, addiexp_w))
    pe_inst_list.append((instances["dummy_max_nop_pe"], f"{top_module}_dummy_max_nop_pe", dummy_max_val, dummy_max_w))
    pe_inst_list.append((instances["getmant_div_pe"], f"{top_module}_getmant_div_pe", getmant_val, getmant_w))
    pe_inst_list.append((instances["subexp_pe"], f"{top_module}_subexp_pe", subexp_val, subexp_w))
    pe_inst_list.append((instances["scalar_mul_d_pe"], f"{top_module}_scalar_mul_d_pe", scalar_mul_d_val, scalar_mul_d_w))
    for i, pe in enumerate(instances["elementwise_mul_pe"]):
        pe_inst_list.append((pe, f"elementwise_mul_pe_{i}", fp_mul_val, fp_mul_w))
    if has_beta:
        for i, pe in enumerate(instances["elementwise_add_pe"]):
            pe_inst_list.append((pe, f"elementwise_add_pe_{i}", elementwise_add_val, elementwise_add_w))
    else:
        for i, pe in enumerate(instances["output_dummy_max_nop_pe"]):
            pe_inst_list.append((pe, f"{top_module}_output_dummy_max_nop_pe_{i}", dummy_max_val, dummy_max_w))

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

    # pass2's log/exp scalar pipeline needs a larger raw_scalar on the broadcast
    # MEM than pass3 does (the log/exp expansion plus the addiexp/subexp gap is
    # deeper). hack_rv_mem_pond_bitstream.py picks raw_scalar=6 for pass2-style
    # (layer_norm_pass2_fp, rms_norm_pass1_fp); match that here.
    broadcast_cfg = json.dumps({
        "type": "get_broadcast_mem",
        "input_stream_size": num_vecs,
        "replicate_factor": num_partial_reduction,
        "raw_scalar": 6,
    })
    instances["broadcast_mem"].add_metadata("lake_rv_config", broadcast_cfg)


def _write_pe_fifos_bypass_config(output_path: str, top_module: str = DEFAULT_TOP_MODULE):
    bypass_cfg = {
        f"{top_module}_accum_pe_0": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
        f"{top_module}_accum_pe_1": {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1},
    }
    bypass_path = os.path.join(output_path, "PE_fifos_bypass_config.json")
    with open(bypass_path, "w") as f:
        json.dump(bypass_cfg, f, indent=2)
    print(f"[INFO] Wrote PE_fifos_bypass_config.json to {bypass_path}")


def build_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    vec_length: int = DEFAULT_VEC_LENGTH,
    num_vecs: int = DEFAULT_NUM_VECS,
    gamma: float = DEFAULT_GAMMA,
    beta: float = DEFAULT_BETA,
    top_module: str = DEFAULT_TOP_MODULE,
    has_beta: bool = DEFAULT_HAS_BETA,
):
    context, top, instances = _build_graph(unroll, top_module=top_module, has_beta=has_beta)
    _configure(context, instances, unroll, vec_length, num_vecs, gamma, beta,
               top_module=top_module, has_beta=has_beta)
    return context, top


def emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design(
    unroll: int, vec_length: int, num_vecs: int, output_path: str,
    gamma: float = DEFAULT_GAMMA, beta: float = DEFAULT_BETA,
    top_module: str = DEFAULT_TOP_MODULE, has_beta: bool = DEFAULT_HAS_BETA,
):
    context, top = build_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_context(
        unroll, vec_length, num_vecs, gamma, beta, top_module=top_module, has_beta=has_beta,
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote {top_module} design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, top_module=top_module)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate layer_norm_pass2 / rms_norm_pass1 strait design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec-length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num-vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--top-module", type=str, default=DEFAULT_TOP_MODULE)
    parser.add_argument("--no-beta", dest="has_beta", action="store_false", default=DEFAULT_HAS_BETA)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design(
        args.unroll, args.vec_length, args.num_vecs, args.output_path,
        gamma=args.gamma, beta=args.beta,
        top_module=args.top_module, has_beta=args.has_beta,
    )
