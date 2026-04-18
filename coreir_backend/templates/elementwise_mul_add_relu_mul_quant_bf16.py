"""
Build the MU * dequant_scale + GLB -> ReLU -> * quant_scale -> e8m0_quant -> bit8_pack CoreIR
graph using pycoreir (zircon_deq_ResReLU_quant_fp).

Halide pipeline per input channel c:
    relu_output(c)     = max(mu_hw_input(c) * dequant_scale + hw_residual_input(c), 0)
    unpacked_result(c) = e8m0_quant(relu_output(c) * quant_scale, bf16_bits(127))
    result(c_pack)     = bit8_pack(unpacked_result(2*c_pack + 1), unpacked_result(2*c_pack))

Unroll model: the C axis is unrolled by `unroll` at the input (2 input lanes per packed
output lane). Output unroll = unroll / 2.

Per input lane c (5 PEs):
  mu_input_io[c].out       -> dequant_pe[c].data0 ; const dequant_scale -> dequant_pe[c].data1
  residual_input_io[c].out -> add_pe[c].data0     ; dequant_pe[c].O0    -> add_pe[c].data1
  add_pe[c].O0             -> max_pe[c].data0     ; const bf16(0.0)     -> max_pe[c].data1
  max_pe[c].O0             -> quant_mul_pe[c].data0 ; const quant_scale -> quant_mul_pe[c].data1
  quant_mul_pe[c].O0       -> quant_pe[c].data0   ; const bf16(127)     -> quant_pe[c].data1

Per packed lane i (c_even=2i, c_odd=2i+1):
  quant_pe[c_odd].O0  -> pack_pe[i].data0   (first arg -> upper byte)
  quant_pe[c_even].O0 -> pack_pe[i].data1   (second arg -> lower byte)
  pack_pe[i].O0       -> output_io[i].in
"""

import os
import json
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
import strait.coreir_backend.utils.build_pe_inst as build_pe_inst

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_TENSOR_SIZE = 50176
DEFAULT_DEQUANT_SCALE = 0.5
DEFAULT_QUANT_SCALE = 0.5
# Halide uses reinterpret(bf16, uint16_t(127)); use raw bits 127 as the bf16 operand.
E8M0_BIAS_BITS_BF16 = 127

# MU input: "_mu_" substring triggers add_mu_prefix_to_io in strait.
MU_INPUT_BASE = "mu_hw_input_stencil"
RES_INPUT_BASE = "hw_residual_input_stencil"
OUTPUT_BASE = "hw_output_stencil"


def _io_names(base: str, idx: int, rw: str):
    self_port = f"{base}_clkwrk_{idx}_op_hcompute_{base}_{idx}_{rw}_0"
    io_prefix = "io16in" if rw == "read" else "io16"
    return self_port, f"{io_prefix}_{self_port}"


def _interface_type(context, unroll: int):
    record = {}
    for c in range(unroll):
        mu_self, _ = _io_names(MU_INPUT_BASE, c, "read")
        res_self, _ = _io_names(RES_INPUT_BASE, c, "read")
        record[mu_self] = context.Array(16, context.BitIn())
        record[res_self] = context.Array(16, context.BitIn())
    for cp in range(unroll // 2):
        out_self, _ = _io_names(OUTPUT_BASE, cp, "write")
        record[out_self] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
    if unroll <= 0 or unroll % 2 != 0:
        raise ValueError(f"unroll ({unroll}) must be a positive even number")

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_ns = context.global_namespace
    pe_module = global_ns.modules["PE"]
    io_module = global_ns.modules["IO"]

    top = global_ns.new_module("elementwise_mul_add_relu_mul_quant_bf16", _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    mu_input_ios = []
    res_input_ios = []
    output_ios = []
    dequant_pes = []
    add_pes = []
    max_pes = []
    quant_mul_pes = []
    quant_pes = []
    pack_pes = []

    for c in range(unroll):
        mu_self, mu_io_name = _io_names(MU_INPUT_BASE, c, "read")
        res_self, res_io_name = _io_names(RES_INPUT_BASE, c, "read")

        mu_io = defn.add_module_instance(mu_io_name, io_module, context.new_values({"mode": "in"}))
        res_io = defn.add_module_instance(res_io_name, io_module, context.new_values({"mode": "in"}))
        dequant_pe = defn.add_module_instance(f"dequant_pe_{c}", pe_module)
        add_pe = defn.add_module_instance(f"add_pe_{c}", pe_module)
        max_pe = defn.add_module_instance(f"max_pe_{c}", pe_module)
        quant_mul_pe = defn.add_module_instance(f"quant_mul_pe_{c}", pe_module)
        quant_pe = defn.add_module_instance(f"quant_pe_{c}", pe_module)

        mu_input_ios.append(mu_io)
        res_input_ios.append(res_io)
        dequant_pes.append(dequant_pe)
        add_pes.append(add_pe)
        max_pes.append(max_pe)
        quant_mul_pes.append(quant_mul_pe)
        quant_pes.append(quant_pe)

        defn.connect(iface.select(mu_self), mu_io.select("in"))
        defn.connect(iface.select(res_self), res_io.select("in"))

        # MU * dequant_scale
        defn.connect(mu_io.select("out"), dequant_pe.select("data0"))
        # residual + dequant_mul result (Halide convention: residual -> data0, mul -> data1)
        defn.connect(res_io.select("out"), add_pe.select("data0"))
        defn.connect(dequant_pe.select("O0"), add_pe.select("data1"))
        # ReLU
        defn.connect(add_pe.select("O0"), max_pe.select("data0"))
        # relu * quant_scale
        defn.connect(max_pe.select("O0"), quant_mul_pe.select("data0"))
        # e8m0_quant
        defn.connect(quant_mul_pe.select("O0"), quant_pe.select("data0"))

    for cp in range(unroll // 2):
        c_even = 2 * cp
        c_odd = 2 * cp + 1
        out_self, out_io_name = _io_names(OUTPUT_BASE, cp, "write")
        output_io = defn.add_module_instance(out_io_name, io_module, context.new_values({"mode": "out"}))
        pack_pe = defn.add_module_instance(f"pack_pe_{cp}", pe_module)

        output_ios.append(output_io)
        pack_pes.append(pack_pe)

        defn.connect(output_io.select("out"), iface.select(out_self))
        defn.connect(quant_pes[c_odd].select("O0"), pack_pe.select("data0"))
        defn.connect(quant_pes[c_even].select("O0"), pack_pe.select("data1"))
        defn.connect(pack_pe.select("O0"), output_io.select("in"))

    top.definition = defn
    context.set_top(top)

    return context, top, {
        "mu_input_io": mu_input_ios,
        "res_input_io": res_input_ios,
        "output_io": output_ios,
        "dequant_pe": dequant_pes,
        "add_pe": add_pes,
        "max_pe": max_pes,
        "quant_mul_pe": quant_mul_pes,
        "quant_pe": quant_pes,
        "pack_pe": pack_pes,
    }


def _configure(context, instances, unroll: int, tensor_size: int, dequant_scale: float, quant_scale: float):
    if tensor_size % unroll != 0:
        raise ValueError(f"tensor_size ({tensor_size}) must be divisible by unroll ({unroll})")
    extent = tensor_size // unroll

    defn = instances["dequant_pe"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    dequant_inst, dequant_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "fp_mul",
        data0=("ext", None),
        data1=("const", build_pe_inst.bf16_bits_from_float(dequant_scale)),
    )
    add_inst, add_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "fp_add",
        data0=("ext", None),
        data1=("ext", None),
    )
    # max(x, 0.0) implements ReLU.
    max_inst, max_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "fp_max",
        data0=("ext", None),
        data1=("const", build_pe_inst.bf16_bits_from_float(0.0)),
    )
    quant_mul_inst, quant_mul_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "fp_mul",
        data0=("ext", None),
        data1=("const", build_pe_inst.bf16_bits_from_float(quant_scale)),
    )
    quant_inst, quant_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "e8m0_quant",
        data0=("ext", None),
        data1=("const", E8M0_BIAS_BITS_BF16),
    )
    pack_inst, pack_w = build_pe_inst.pe_inst_to_bits_with_operands(
        "bit8_pack",
        data0=("ext", None),
        data1=("ext", None),
    )

    def _wire_const(name_prefix, pe_list, inst_val, inst_w):
        for i, pe in enumerate(pe_list):
            c = defn.add_generator_instance(
                f"{name_prefix}_{i}",
                const_generator,
                context.new_values({"width": inst_w}),
                context.new_values({"value": BitVector[inst_w](inst_val)}),
            )
            defn.connect(c.select("out"), pe.select("inst"))

    _wire_const("const_inst_dequant_pe", instances["dequant_pe"], dequant_inst, dequant_w)
    _wire_const("const_inst_add_pe", instances["add_pe"], add_inst, add_w)
    _wire_const("const_inst_max_pe", instances["max_pe"], max_inst, max_w)
    _wire_const("const_inst_quant_mul_pe", instances["quant_mul_pe"], quant_mul_inst, quant_mul_w)
    _wire_const("const_inst_quant_pe", instances["quant_pe"], quant_inst, quant_w)
    _wire_const("const_inst_pack_pe", instances["pack_pe"], pack_inst, pack_w)

    glb2out = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    })
    in2glb = json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    })

    for io_in in instances["mu_input_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_in in instances["res_input_io"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["output_io"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_elementwise_mul_add_relu_mul_quant_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
    dequant_scale: float = DEFAULT_DEQUANT_SCALE,
    quant_scale: float = DEFAULT_QUANT_SCALE,
):
    """
    Build the zircon_deq_ResReLU_quant_fp strait design.

    Args:
        unroll: number of input lanes (must be even); output lanes = unroll / 2.
        tensor_size: total input tensor size; each input IO extent = tensor_size / unroll.
        dequant_scale: bf16 mul constant for the first (dequantize) mul stage.
        quant_scale: bf16 mul constant for the second (quantize) mul stage before e8m0_quant.
    """
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, tensor_size, dequant_scale, quant_scale)
    return context, top


def emit_elementwise_mul_add_relu_mul_quant_bf16_design(
    unroll: int,
    tensor_size: int,
    output_path: str,
    dequant_scale: float = DEFAULT_DEQUANT_SCALE,
    quant_scale: float = DEFAULT_QUANT_SCALE,
):
    context, top = build_elementwise_mul_add_relu_mul_quant_bf16_context(
        unroll, tensor_size, dequant_scale, quant_scale
    )
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_add_relu_mul_quant_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate elementwise_mul_add_relu_mul_quant_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--tensor-size", type=int, default=DEFAULT_TENSOR_SIZE)
    parser.add_argument("--dequant-scale", type=float, default=DEFAULT_DEQUANT_SCALE)
    parser.add_argument("--quant-scale", type=float, default=DEFAULT_QUANT_SCALE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_add_relu_mul_quant_bf16_design(
        args.unroll, args.tensor_size, args.output_path, args.dequant_scale, args.quant_scale
    )
