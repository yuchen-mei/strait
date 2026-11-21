"""
Build the MU-output x broadcast-scalar CoreIR graph using pycoreir (a GEMM epilogue).

    out[row, col] = mu[row, col] * scale[group(col), row],   group(col) = col // group_cols

For Mamba2 C@states this is Y_off[l, h*64 + p] * exp(a_cum)[h, l]: every 32-column MU output block lies
inside one group (head), so each MU output word needs ONE scale value, shared by all lanes.

Per lane i (one lane per MU output column of a 32-wide MU word):
  mu_hw_input IO.out          -> mul_pe_i.data0
  hw_broadcast_input IO.out   -> mul_pe_i.data1   (a single IO, fanned out to every lane)
  mul_pe_i (fp_mul) O0        -> hw_output IO.in

The broadcast IO is one unpacked GLB stream holding the run's scales group-major, address = g * R + row
(R = x1 * x2 rows per run). Its address generator replays them in MU output order, x1 (innermost) -> k1 -> x2 -> k2,
with k1 split into (blocks per group, groups per k1 loop), one value per handshake:
  extent = [x1, bpg, k1 / bpg, x2, k2]
  stride = [1, -(x1 - 1), R - (x1 - 1), 1 - (k1 / bpg - 1) * R, 1]     (relative strides)
The MU-tiling address hack must leave this IO alone (NO_MU_TILING_ADDR_HACK in parse_design_meta.py). The template
does not set packing: the app's GLB bank config makes the broadcast IO unpacked (HW_BROADCAST_INPUT_STENCIL_PACKED=0)
and the outputs E64-packed, as for stable_softmax_pass2's vec_max.

Logical IO names (between io16[in]_ and _clkwrk; they match Halide's design_meta_halide.json):
  mu_hw_input_stencil (gets the MU_ prefix via add_mu_prefix_to_io), hw_broadcast_input_stencil, hw_output_stencil.

Templated design parameters:
- unroll: lanes (MU output columns per word, 32 in E64 multi-bank mode).
- tensor_size: output elements of one run; each MU/output IO streams tensor_size // unroll words.
- x1, x2, k1, k2: the run's MU loop bounds (L1 rows, L2 rows, L1 and L2 32-column blocks).
- group_cols: output columns sharing one scale (64 = one Mamba2 head).
"""

import os
import json
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands

HEADERS_DIR = list(headers_pkg.__path__)[0]

MU_OC0 = 32
_MU_INPUT = "mu_hw_input_stencil"
_BROADCAST = "hw_broadcast_input_stencil"
_OUTPUT = "hw_output_stencil"


def _lane_port_names(i: int):
    """Return (mu_self, output_self, mu_io, output_io) for lane i."""
    mu_self = f"{_MU_INPUT}_clkwrk_{i}_op_hcompute_{_MU_INPUT}_{i}_read_0"
    output_self = f"{_OUTPUT}_clkwrk_{i}_op_hcompute_{_OUTPUT}_{i}_write_0"
    return mu_self, output_self, f"io16in_{mu_self}", f"io16_{output_self}"


# The single shared broadcast IO (clkwrk_0, so the logical name is preserved)
_BROADCAST_SELF = f"{_BROADCAST}_clkwrk_0_op_hcompute_{_BROADCAST}_0_read_0"
_BROADCAST_IO = f"io16in_{_BROADCAST_SELF}"


def broadcast_address_pattern(x1: int, x2: int, k1: int, k2: int, group_cols: int):
    """The broadcast IO's GLB read pattern (innermost first, relative strides) and its scale count."""
    if group_cols % MU_OC0 != 0:
        raise ValueError(f"group_cols ({group_cols}) must be a multiple of {MU_OC0}")
    bpg = group_cols // MU_OC0              # 32-column MU blocks per group
    if k1 % bpg != 0:
        raise ValueError(f"k1 ({k1}) must be a multiple of the blocks per group ({bpg})")
    gpk = k1 // bpg                         # groups per k1 loop
    rows = x1 * x2
    extent = [x1, bpg, gpk, x2, k2]
    stride = [1, -(x1 - 1), rows - (x1 - 1), 1 - (gpk - 1) * rows, 1]
    if min(extent) < 2:
        raise ValueError(f"every extent must be >= 2 (the GLB DMA encodes extent - 2): {extent}")
    num_scales = gpk * k2 * rows
    return extent, stride, num_scales


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        mu_self, output_self, _, _ = _lane_port_names(i)
        record[mu_self] = context.Array(16, context.BitIn())
        if i == 0:
            record[_BROADCAST_SELF] = context.Array(16, context.BitIn())
        record[output_self] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_graph(unroll: int):
    """IO instances and one fp_mul PE per lane. Returns (context, module, instances)."""
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    pe_module = global_namespace.modules["PE"]
    io_module = global_namespace.modules["IO"]

    module = global_namespace.new_module("elementwise_mul_broadcast_bf16", _interface_type(context, unroll))
    defn = module.new_definition()
    iface = defn.interface

    broadcast_io = defn.add_module_instance(_BROADCAST_IO, io_module, context.new_values({"mode": "in"}))
    defn.connect(iface.select(_BROADCAST_SELF), broadcast_io.select("in"))

    mu_io_list, output_io_list, pe_list = [], [], []
    for i in range(unroll):
        mu_self, output_self, mu_io_name, output_io_name = _lane_port_names(i)
        mu_io = defn.add_module_instance(mu_io_name, io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(output_io_name, io_module, context.new_values({"mode": "out"}))
        pe_inst = defn.add_module_instance(f"mul_pe_{i}", pe_module)

        defn.connect(iface.select(mu_self), mu_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(output_self))
        defn.connect(mu_io.select("out"), pe_inst.select("data0"))
        defn.connect(broadcast_io.select("out"), pe_inst.select("data1"))
        defn.connect(pe_inst.select("O0"), output_io.select("in"))

        mu_io_list.append(mu_io)
        output_io_list.append(output_io)
        pe_list.append(pe_inst)

    module.definition = defn
    context.set_top(module)
    instances = {"io_mu": mu_io_list, "io_broadcast": broadcast_io, "io_output": output_io_list, "pe": pe_list}
    return context, module, instances


def _configure(context, instances, unroll: int, tensor_size: int, x1: int, x2: int, k1: int, k2: int, group_cols: int):
    """PE instruction constants and IO address metadata."""
    extent = tensor_size // unroll
    if x1 * k1 * x2 * k2 != extent:
        raise ValueError(f"MU loops x1*k1*x2*k2 = {x1 * k1 * x2 * k2} must equal the per-lane extent {extent}")
    defn = instances["pe"][0].module_def
    const_generator = context.get_lib("coreir").generators["const"]

    fp_mul_val, fp_mul_w = pe_inst_to_bits_with_operands("fp_mul", data0=("ext", None), data1=("ext", None))
    for i, pe_inst in enumerate(instances["pe"]):
        pe_inst_const = defn.add_generator_instance(
            f"const_inst_mul_pe_{i}",
            const_generator,
            context.new_values({"width": fp_mul_w}),
            context.new_values({"value": BitVector[fp_mul_w](fp_mul_val)}),
        )
        defn.connect(pe_inst_const.select("out"), pe_inst.select("inst"))

    # MU inputs: not read from the GLB (ignored); outputs: rewritten by the MU-tiling address hack
    linear_glb2out = json.dumps({
        "cycle_starting_addr": [0], "cycle_stride": [1], "dimensionality": 1,
        "extent": [extent], "read_data_starting_addr": [0], "read_data_stride": [1],
    })
    linear_in2glb = json.dumps({
        "cycle_starting_addr": [0], "cycle_stride": [1], "dimensionality": 1,
        "extent": [extent], "write_data_starting_addr": [0], "write_data_stride": [1],
    })
    b_extent, b_stride, _ = broadcast_address_pattern(x1, x2, k1, k2, group_cols)
    broadcast_glb2out = json.dumps({
        "cycle_starting_addr": [0], "cycle_stride": [1] * len(b_extent), "dimensionality": len(b_extent),
        "extent": b_extent, "read_data_starting_addr": [0], "read_data_stride": b_stride,
    })

    for io_in in instances["io_mu"]:
        io_in.add_metadata("glb2out_0", linear_glb2out)
    instances["io_broadcast"].add_metadata("glb2out_0", broadcast_glb2out)
    for io_out in instances["io_output"]:
        io_out.add_metadata("in2glb_0", linear_in2glb)


def build_elementwise_mul_broadcast_bf16_context(unroll: int, tensor_size: int, x1: int, x2: int, k1: int, k2: int,
                                                 group_cols: int):
    context, module, instances = _build_graph(unroll)
    _configure(context, instances, unroll, tensor_size, x1, x2, k1, k2, group_cols)
    return context, module


def emit_elementwise_mul_broadcast_bf16_design(unroll: int, tensor_size: int, x1: int, x2: int, k1: int, k2: int,
                                               group_cols: int, output_path: str):
    """Build and write the elementwise_mul_broadcast_bf16 design_top.json."""
    context, module = build_elementwise_mul_broadcast_bf16_context(unroll, tensor_size, x1, x2, k1, k2, group_cols)
    out_file = os.path.join(output_path, "design_top.json")
    module.save_to_file(out_file)
    print(f"[INFO] Wrote elementwise_mul_broadcast_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate elementwise_mul_broadcast_bf16 CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--tensor-size", type=int, default=262144)
    parser.add_argument("--x1", type=int, default=128)
    parser.add_argument("--x2", type=int, default=2)
    parser.add_argument("--k1", type=int, default=8)
    parser.add_argument("--k2", type=int, default=4)
    parser.add_argument("--group-cols", type=int, default=64)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_elementwise_mul_broadcast_bf16_design(args.unroll, args.tensor_size, args.x1, args.x2, args.k1, args.k2,
                                               args.group_cols, args.output_path)
