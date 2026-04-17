"""
Build the NOP (no-operation / passthrough) CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of lanes (= glb_i = glb_o from halide_gen_args).
- tensor_size: total tensor size; each IO has extent tensor_size // unroll.

For each unrolled lane i:
  input IO.out -> output IO.in  (direct passthrough)

Architecture per lane:
  - input IO  -> output IO  (direct wire, no PEs or MEMs)
"""

import os
import json
from pathlib import Path

import coreir

import strait.coreir_backend.utils.headers as headers_pkg

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_TENSOR_SIZE = 50176


def _nop_bf16_interface_type(context, unroll: int):
    """
    Define the top-level interface: per lane, one input IO and one output IO.
    """
    record = {}
    for i in range(unroll):
        record[f"mu_hw_input_stencil_clkwrk_{i}_op_hcompute_mu_hw_input_glb_stencil_{i}_read_0"] = context.Array(16, context.BitIn())
        record[f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_nop_bf16_graph(unroll: int):
    """
    Structural construction: create modules, instances, and connections
    for the NOP passthrough graph with `unroll` lanes.

    Returns (context, nop_module, instances_dict).
    """
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    io_module = global_namespace.modules["IO"]

    nop_module = global_namespace.new_module(
        "zircon_nop",
        _nop_bf16_interface_type(context, unroll),
    )
    nop_defn = nop_module.new_definition()
    nop_iface = nop_defn.interface

    io_input_list = []
    io_output_list = []

    for i in range(unroll):
        # ── IO and self-port names ────────────────────────────────────
        input_io_name = f"io16in_mu_hw_input_stencil_clkwrk_{i}_op_hcompute_mu_hw_input_glb_stencil_{i}_read_0"
        output_io_name = f"io16_hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"

        input_self = f"mu_hw_input_stencil_clkwrk_{i}_op_hcompute_mu_hw_input_glb_stencil_{i}_read_0"
        output_self = f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil_{i}_write_0"

        # ── Create IO instances ──────────────────────────────────────
        input_io = nop_defn.add_module_instance(input_io_name, io_module, context.new_values({"mode": "in"}))
        output_io = nop_defn.add_module_instance(output_io_name, io_module, context.new_values({"mode": "out"}))

        io_input_list.append(input_io)
        io_output_list.append(output_io)

        # ══════════════════════════════════════════════════════════════
        # Connections
        # ══════════════════════════════════════════════════════════════

        # --- Self <-> IO ---
        nop_defn.connect(nop_iface.select(input_self), input_io.select("in"))
        nop_defn.connect(output_io.select("out"), nop_iface.select(output_self))

        # --- Input IO -> Output IO (direct passthrough) ---
        nop_defn.connect(input_io.select("out"), output_io.select("in"))

    nop_module.definition = nop_defn
    context.set_top(nop_module)

    instances = {
        "io_input": io_input_list,
        "io_output": io_output_list,
    }
    return context, nop_module, instances


def _configure_nop_bf16(instances, unroll: int, tensor_size: int):
    """
    Configuration step: set IO metadata with extents.
    """
    extent = tensor_size // unroll

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

    for io_in in instances["io_input"]:
        io_in.add_metadata("glb2out_0", glb2out)
    for io_out in instances["io_output"]:
        io_out.add_metadata("in2glb_0", in2glb)


def build_nop_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    tensor_size: int = DEFAULT_TENSOR_SIZE,
):
    """
    Build the NOP passthrough design with templated parameters.

    Args:
        unroll: Number of parallel lanes (= glb_i = glb_o).
        tensor_size: Total tensor size.

    Returns:
        (context, nop_module)
    """
    context, nop_module, instances = _build_nop_bf16_graph(unroll)
    _configure_nop_bf16(instances, unroll, tensor_size)
    return context, nop_module


def emit_nop_bf16_design(unroll: int, tensor_size: int, output_path: str):
    """Build and write the zircon_nop design_top.json."""
    context, nop_module = build_nop_bf16_context(unroll, tensor_size)
    out_file = os.path.join(output_path, "design_top.json")
    nop_module.save_to_file(out_file)
    print(f"[INFO] Wrote nop_bf16 design_top.json to {out_file}")
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate NOP CoreIR design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--tensor-size", type=int, default=DEFAULT_TENSOR_SIZE)
    parser.add_argument("--output-path", type=str, default=".")
    args = parser.parse_args()
    emit_nop_bf16_design(args.unroll, args.tensor_size, args.output_path)
