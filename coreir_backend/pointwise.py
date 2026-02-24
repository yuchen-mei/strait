"""
Build the pointwise CoreIR graph using pycoreir
"""

import os
import json
import coreir
from hwtypes import BitVector
import strait.utils.headers as headers_pkg

HEADERS_DIR = list(headers_pkg.__path__)[0]

# Instance names used in both construction and configuration
IO_IN_NAME = "io16in_hw_input_stencil_op_hcompute_hw_input_global_wrapper_stencil_read_0"
IO_OUT_NAME = "io16_hw_output_stencil_op_hcompute_hw_output_stencil_write_0"
PE_NAME = "op_hcompute_mult_stencil$inner_compute$mul_i3072_i444"

# Names for configuration-only const instances
C0_NAME = "op_hcompute_mult_stencil$inner_compute$c0"
C1_NAME = "op_hcompute_mult_stencil$inner_compute$c1"

# Default configuration values
PE_INST_VAL = 0x020000000041ABFC8004C
IO_EXTENT = 4096


def _pointwise_interface_type(c):
    """
    Define interface type for top-level pointwise module.
    Minimal interface includes only the input/output ports connected to IO instances.
    """
    return c.Record({
        "hw_input_stencil_op_hcompute_hw_input_global_wrapper_stencil_read_0": c.Array(16, c.BitIn()),
        "hw_output_stencil_op_hcompute_hw_output_stencil_write_0": c.Array(16, c.Bit()),
    })


def _build_pointwise_graph():
    """
    Pure structural construction: create modules, instances, and connections.

    Returns (context, pointwise_module, instances_dict).
    """
    c = coreir.Context()
    # Load headers
    for name in ("lassen_header.json", "io_header.json", "bit_io_header.json"):
        c.load_header(os.path.join(HEADERS_DIR, name))

    g = c.global_namespace
    PE = g.modules["PE"]
    IO = g.modules["IO"]

    pointwise = g.new_module("pointwise", _pointwise_interface_type(c))
    defn = pointwise.new_definition()

    # IOs and PE instances
    io_in = defn.add_module_instance(IO_IN_NAME, IO, c.new_values({"mode": "in"}))
    io_out = defn.add_module_instance(IO_OUT_NAME, IO, c.new_values({"mode": "out"}))
    pe_inst = defn.add_module_instance(PE_NAME, PE)

    # Pure data-path connections only.
    self_sel = defn.interface
    defn.connect(pe_inst.select("O0"), io_out.select("in"))
    defn.connect(io_out.select("out"), self_sel.select("hw_output_stencil_op_hcompute_hw_output_stencil_write_0"))
    defn.connect(self_sel.select("hw_input_stencil_op_hcompute_hw_input_global_wrapper_stencil_read_0"), io_in.select("in"))
    defn.connect(io_in.select("out"), pe_inst.select("data0"))

    pointwise.definition = defn
    c.set_top(pointwise)

    instances = {
        "io_in": io_in,
        "io_out": io_out,
        "pe": pe_inst,
    }
    return c, pointwise, instances


def _configure_io_metadata(io_in, io_out, extent: int = IO_EXTENT):
    """
    Attach metadata to IO instances.
    """
    # Input: GLB to CGRA
    io_in.add_metadata("glb2out_0", json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "read_data_starting_addr": [0],
        "read_data_stride": [1],
    }))

    # Output: CGRA to GLB
    io_out.add_metadata("in2glb_0", json.dumps({
        "cycle_starting_addr": [0],
        "cycle_stride": [1],
        "dimensionality": 1,
        "extent": [extent],
        "write_data_starting_addr": [0],
        "write_data_stride": [1],
    }))


def _configure_pointwise(c, instances, pe_inst_val: int = PE_INST_VAL, io_extent: int = IO_EXTENT):
    """
    Configuration step: set PE instruction constant and IO metadata.
    """
    io_in = instances["io_in"]
    io_out = instances["io_out"]
    pe_inst = instances["pe"]

    # Add and connect const for PE.inst
    defn = pe_inst.module_def
    const_gen = c.get_lib("coreir").generators["const"]
    c0 = defn.add_generator_instance(
        C0_NAME,
        const_gen,
        c.new_values({"width": 84}),
        c.new_values({"value": BitVector[84](pe_inst_val)}),
    )
    defn.connect(c0.select("out"), pe_inst.select("inst"))
    instances["c0"] = c0

    # Add and connect const for PE.clk_en
    corebit_const = c.get_lib("corebit").modules["const"]
    c1 = defn.add_module_instance(C1_NAME, corebit_const, c.new_values({"value": True}))
    defn.connect(c1.select("out"), pe_inst.select("clk_en"))
    instances["c1"] = c1

    # Attach IO metadata.
    _configure_io_metadata(io_in, io_out, extent=io_extent)


def build_pointwise_context():
    """
    Build the pointwise design by first constructing the bare CoreIR graph
    (instances + connections), then applying configuration (PE const and IO
    metadata).
    """
    c, pointwise, instances = _build_pointwise_graph()
    _configure_pointwise(c, instances)
    return c, pointwise


if __name__ == "__main__":
    c, top = build_pointwise_context()
    out_path = os.path.join(os.path.dirname(__file__), "pointwise_top.json")
    top.save_to_file(out_path)
    print(f"Wrote {out_path}")
