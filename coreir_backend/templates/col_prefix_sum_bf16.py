"""Build the BF16 cumsum along dim -2 (down the rows) CoreIR graph for Zircon.

    output[b, r, c] = input[b, 0, c] + input[b, 1, c] + ... + input[b, r, c]

The tensor is num_mats matrices of scan_rows rows and vec_length columns,
stored row-major as an activation: each cycle the ``unroll`` lanes hold
``unroll`` consecutive columns of one row (4 GLB tiles for 32 lanes). Lane j
therefore sees column m * unroll + j for m = 0 .. G - 1 (G = vec_length /
unroll column groups), then the next row, then the next matrix. Each output
depends only on the same column one row up, which reached the same lane G
tokens earlier, so the lanes need no data from each other and the scan needs
no transpose:

    y[r, m] = y[r - 1, m] + x[r, m]

Per lane, one fp_add PE and the Pond of its tile form the recurrence. The
Pond (``get_col_cumsum_pond``) is a ring of running sums; read port 0 gives
the PE the sum of the same column one row up, the PE writes the new sum back,
and read port 1 sends the new sum to the output IO. The loop stays inside
the tile, so its latency does not depend on placement; with G >= 2 column
groups a lane takes one value per cycle. The clear port zeroes the Pond
before each matrix.

The PE output feeds only its own Pond, so there is no fan-out whose stall
the in-tile Pond write would miss, and the PE's output FIFO is bypassed (as
in the accumulator templates) to keep the loop short. The Pond's write side
cannot stall its writer; read port 0 holds the PE back once the sums it would
overwrite are unread, so a slow output stalls the loop instead of losing data.

Templated design parameters:
- unroll: lanes (32 for an activation in 4 GLB tiles).
- vec_length: columns; a multiple of unroll.
- num_vecs: rows of all matrices together (num_mats * scan_rows).
- scan_rows: rows of one matrix (the scanned dimension).
"""

import json
import os
from pathlib import Path

import coreir
from hwtypes import BitVector

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.build_pe_inst import pe_inst_to_bits_with_operands

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_VEC_LENGTH = 128
DEFAULT_NUM_VECS = 1024
DEFAULT_SCAN_ROWS = 128
TOP_MODULE = "col_prefix_sum_fp"
POND_DEPTH = 32
MAX_EXTENT = 1024  # Zircon MEM and Pond extent counters are 11-bit signed


def _stencil_suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def _input_self_port(i: int) -> str:
    return f"input_host_stencil_clkwrk_{i}_op_hcompute_input_glb_stencil{_stencil_suffix(i)}_read_0"


def _output_self_port(i: int) -> str:
    return f"hw_output_stencil_clkwrk_{i}_op_hcompute_hw_output_stencil{_stencil_suffix(i)}_write_0"


def _interface_type(context, unroll: int):
    record = {}
    for i in range(unroll):
        record[_input_self_port(i)] = context.Array(16, context.BitIn())
        record[_output_self_port(i)] = context.Array(16, context.Bit())
    return context.Record(record)


def _shape(unroll: int, vec_length: int, num_vecs: int, scan_rows: int):
    """Return (column groups per row, matrices) after checking the limits."""
    if vec_length % unroll:
        raise ValueError(f"[ERROR] vec_length {vec_length} must be a multiple of unroll {unroll}")
    groups = vec_length // unroll
    if num_vecs % scan_rows:
        raise ValueError(f"[ERROR] num_vecs {num_vecs} must be a multiple of scan_rows {scan_rows}")
    mats = num_vecs // scan_rows
    if not 1 <= groups < POND_DEPTH - 2:
        raise ValueError(f"[ERROR] {groups} column groups per lane do not fit the 32-word Pond ring")
    if not 2 <= groups * scan_rows <= MAX_EXTENT:
        raise ValueError(f"[ERROR] {groups} groups x {scan_rows} rows exceed {MAX_EXTENT} tokens per matrix")
    if not 2 <= mats <= MAX_EXTENT:
        raise ValueError(f"[ERROR] {mats} matrices must lie in [2, {MAX_EXTENT}]")
    return groups, mats


def _sum_pe_name(i: int) -> str:
    return f"lane{i}_sum_pe"


def _build_graph(unroll: int):
    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    namespace = context.global_namespace
    pe_module = namespace.modules["PE"]
    io_module = namespace.modules["IO"]
    pond_gen = context.get_lib("cgralib").generators["Pond"]
    const_gen = context.get_lib("coreir").generators["const"]

    top = namespace.new_module(TOP_MODULE, _interface_type(context, unroll))
    defn = top.new_definition()
    iface = defn.interface

    corebit_const = context.get_namespace("corebit").modules["const"]
    clk_en = defn.add_module_instance(
        f"{TOP_MODULE}_clk_en_const", corebit_const, context.new_values({"value": True})
    )
    instances = {"input_io": [], "output_io": [], "sum_pe": [], "sum_pond": []}
    for i in range(unroll):
        input_io = defn.add_module_instance(f"io16in_{_input_self_port(i)}", io_module, context.new_values({"mode": "in"}))
        output_io = defn.add_module_instance(f"io16_{_output_self_port(i)}", io_module, context.new_values({"mode": "out"}))
        defn.connect(iface.select(_input_self_port(i)), input_io.select("in"))
        defn.connect(output_io.select("out"), iface.select(_output_self_port(i)))

        pe = defn.add_module_instance(_sum_pe_name(i), pe_module)
        pond = defn.add_generator_instance(
            # The mapper's Pond packing ILP matches names as substrings: no
            # Pond or PE name may be part of another lane's name.
            f"lane{i}_col_sum_pond",
            pond_gen,
            context.new_values({"ID": "", "has_stencil_valid": True, "num_inputs": 2, "num_outputs": 2, "width": 16}),
            context.new_values({"config": {}, "mode": "pond"}),
        )
        for key, value in [("config", {}), ("is_rom", False), ("mode", "pond"), ("width", 16)]:
            pond.add_metadata(key, json.dumps(value))
        defn.connect(clk_en.select("out"), pond.select("clk_en"))

        # y = y_up + x: the sum one row up from Pond port r0, the input from the IO
        defn.connect(pond.select("data_out_pond_0"), pe.select("data0"))
        defn.connect(input_io.select("out"), pe.select("data1"))
        defn.connect(pe.select("O0"), pond.select("data_in_pond_0"))
        defn.connect(pond.select("data_out_pond_1"), output_io.select("in"))

        instances["input_io"].append(input_io)
        instances["output_io"].append(output_io)
        instances["sum_pe"].append(pe)
        instances["sum_pond"].append(pond)

    top.definition = defn
    context.set_top(top)
    return context, top, instances


def _configure(context, instances, unroll: int, vec_length: int, num_vecs: int, scan_rows: int):
    groups, mats = _shape(unroll, vec_length, num_vecs, scan_rows)
    defn = instances["output_io"][0].module_def
    const_gen = context.get_lib("coreir").generators["const"]

    val, width = pe_inst_to_bits_with_operands("fp_add", data0=("ext", None), data1=("ext", None))
    for pe in instances["sum_pe"]:
        c = defn.add_generator_instance(
            f"const_inst_{pe.name}", const_gen,
            context.new_values({"width": width}), context.new_values({"value": BitVector[width](val)}),
        )
        defn.connect(c.select("out"), pe.select("inst"))

    tokens = groups * num_vecs  # per lane

    def stream(key):
        return json.dumps(
            {
                "cycle_starting_addr": [0],
                "cycle_stride": [1],
                "dimensionality": 1,
                "extent": [tokens],
                f"{key}_data_starting_addr": [0],
                f"{key}_data_stride": [1],
            }
        )

    for io in instances["input_io"]:
        io.add_metadata("glb2out_0", stream("read"))
    for io in instances["output_io"]:
        io.add_metadata("in2glb_0", stream("write"))

    pond_cfg = json.dumps({"type": "get_col_cumsum_pond", "num_groups": groups, "num_rows": scan_rows, "num_mats": mats})
    for pond in instances["sum_pond"]:
        pond.add_metadata("lake_rv_config", pond_cfg)


def _write_pe_fifos_bypass_config(output_path: str, unroll: int):
    """Bypass the output FIFO of every sum PE: its only sink is the Pond of its tile."""
    cfg = {_sum_pe_name(i): {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1} for i in range(unroll)}
    with open(os.path.join(output_path, "PE_fifos_bypass_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def emit_col_prefix_sum_bf16_design(unroll: int, vec_length: int, num_vecs: int, scan_rows: int, output_path: str):
    """Write design_top.json and PE_fifos_bypass_config.json for one dim -2 cumsum kernel."""
    context, top, instances = _build_graph(unroll)
    _configure(context, instances, unroll, vec_length, num_vecs, scan_rows)
    out_file = os.path.join(output_path, "design_top.json")
    top.save_to_file(out_file)
    print(f"[INFO] Wrote col_prefix_sum_bf16 design_top.json to {out_file}")
    _write_pe_fifos_bypass_config(output_path, unroll)
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the dim -2 cumsum design_top.json")
    parser.add_argument("--unroll", type=int, default=DEFAULT_UNROLL)
    parser.add_argument("--vec_length", type=int, default=DEFAULT_VEC_LENGTH)
    parser.add_argument("--num_vecs", type=int, default=DEFAULT_NUM_VECS)
    parser.add_argument("--scan_rows", type=int, default=DEFAULT_SCAN_ROWS)
    parser.add_argument("--output_path", type=str, default=".")
    args = parser.parse_args()
    emit_col_prefix_sum_bf16_design(args.unroll, args.vec_length, args.num_vecs, args.scan_rows, args.output_path)
