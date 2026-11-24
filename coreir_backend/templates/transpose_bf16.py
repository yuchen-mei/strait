"""
Build the Transpose CoreIR graph using pycoreir.

Templated design parameters:
- unroll: number of output lanes (MEMs and output IOs); default 32.
- input_shape: 2D input tensor shape (X, Y); default (128, 768).
- kernel_id: which CGRA kernel slice to configure (0-indexed).
- IO_idx_per_tile: which of the NUM_INPUT_IOS input IOs streams data to MEMs (0-indexed).

Compute graph:
  NUM_INPUT_IOS input IOs (GLB_BANK_WIDTH // CGRA_WORD_WIDTH * GLB_BANK_PER_TILE = 8)
    -> input IO at IO_idx_per_tile broadcasts to `unroll` MEMs
    -> each MEM's data_out_0 -> one output IO
    -> remaining input IOs are dangling

Batched transpose (batch > 1 or in_lanes != unroll): input [batch, X, Y] -> output [batch, Y, X], both in their
natural lane layouts. The input [batch * X rows, Y cols] is spread over in_lanes lanes (lane = y mod in_lanes, word =
(b X + x) Y / in_lanes + y div in_lanes); the output [batch * Y rows, X cols] over the `unroll` output lanes (lane = x mod
unroll, word = (b Y + y) X / unroll + x div unroll). Pass kernel_id reads input lane kernel_id, i.e. the output rows
(b, y) with y = kernel_id mod in_lanes, each as X values in x order (the stream the MEMs expect); in_lanes passes write
the whole output. E.g. Mamba2 decay_to_end [4, 64, 256] -> [4, 256, 64]: batch 4, X 64, Y 256, in_lanes 32.
"""

import json
import os
from pathlib import Path

import coreir

import strait.coreir_backend.utils.headers as headers_pkg
from strait.coreir_backend.utils.coreir_helpers import make_mem_genargs

HEADERS_DIR = list(headers_pkg.__path__)[0]

DEFAULT_UNROLL = 32
DEFAULT_INPUT_SHAPE = (128, 768)

CGRA_WORD_WIDTH = 16
GLB_BANK_WIDTH = 64
GLB_BANK_PER_TILE = 2
NUM_INPUT_IOS = GLB_BANK_WIDTH // CGRA_WORD_WIDTH * GLB_BANK_PER_TILE


def _transpose_bf16_interface_type(context, unroll: int):
    """NUM_INPUT_IOS 16-bit input ports; one 16-bit output port per lane."""
    record = {}
    for i in range(NUM_INPUT_IOS):
        record[f"input_in_{i}"] = context.Array(16, context.BitIn())
    for i in range(unroll):
        record[f"lane_{i}_out"] = context.Array(16, context.Bit())
    return context.Record(record)


def _build_transpose_bf16_graph(
    unroll: int,
    IO_idx_per_tile: int = 0,
    input_name: str = "input",
    output_name: str = "output",
    output_stencil: str = None,
    top_module: str = "transpose_bf16",
):
    """
    Structural construction for transpose compute graph.

    NUM_INPUT_IOS input IOs are created. The one at IO_idx_per_tile broadcasts
    its output to all `unroll` MEMs (data_in_0). The remaining input IOs are
    dangling (connected to the interface but not to any MEMs).
    Each MEM's data_out_0 feeds one output IO.
    output_stencil, if given, is the output IOs' logical name (e.g. "hw_output_stencil" for a Halide app); by default
    it is "output_<output_name>_stencil".
    """
    assert 0 <= IO_idx_per_tile < NUM_INPUT_IOS, f"IO_idx_per_tile {IO_idx_per_tile} must be in range [0, {NUM_INPUT_IOS})."
    if output_stencil is None:
        output_stencil = f"output_{output_name}_stencil"

    context = coreir.Context()
    for path in sorted(Path(HEADERS_DIR).glob("*.json")):
        context.load_header(str(path))
    context.load_library("cgralib")

    global_namespace = context.global_namespace
    io_module = global_namespace.modules["IO"]
    mem_gen = context.get_lib("cgralib").generators["Mem"]

    transpose_module = global_namespace.new_module(
        top_module,
        _transpose_bf16_interface_type(context, unroll),
    )
    transpose_defn = transpose_module.new_definition()
    transpose_iface = transpose_defn.interface

    mem_genargs = make_mem_genargs(context)

    # Create NUM_INPUT_IOS input IOs; only IO_idx_per_tile connects to active MEMs.
    # Dangling IOs are connected to the interface only (no fabric connections).
    io_in_list = []
    for j in range(NUM_INPUT_IOS):
        io_in_name = f"io16in_{input_name}_stencil_clkwrk_{j}_op_hcompute_stencil_{j}_read_0"
        io_in = transpose_defn.add_module_instance(io_in_name, io_module, context.new_values({"mode": "in"}))
        transpose_defn.connect(transpose_iface.select(f"input_in_{j}"), io_in.select("in"))
        io_in_list.append(io_in)

    active_io_in = io_in_list[IO_idx_per_tile]

    mem_list = []
    io_out_list = []

    for i in range(unroll):
        io_out_name = f"io16_{output_stencil}_clkwrk_{i}_op_hcompute_stencil_{i}_write_0"
        io_out = transpose_defn.add_module_instance(io_out_name, io_module, context.new_values({"mode": "out"}))
        io_out_list.append(io_out)

        mem = transpose_defn.add_generator_instance(
            f"mem_lane{i}",
            mem_gen,
            mem_genargs,
            context.new_values({"config": {}, "mode": "lake"}),
        )
        for _key, _val in [
            ("config", json.dumps({})),
            ("is_rom", json.dumps(False)),
            ("mode", json.dumps("lake")),
            ("width", json.dumps(16)),
        ]:
            mem.add_metadata(_key, _val)
        mem_list.append(mem)

        # Active input IO broadcasts to every MEM's write port
        transpose_defn.connect(active_io_in.select("out"), mem.select("data_in_0"))

        # each MEM's read port 0 -> corresponding output IO -> interface
        transpose_defn.connect(mem.select("data_out_0"), io_out.select("in"))
        transpose_defn.connect(io_out.select("out"), transpose_iface.select(f"lane_{i}_out"))

    transpose_module.definition = transpose_defn
    context.set_top(transpose_module)

    instances = {
        "io_in": io_in_list,
        "io_out": io_out_list,
        "mem": mem_list,
    }
    return context, transpose_module, instances


def _relative_stream(start: int, levels, key: str) -> dict:
    """
    A GLB DMA stream dict (relative strides, SKIP_GLB_DMA_STRIDE_ADJUSTMENT) from (extent, absolute word step) levels,
    innermost first. Levels of extent 1 are dropped, and a level whose step is the inner level's span is merged into it.
    """
    merged = []
    for extent, step in levels:
        if extent == 1:
            continue
        if merged and step == merged[-1][1] * merged[-1][0]:
            merged[-1] = (merged[-1][0] * extent, merged[-1][1])
        else:
            merged.append((extent, step))
    if not merged:
        merged = [(1, 1)]
    strides = []
    for i, (_, step) in enumerate(merged):
        strides.append(step - sum(s * (e - 1) for e, s in merged[:i]))
    return {
        "dimensionality": len(merged),
        f"{key}_data_starting_addr": [start],
        f"{key}_data_stride": strides,
        "extent": [e for e, _ in merged],
        "cycle_starting_addr": [0],
        "cycle_stride": [1] * len(merged),
    }


def transpose_streams(unroll: int, input_shape: tuple, kernel_id: int, batch: int = 1, in_lanes: int = None):
    """
    Input read and output write streams of one batched-transpose pass (see the module docstring for the layouts).
    Returns (glb2out, in2glb, mem_Y): mem_Y is the Y the filter_mem_transpose config needs so that its
    X * Y / unroll / unroll words per MEM equal the pass's batch * X * Y / in_lanes values / unroll.
    """
    X, Y = input_shape
    in_lanes = unroll if in_lanes is None else in_lanes
    assert X % unroll == 0, f"X dimension {X} must be divisible by the {unroll} output lanes."
    assert Y % in_lanes == 0, f"Y dimension {Y} must be divisible by the {in_lanes} input lanes."
    assert 0 <= kernel_id < in_lanes, f"kernel_id {kernel_id} must be in range [0, {in_lanes})."
    g_in, g_out = Y // in_lanes, X // unroll
    # Input lane kernel_id: for b, for y group, for x: word (b X + x) g_in + y group
    glb2out = _relative_stream(0, [(X, g_in), (g_in, 1), (batch, X * g_in)], "read")
    # Output lane i keeps x = i mod unroll: for b, for y group, for x group: word (b Y + y group * in_lanes + kernel_id)
    # g_out + x group
    in2glb = _relative_stream(kernel_id * g_out, [(g_out, 1), (g_in, in_lanes * g_out), (batch, Y * g_out)], "write")
    return glb2out, in2glb, batch * Y * unroll // in_lanes


def _configure_transpose_bf16(instances, unroll: int, input_shape: tuple, kernel_id: int, batch: int = 1,
                              in_lanes: int = None):
    """
    Configure IO DMA metadata for one CGRA kernel slice.

    input_shape is (X, Y); the transposed output shape is (Y, X).
    kernel_id selects which slice of the output GLB address space this kernel writes to.
    With batch > 1 or in_lanes != unroll: the batched transpose of the module docstring.
    """
    if batch != 1 or in_lanes not in (None, unroll):
        glb2out, in2glb, mem_Y = transpose_streams(unroll, input_shape, kernel_id, batch, in_lanes)
        for io_in in instances["io_in"]:
            io_in.add_metadata("glb2out_0", json.dumps(glb2out))
        for io_out in instances["io_out"]:
            io_out.add_metadata("in2glb_0", json.dumps(in2glb))
        word_width_per_bank_row = GLB_BANK_WIDTH // CGRA_WORD_WIDTH
        for i, mem in enumerate(instances["mem"]):
            mem.add_metadata(
                "lake_rv_config",
                json.dumps(
                    {
                        "type": "filter_mem_transpose",
                        "X": input_shape[0],
                        "Y": mem_Y,
                        "output_glb_bank_idx": i // word_width_per_bank_row,
                        "lane_idx_within_bank": i % word_width_per_bank_row,
                        "unroll": unroll,
                    }
                ),
            )
        return

    X, Y = input_shape
    K = kernel_id
    assert X % unroll == 0, f"X dimension {X} must be divisible by unroll factor {unroll}."
    assert Y % unroll == 0, f"Y dimension {Y} must be divisible by unroll factor {unroll}."

    word_width_per_bank_row = GLB_BANK_WIDTH // CGRA_WORD_WIDTH

    glb2out = json.dumps(
        {
            "dimensionality": 2,
            "read_data_starting_addr": [0],
            "read_data_stride": [Y // unroll, -(Y // unroll * (X - 1)) + 1],
            "extent": [X, Y // unroll],
            "cycle_starting_addr": [0],
            "cycle_stride": [1, 1],
        }
    )
    for io_in in instances["io_in"]:
        io_in.add_metadata("glb2out_0", glb2out)

    in2glb = json.dumps(
        {
            "dimensionality": 2,
            "write_data_starting_addr": [K * X // unroll],
            "write_data_stride": [1, X + 1 - X // unroll],
            "extent": [X // unroll, Y // unroll],
            "cycle_starting_addr": [0],
            "cycle_stride": [1, 1],
        }
    )
    for io_out in instances["io_out"]:
        io_out.add_metadata("in2glb_0", in2glb)

    for i, mem in enumerate(instances["mem"]):
        output_glb_bank_idx = i // word_width_per_bank_row
        lane_idx_within_bank = i % word_width_per_bank_row
        mem.add_metadata(
            "lake_rv_config",
            json.dumps(
                {
                    "type": "filter_mem_transpose",
                    "X": X,
                    "Y": Y,
                    "output_glb_bank_idx": output_glb_bank_idx,
                    "lane_idx_within_bank": lane_idx_within_bank,
                    "unroll": unroll,
                }
            ),
        )


def build_transpose_bf16_context(
    unroll: int = DEFAULT_UNROLL,
    input_shape: tuple = DEFAULT_INPUT_SHAPE,
    kernel_id: int = 0,
    IO_idx_per_tile: int = 0,
    input_name: str = "input",
    output_name: str = "output",
    batch: int = 1,
    in_lanes: int = None,
    output_stencil: str = None,
    top_module: str = "transpose_bf16",
):
    """
    Build the transpose BF16 design.

    Args:
        unroll: Number of output lanes (MEMs and output IOs).
        input_shape: 2D input tensor shape (X, Y). Output shape is (Y, X).
        kernel_id: Which CGRA kernel slice to configure (0-indexed).
        IO_idx_per_tile: Which of the NUM_INPUT_IOS input IOs streams data to MEMs (0-indexed).
        input_name: Tensor name embedded in input IO instance names.
        output_name: Tensor name embedded in output IO instance names.
        batch: Number of independent (X, Y) matrices (batched transpose; see the module docstring).
        in_lanes: Lanes the input is spread over (default unroll); one pass per input lane.
        output_stencil: Logical output IO name (default "output_<output_name>_stencil").
        top_module: Name of the top module.

    Returns:
        (context, transpose_module)
    """
    context, transpose_module, instances = _build_transpose_bf16_graph(
        unroll, IO_idx_per_tile, input_name, output_name, output_stencil, top_module
    )
    _configure_transpose_bf16(instances, unroll, input_shape, kernel_id, batch, in_lanes)
    return context, transpose_module


def emit_transpose_bf16_coreir_json(
    kernel: dict,
    output_path: str,
    unroll: int = DEFAULT_UNROLL,
    kernel_id: int = 0,
):
    kernel_inputs = kernel.get("inputs", {})
    kernel_outputs = kernel.get("outputs", {})
    assert len(kernel_inputs) == 1, "Transpose BF16 kernel must have 1 input."

    input_name = next(iter(kernel_inputs))
    output_name = next(iter(kernel_outputs.values())).get("node")

    for _, input_info in kernel_inputs.items():
        shape = tuple(input_info.get("shape"))
        assert len(shape) == 2, f"Transpose BF16 expects a 2D input shape, got {shape}."

    IO_idx_per_tile = kernel_id % NUM_INPUT_IOS
    context, transpose_top = build_transpose_bf16_context(unroll, shape, kernel_id, IO_idx_per_tile, input_name, output_name)
    transpose_top.save_to_file(os.path.join(output_path, "design_top.json"))
    print(f"[INFO] Wrote transpose_bf16_coreir_json to {os.path.join(output_path, 'design_top.json')}")


def emit_transpose_bf16_design(
    unroll: int,
    input_shape: tuple,
    kernel_id: int,
    output_path: str,
    batch: int = 1,
    in_lanes: int = None,
    input_name: str = "input_host",
    output_stencil: str = "hw_output_stencil",
    top_module: str = "transpose_fp",
):
    """
    Write design_top.json of one (batched) transpose pass for a Halide app: input IOs input_host_stencil (the active one
    is kernel_id's lane within its GLB tile), output IOs hw_output_stencil.
    """
    IO_idx_per_tile = kernel_id % NUM_INPUT_IOS
    _, transpose_top = build_transpose_bf16_context(
        unroll, input_shape, kernel_id, IO_idx_per_tile, input_name, batch=batch, in_lanes=in_lanes,
        output_stencil=output_stencil, top_module=top_module,
    )
    transpose_top.save_to_file(os.path.join(output_path, "design_top.json"))
    print(f"[INFO] Wrote transpose_bf16 design (batch {batch}, input {tuple(input_shape)}, pass {kernel_id}) to "
          f"{os.path.join(output_path, 'design_top.json')}")


if __name__ == "__main__":
    _, transpose_module = build_transpose_bf16_context()
    transpose_module.save_to_file("transpose_bf16_coreir.json")
    print("[INFO] Wrote transpose_bf16_coreir.json")
