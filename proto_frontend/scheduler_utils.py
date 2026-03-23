"""
Scheduler utility helpers
"""

import json
import re

SA_BANK_OFFSET = 12
MAX_BANK_IDX = 27

GLB_BANK_WIDTH = 64
CGRA_WORD_WIDTH = 16
GLB_BANK_PER_TILE = 2

WORDS_PER_BANK = GLB_BANK_WIDTH // CGRA_WORD_WIDTH # = 4
IOS_PER_TILE = GLB_BANK_PER_TILE * WORDS_PER_BANK  # = 8

# ---------------------------------------------------------------------------
# GLB Bank helpers
# ---------------------------------------------------------------------------


def _num_banks(datatype: str) -> int:
    """Return the number of unique bank indices for a given datatype."""
    if datatype == "bfloat16":
        return 8  # 32 entries / 4
    elif datatype == "int8":
        return 4  # 16 entries / 4
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")


def _allocate_banks(start: int, count: int) -> tuple:
    """
    Build a glb_bank_idx list of count unique indices starting at start, each
    repeated 4 times for E64 packing, and return (glb_list, next_start_bank).

    Overflow past MAX_BANK_IDX bounces back to SA_BANK_OFFSET.

    Example (SA_BANK_OFFSET=12, MAX_BANK_IDX=27):
      start=20, count=8 -> banks [20..27], next=12  (bounces)
      start=12, count=8 -> banks [12..19], next=20
    """
    bank_span = MAX_BANK_IDX - SA_BANK_OFFSET + 1
    result = []
    current_pos = start
    for _ in range(count):
        bank = SA_BANK_OFFSET + (current_pos - MAX_BANK_IDX - 1) % bank_span if current_pos > MAX_BANK_IDX else current_pos
        result.extend([bank] * 4)
        current_pos += 1
    next_start = SA_BANK_OFFSET + (current_pos - MAX_BANK_IDX - 1) % bank_span if current_pos > MAX_BANK_IDX else current_pos
    return result, next_start


# ---------------------------------------------------------------------------
# Op / tensor protobuf extraction helpers
# ---------------------------------------------------------------------------


def _normalize_operation_name(target: str) -> str:
    """Extract op name (mul, add, softmax, etc.) from target string."""
    if not target:
        raise ValueError("[ERROR] Op target is empty, cannot normalize operation name")
    # e.g. aten::mul.default -> mul; quantize_mx -> quantize_mx
    match = re.search(r"::(\w+)(?:\.\w+)?$", target)
    if match:
        return match.group(1)
    return target.split(".")[0] if "." in target else target


def _raw_tensors_from_argument(arg) -> list:
    """Extract raw tensor protobufs from an Argument (tensor or tensor_list)."""
    arg_type = arg.WhichOneof("arg_type")
    if arg_type == "tensor":
        return [arg.tensor]
    if arg_type == "tensor_list":
        return list(arg.tensor_list.tensors)
    return []


def _get_input_pairs(op) -> list:
    """Return list of (key, tensor_proto) for all inputs of op."""
    pairs = []
    for arg_idx, arg in enumerate(op.op.args):
        raw_tensors = _raw_tensors_from_argument(arg)
        for tensor_idx, tensor in enumerate(raw_tensors):
            pairs.append((str(arg_idx) if len(raw_tensors) == 1 else f"{arg_idx}_{tensor_idx}", tensor))
    for kw_key, arg in op.op.kwargs.items():
        raw_tensors = _raw_tensors_from_argument(arg)
        for tensor_idx, tensor in enumerate(raw_tensors):
            pairs.append((kw_key if len(raw_tensors) == 1 else f"{kw_key}_{tensor_idx}", tensor))
    return pairs


def _get_output_pairs(op) -> list:
    """Return list of (key, tensor_proto) for all outputs of op."""
    return_type = op.WhichOneof("return_type")
    if return_type == "output":
        return [("0", op.output)]
    elif return_type == "outputs":
        return [(str(tensor_idx), tensor) for tensor_idx, tensor in enumerate(op.outputs.tensors)]
    return []


def _allocate_sequential(pairs: list, start_bank: int) -> tuple:
    """
    Allocate glb_bank_idx for each (key, tensor) pair sequentially.
    Returns (list of (key, tensor, glb), next_bank).
    """
    current_bank = start_bank
    allocations = []
    for key, tensor in pairs:
        glb_banks, current_bank = _allocate_banks(current_bank, _num_banks(tensor.dtype or ""))
        allocations.append((key, tensor, glb_banks))
    return allocations, current_bank


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _e64_packing_for_graph(glb_for_graph: list) -> list:
    """Return a list parallel to glb_for_graph: 1 if the bank appears exactly 4 times, else 0."""
    counts = {}
    for b in glb_for_graph:
        counts[b] = counts.get(b, 0) + 1
    return [1 if counts[b] == 4 else 0 for b in glb_for_graph]


def _tensor_to_dict(tensor, is_first_pass, is_last_pass, glb_for_data, glb_for_graph) -> dict:
    return {
        "node": tensor.node,
        "shape": list(tensor.shape),
        "datatype": tensor.dtype or "",
        "is_first_pass": is_first_pass,
        "is_last_pass": is_last_pass,
        "glb_bank_idx_for_data": glb_for_data,
        "glb_bank_idx_for_graph": glb_for_graph,
        "e64_packing_for_graph": _e64_packing_for_graph(glb_for_graph),
    }


def _simple_op_to_json(op) -> dict:
    """Convert a non-decomposed op to JSON (is_first_pass=1, is_last_pass=1)."""
    input_allocs, next_bank = _allocate_sequential(_get_input_pairs(op), SA_BANK_OFFSET)
    output_allocs, _ = _allocate_sequential(_get_output_pairs(op), next_bank)
    operation = _normalize_operation_name(op.op.target) or op.op.name
    return {
        "operation": operation,
        "name": op.op.name,
        "inputs": {key: _tensor_to_dict(tensor, 1, 1, glb_banks, glb_banks) for key, tensor, glb_banks in input_allocs},
        "outputs": {key: _tensor_to_dict(tensor, 1, 1, glb_banks, glb_banks) for key, tensor, glb_banks in output_allocs},
    }


def _stage_decomp_group_to_json(group: list) -> list:
    """
    Generate JSON for a stage-decomposed group.

    Each pass's output banks become the next pass's input banks.

    Example for softmax (1 bfloat16 input, 1 bfloat16 output):
      pass1 input : [12..19],  pass1 output: [20..27]
      pass2 input : [20..27],  pass2 output: [12..19]  (bounces)
    """
    total = len(group)
    result = []

    op = group[0]
    base_operation = _normalize_operation_name(op.op.target) or op.op.name
    input_allocs, mid_bank = _allocate_sequential(_get_input_pairs(op), SA_BANK_OFFSET)
    output_allocs, prev_next_bank = _allocate_sequential(_get_output_pairs(op), mid_bank)
    is_last = 1 if total == 1 else 0
    result.append(
        {
            "operation": f"{base_operation}_pass1",
            "name": op.op.name,
            "inputs": {key: _tensor_to_dict(tensor, 1, is_last, glb_banks, glb_banks) for key, tensor, glb_banks in input_allocs},
            "outputs": {key: _tensor_to_dict(tensor, 1, is_last, glb_banks, glb_banks) for key, tensor, glb_banks in output_allocs},
        }
    )

    prev_output_allocs = output_allocs
    for pass_idx in range(1, total):
        op = group[pass_idx]
        is_last = 1 if pass_idx == total - 1 else 0

        # Reuse previous pass's output banks as inputs; spill to fresh banks if needed.
        input_allocs = []
        for input_idx, (key, tensor) in enumerate(_get_input_pairs(op)):
            if input_idx < len(prev_output_allocs):
                _, _, glb_banks = prev_output_allocs[input_idx]
            else:
                glb_banks, prev_next_bank = _allocate_banks(prev_next_bank, _num_banks(tensor.dtype or ""))
            input_allocs.append((key, tensor, glb_banks))

        output_allocs, prev_next_bank = _allocate_sequential(_get_output_pairs(op), prev_next_bank)
        prev_output_allocs = output_allocs

        result.append(
            {
                "operation": f"{base_operation}_pass{pass_idx + 1}",
                "name": op.op.name,
                "inputs": {key: _tensor_to_dict(tensor, 0, is_last, glb_banks, glb_banks) for key, tensor, glb_banks in input_allocs},
                "outputs": {key: _tensor_to_dict(tensor, 0, is_last, glb_banks, glb_banks) for key, tensor, glb_banks in output_allocs},
            }
        )

    return result


def _tensor_decomp_group_to_json(group: list) -> list:
    """
    Generate JSON for a tensor-decomposed group.

    Each pass gets (n_banks / total) banks for graph; data always holds the full range.

    Example for silu into 2 passes (bfloat16):
      pass0 input  data/graph: [12..19] / [12..15]
      pass0 output data/graph: [20..27] / [20..23]
      pass1 input  data/graph: [12..19] / [16..19]
      pass1 output data/graph: [20..27] / [24..27]
    """
    total = len(group)
    first_op = group[0]
    orig_input_allocs, mid_bank = _allocate_sequential(_get_input_pairs(first_op), SA_BANK_OFFSET)
    orig_output_allocs, _ = _allocate_sequential(_get_output_pairs(first_op), mid_bank)

    def _split_glb(orig_glb, pass_idx):
        unique = orig_glb[::4]
        num_sub = len(unique) // total
        sub = unique[pass_idx * num_sub : (pass_idx + 1) * num_sub]
        return [b for b in sub for _ in range(4)]

    base_operation = _normalize_operation_name(first_op.op.target) or first_op.op.name
    result = []
    for pass_idx, op in enumerate(group):
        is_first_pass, is_last_pass = (1 if pass_idx == 0 else 0), (1 if pass_idx == total - 1 else 0)
        result.append(
            {
                "operation": base_operation,
                "name": op.op.name,
                "inputs": {
                    key: _tensor_to_dict(tensor, is_first_pass, is_last_pass, glb_banks, _split_glb(glb_banks, pass_idx))
                    for (key, tensor), (_, _, glb_banks) in zip(_get_input_pairs(op), orig_input_allocs)
                },
                "outputs": {
                    key: _tensor_to_dict(tensor, is_first_pass, is_last_pass, glb_banks, _split_glb(glb_banks, pass_idx))
                    for (key, tensor), (_, _, glb_banks) in zip(_get_output_pairs(op), orig_output_allocs)
                },
            }
        )
    return result


def _transpose_decomp_group_to_json(group: list) -> list:
    """
    Generate JSON for a transpose-decomposed group.

    group has n_banks // GLB_BANK_PER_TILE * IOS_PER_TILE ops (e.g., 32 for bfloat16).
    pass_idx encodes both tile and IO: tile_idx = pass_idx // IOS_PER_TILE,
    io_idx = pass_idx % IOS_PER_TILE.

    Outputs: full original banks for both data and graph.
    Inputs:  data = full original range; graph = tile banks repeated WORDS_PER_BANK times.

    Example for bfloat16 starting at bank 12 (32 passes total):
      pass0..7  (tile0, banks 12,13): IO_idx_per_tile=0..7, graph=[12,12,12,12,13,13,13,13]
      pass8..15 (tile1, banks 14,15): IO_idx_per_tile=0..7, graph=[14,14,14,14,15,15,15,15]
      ...
    """
    total = len(group)
    first_op = group[0]
    orig_input_allocs, mid_bank = _allocate_sequential(_get_input_pairs(first_op), SA_BANK_OFFSET)
    orig_output_allocs, _ = _allocate_sequential(_get_output_pairs(first_op), mid_bank)

    base_operation = _normalize_operation_name(first_op.op.target) or first_op.op.name
    result = []
    for pass_idx, op in enumerate(group):
        tile_idx = pass_idx // IOS_PER_TILE
        is_first_pass = 1 if pass_idx == 0 else 0
        is_last_pass = 1 if pass_idx == total - 1 else 0

        inputs = {}
        for (key, tensor), (_, _, orig_glb_banks) in zip(_get_input_pairs(op), orig_input_allocs):
            all_banks = orig_glb_banks[::4]
            tile_banks = all_banks[tile_idx * GLB_BANK_PER_TILE : (tile_idx + 1) * GLB_BANK_PER_TILE]
            tile_graph = [b for b in tile_banks for _ in range(WORDS_PER_BANK)]
            inputs[key] = _tensor_to_dict(tensor, is_first_pass, is_last_pass, orig_glb_banks, tile_graph)

        result.append(
            {
                "operation": base_operation,
                "name": op.op.name,
                "kernel_id": pass_idx,
                "inputs": inputs,
                "outputs": {
                    key: _tensor_to_dict(tensor, is_first_pass, is_last_pass, glb_banks, glb_banks)
                    for (key, tensor), (_, _, glb_banks) in zip(_get_output_pairs(op), orig_output_allocs)
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _dumps_compact(obj, indent=2) -> str:
    """Serialize obj to JSON with indentation but lists on a single line."""
    raw = json.dumps(obj, indent=indent)
    return re.sub(
        r"\[([^\[\]]*)\]",
        lambda match: "[" + re.sub(r"\s+", " ", match.group(1).strip()) + "]",
        raw,
        flags=re.DOTALL,
    )
