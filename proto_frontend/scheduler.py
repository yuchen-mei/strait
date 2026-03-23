"""
Scheduler framework: load model protobuf, apply transformations
(collapse_nops, decomposition, tiling, fusion), and generate new ops.
Final output: json of transformed ops with operation, inputs and outputs.

See scheduler_utils.py for bank allocation and JSON serialization helpers.
"""

import argparse
import os

import strait.proto_frontend.param_pb2 as param_pb2
from google.protobuf import text_format

from strait.proto_frontend.scheduler_utils import (
    _normalize_operation_name,
    _num_banks,
    _get_input_pairs,
    _get_output_pairs,
    _simple_op_to_json,
    _stage_decomp_group_to_json,
    _tensor_decomp_group_to_json,
    _transpose_decomp_group_to_json,
    _dumps_compact,
    GLB_BANK_PER_TILE,
    IOS_PER_TILE,
)


def load_model(model_protobuf_path: str) -> param_pb2.Model:
    """Load protobuf from a model.txt file."""
    params = param_pb2.Model()
    with open(model_protobuf_path, "r") as f:
        text_format.Parse(f.read(), params)
    return params


def collapse_nops(params: param_pb2.Model) -> None:
    """Assert no fused_op in the model, then remove all no-ops from params.ops."""
    for op_idx, op in enumerate(params.ops):
        if op.WhichOneof("op_type") == "fused_op":
            raise AssertionError(f"fused_op not supported: op at index {op_idx} is a fused_op")

    new_ops = []
    for op in params.ops:
        if op.op.op == "nop":
            continue
        new_op = param_pb2.Operation()
        new_op.CopyFrom(op)
        new_ops.append(new_op)
    params.ClearField("ops")
    for op in new_ops:
        params.ops.add().CopyFrom(op)


# Per-op decomposition configuration: op_type -> (decomp_type, num_passes).
# decomp_type: "tensor", "stage", or "transpose".
# num_passes: for "tensor", must divide n_banks for every tensor dtype in the op
#             (n_banks=8 for bfloat16, 4 for int8); for "stage", must be >= 2;
#             for "transpose", use None to auto-derive (= n_banks from first input).
_OP_DECOMP_CONFIG: dict = {
    "silu": ("tensor", 2),
    "transpose": ("transpose", None),
}


def _validate_decomp(op, decomp_type: str, num_passes: int) -> None:
    """Raise ValueError if num_passes is invalid for the given decomp_type and op."""
    if decomp_type == "stage":
        if num_passes < 2:
            raise ValueError(f"stage decomposition requires num_passes >= 2, got {num_passes}")
    elif decomp_type == "tensor":
        all_tensors = [t for _, t in _get_input_pairs(op)] + [t for _, t in _get_output_pairs(op)]
        for tensor in all_tensors:
            dtype = tensor.dtype or ""
            if not dtype:
                continue
            bank_count = _num_banks(dtype)
            if bank_count % num_passes != 0:
                raise ValueError(
                    f"tensor decomposition: num_passes={num_passes} does not evenly "
                    f"divide n_banks={bank_count} (dtype={dtype}); valid values are factors of {bank_count}"
                )
    elif decomp_type == "transpose":
        if num_passes < 1:
            raise ValueError(f"transpose decomposition requires num_passes >= 1, got {num_passes}")
    else:
        raise ValueError(f"Unknown decomp_type: {decomp_type!r}")


def decomposition(params: param_pb2.Model) -> dict:
    """
    Decompose ops according to _OP_DECOMP_CONFIG.

    Each matched op is replaced by num_passes copies named {original}_pass1 .. _passN.

    decomp_type rules:
      "tensor"   : bank count divided by num_passes per tensor; pass i uses the i-th
                   sub-range of the original bank range.
      "stage"    : bank count unchanged; each pass's output banks become the next
                   pass's input banks.
      "transpose": output banks unchanged; each pass's input graph banks = one even
                   bank from the original input range (num_passes auto-derived as
                   n_banks // 2 from the first input tensor dtype).

    Returns pass_info: dict mapping op_name -> {pass_idx, total_passes, decomp_type, group_id}.
    """
    pass_info = {}
    new_ops = []
    for op in params.ops:
        op_type = _normalize_operation_name(op.op.target) or op.op.name
        config = _OP_DECOMP_CONFIG.get(op_type)
        if config is not None:
            decomp_type, num_passes = config
            if decomp_type == "transpose" and num_passes is None:
                # Auto-derive: one pass per GLB tile IO = n_banks // GLB_BANK_PER_TILE * IOS_PER_TILE.
                first_dtype = next(
                    (tensor.dtype for _, tensor in _get_input_pairs(op) if tensor.dtype),
                    None,
                )
                if first_dtype is None:
                    raise ValueError(f"transpose decomp: no input tensor with dtype found on op {op.op.name!r}")
                num_passes = _num_banks(first_dtype) // GLB_BANK_PER_TILE * IOS_PER_TILE
            _validate_decomp(op, decomp_type, num_passes)
            original_name = op.op.name
            for pass_index in range(num_passes):
                new_op = param_pb2.Operation()
                new_op.CopyFrom(op)
                new_op.op.name = f"{original_name}_pass{pass_index + 1}"
                new_ops.append(new_op)
                pass_info[new_op.op.name] = {
                    "pass_idx": pass_index,
                    "total_passes": num_passes,
                    "decomp_type": decomp_type,
                    "group_id": original_name,
                }
        else:
            new_op = param_pb2.Operation()
            new_op.CopyFrom(op)
            new_ops.append(new_op)
    params.ClearField("ops")
    for op in new_ops:
        params.ops.add().CopyFrom(op)
    return pass_info


def tiling(params: param_pb2.Model):
    """TODO: Implement tiling"""
    pass


def fusion(params: param_pb2.Model):
    """TODO: Implement fusion"""
    pass


def schedule(params: param_pb2.Model) -> dict:
    """
    Run the full scheduler pipeline: collapse_nops -> decomposition -> tiling -> fusion.
    Mutates params in place and returns pass_info dict from decomposition.
    """
    collapse_nops(params)
    pass_info = decomposition(params)
    tiling(params)
    fusion(params)
    return pass_info


def params_to_json(params: param_pb2.Model, pass_info: dict = None) -> list:
    """
    Convert scheduled params to a JSON list of ops.
    Each op: {operation, name, inputs, outputs}.
    Each tensor value: {node, shape, datatype, is_first_pass, is_last_pass,
                        glb_bank_idx_for_data, glb_bank_idx_for_graph}.

    Decomposed op groups are processed together so bank relationships between
    passes (stage: shared banks; tensor: split banks) are correctly assigned.
    """
    if pass_info is None:
        pass_info = {}

    ops = list(params.ops)
    result = []
    op_idx = 0
    while op_idx < len(ops):
        op = ops[op_idx]
        info = pass_info.get(op.op.name)
        if info and info["pass_idx"] == 0:
            # Collect all passes belonging to this decomposition group.
            group_id = info["group_id"]
            total_passes = info["total_passes"]
            decomp_type = info["decomp_type"]
            group = [op]
            scan_idx = op_idx + 1
            while scan_idx < len(ops) and len(group) < total_passes:
                next_info = pass_info.get(ops[scan_idx].op.name)
                if next_info and next_info["group_id"] == group_id:
                    group.append(ops[scan_idx])
                    scan_idx += 1
                else:
                    break
            if decomp_type == "stage":
                result.extend(_stage_decomp_group_to_json(group))
            elif decomp_type == "tensor":
                result.extend(_tensor_decomp_group_to_json(group))
            elif decomp_type == "transpose":
                result.extend(_transpose_decomp_group_to_json(group))
            else:
                raise ValueError(f"Unknown decomp_type: {decomp_type}")
            op_idx = scan_idx
        else:
            result.append(_simple_op_to_json(op))
            op_idx += 1
    return result


def protobuf_to_scheduled_ops(model_txt_path: str, output_json_path: str) -> None:
    """Load model.txt, schedule, write json of transformed ops."""
    params = load_model(model_txt_path)
    pass_info = schedule(params)
    transformed_ops = params_to_json(params, pass_info)
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        f.write(_dumps_compact(transformed_ops))
    print(f"[INFO] Wrote {len(transformed_ops)} ops to {output_json_path}.")


def main():
    parser = argparse.ArgumentParser(description="Load model.txt, schedule, write json of transformed ops.")
    parser.add_argument("model_txt", help="Path to model.txt.")
    parser.add_argument("-o", "--output", required=True, help="Output json path.")
    args = parser.parse_args()

    if not os.path.isfile(args.model_txt):
        raise FileNotFoundError(f"[ERROR] Model file not found: {args.model_txt}")

    protobuf_to_scheduled_ops(args.model_txt, args.output)


if __name__ == "__main__":
    main()
