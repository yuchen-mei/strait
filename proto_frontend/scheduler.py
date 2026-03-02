"""
Scheduler framework: load model protobuf, apply transformations
(collapse_nops, fusion, tiling, fission), and generate new ops.
Final output: json of transformed ops with operation, inputs and outputs.
"""
import argparse
import json
import os
import re

import strait.proto_frontend.param_pb2 as param_pb2
from google.protobuf import text_format


def load_model(model_protobuf_path: str) -> param_pb2.Model:
    """Load protobuf from a model.txt file."""
    params = param_pb2.Model()
    with open(model_protobuf_path, "r") as f:
        text_format.Parse(f.read(), params)
    return params

def collapse_nops(params: param_pb2.Model) -> None:
    """
    Assert no fused_op in the model, then remove all no-ops from params.ops.
    """
    for i, op in enumerate(params.ops):
        if op.WhichOneof("op_type") == "fused_op":
            raise AssertionError(f"fused_op not supported: op at index {i} is a fused_op")

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

def fission(params: param_pb2.Model) -> None:
    """
    TODO: Implement fission
    """
    pass

def tiling(params: param_pb2.Model) -> None:
    """
    TODO: Implement tiling
    """
    pass

def fusion(params: param_pb2.Model) -> None:
    """
    TODO: Implement fusion
    """
    pass

def schedule(params: param_pb2.Model) -> param_pb2.Model:
    """
    Run the full scheduler pipeline: collapse_nops -> fission -> tiling -> fusion.
    Returns the transformed params (mutated in place).
    """
    collapse_nops(params)
    fission(params)
    tiling(params)
    fusion(params)
    return params

def get_schedule(params: param_pb2.Model):
    """
    Return the list of op names in schedule order.
    """
    return [op.op.name for op in params.ops]

def _tensor_to_dict(tensor) -> dict:
    """Convert a tensor protobuf to {node, shape, datatype} dict."""
    return {
        "node": tensor.node,
        "shape": list(tensor.shape),
        "datatype": tensor.dtype or "",
    }

def _extract_tensors_from_argument(arg) -> list:
    """Extract tensor dicts from an Argument (tensor or tensor_list)."""
    which = arg.WhichOneof("arg_type")
    if which == "tensor":
        return [_tensor_to_dict(arg.tensor)]
    if which == "tensor_list":
        return [_tensor_to_dict(t) for t in arg.tensor_list.tensors]
    return []

def _extract_inputs_from_op_overload(op_overload) -> dict:
    """Build inputs dict from OpOverload args and kwargs. Keys: arg index or kwarg name."""
    inputs = {}
    for i, arg in enumerate(op_overload.args):
        tensors = _extract_tensors_from_argument(arg)
        for j, t in enumerate(tensors):
            key = str(i) if len(tensors) == 1 else f"{i}_{j}"
            inputs[key] = t
    for key, arg in op_overload.kwargs.items():
        tensors = _extract_tensors_from_argument(arg)
        for j, t in enumerate(tensors):
            k = key if len(tensors) == 1 else f"{key}_{j}"
            inputs[k] = t
    return inputs

def _extract_outputs_from_operation(op) -> dict:
    """Build outputs dict from Operation output or outputs."""
    which = op.WhichOneof("return_type")
    if which == "output":
        return {"0": _tensor_to_dict(op.output)}
    if which == "outputs":
        return {str(i): _tensor_to_dict(t) for i, t in enumerate(op.outputs.tensors)}
    return {}

def _normalize_operation_name(target: str) -> str:
    """Extract op name (mul, add, softmax, etc.) from target string."""
    if not target:
        return "unknown"
    # e.g. aten::mul.default -> mul; quantize_mx -> quantize_mx
    m = re.search(r"::(\w+)(?:\.\w+)?$", target)
    if m:
        return m.group(1)
    return target.split(".")[0] if "." in target else target

def _operation_to_json(op) -> dict:
    """Convert a single op to {operation, name, inputs, outputs} json dict."""
    operation = _normalize_operation_name(op.op.target) or op.op.name
    inputs = _extract_inputs_from_op_overload(op.op)
    outputs = _extract_outputs_from_operation(op)
    return {
        "operation": operation,
        "name": op.op.name,
        "inputs": inputs,
        "outputs": outputs,
    }

def params_to_json(params: param_pb2.Model) -> list:
    """
    Convert scheduled params to json list of transformed ops.
    Each op: {operation, name, inputs and outputs}.
    name: original op name from protobuf.
    inputs/outputs: dict of name -> {node, shape, datatype}.
    """
    return [_operation_to_json(op) for op in params.ops]

def protobuf_to_scheduled_ops(model_txt_path: str, output_json_path: str) -> None:
    """
    Load model.txt, schedule, write json of transformed ops.
    """
    params = load_model(model_txt_path)
    schedule(params)
    transformed_ops = params_to_json(params)
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(transformed_ops, f, indent=2)
    print(f"[INFO] Wrote {len(transformed_ops)} ops to {output_json_path}.")


def main():
    parser = argparse.ArgumentParser(description="Load model.txt, schedule, write json of transformed ops.")
    parser.add_argument("model_txt", help="Path to model.txt.")
    parser.add_argument("-o", "--output", required=True, help="Output json path.")
    args = parser.parse_args()

    if not os.path.isfile(args.model_txt):
        raise FileNotFoundError(f"Model file not found: {args.model_txt}")

    protobuf_to_scheduled_ops(args.model_txt, args.output)


if __name__ == "__main__":
    main()
