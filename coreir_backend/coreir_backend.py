import os
import json
import importlib
import pkgutil
from typing import Dict, Any

# Import the utility packages
from strait.coreir_backend.utils.bin2raw import convert_bin_to_raw
from strait.coreir_backend.utils.bin2txt import convert_bin_to_txt
import strait.coreir_backend.templates as templates_pkg

# Dynamically load all templates in the templates package.
loaded_templates = {}
for _, module_name, _ in pkgutil.iter_modules(templates_pkg.__path__):
    full_module_name = f"{templates_pkg.__name__}.{module_name}"
    loaded_templates[module_name] = importlib.import_module(full_module_name)
print(f"[INFO] Discovered templates: {list(loaded_templates.keys())}")

CGRA_DATA_WIDTH = 16


def _io_logical_names_from_design_top(design_top_path: str):
    """
    Read design_top.json and return (input_names, output_names) in instance order.
    Names are the logical names parse_design_meta will look up (between io16in_/io16_ and _clkwrk).
    """
    with open(design_top_path, "r") as f:
        design_top = json.load(f)
    top_name = design_top["top"]
    _, _, design_name = top_name.partition("global.")
    instances = design_top["namespaces"]["global"]["modules"][design_name]["instances"]
    input_names = []
    output_names = []
    seen_inputs = set()
    seen_outputs = set()
    for inst in instances:
        end_delim = "_clkwrk" if "clkwrk" in inst else "_op_hcompute"
        if inst.startswith("io16in_"):
            starti = inst.find("io16in_") + len("io16in_")
            endi = inst.find(end_delim, starti)
            name = inst[starti:endi]
            if name not in seen_inputs:
                input_names.append(name)
                seen_inputs.add(name)
        elif inst.startswith("io16_"):
            starti = inst.find("io16_") + len("io16_")
            endi = inst.find(end_delim, starti)
            name = inst[starti:endi]
            if name not in seen_outputs:
                output_names.append(name)
                seen_outputs.add(name)
    return input_names, output_names


class CoreIRBackend:
    def __init__(self, scheduled_ops_path: str, tensor_files_path: str, output_dir: str):
        self.scheduled_ops_path = scheduled_ops_path
        self.tensor_files_path = tensor_files_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the scheduled operations
        with open(self.scheduled_ops_path, 'r') as f:
            self.scheduled_ops = json.load(f)

    def run(self):
        """Iterates through all scheduled kernels and processes them."""
        print(f"[INFO] Starting CoreIR Backend generation for {len(self.scheduled_ops)} kernels.")
        for kernel in self.scheduled_ops:
            self._get_kernel_tensor_files(kernel)
            self._get_kernel_coreir_json(kernel)
            self._get_kernel_design_meta_halide_json(kernel)

    def _get_template_name(self, kernel: Dict[str, Any]) -> str:
        """
        Mapping each kernel to its corresponding CoreIR template name.
        """
        operation = kernel.get("operation")
        first_input_dtype = next(iter(kernel.get("inputs", {}).values())).get("datatype")

        # Elementwise operations
        if operation in ["silu", "gelu"]:
            if first_input_dtype == "int8":
                return "elementwise_swish_int8"
            elif first_input_dtype == "bfloat16":
                return "elementwise_swish_bf16"
            else:
                raise NotImplementedError(
                    f"[TODO] No template mapping defined for operation: '{operation}' with datatype: '{first_input_dtype}'! "
                    f"Please implement it in _get_template_name."
                )
        elif operation in ["mul", "add", "sub", "div", "reciprocal", "sqrt", "pow", "exp", "log"]:
            if first_input_dtype == "int8":
                return f"elementwise_{operation}_int8"
            elif first_input_dtype == "bfloat16":
                return f"elementwise_{operation}_bf16"
            else:
                raise NotImplementedError(
                    f"[TODO] No template mapping defined for operation: '{operation}' with datatype: '{first_input_dtype}'! "
                    f"Please implement it in _get_template_name."
                )

        # Transpose
        elif operation == "transpose":
            if first_input_dtype == "bfloat16":
                return "transpose_bf16"
            elif first_input_dtype == "int8":
                return "transpose_int8"
            else:
                raise NotImplementedError(
                    f"[TODO] No template mapping defined for operation: '{operation}' with datatype: '{first_input_dtype}'! "
                    f"Please implement it in _get_template_name."
                )
        else:
            raise NotImplementedError(f"[TODO] No template mapping defined for operation: '{operation}'! Please implement it in _get_template_name.")

    def _get_template_params(self, kernel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract template parameters from a scheduled kernel entry.

        - unroll: derived from the length of glb_bank_idx_for_graph on the first input.
        - beta: operation-specific scale for swish-family ops (silu=1.0, gelu=1.702).
        """
        operation = kernel.get("operation")
        params = {}

        if operation == "transpose":
            # For transpose, input graph banks hold one bank per pass.
            # In Zircon, fabric can only load from odd banks in E64 MB mode, 
            # and we actually only consume one software IO with 16-bit slice for each pass.
            # Derive unroll from the output, which retains the full bank set.
            first_output = next(iter(kernel.get("outputs", {}).values()), {})
            params["unroll"] = len(first_output.get("glb_bank_idx_for_graph", []))
            params["kernel_id"] = kernel.get("kernel_id", 0)
        else:
            first_input = next(iter(kernel.get("inputs", {}).values()), {})
            params["unroll"] = len(first_input.get("glb_bank_idx_for_graph", []))

        if operation == "silu":
            params["beta"] = 1.0
        elif operation == "gelu":
            params["beta"] = 1.702

        return params

    def _get_kernel_tensor_files(self, kernel: Dict[str, Any]):
        """Convert tensor files from bin to raw and txt formats from proto_frontend to coreir_backend."""
        layer_dir = os.path.join(self.output_dir, kernel.get("name"))
        os.makedirs(layer_dir, exist_ok=True)
        kernel_inputs = kernel.get("inputs", {})
        kernel_outputs = kernel.get("outputs", {})

        for input_type, input in kernel_inputs.items():
            convert_bin_to_raw(
                input_path = os.path.join(self.tensor_files_path, input.get("node") + ".bin"),
                output_path = os.path.join(layer_dir, "input_" + input.get("node") + ".raw"),
                target_dtype = input.get("datatype")
            )
            convert_bin_to_txt(
                input_path = os.path.join(self.tensor_files_path, input.get("node") + ".bin"),
                output_path = os.path.join(layer_dir, "input_" + input.get("node") + ".txt"),
                target_dtype = input.get("datatype")
            )

        for output_type, output in kernel_outputs.items():
            convert_bin_to_raw(
                input_path = os.path.join(self.tensor_files_path, output.get("node") + ".bin"),
                output_path = os.path.join(layer_dir, "output_" + output.get("node") + ".raw"),
                target_dtype = output.get("datatype")
            )
            convert_bin_to_txt(
                input_path = os.path.join(self.tensor_files_path, output.get("node") + ".bin"),
                output_path = os.path.join(layer_dir, "output_" + output.get("node") + ".txt"),
                target_dtype = output.get("datatype")
            )

    def _get_kernel_coreir_json(self, kernel: Dict[str, Any]):
        """Dynamically routes the kernel to the correct template module and function."""
        kernel_name = kernel.get("name")
        operation = kernel.get("operation")

        # Get template name and parameters from operation
        template_name = self._get_template_name(kernel)
        template_params = self._get_template_params(kernel)

        # Retrieve the emit function name from the template module
        emit_func_name = f"emit_{template_name}_coreir_json"

        # Ensure template exists
        if template_name not in loaded_templates:
            raise ValueError(f"[ERROR] Missing template file '{template_name}.py'.")

        template_module = loaded_templates[template_name]

        if not hasattr(template_module, emit_func_name):
            raise ValueError(f"[ERROR] Module '{template_name}' missing function '{emit_func_name}'.")

        # Grab the actual emit function from the module
        emit_func = getattr(template_module, emit_func_name)

        # Set up the output directory and path
        layer_dir = os.path.join(self.output_dir, kernel_name)
        os.makedirs(layer_dir, exist_ok=True)

        print(f"[INFO] Routing {kernel_name} ({operation}) -> {template_name}.py -> {layer_dir}/design_top.json")

        # ===============================
        # Execute the template function
        # ===============================
        try:
            emit_func(kernel=kernel, output_path=layer_dir, **template_params)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to emit json for {kernel_name}: {e}")

        # Dump a per-kernel scheduled_ops.json so the IO placer can consume
        # glb_bank_idx_for_graph without needing to know the frontend path.
        sched_ops_path = os.path.join(layer_dir, "scheduled_ops.json")
        with open(sched_ops_path, "w") as f:
            json.dump([kernel], f, indent=2)
        print(f"[INFO] Wrote per-kernel scheduled_ops.json to {sched_ops_path}")

    def _get_kernel_design_meta_halide_json(self, kernel: Dict[str, Any]):
        """
        Generate design_meta_halide.json so IO names match design_top.json instance names.
        parse_design_meta.findIO looks up the logical name between io16in_/io16_ and _clkwrk.
        Derive the list of names from design_top.json and create one meta entry per name.
        """
        layer_dir = os.path.join(self.output_dir, kernel.get("name"))
        os.makedirs(layer_dir, exist_ok=True)
        design_top_path = os.path.join(layer_dir, "design_top.json")
        if not os.path.isfile(design_top_path):
            raise FileNotFoundError(f"design_top.json not found at {design_top_path}; emit coreir before design_meta.")

        input_names, output_names = _io_logical_names_from_design_top(design_top_path)
        kernel_inputs = [inp for inp in kernel.get("inputs", {}).values() if "_tensor_constant" not in inp.get("node", "")]
        kernel_outputs = list(kernel.get("outputs", {}).values())
        default_input = kernel_inputs[0] if kernel_inputs else {"node": "input", "shape": [1]}
        default_output = kernel_outputs[0] if kernel_outputs else {"node": "output", "shape": [1]}

        design_meta_halide_dict = {
            "IOs": {"inputs": [], "outputs": [], "mu_inputs": []},
            "testing": {},
        }
        for i, name in enumerate(input_names):
            inp = kernel_inputs[i] if i < len(kernel_inputs) else default_input
            design_meta_halide_dict["IOs"]["inputs"].append({
                "bitwidth": CGRA_DATA_WIDTH,
                "datafile": "input_" + inp.get("node") + ".raw",
                "name": name,
                "shape": inp.get("shape"),
            })
        for i, name in enumerate(output_names):
            out = kernel_outputs[i] if i < len(kernel_outputs) else default_output
            design_meta_halide_dict["IOs"]["outputs"].append({
                "bitwidth": CGRA_DATA_WIDTH,
                "datafile": "output_" + out.get("node") + ".raw",
                "name": name,
                "shape": out.get("shape"),
            })

        out_path = os.path.join(layer_dir, "design_meta_halide.json")
        with open(out_path, "w") as f:
            json.dump(design_meta_halide_dict, f, indent=2)
        print(f"[INFO] Wrote design meta halide json to {out_path}")


if __name__ == "__main__":
    backend = CoreIRBackend(
        scheduled_ops_path="/aha/strait/proto_frontend/_generated_scheduled_ops/pointwise/int8/scheduled_ops.json",
        tensor_files_path="/aha/strait/proto_frontend/_generated_protobuf/pointwise/int8/tensor_files",
        output_dir="/aha/strait/_generated_coreirs/pointwise/int8"
    )
    backend.run()