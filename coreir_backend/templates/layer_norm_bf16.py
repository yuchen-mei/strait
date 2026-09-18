"""Single-invocation LayerNorm: center, normalize, and per-channel affine.

Compose the proven ready/valid reduction templates without intermediate GLB
stores. Two sets of dual-read CGRA memories retain x and x - mean respectively.
A third MEM set decouples normalized lane outputs from coefficient IO.
At 16 lanes, each external tensor uses two GLB tiles in E64 multi-bank mode.
The existing log/exp reciprocal-square-root approximation and accumulator FIFO
bypasses are preserved. Like the split implementation, epsilon is omitted.
"""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .reduction_sum_mul_elementwise_sub_bf16 import (
    emit_reduction_sum_mul_elementwise_sub_bf16_design,
)
from .reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16 import (
    emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design,
)
from .elementwise_mul_add_mul_add_bf16 import (
    emit_elementwise_mul_add_mul_add_bf16_design,
)


def emit_layer_norm_bf16_design(unroll, vec_length, num_vecs, output_path):
    if unroll != 16:
        raise ValueError("The single-pass GLB layout requires 16 lanes")
    if num_vecs < 1 or vec_length < 8 * unroll or vec_length % (8 * unroll):
        raise ValueError("Rows must be positive and width a positive multiple of 128")

    with TemporaryDirectory() as temporary:
        graphs, bypass = {}, {}
        emitters = {
            "mean": emit_reduction_sum_mul_elementwise_sub_bf16_design,
            "norm": emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design,
            "affine": emit_elementwise_mul_add_mul_add_bf16_design,
        }
        for stage, emit in emitters.items():
            directory = Path(temporary) / stage
            directory.mkdir()
            emit(unroll, vec_length, num_vecs, str(directory))
            graphs[stage] = json.loads((directory / "design_top.json").read_text())
            fifo_path = directory / "PE_fifos_bypass_config.json"
            if fifo_path.exists():
                bypass.update({stage + "_" + name: value
                               for name, value in json.loads(fifo_path.read_text()).items()})

    def module(graph):
        return graph["namespaces"]["global"]["modules"][graph["top"].split(".")[-1]]

    result = copy.deepcopy(graphs["affine"])
    fused = module(result)
    instances, connections = fused["instances"], fused["connections"]
    # Replace the affine input edges with mean/normalization and FIFO stages.
    connections[:] = [edge for edge in connections
                       if not any(port.startswith("io16in_input_host_") for port in edge)
                       or any(port.startswith("self.") for port in edge)]

    for stage in ("mean", "norm"):
        source = module(graphs[stage])
        internal = {name for name, inst in source["instances"].items()
                    if inst.get("modref") != "global.IO"}
        instances.update({stage + "_" + name: copy.deepcopy(source["instances"][name])
                          for name in internal})
        for name in internal:
            if name.startswith("tile_input_lane_") or name.endswith("_filter_mem"):
                metadata = instances[stage + "_" + name]["metadata"]
                config = metadata["lake_rv_config"]
                if isinstance(config, str):
                    config = json.loads(config)
                # Half-row blocks retain a full block of SRAM write margin
                # without delaying each reduction by another complete row.
                config["row_size"] = vec_length // (2 * unroll)
                metadata["lake_rv_config"] = config
        for edge in source["connections"]:
            if all(port.split(".")[0] in internal for port in edge):
                connections.append([stage + "_" + port for port in edge])

    # Decouple every broadcast lane from the coupled GLB coefficient streams.
    # Deep FIFOs absorb path skew while the affine PEs consume matching tokens.
    affine_fifo_lanes = range(unroll)
    for lane in affine_fifo_lanes:
        fifo = f"z_norm_affine_fifo_lane_{lane}"
        instances[fifo] = copy.deepcopy(instances[f"norm_tile_input_lane_{lane}"])
        instances[fifo]["metadata"]["lake_rv_config"]["type"] = "fifo"
        # Only latency matching is needed here, so release data in short
        # blocks instead of waiting for half a reduction row.
        instances[fifo]["metadata"]["lake_rv_config"]["row_size"] = 8
        clock_edge = next(edge for edge in connections
                          if f"norm_tile_input_lane_{lane}.clk_en" in edge)
        connections.append([port.replace(f"norm_tile_input_lane_{lane}.", fifo + ".")
                            for port in clock_edge])
        connections.append([fifo + ".data_out_0", f"mul_vec_pe_{lane}.data0"])

    for lane in range(unroll):
        input_io = next(name for name in instances
                        if name.startswith("io16in_input_host_") and f"_clkwrk_{lane}_" in name)
        connections.extend([
            [input_io + ".out", f"mean_tile_input_lane_{lane}.data_in_0"],
            [f"mean_elementwise_add_pe_{lane}.O0", f"norm_tile_input_lane_{lane}.data_in_0"],
            [f"norm_elementwise_mul_pe_{lane}.O0",
             f"z_norm_affine_fifo_lane_{lane}.data_in_0" if lane in affine_fifo_lanes
             else f"mul_vec_pe_{lane}.data0"],
        ])

    modules = result["namespaces"]["global"]["modules"]
    del modules[result["top"].split(".")[-1]]
    modules["layer_norm_fp"] = fused
    result["top"] = "global.layer_norm_fp"
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "design_top.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    (directory / "PE_fifos_bypass_config.json").write_text(json.dumps(bypass, indent=2) + "\n")
    return str(destination)
