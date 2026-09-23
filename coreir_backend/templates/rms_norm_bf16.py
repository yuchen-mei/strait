"""Streaming 16-lane RMSNorm with per-channel gamma and no GLB intermediates.

Reuse the split flow's BF16 reduction/rsqrt and broadcast-weight multiply.
Dual-read MEMs retain activations; deep FIFOs decouple normalized lanes from
gamma and gamma outputs from the coupled GLB stream. The existing arithmetic
(including no epsilon) stays unchanged. All buffering is configured on the
taped-out hardware. For 32 output lanes, the existing output MEMs scatter
even/odd elements into canonical 32-lane stripes; input and compute stay at
16 lanes and gamma DMA remains sequential within each row.
"""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .elementwise_mul_bf16 import emit_elementwise_mul_bf16_design, _lane_port_names
from .reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16 import (
    emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design,
)


def emit_rms_norm_bf16_design(unroll, vec_length, num_vecs, output_path,
                            output_unroll=None):
    output_unroll = unroll if output_unroll is None else output_unroll
    if unroll != 16:
        raise ValueError("The single-pass input and compute graph require 16 lanes")
    if output_unroll not in (unroll, 2 * unroll):
        raise ValueError("Activation output must have 16 or 32 lanes")
    if num_vecs < 1 or vec_length < 8 * unroll or vec_length % (8 * unroll):
        raise ValueError("Rows must be positive and width a positive multiple of 128")
    if vec_length * num_vecs > 131072:
        raise ValueError("Activation must fit in two 128-KiB GLB tiles")

    def module(graph):
        return graph["namespaces"]["global"]["modules"][graph["top"].split(".")[-1]]

    with TemporaryDirectory() as temporary:
        emit_reduction_sum_of_sqr_sqrt_recip_mul_elementwise_mul_add_bf16_design(
            unroll, vec_length, num_vecs, temporary,
        )
        source = module(json.loads((Path(temporary) / "design_top.json").read_text()))
        bypass = json.loads((Path(temporary) / "PE_fifos_bypass_config.json").read_text())
        emit_elementwise_mul_bf16_design(
            unroll, vec_length * num_vecs, temporary,
            mode="input_x_weight_broadcast", num_vecs=num_vecs,
            top_module="rms_norm_fp",
        )
        result = json.loads((Path(temporary) / "design_top.json").read_text())

    fused = module(result)
    instances, connections = fused["instances"], fused["connections"]
    scatter_outputs = output_unroll != unroll
    if scatter_outputs:
        for lane in range(output_unroll):
            _, _, output_port, _, _, output_io = _lane_port_names(
                "input_x_weight_broadcast", lane)
            if lane >= unroll:
                original = _lane_port_names("input_x_weight_broadcast", lane - unroll)[5]
                instances[output_io] = copy.deepcopy(instances[original])
                fused["type"][1].append([output_port, ["Array", 16, "Bit"]])
                connections.append([output_io + ".out", "self." + output_port])
            # Each reader emits a canonical stripe, so output DMA is sequential.
            instances[output_io]["metadata"]["in2glb_0"]["extent"] = [
                vec_length * num_vecs // output_unroll]
    internal = {name for name, inst in source["instances"].items()
                if inst.get("modref") != "global.IO"}
    for name in internal:
        instances["norm_" + name] = copy.deepcopy(source["instances"][name])
        if name.startswith("tile_input_lane_") or name.endswith("_filter_mem"):
            metadata = instances["norm_" + name]["metadata"]
            config = metadata["lake_rv_config"]
            if isinstance(config, str):
                config = json.loads(config)
            config["row_size"] = vec_length // (2 * unroll)
            metadata["lake_rv_config"] = config
    connections.extend([["norm_" + port for port in edge]
                        for edge in source["connections"]
                        if all(port.split(".")[0] in internal for port in edge)])

    for lane in range(unroll):
        activation = f"norm_tile_input_lane_{lane}"
        fifo = f"z_norm_affine_fifo_lane_{lane}"
        instances[fifo] = copy.deepcopy(instances[activation])
        instances[fifo]["metadata"]["lake_rv_config"].update(type="fifo", row_size=8)
        clock_edge = next(edge for edge in connections if activation + ".clk_en" in edge)
        connections.append([port.replace(activation + ".", fifo + ".") for port in clock_edge])
        input_io = next(name for name in instances
                        if name.startswith("io16in_input_host_") and f"_clkwrk_{lane}_" in name)
        connections.remove(next(edge for edge in connections
                                if input_io + ".out" in edge and f"pe_lane{lane}.data0" in edge))
        # Use the normalization template's output multiplier.
        output_io = next(name for name, inst in source["instances"].items()
                         if name.startswith("io16_") and f"_clkwrk_{lane}_" in name)
        output_edge = next(edge for edge in source["connections"] if output_io + ".in" in edge)
        normalized = next(port for port in output_edge if port != output_io + ".in")
        connections.extend([
            [input_io + ".out", activation + ".data_in_0"],
            ["norm_" + normalized, fifo + ".data_in_0"],
            [fifo + ".data_out_0", f"pe_lane{lane}.data0"],
        ])

        # Fast gamma PEs feed balancing Ponds. Give every Pond a deep output
        # FIFO so GLB lane skew cannot overflow its latency-only schedule.
        output_fifo = f"zz_affine_output_fifo_lane_{lane}"
        instances[output_fifo] = copy.deepcopy(instances[fifo])
        if scatter_outputs:
            # Reuse LayerNorm's 16-input scatter schedule: reader 0 selects
            # even elements and reader 1 odd elements of this compute lane.
            instances[output_fifo]["metadata"]["lake_rv_config"]["type"] = "get_filter_mem_two_streams"
        connections.append([port.replace(activation + ".", output_fifo + ".")
                            for port in clock_edge])
        final_io = next(name for name in instances
                        if name.startswith("io16_") and f"_clkwrk_{lane}_" in name)
        connections.remove(next(edge for edge in connections
                                if final_io + ".in" in edge and f"pe_lane{lane}.O0" in edge))
        connections.extend([
            [f"pe_lane{lane}.O0", output_fifo + ".data_in_0"],
            [output_fifo + ".data_out_0", final_io + ".in"],
        ])
        if scatter_outputs:
            upper_io = _lane_port_names("input_x_weight_broadcast", lane + unroll)[5]
            connections.append([output_fifo + ".data_out_1", upper_io + ".in"])

    # E64 packing consumes IO dictionary order; keep all 32 output lanes in
    # numeric order, as in the LayerNorm emitter.
    io_names = sorted(
        (name for name, inst in instances.items() if inst.get("modref") == "global.IO"),
        key=lambda name: (name.split("_clkwrk_")[0],
                          int(name.split("_clkwrk_")[1].split("_")[0])),
    )
    fused["instances"] = {name: inst for name, inst in instances.items()
                          if inst.get("modref") != "global.IO"}
    fused["instances"].update({name: instances[name] for name in io_names})

    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "design_top.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    (directory / "PE_fifos_bypass_config.json").write_text(
        json.dumps({"norm_" + name: value for name, value in bypass.items()}, indent=2) + "\n")
    return str(destination)
