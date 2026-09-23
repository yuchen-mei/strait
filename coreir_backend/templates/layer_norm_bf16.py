"""Single-invocation LayerNorm: center, normalize, and per-channel affine.

Compose the proven ready/valid reduction templates without intermediate GLB
stores. Two sets of dual-read CGRA memories retain x and x - mean respectively.
A third MEM set scatters affine results into the consumer's 32-lane GLB
layout; eight paired FIFO MEMs retain buffering before affine. With 16
output lanes, the third MEM set instead supplies those pre-affine FIFOs.
The compute graph has 16 lanes. A 32-lane activation layout can be consumed
directly from four GLB tiles: the existing mean-stage MEMs merge pairs of
lanes in four-word blocks and retain both reads. Reduction MEMs restore the
original partial-sum order; coefficient DMA follows the compute order. With
32 output lanes, affine-boundary MEMs split alternating blocks into canonical
output stripes. With 16 output lanes, output DMA restores channel order.
Neither mode needs an intermediate GLB transfer or a host tensor permutation.
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
    _input_io_name,
    _input_self_port,
    _output_io_name,
    _output_self_port,
)


def emit_layer_norm_bf16_design(unroll, vec_length, num_vecs, output_path,
                              input_unroll=None, output_unroll=None):
    input_unroll = unroll if input_unroll is None else input_unroll
    output_unroll = unroll if output_unroll is None else output_unroll
    if unroll != 16:
        raise ValueError("The single-pass compute graph requires 16 lanes")
    if input_unroll not in (unroll, 2 * unroll):
        raise ValueError("Activation input must have 16 or 32 lanes")
    if output_unroll not in (unroll, 2 * unroll):
        raise ValueError("Activation output must have 16 or 32 lanes")
    if num_vecs < 1 or vec_length < 8 * unroll or vec_length % (8 * unroll):
        raise ValueError("Rows must be positive and width a positive multiple of 128")
    if input_unroll != unroll:
        # Validate the configuration before constructing the graph. With four
        # word write aggregators, each interleave block holds eight-word pairs.
        from lake.spec.hack_rv_mem_pond_bitstream import get_merge_dual_read_mem
        layout_block = max(8, vec_length // (2 * unroll))
        if layout_block % 8:
            layout_block = vec_length // unroll
        get_merge_dual_read_mem(vec_length * num_vecs // input_unroll, layout_block)

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
    if input_unroll != unroll:
        # Gamma and beta stay at 16 lanes; activation IO has its own width.
        for lane in range(input_unroll):
            name = _input_io_name(lane)
            if lane >= unroll:
                instances[name] = copy.deepcopy(instances[_input_io_name(0)])
                fused["type"][1].append([_input_self_port(lane), ["Array", 16, "BitIn"]])
                connections.append(["self." + _input_self_port(lane), name + ".in"])
            instances[name]["metadata"]["glb2out_0"]["extent"] = [
                vec_length * num_vecs // input_unroll]
        # MEMs emit four low-half words followed by four high-half words.
        # Within each eight-step group the logical indices are 0,2,4,6,1,3,5,7.
        # DMA strides are rollover deltas (SKIP_GLB_DMA_STRIDE_ADJUSTMENT=1).
        # Coefficients stay in their original GLB layout; no host repacking.
        per_row = vec_length // unroll
        for name, instance in instances.items():
            if name.startswith(("io16in_weight_host_", "io16in_bias_host_")):
                instance["metadata"]["glb2out_0"].update({
                    "dimensionality": 4,
                    "extent": [4, 2, per_row // 8, num_vecs],
                    "cycle_stride": [1, 1, 1, 1],
                    "read_data_stride": [2, -5, 1, 1 - per_row],
                })
            elif name.startswith("io16_hw_output_"):
                instance["metadata"]["in2glb_0"].update({
                    "dimensionality": 3,
                    "extent": [4, 2, per_row * num_vecs // 8],
                    "cycle_stride": [1, 1, 1],
                    "write_data_stride": [2, -5, 1],
                })
    scatter_outputs = output_unroll != unroll
    if scatter_outputs:
        # Each reader emits a canonical 32-lane stripe, so output DMA is
        # sequential. All permutation stays inside the streaming MEMs.
        for lane in range(output_unroll):
            name = _output_io_name(lane)
            if lane >= unroll:
                instances[name] = copy.deepcopy(instances[_output_io_name(0)])
                fused["type"][1].append([_output_self_port(lane), ["Array", 16, "Bit"]])
                connections.append([name + ".out", "self." + _output_self_port(lane)])
            instances[name]["metadata"]["in2glb_0"].update({
                "dimensionality": 1,
                "extent": [vec_length * num_vecs // output_unroll],
                "cycle_stride": [1],
                "write_data_stride": [1],
            })
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
                if stage == "mean" and name.startswith("tile_input_lane_") and input_unroll != unroll:
                    config = {
                        "type": "merge_dual_read",
                        "single_input_stream_size": vec_length * num_vecs // input_unroll,
                        "row_size": layout_block,
                    }
                elif name.endswith("_filter_mem") and input_unroll != unroll:
                    config["type"] = "deinterleave_blocks"
                    config["row_size"] = layout_block
                metadata["lake_rv_config"] = config
        for edge in source["connections"]:
            if all(port.split(".")[0] in internal for port in edge):
                connections.append([stage + "_" + port for port in edge])

    # Reuse the original 16 MEMs for output scatter. Eight additional MEMs
    # each buffer two independent normalized lanes before affine.
    # Existing affine balancing Ponds are inserted by the global mapping pass.
    for lane in range(unroll):
        fifo = (f"z_output_scatter_lane_{lane}" if scatter_outputs
                else f"z_norm_affine_fifo_lane_{lane}")
        instances[fifo] = copy.deepcopy(instances[f"norm_tile_input_lane_{lane}"])
        config = instances[fifo]["metadata"]["lake_rv_config"]
        config["type"] = "fifo"
        if scatter_outputs:
            config["type"] = ("deinterleave_blocks" if input_unroll != unroll
                              else "get_filter_mem_two_streams")
        # Release data in short blocks instead of waiting for half a row.
        config["row_size"] = 8
        clock_edge = next(edge for edge in connections
                          if f"norm_tile_input_lane_{lane}.clk_en" in edge)
        connections.append([port.replace(f"norm_tile_input_lane_{lane}.", fifo + ".")
                            for port in clock_edge])
        if scatter_outputs:
            direct_output = {f"add_vec_pe_{lane}.O0", _output_io_name(lane) + ".in"}
            connections.remove(next(edge for edge in connections if set(edge) == direct_output))
            connections.extend([
                [f"add_vec_pe_{lane}.O0", fifo + ".data_in_0"],
                [fifo + ".data_out_0", _output_io_name(lane) + ".in"],
                [fifo + ".data_out_1", _output_io_name(lane + unroll) + ".in"],
            ])
        else:
            connections.append([fifo + ".data_out_0", f"mul_vec_pe_{lane}.data0"])

    if scatter_outputs:
        for pair in range(unroll // 2):
            # Sort after scatter MEMs to preserve the existing placement IDs.
            fifo = f"zz_norm_affine_pair_fifo_{pair}"
            source = f"norm_tile_input_lane_{2 * pair}"
            instances[fifo] = copy.deepcopy(instances[source])
            instances[fifo]["metadata"]["lake_rv_config"].update({
                "type": "paired_fifo", "row_size": 8,
            })
            clock_edge = next(edge for edge in connections if source + ".clk_en" in edge)
            connections.append([port.replace(source + ".", fifo + ".") for port in clock_edge])
            for port in (0, 1):
                connections.append([fifo + f".data_out_{port}",
                                    f"mul_vec_pe_{2 * pair + port}.data0"])

    for lane in range(unroll):
        input_io = next(name for name in instances
                        if name.startswith("io16in_input_host_") and f"_clkwrk_{lane}_" in name)
        connections.extend([
            [input_io + ".out", f"mean_tile_input_lane_{lane}.data_in_0"],
            [f"mean_elementwise_add_pe_{lane}.O0", f"norm_tile_input_lane_{lane}.data_in_0"],
            [f"norm_elementwise_mul_pe_{lane}.O0",
             f"zz_norm_affine_pair_fifo_{lane // 2}.data_in_{lane % 2}" if scatter_outputs
             else f"z_norm_affine_fifo_lane_{lane}.data_in_0"],
        ])
        if input_unroll != unroll:
            connections.append([_input_io_name(lane + unroll) + ".out",
                                f"mean_tile_input_lane_{lane}.data_in_1"])

    # The metadata parser consumes IO dictionary order when packing four
    # lanes into E64 banks. Emit every tensor in numeric lane order, including
    # the newly added activation lanes, even when called outside the Halide
    # flow's global IO-sorting pass.
    io_names = sorted(
        (name for name, inst in instances.items() if inst.get("modref") == "global.IO"),
        key=lambda name: (name.split("_clkwrk_")[0],
                          int(name.split("_clkwrk_")[1].split("_")[0])),
    )
    fused["instances"] = {name: inst for name, inst in instances.items()
                          if inst.get("modref") != "global.IO"}
    fused["instances"].update({name: instances[name] for name in io_names})

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
