"""Shared GLB layout conversion for the 16-lane normalization templates."""

import copy

from ..elementwise_mul_bf16 import _lane_port_names


def input_merge_config(unroll, vec_length, num_vecs, input_unroll):
    """Validate and describe the existing MEM's four-word-block gather."""
    from lake.spec.hack_rv_mem_pond_bitstream import get_merge_dual_read_mem

    block = max(8, vec_length // (2 * unroll))
    if block % 8:
        block = vec_length // unroll
    stream_size = vec_length * num_vecs // input_unroll
    get_merge_dual_read_mem(stream_size, block)
    return {"type": "merge_dual_read",
            "single_input_stream_size": stream_size, "row_size": block}


def configure_norm_io_layout(module, unroll, vec_length, num_vecs,
                             input_unroll, output_unroll):
    """Match activation IO and coefficient DMA to the MEM compute order.

    A gathered lane emits four low-half words followed by four high-half
    words. Gamma/beta retain their canonical storage and 16-lane interface;
    their DMA visits indices 0,2,4,6,1,3,5,7 within each eight-step group.
    Scatter readers restore canonical stripes for sequential 32-lane stores.
    With 16 output lanes, output DMA performs that restoration instead.
    """
    instances, connections = module["instances"], module["connections"]

    def ports(lane):
        return _lane_port_names("input_x_weight_broadcast", lane)

    if input_unroll != unroll:
        for lane in range(input_unroll):
            input_port, _, _, name, _, _ = ports(lane)
            if lane >= unroll:
                instances[name] = copy.deepcopy(instances[ports(0)[3]])
                module["type"][1].append([input_port, ["Array", 16, "BitIn"]])
                connections.append(["self." + input_port, name + ".in"])
            instances[name]["metadata"]["glb2out_0"]["extent"] = [
                vec_length * num_vecs // input_unroll]

        per_row = vec_length // unroll
        # Strides are rollover deltas (SKIP_GLB_DMA_STRIDE_ADJUSTMENT=1).
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

    if output_unroll != unroll:
        for lane in range(output_unroll):
            _, _, output_port, _, _, name = ports(lane)
            if lane >= unroll:
                instances[name] = copy.deepcopy(instances[ports(0)[5]])
                module["type"][1].append([output_port, ["Array", 16, "Bit"]])
                connections.append([name + ".out", "self." + output_port])
            instances[name]["metadata"]["in2glb_0"].update({
                "dimensionality": 1,
                "extent": [vec_length * num_vecs // output_unroll],
                "cycle_stride": [1],
                "write_data_stride": [1],
            })
