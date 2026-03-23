"""
Shared helpers for building CoreIR graphs in strait coreir_backend templates.
"""

# Genargs for the cgralib Mem generator.
# is_rom=True is required to match the peak IR interface (10 input ports):
#   rst_n, clk_en, data_in_0, chain_data_in_0, data_in_1, chain_data_in_1,
#   wen_in_0, ren_in_0, addr_in_0, flush
# Whether a Mem instance behaves as ROM or MEM is controlled by metadata
# (is_rom, mode, init), not by these genargs.
_BASE_MEM_GENARGS = {
    "ID": "",
    "ctrl_width": 16,
    "has_chain_en": False,
    "has_external_addrgen": False,
    "has_flush": True,
    "has_read_valid": False,
    "has_reset": False,
    "has_stencil_valid": True,
    "has_valid": False,
    "is_rom": True,
    "num_inputs": 2,
    "num_outputs": 2,
    "use_prebuilt_mem": True,
    "width": 16,
}


def make_mem_genargs(context):
    """
    Return a coreir Values object for the cgralib Mem generator.

    The genargs always use is_rom=True to produce the 10-port interface
    expected by the MetaMapper peak IR. ROM vs MEM behavior is set via
    instance metadata (is_rom, mode, init) by the caller.

    Args:
        context: active coreir.Context

    Returns:
        coreir Values object suitable for add_generator_instance genargs.
    """
    return context.new_values(_BASE_MEM_GENARGS)
