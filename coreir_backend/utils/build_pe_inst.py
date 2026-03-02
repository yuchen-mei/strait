#!/usr/bin/env python3
import sys, os
import json
import struct
from pathlib import Path

from hwtypes import BitVector, Bit
from peak.assembler import Assembler
from peak.mapper import read_serialized_bindings
from peak.mapper.mapper import _create_path_to_adt
from peak import family as peak_family

from lassen.sim import PE_fc as lassen_fc
from lassen.mode import Mode_t


def _default_for_type(t, fam):
    '''
    Placeholder default values for rewrite_rule inputs.
    '''
    if hasattr(t, "size"):
        return fam.BitVector[t.size](0)
    if t is fam.Bit:
        return fam.Bit(0)
    return fam.BitVector[16](0)


def _base_inst_bitvector(op_name: str, repo_root: Path = Path("/aha"), fam=None):
    """
    Get the base PE instruction as a BitVector, using the rewrite-rule mapping.
    This corresponds to the unconstrained single-PE op (all operands defaulted).
    """
    rewrite_rules_path = os.path.join(
        repo_root, "lassen", "lassen", "rewrite_rules", f"{op_name}.json"
    )
    if not os.path.isfile(rewrite_rules_path):
        raise ValueError(
            f"Unknown PE op: {op_name} (no rewrite rule at {rewrite_rules_path})"
        )

    peak_eq = __import__(f"lassen.rewrite_rules.{op_name}", fromlist=[op_name])
    ir_fc = getattr(peak_eq, op_name + "_fc")

    with open(rewrite_rules_path, "r") as f:
        serialized = json.load(f)

    if fam is None:
        fam = peak_family.PyFamily()

    rewrite_rules = read_serialized_bindings(serialized, ir_fc, lassen_fc)

    ir_path_types = _create_path_to_adt(rewrite_rules.ir_fc(fam).input_t)
    arch_path_types = _create_path_to_adt(rewrite_rules.arch_fc(fam).input_t)
    ir_paths, arch_paths = rewrite_rules.get_input_paths()

    ir_vals = {p: _default_for_type(ir_path_types[p], fam) for p in ir_paths}
    arch_vals = {p: _default_for_type(arch_path_types[p], fam) for p in arch_paths}

    _, arch_inputs = rewrite_rules.build_inputs(ir_vals, arch_vals, fam)
    inst_val = arch_inputs["inst"]  # AssembledADT[Inst, Assembler, BitVector]
    bv = inst_val._value_           # BitVector[84]
    return bv, fam


def bf16_bits_from_float(x: float) -> int:
    """
    Return the 16-bit bfloat16 bit pattern for a Python float
    """
    packed = struct.pack("f", x)
    u32 = struct.unpack("I", packed)[0]
    return u32 >> 16


def pe_inst_to_bits(op_name: str, repo_root: Path = Path("/aha")):
    '''
    Get 84-bit instruction for a single-PE op from its rewrite_rule json.
    This returns the "base" instruction with all operands using default
    modes/values from the rewrite rule.
    '''
    bv, _ = _base_inst_bitvector(op_name, repo_root)
    return int(bv), len(bv)


def pe_inst_to_bits_with_operands(
    op_name: str,
    repo_root: Path = Path("/aha"),
    *,
    data0=None,
    data1=None,
    data2=None,
    bit0=None,
    bit1=None,
    bit2=None,
):
    """
    Get 84-bit instruction for a single-PE op, specialized for operand sources.

    Each operand argument can be:
      - None: leave whatever the rewrite rule produced
      - ("ext", None): operand comes from the external PE port (bypass mode)
      - ("const", value): operand is a constant encoded in the instruction

    data* constants are 16-bit values, bit* constants are 0/1.
    """
    # Build base instruction bits and family
    bv, fam = _base_inst_bitvector(op_name, repo_root)

    # Disassemble to Inst so we can patch modes/consts
    PE = lassen_fc(fam)
    inst_type = PE.input_t.field_dict["inst"]
    assembler = Assembler(inst_type)
    inst = assembler.disassemble(bv)

    def _patch_data(reg_field, data_field, src):
        if src is None:
            return
        kind, val = src
        if kind == "ext":
            setattr(inst, reg_field, Mode_t.BYPASS)
        elif kind == "const":
            setattr(inst, reg_field, Mode_t.CONST)
            setattr(inst, data_field, BitVector[16](val))
        else:
            raise ValueError(f"Unknown data src kind {kind} for {reg_field}")

    def _patch_bit(reg_field, bit_field, src):
        if src is None:
            return
        kind, val = src
        if kind == "ext":
            setattr(inst, reg_field, Mode_t.BYPASS)
        elif kind == "const":
            setattr(inst, reg_field, Mode_t.CONST)
            setattr(inst, bit_field, Bit(bool(val)))
        else:
            raise ValueError(f"Unknown bit src kind {kind} for {reg_field}")

    _patch_data("rega", "data0", data0)
    _patch_data("regb", "data1", data1)
    _patch_data("regc", "data2", data2)
    _patch_bit("regd", "bit0", bit0)
    _patch_bit("rege", "bit1", bit1)
    _patch_bit("regf", "bit2", bit2)

    final_bv = assembler.assemble(inst)
    return int(final_bv), len(final_bv)


def main():
    # Test the base instruction
    if len(sys.argv) < 2:
        print("Usage: build_inst.py OP_NAME [REPO_ROOT]")
        print("  Example: build_inst.py e8m0_quant /aha")
        sys.exit(1)

    op_name = sys.argv[1]
    repo_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("/aha")

    config_int, width = pe_inst_to_bits(op_name, repo_root)

    print(f"Op: {op_name}")
    print(f"Width: {width} bits")
    hex_str = f"{width}'h{config_int:0{width // 4}x}"
    print(f"Instruction hex: {hex_str}")

    # Test the instruction with operands (integer mul)
    cfg, width = pe_inst_to_bits_with_operands(
        "mul",
        Path("/aha"),
        data0=("ext", None),
        data1=("const", 2),
    )
    print(f"Op: mul with operands: data0=ext, data1=const(2)")
    print(f"Width: {width} bits")
    hex_str = f"{width}'h{cfg:0{width // 4}x}"
    print(f"Instruction hex: {hex_str}")

    # Example: fp_mul with bf16(2.0) constant
    cfg_bf16, width = pe_inst_to_bits_with_operands(
        "fp_mul",
        Path("/aha"),
        data0=("ext", None),
        data1=("const", bf16_bits_from_float(2.0)),
    )
    print(f"Op: fp_mul with operands: data0=ext, data1=const(bf16(2.0))")
    print(f"Width: {width} bits")
    hex_str = f"{width}'h{cfg_bf16:0{width // 4}x}"
    print(f"Instruction hex: {hex_str}")


if __name__ == "__main__":
    main()
