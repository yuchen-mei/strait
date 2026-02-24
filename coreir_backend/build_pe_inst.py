#!/usr/bin/env python3
import sys, os
import json
from pathlib import Path

from peak.mapper import read_serialized_bindings
from peak.mapper.mapper import _create_path_to_adt
from peak import family as peak_family

from lassen.sim import PE_fc as lassen_fc


def _default_for_type(t, fam):
    '''
    Placeholder default values for rewrite-rule inputs.
    '''
    if hasattr(t, "size"):
        return fam.BitVector[t.size](0)
    if t is fam.Bit:
        return fam.Bit(0)
    return fam.BitVector[16](0)


def pe_inst_to_bits(op_name: str, repo_root: Path):
    '''
    Get 84-bit instruction for a single-PE op from its rewrite_rule json.
    '''
    rewrite_rules_path = os.path.join(repo_root, "lassen", "lassen", "rewrite_rules", f"{op_name}.json")
    if not os.path.isfile(rewrite_rules_path):
        raise ValueError(f"Unknown PE op: {op_name} (no rewrite rule at {rewrite_rules_path})")

    peak_eq = __import__(f"lassen.rewrite_rules.{op_name}", fromlist=[op_name])
    ir_fc = getattr(peak_eq, op_name + "_fc")

    with open(rewrite_rules_path, "r") as f:
        serialized = json.load(f)

    rewrite_rules = read_serialized_bindings(serialized, ir_fc, lassen_fc)
    fam = peak_family.PyFamily()

    ir_path_types = _create_path_to_adt(rewrite_rules.ir_fc(fam).input_t)
    arch_path_types = _create_path_to_adt(rewrite_rules.arch_fc(fam).input_t)
    ir_paths, arch_paths = rewrite_rules.get_input_paths()

    ir_vals = {p: _default_for_type(ir_path_types[p], fam) for p in ir_paths}
    arch_vals = {p: _default_for_type(arch_path_types[p], fam) for p in arch_paths}

    _, arch_inputs = rewrite_rules.build_inputs(ir_vals, arch_vals, fam)
    inst_val = arch_inputs["inst"]
    bv = inst_val._value_

    return int(bv), len(bv)


def main():
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


if __name__ == "__main__":
    main()
