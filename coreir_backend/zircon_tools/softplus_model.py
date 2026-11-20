"""Bit-accurate model of the Zircon bias-add + softplus PE chain, and its accuracy."""

import math
import sys

import numpy as np

import zircon_pe_model as M


def exp_rom_table():
    return M.f2b([2.0 ** (i / 128.0) for i in range(128)] + [2.0 ** (i / 128.0) for i in range(-128, 0)])


def ln_rom_table():
    """ln ROM indexed by the low byte of v = 1 + e, with v in (1, 2].

    v in [1, 2) has biased exponent 127, so its low byte is 0x80 | mantissa and
    entries 128..255 hold ln(1 + m / 128). v = 2.0 (0x4000) has low byte 0, so
    entry 0 holds ln(2). Entries 1..127 are never addressed.
    """
    vals = [-0.0] * 256
    vals[0] = math.log(2.0)
    for m in range(1, 128):
        vals[128 + m] = math.log1p(m / 128.0)
    return M.f2b(vals)


C = {k: M.f2b(v) for k, v in dict(zero=0.0, one=1.0, ln2_inv=1.0 / math.log(2.0)).items()}


def softplus_chain(x, bias):
    s = M.fp_add(x, bias)
    t = M.fp_max(s, C["zero"])
    u = M.or_(s, 0x8000)
    z = M.fp_mul(u, C["ln2_inv"])
    frac = M.fp_getffrac(z)
    n = M.fp_getfint(z)
    e = M.fp_addiexp(M.rom(exp_rom_table(), frac), n)
    v = M.fp_add(e, C["one"])
    w = M.fp_sub(v, C["one"])
    d = M.fp_sub(e, w)
    l = M.rom(ln_rom_table(), M.and_(v, 0x00FF))
    l1 = M.fp_add(l, d)
    return M.fp_add(t, l1)


if __name__ == "__main__":
    s = np.linspace(-20, 20, 40001)
    sb = M.f2b(s)
    zero = np.zeros_like(sb)
    out = M.b2f(softplus_chain(sb, zero))
    ref = np.log1p(np.exp(-np.abs(M.b2f(sb)))) + np.maximum(M.b2f(sb), 0)
    rel = np.abs(out - ref) / ref
    for lo, hi in [(-20, -12), (-12, -5), (-5, -1), (-1, 1), (1, 5), (5, 20)]:
        m = (M.b2f(sb) >= lo) & (M.b2f(sb) < hi)
        print(f"s in [{lo:4},{hi:4}): max rel err {rel[m].max()*100:6.3f}%  mean {rel[m].mean()*100:6.3f}%")
    # bf16 rounding of the exact result alone
    rb = np.abs(M.b2f(M.f2b(ref)) - ref) / ref
    print(f"bf16 rounding floor: max {rb.max()*100:.3f}%")
    print("regression tolerance pass:", np.all(np.abs(out - ref) <= 1.5e-4 + 0.2 * np.abs(ref)))
