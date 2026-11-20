"""Vectorized bit-level model of the Zircon PE ops used by the Nemotron kernels.

All values are uint16 numpy arrays holding BF16 (or integer) bit patterns.
Float ops round the exact result to BF16 with round-to-nearest-even; the
custom ops follow lassen/float/float_custom.py line by line.
"""

import numpy as np

U16 = np.uint16


def f2b(x):
    """float -> BF16 bits: round to nearest even, then flush results below the
    normal range to a signed zero (the Zircon FPU has no denormals)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x) & (x != 0)
    ax = np.abs(np.where(finite, x, 1.0))
    e = np.floor(np.log2(ax))
    # guard against log2 rounding at exact powers of two
    e = np.where(2.0 ** (e + 1) <= ax, e + 1, e)
    e = np.where(2.0**e > ax, e - 1, e)
    q = 2.0 ** (e - 7)
    r = np.rint(ax / q) * q
    r = np.where(r < 2.0**-126, 0.0, r)
    r = np.where(finite, np.copysign(r, x), x)
    with np.errstate(over="ignore"):
        bits32 = r.astype(np.float32).view(np.uint32)
    out = np.atleast_1d((bits32 >> 16).astype(np.uint16))
    out[np.atleast_1d(np.isnan(x))] = 0x7FC0
    return out.reshape(x.shape)


def b2f(b):
    """BF16 bits -> float, reading denormal inputs as a signed zero."""
    b = np.asarray(b, dtype=np.uint16)
    shape = b.shape
    b = np.atleast_1d(np.where((b & 0x7F80) == 0, b & 0x8000, b)).astype(np.uint32) << 16
    return b.view(np.float32).astype(np.float64).reshape(shape)


def fp_add(a, b):
    return f2b(b2f(a) + b2f(b))


def fp_sub(a, b):
    return f2b(b2f(a) - b2f(b))


def fp_mul(a, b):
    """BF16 multiply as in garnet_PE.v PEGEN_mul_hack.

    CW_fp_mult forms the product with 3 extra fraction bits, rounded toward
    zero; the wrapper then rounds to nearest even using only those 3 bits
    (guard g, round r, bit x0): it rounds up when g & (r | x0) or when g and
    the result LSB are set. Bits below x0 are lost, so a product just above a
    halfway point can round down. Normal operands only; zero gives zero and
    results outside the normal range are flushed or saturated.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    sign = (a ^ b) & 0x8000
    ea, eb = _exp(a), _exp(b)
    sa = (a & 0x7F) | 0x80
    sb = (b & 0x7F) | 0x80
    prod = sa * sb  # 16-bit significand product in [2^14, 2^16)
    top = prod >= (1 << 15)
    shift = np.where(top, 8, 7)
    e = ea + eb - 127 + top.astype(np.int64)
    kept = prod >> shift  # 8 bits including the hidden one
    g = (prod >> (shift - 1)) & 1
    r = (prod >> (shift - 2)) & 1
    x0 = (prod >> (shift - 3)) & 1
    up = (g & (r | x0)) | (kept & 1 & g)
    kept = kept + up
    carry = kept >= (1 << 8)
    kept = np.where(carry, kept >> 1, kept)
    e = e + carry.astype(np.int64)
    out = sign | (np.clip(e, 0, 255) << 7) | (kept & 0x7F)
    zero = (ea == 0) | (eb == 0) | (e <= 0)
    out = np.where(zero, sign, out)
    out = np.where(e >= 255, sign | 0x7F80, out)
    return out.astype(U16)


def fp_max(a, b):
    """lassen FP_max: a - b computed in BF16; if the result is negative take b."""
    a = np.asarray(a, dtype=U16)
    b = np.asarray(b, dtype=U16)
    diff = fp_sub(a, b)
    return np.where((diff & 0x8000) != 0, b, a).astype(U16)


def fp_abs_max(a, b):
    a = np.asarray(a, dtype=U16) & 0x7FFF
    b = np.asarray(b, dtype=U16) & 0x7FFF
    diff = fp_sub(a, b)
    return np.where((diff & 0x8000) != 0, b, a).astype(U16)


def or_(a, b):
    return (np.asarray(a, dtype=U16) | np.asarray(b, dtype=U16)).astype(U16)


def and_(a, b):
    return (np.asarray(a, dtype=U16) & np.asarray(b, dtype=U16)).astype(U16)


def _exp(a):
    return ((np.asarray(a, dtype=np.int64) >> 7) & 0xFF)


def fp_getmant(a):
    return (np.asarray(a, dtype=U16) & 0x7F).astype(U16)


def fp_addiexp(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    sign = a & 0x8000
    exp = (_exp(a) + (b & 0xFF)) & 0xFF
    return (sign | (exp << 7) | (a & 0x7F)).astype(U16)


def fp_subexp(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    exp = (_exp(a) - _exp(b) + 127) & 0xFF
    return ((a & 0x8000) | (b & 0x8000) | (exp << 7) | (a & 0x7F)).astype(U16)


def fp_getfint(a):
    a = np.asarray(a, dtype=np.int64)
    mant = (a & 0x7F) | 0x80
    ue = _exp(a) - 127
    shifted = np.where((ue < 0) | (ue >= 23), 0, (mant << np.clip(ue, 0, 22)) & ((1 << 23) - 1))
    res = (shifted >> 7) & 0xFFFF
    res = np.where((a & 0x8000) != 0, (-res) & 0xFFFF, res)
    return res.astype(U16)


def fp_getffrac(a):
    a = np.asarray(a, dtype=np.int64)
    mant = (a & 0x7F) | 0x80
    ue = _exp(a) - 127
    shifted = np.where(ue < 0, mant >> np.clip(-ue, 0, 15), (mant << np.clip(ue, 0, 15)) & 0xFFFF)
    res = shifted & 0x7F
    res = np.where((a & 0x8000) != 0, (-res) & 0xFFFF, res)
    return (res & 0xFF).astype(U16)


def fp_cnvexp2f(a):
    """Unbiased exponent of a (biased exponent minus 127), as a BF16 value."""
    a = np.asarray(a, dtype=np.int64)
    return f2b((_exp(a) - 127).astype(np.float64))


def get_shared_exp(a):
    e = _exp(a)
    return np.where(e == 0, 127, (e - 6) & 0xFF).astype(U16)


def e8m0_quant(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    mant = (a & 0x7F) | 0x80
    q = _exp(a) - (b & 0xFF)
    shifted = np.where(q < 0, mant >> np.clip(-q, 0, 22), (mant << np.clip(q, 0, 22)) & ((1 << 23) - 1))
    rnd = (shifted >> 6) & 1
    u = ((shifted >> 7) + rnd) & 0xFF
    s = np.where((a & 0x8000) != 0, (-u) & 0xFF, u)
    return s.astype(U16)


def bit8_pack(a, b):
    return ((((np.asarray(a, dtype=np.int64) & 0xFF) << 8) | (np.asarray(b, dtype=np.int64) & 0xFF))).astype(U16)


def rom(table, idx):
    table = np.asarray(table, dtype=U16)
    return table[np.asarray(idx, dtype=np.int64) % len(table)]
