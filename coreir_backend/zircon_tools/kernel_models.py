"""Bit-accurate models of the Zircon Nemotron kernels, compared bit for bit with an RTL run.

Run after `aha regress` of the app (it reads the app's bin/ inputs and the
simulated output the regression saved):

    python kernel_models.py softplus <app_dir>   # add_softplus_fp
    python kernel_models.py conv1d   <app_dir>   # depthwise_conv1d_silu_fp
    python kernel_models.py cumsum   <app_dir>   # row_prefix_sum_fp  (cumsum dim -1)
    python kernel_models.py colsum   <app_dir>   # col_prefix_sum_fp  (cumsum dim -2)
    python kernel_models.py exact    <app_dir>   # fused_e8m0_quant_single_IO (vs process.cpp gold)

<app_dir> is e.g. /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/col_prefix_sum_fp.
Shapes come from bin/halide_gen_args. The models redo every PE operation in
the kernel's order with the Zircon PE arithmetic (zircon_pe_model) and the
template's own ROM tables; they also print the error against an FP64
reference. Exit status 1 if any output differs.
"""

import math
import sys
from pathlib import Path

import numpy as np

import zircon_pe_model as M
import softplus_model

AHA_ROOT = Path(__file__).resolve().parents[3]  # .../strait/coreir_backend/zircon_tools
sys.path.insert(0, str(AHA_ROOT))


def gen_args(app_dir):
    """The app's HALIDE_GEN_ARGS as a dict (bin/halide_gen_args)."""
    text = (Path(app_dir) / "bin/halide_gen_args").read_text()
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def load_raw(path):
    return np.fromfile(path, dtype=">u2").astype(np.uint16)


def brent_kung(lanes):
    """lanes: list of uint16 arrays (one per lane). Returns inclusive prefix lanes."""
    n = len(lanes)
    v = list(lanes)
    levels = int(math.log2(n))
    for d in range(levels):
        step = 1 << (d + 1)
        for i in range(step - 1, n, step):
            v[i] = M.fp_add(v[i - step // 2], v[i])
    for d in range(levels - 2, -1, -1):
        step = 1 << (d + 1)
        for i in range(step - 1 + step // 2, n, step):
            v[i] = M.fp_add(v[i - step // 2], v[i])
    return v


def cumsum_model(x, unroll):
    """x: [rows, length] uint16. Same additions, in the same order, as row_prefix_sum_bf16."""
    rows, length = x.shape
    groups = length // unroll
    xg = x.reshape(rows, groups, unroll)
    local = brent_kung([xg[:, :, i] for i in range(unroll)])  # each [rows, groups]
    total = local[unroll - 1]
    zero = np.zeros(rows, dtype=np.uint16)
    # carry of group g: shift d gives total[g - d], or 0 for g < d; shift 1
    # passes a nop PE, and the shifts are added in the order d = 2, 3, ...
    shifted = lambda g, d: total[:, g - d] if g >= d else zero
    out = np.zeros_like(xg)
    for g in range(groups):
        carry = shifted(g, 1)
        for d in range(2, groups):
            carry = M.fp_add(shifted(g, d), carry)
        for i in range(unroll):
            out[:, g, i] = M.fp_add(local[i][:, g], carry)
    return out.reshape(rows, length)


def conv1d_silu_model(x, w, b, T):
    """x: [length, channels], w: [channels, 4], b: [channels] uint16; T: the template module."""
    length, channels = x.shape
    zero = np.zeros_like(x)
    def delayed(d):
        out = zero.copy()
        out[d:] = x[: length - d]
        return out
    # Transposed form: product k is w_k * x[t], delayed by 3 - k positions with +0 before t = 0.
    def delay(v, d):
        out = np.zeros_like(v)
        out[d:] = v[: length - d]
        return out
    m = [delay(M.fp_mul(x, np.broadcast_to(w[:, k], x.shape)), 3 - k) for k in range(4)]
    # ((tap3 + tap2) + (tap1 + tap0)) + bias, as in the template
    y = M.fp_add(M.fp_add(M.fp_add(m[3], m[2]), M.fp_add(m[1], m[0])), np.broadcast_to(b, x.shape))
    c = lambda v: np.uint16(T.bf16_bits_rne(v))
    n = M.fp_mul(y, c(-1.0 / math.log(2.0)))
    e = M.fp_addiexp(M.rom(np.array(T.exp_rom_table(), dtype=np.uint16), M.fp_getffrac(n)), M.fp_getfint(n))
    d = M.fp_add(e, c(1.0))
    inv = M.fp_subexp(M.rom(np.array(T.div_rom_table(), dtype=np.uint16), M.fp_getmant(d)), d)
    return y, M.fp_mul(inv, y)


def compare(name, model_bits, sim_bits, ref_float=None):
    model_bits = np.asarray(model_bits, dtype=np.uint16).ravel()
    sim_bits = np.asarray(sim_bits, dtype=np.uint16).ravel()
    assert model_bits.shape == sim_bits.shape, (model_bits.shape, sim_bits.shape)
    mism = np.nonzero(model_bits != sim_bits)[0]
    print(f"[INFO] {name}: {len(sim_bits)} outputs, {len(mism)} differ from the bit-accurate model")
    for i in mism[:10]:
        print(f"    index {i}: model {model_bits[i]:04x} ({M.b2f(model_bits[i]):.6g}) sim {sim_bits[i]:04x} ({M.b2f(sim_bits[i]):.6g})")
    if ref_float is not None:
        ref = np.asarray(ref_float, dtype=np.float64).ravel()
        sim = M.b2f(sim_bits)
        nz = ref != 0
        rel = np.abs(sim[nz] - ref[nz]) / np.abs(ref[nz])
        print(f"[INFO] {name}: relative error vs fp64 reference: max {rel.max()*100:.3f}%, mean {rel.mean()*100:.3f}%")
    return len(mism)


def sim_bits_from_npy(app_dir, name="hw_output"):
    sim = np.load(Path(app_dir) / "bin" / f"sim_{name}_array_fp.npy").astype(np.float32)
    return (sim.view(np.uint32) >> 16).astype(np.uint16)


if __name__ == "__main__":
    kind, app = sys.argv[1], Path(sys.argv[2])
    if kind == "softplus":
        x = load_raw(app / "bin/input_host_stencil.raw")
        bias = load_raw(app / "bin/bias_host_stencil.raw")
        rows = len(bias)
        x = x.reshape(rows, -1)
        b = np.repeat(bias[:, None], x.shape[1], axis=1)
        # Use the template's own ROM tables and constants, so the model checks
        # the PE chain rather than a second copy of the table generation.
        import strait.coreir_backend.templates.elementwise_broadcast_add_softplus_bf16 as T
        softplus_model.exp_rom_table = lambda: np.array(T.exp_rom_table(), dtype=np.uint16)
        softplus_model.ln_rom_table = lambda: np.array(T.ln_rom_table(), dtype=np.uint16)
        softplus_model.C["ln2_inv"] = np.uint16(T.bf16_bits_rne(1.0 / math.log(2.0)))
        model = softplus_model.softplus_chain(x, b)
        s = M.b2f(x) + M.b2f(b)
        ref = np.maximum(s, 0) + np.log1p(np.exp(-np.abs(s)))
        bad = compare("softplus", model, sim_bits_from_npy(app), ref)
    elif kind == "cumsum":
        g = gen_args(app)
        unroll = int(sys.argv[3]) if len(sys.argv) > 3 else int(g["glb_i"])
        x = load_raw(app / "bin/input_host_stencil.raw")
        rows = int(sys.argv[4]) if len(sys.argv) > 4 else int(g["vec_height"])
        length = x.size // rows
        x = x.reshape(rows, length)
        model = cumsum_model(x, unroll)
        ref = np.cumsum(M.b2f(x), axis=1)
        bad = compare("cumsum", model, sim_bits_from_npy(app), ref)
    elif kind == "colsum":
        # cumsum along dim -2: rows per matrix and columns from the gen args
        g = gen_args(app)
        scan_rows = int(sys.argv[3]) if len(sys.argv) > 3 else int(g["scan_rows"])
        cols = int(sys.argv[4]) if len(sys.argv) > 4 else int(g["vec_width"])
        x = load_raw(app / "bin/input_host_stencil.raw").reshape(-1, cols)
        model = np.zeros_like(x)
        for b in range(x.shape[0] // scan_rows):
            acc = np.zeros(cols, dtype=np.uint16)  # the cleared Pond: +0
            for r in range(scan_rows):
                acc = M.fp_add(acc, x[b * scan_rows + r])  # data0 = sum one row up, data1 = input
                model[b * scan_rows + r] = acc
        xf = M.b2f(x).astype(np.float64).reshape(-1, scan_rows, cols)
        ref = np.cumsum(xf, axis=1).reshape(-1, cols)
        bad = compare("colsum", model.reshape(-1), sim_bits_from_npy(app), ref.reshape(-1))
    elif kind == "conv1d":
        import strait.coreir_backend.templates.depthwise_conv1d_silu_bf16 as T
        b = load_raw(app / "bin/bias_host_stencil.raw")
        channels = len(b)
        w = load_raw(app / "bin/weight_host_stencil.raw").reshape(channels, 4)
        x = load_raw(app / "bin/input_host_stencil.raw").reshape(-1, channels)
        y, model = conv1d_silu_model(x, w, b, T)
        xf, wf, bf = M.b2f(x), M.b2f(w), M.b2f(b)
        yr = np.broadcast_to(bf, xf.shape).copy()
        for k in range(4):
            sh = np.zeros_like(xf)
            sh[3 - k:] = xf[: xf.shape[0] - (3 - k)]
            yr += sh * wf[:, k]
        ref = yr / (1.0 + np.exp(-yr))
        bad = compare("conv1d_silu", model, sim_bits_from_npy(app), ref)
    elif kind == "exact":
        # Every output of the app, bit for bit against the gold raw file written by process.cpp.
        import json
        from aha.util.test import _load_hex_txt
        meta = json.loads((app / "bin/design_meta.json").read_text())
        bad = 0
        for out in meta["IOs"]["outputs"]:
            name = Path(out["datafile"]).stem
            gold = load_raw(app / "bin" / out["datafile"])
            sim = np.asarray(_load_hex_txt(f"{AHA_ROOT}/garnet/tests/test_app/{name}.txt", block_index=0), dtype=np.uint16)
            n = min(len(gold), len(sim))
            if len(gold) != len(sim):
                print(f"[WARNING] {name}: gold has {len(gold)} words, sim {len(sim)}")
            mism = np.nonzero(gold[:n] != sim[:n])[0]
            print(f"[INFO] {name}: {n} words, {len(mism)} differ from gold")
            for i in mism[:8]:
                print(f"    index {i}: gold {gold[i]:04x} sim {sim[i]:04x}")
            bad += len(mism) + abs(len(gold) - len(sim))
    sys.exit(1 if bad else 0)
