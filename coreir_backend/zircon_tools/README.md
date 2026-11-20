# Zircon kernel tools

Helpers for the Zircon strait templates (`../templates`). They are not part
of the compile flow: every kernel runs with a plain `aha regress`, and these
scripts only check or analyze a finished run.

| Script | Purpose |
|---|---|
| `kernel_models.py` | Bit-accurate models of the Nemotron kernels. They redo every PE operation in the kernel's order with the Zircon PE arithmetic and compare the RTL output bit for bit. They also print the error against an FP64 reference. |
| `zircon_pe_model.py` | Zircon PE arithmetic in numpy: BF16 `fp_add` (RNE), `fp_mul` as `CW_fp_mult` with 3 extra bits truncated then rounded, `fp_getffrac`/`fp_getfint`/`fp_addiexp`/`fp_getmant`/`fp_subexp`. |
| `softplus_model.py` | The softplus PE chain, used by `kernel_models.py`. |
| `pb_plan.py` | Path-balancing analysis of a placed and routed app: fork-join latencies from `design.route`, the slack left on rigid joins and the FIFO Pond occupancy. It can also plan path-balance Ponds. |

## Bit-exact check after a regression

Run inside the container after `aha regress` of the app; the model reads
the app's `bin/`:

```
cd /aha/strait/coreir_backend/zircon_tools
python kernel_models.py colsum   /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/col_prefix_sum_fp
python kernel_models.py cumsum   /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/row_prefix_sum_fp
python kernel_models.py conv1d   /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/depthwise_conv1d_silu_fp
python kernel_models.py softplus /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/add_softplus_fp
python kernel_models.py exact    /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/fused_e8m0_quant_single_IO
```

`exact` compares with the `process.cpp` gold through the simulator's output
in `garnet/tests/test_app`, so run it right after that app's regression.
The exit status is 1 if any output differs.

## Path-balancing analysis

```
python pb_plan.py /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/<app>/bin --stream <tokens per lane> --no-balance
```

It reports the slack left on rigid fork-joins (a rate loss) and the largest
FIFO Pond occupancy, which must stay below 28 words. See the docstring for
the latency model, its rate-1 limitation, and how to insert planned Ponds.
