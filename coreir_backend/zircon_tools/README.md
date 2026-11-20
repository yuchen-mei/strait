# Zircon kernel tools

Helpers for the Zircon strait templates (`../templates`). They are not part
of the compile flow: every kernel runs with a plain `aha regress`, whose
result check is the validation.

| Script | Purpose |
|---|---|
| `pb_plan.py` | Path-balancing analysis of a placed and routed app: fork-join latencies from `design.route`, the slack left on rigid joins and the FIFO Pond occupancy. It can also plan path-balance Ponds. |

## Path-balancing analysis

Run inside the container after `aha regress` of the app; it reads the app's
`bin/`:

```
python pb_plan.py /aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/<app>/bin --stream <tokens per lane> --no-balance
```

It reports the slack left on rigid fork-joins (a rate loss) and the largest
FIFO Pond occupancy, which must stay below 28 words. See the docstring for
the latency model, its rate-1 limitation, and how to insert planned Ponds.
