# Stage-2 Deferred Plot Rendering Evidence

Source commit: `78cfec9`

Optimization:
- Stage-2 online curve persistence now defers PNG/PDF rendering by default.
- Required NPZ curve data is still written on every persistence refresh.
- Explicit `render_plots=True` and `RFR_STAGE2_RENDER_PLOTS=1` retain live
  rendering; the offline regeneration command already requests rendering
  explicitly.

Server:
- Host: `f1ac06029e4a`
- GPU inventory: one NVIDIA GeForce RTX 4090 with 24564 MiB
- CPU inventory: 20 logical CPUs
- Python: `3.10.19` from the `llm_ist` environment
- Source checkout: clean detached checkout at `78cfec9`; the canonical server
  branch was restored clean at `f8fc048` after the gate.

Verification:
- RED test/support commit: `84f274a`
- RED focused test: `red/red.rc` is `1`; the old default returned `True`.
- GREEN compile: `green/py_compile.rc` is `0`.
- GREEN focused test: `green/focused.rc` is `0` (`1` test).
- GREEN related suites: `green/full.rc` is `0` (`39` tests).
- The full suite emitted existing test-only unclosed-file `ResourceWarning`
  messages; no test failed.
- Resource snapshot: `meta/server_resource_snapshot.rc` is `0`.

Benchmark:
- Workload: 60,000 curve points, three repeats per path.
- Before default median: `1.0683508989750408s` with PNG output.
- After default median: `0.0019046380184590816s` with NPZ only.
- Default-path speedup: `560.9207044178262x`.
- Explicit rendering median: `1.0725533899967559s`; PNG output remains
  available.
- Before/default and after/explicit NPZ arrays are equal.
- At 100 online refreshes, measured savings project to `106.64462609565817s`.

Scope:
- No Stage-2 RL policy, reward, action, evaluation, or training-data semantics
  changed.
- This is a CPU/reporting hot-path optimization; it does not make a multi-GPU
  scaling claim on the single-GPU replacement server.
