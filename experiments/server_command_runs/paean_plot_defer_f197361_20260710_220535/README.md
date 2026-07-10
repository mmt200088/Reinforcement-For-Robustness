# Paean Plot Deferral Benchmark

Source commit: f19736112c12b62b29b2958158bb751ca806b4a2

## Result

- Fixed-action evaluation keeps JSON, Markdown, and the driver combined HTML.
- The driver now defaults RFR_PAEAN_RENDER_PLOTS=0, while Paean itself defaults
  to rendering and an explicit value of 1 restores the original behavior.
- The two internal Paean PNGs cost 0.921s on a cold
  two-candidate render and 0.728s at warm median.
- This serial section was 4.39%
  of the prior stable 20.99s two-candidate batch.
- The final end-to-end run produced zero internal PNGs, retained the combined
  HTML, and matched baseline/candidate semantics exactly.

## Verification

- Python compilation: rc=0.
- Related unit tests: 60 passed.
- End-to-end driver: rc=0.
- Paean default render gate: enabled.
- Driver default render gate: disabled.
- Explicit RFR_PAEAN_RENDER_PLOTS=1: enabled.
- Semantic differences versus the accepted deterministic batch: only manifest
  path and measured time_ms.

## Timing Scope

The plot-deferred end-to-end run took
31.56s, but
host scheduling variance made it slower than the prior stable run. It is not
used for an end-to-end speedup claim. The performance result is the isolated
0.92s synchronous rendering section that the new driver path no longer calls.

## Retained Diagnostics

- before/benchmark.rc=1 is a benchmark-fixture path error caused by the server
  sparse checkout; before_retry/benchmark.rc=0 is the valid baseline.
- red/focused.rc=1 is the expected TDD RED result.

See comparison.json and after_e2e/parity.json for machine-readable evidence.
