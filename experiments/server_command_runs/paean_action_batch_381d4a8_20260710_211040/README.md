# Paean Fixed-Action Batch Benchmark

Source commit: 381d4a81b9802289fb306e29d64720b5a1813390

## Result

- Legacy: 2 Paean/Hugging Face processes, 44.96s.
- Optimized: 1 Paean/Hugging Face process with 2 candidates, 20.99s.
- Speedup: 2.142x.
- Wall-time reduction: 53.31%.
- Current independent deterministic references total: 43.55s; batch speedup 2.075x.
- Peak GPU memory remained 4071 MiB.
- Average sampled GPU utilization rose from 9.87% to 14.29%.

## Semantic Check

- Clean baseline fields match both independently launched references exactly.
- Every candidate field matches its independent deterministic reference exactly except expected runtime/path metadata (time_ms and artifact paths).
- Action vectors, optimizer outputs, cost/fusion totals, validity, loss, accuracy, and F1 are exact.
- Old sampled candidate metrics are not a bitwise oracle because the old independent BLB noise generator was seeded from OS entropy.
- All final commands exited 0; Python compilation and 58 related tests passed.

## Work Removed

- Model/tokenizer/dataset initialization: 2 -> 1.
- Clean baseline evaluation: 2 -> 1.
- Paean launcher process: 2 -> 1.
- Unused cost-matched prefilter attempts: 10,000 -> 0.

## Retained Diagnostic Runs

- `red/*.rc=1` files are the expected TDD failures before each implementation.
- `green/tests.rc=1` is the first post-refactor suite run. Its only failure was
  an existing static test that still inspected `_run_one()` after command
  construction moved to `_final_eval_command()`; `green/tests_final.rc=0` and
  the later full gate supersede it.
- `after/` is the first successful batch launch, retained because it exposed
  that the independent BLB noise generator used OS entropy. The final source
  isolates that generator, and `final_batch/` plus `final_single_refs/` provide
  the accepted parity evidence.

See comparison.json and parity.json for machine-readable evidence.
