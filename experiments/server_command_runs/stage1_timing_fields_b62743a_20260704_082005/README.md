# Stage-1 Timing Fields Evidence

Final source commit: `b62743a`

This evidence covers the Stage-1 rollout timing-field update. The change adds
the remaining diagnostics needed before the Stage-1 1GPU vs 4GPU throughput
gate:

- worker wall seconds remain in the existing `[stage1-rollout]` line;
- eval-cache hit/miss/hit-rate remains in the existing cache line;
- model-forward wall seconds and forward call count are captured from worker
  cache-miss `_run_evaluation()` calls;
- report-write wall seconds are emitted as `report_write` while preserving the
  existing `detail` field for older parsers.

The fields are written to the existing Stage-1 training log and parsed by the
existing Stage-1 parallel report tool. This is instrumentation for the
throughput gate; it is not itself a 1GPU vs 4GPU speedup claim.

## Server Verification

- Initial RED:
  `/hy-tmp/rfr_stage1_timing_red_1369547_20260704_081516`
- Intermediate RED:
  `/hy-tmp/rfr_stage1_timing_green_1369547_20260704_081726`
- Precommit GREEN:
  `/hy-tmp/rfr_stage1_timing_green2_1369547_20260704_081909`
- Final GREEN from committed source:
  `/hy-tmp/rfr_stage1_timing_final_b62743a_20260704_082005`

Initial RED failed because the parser did not recognize the new timing fields
and the evaluator did not emit them. The intermediate RED caught the `__new__`
unit-test path that bypasses `__init__`, so the timing state now initializes
both eagerly and lazily.

Final GREEN result: `SOURCE_COMMIT=b62743a`, `PY_COMPILE_RC=0`, `TEST_RC=0`,
with `tests.test_stage1_eval_accel`, `tests.test_stage1_parallel_semantics`,
and `tests.test_stage1_parallel_report` all passing (`50` tests).

## Local Contents

- `initial_red.log`: focused RED for missing parser/source timing fields.
- `intermediate_lazy_init_red.log`: broader Stage-1 gate exposing lazy-init
  coverage needed for timing state.
- `precommit_green.log`: server GREEN before the source commit.
- `green.log`: final server GREEN from committed source `b62743a`.
