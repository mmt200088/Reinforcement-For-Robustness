# Stage-1 Parallel Episode Count Diagnostics (`4834b2f`)

Purpose: fix the Stage-1 parallel rollout total timing line so `episodes=`
uses actual worker episode counts instead of `num_workers * floor(episodes_per_worker)`.

Why: the prior 170-episode 4GPU gate assigned episodes as `43/43/42/42`, but
`[stage1-rollout-total]` reported `168`. That under-counted 4GPU throughput
and made the A/B comparison less precise.

Evidence:

- `red.log`: the new test failed on `4d4c713` because the old source still used
  `len(workers) * episodes_per_worker`.
- `green.log`: after the fix, `py_compile` passed and
  `tests.test_stage1_parallel_semantics` plus `tests.test_stage1_parallel_report`
  passed (`25` tests).

This is a diagnostics fix. It does not change Stage-1 reward, action sampling,
validation split, PPO update semantics, or worker scheduling.
