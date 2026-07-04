# Stage-1 GPU A/B Gate Rerun (`180d319`)

Purpose: rerun the formal Stage-1 MRPC 170-episode 1GPU vs 4GPU gate after
the `4834b2f` episode-count diagnostics fix.

Source commit: `180d319`

Server run: `/hy-tmp/rfr_stage1_ab_180d319_20260704_090423_min170_wait`

Result:

- Both `g1` and `g4` completed with `launcher_rc=0`, `wait_rc=0`, and
  `COMPLETED`.
- The diagnostics fix worked: `g4` now reports `total_episodes=170` with
  worker counts `43/43/42/42`.
- `g4` used `cuda:0..3`, but it remains slower end-to-end on this gate.

Key numbers from `stage1_ab_out/comparison.json`:

- `g1` wall time: `107s`; parser throughput: `7469.518 ep/h`.
- `g4` wall time: `197s`; parser throughput: `3742.936 ep/h`.
- Wall-clock speedup `g4/g1`: `0.543`.
- Parser-throughput speedup `g4/g1`: `0.501`.
- `g1` model-forward timing: `68.510s`.
- `g4` model-forward timing: `569.828s`.

Conclusion: do not promote Stage-1 defaults to 4GPU yet. The four-worker path
is active, but the small validation-full reward workload and duplicated
per-worker model-forward overhead dominate. The next Stage-1 optimization
should target repeated validation/model-forward cost before another default
promotion attempt.
