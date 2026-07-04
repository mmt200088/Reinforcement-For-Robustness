# Stage-1 Default Batch GPU A/B (`9f3864d`)

Purpose: verify the source change in `9f3864d`, which promotes
`run rl --mode stage1-only` to a Stage-1-specific launcher default batch size
when the user does not pass `--batch-size`.

Source commit: `9f3864d`

Server run: `/hy-tmp/rfr_stage1_default_batch_ab_9f3864d_20260704_092741`

Result:

- Both `g1` and `g4` were launched without an explicit `--batch-size` flag in
  the A/B driver.
- Launcher output for both runs shows Python receiving
  `--batch_size 128 --micro_batch_size 128`.
- Both `g1` and `g4` completed with `launcher_rc=0`, `wait_rc=0`, and
  `COMPLETED`.
- `g4` used `cuda:0..3` and reported 170 episodes with worker counts
  `43/43/42/42`.

Key numbers:

- `g1` wall time: `102s`; parser throughput: `7558.355` ep/h; model-forward
  timing: `67.699s`.
- `g4` wall time: `92s`; parser throughput: `9007.153` ep/h; model-forward
  timing: `174.264s`.
- 4GPU/1GPU wall speedup: `1.109x`.
- 4GPU/1GPU parser-throughput speedup: `1.192x`.

Conclusion: the new Stage-1-only launcher default applies at runtime and keeps
the previously observed batch-128 four-GPU improvement without requiring users
to remember `--batch-size 128`.
