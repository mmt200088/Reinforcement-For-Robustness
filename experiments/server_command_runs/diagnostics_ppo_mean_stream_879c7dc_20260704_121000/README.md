# Diagnostics PPO Warning Mean Streaming

Source commit: `879c7dc`

This run verifies that `blb_stage2_rl/diagnostics.py` computes the recent
three-update PPO warning means through the shared streaming `mean_or_default()`
helper instead of building short Python lists and passing them to `np.mean()`.
The change is limited to human diagnostics summary generation and does not
change PPO, reward, or action-selection semantics.

Server temporary sources:

- RED: `/hy-tmp/diagnostics_ppo_mean_red_5099fad_20260704_121000`
- GREEN: `/hy-tmp/diagnostics_ppo_mean_green_20260704_121000`

Verification:

- `red.rc`: `1`, expected failure on old source because the diagnostics module
  still used `np.mean([u.entropy ...])` / `np.mean([u.clip_fraction ...])`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_blb_diagnostics_static` passed.
