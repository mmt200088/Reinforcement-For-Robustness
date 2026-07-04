# Paean Action Spec Tuple JSON

Source commit: `1d65c55`

This run verifies that `Paean/run_final_eval.py` serializes
`FinalEvalSettings.action_ranges` and `action_fixed` directly. Those fields are
normalized to tuples by `Paean/config.py`, and `json.dumps()` serializes tuples
as JSON arrays, so the previous `list(...)` wrappers were unnecessary per-launch
copies in the final-eval command builder.

Server temporary sources:

- RED: `/hy-tmp/paean_run_final_eval_action_specs_red_e929e31_20260704_122000`
- GREEN: `/hy-tmp/paean_run_final_eval_action_specs_green_20260704_122000`

Verification:

- `red.rc`: `1`, expected failure on old source because the command builder
  still used `json.dumps(list(settings.action_ranges))` and
  `json.dumps(list(settings.action_fixed))`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_paean_run_final_eval_static` passed.
