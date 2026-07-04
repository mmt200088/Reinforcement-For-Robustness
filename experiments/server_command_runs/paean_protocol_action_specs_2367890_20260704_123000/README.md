# Paean Evaluation Protocol Action Spec Reuse

Source commit: `2367890`

This run verifies that `Paean/blb_action_eval.py` keeps
`self.action_ranges` and `self.action_fixed` as normalized tuples in the
`evaluation_protocol` payload until the existing `to_jsonable()` conversion
step. This avoids manual `list(...)` copies while preserving the final JSON
array output. The human log formatting still uses `list(...)` for readability
and was intentionally left unchanged.

Server temporary sources:

- RED: `/hy-tmp/paean_blb_protocol_specs_red_f8be01f_20260704_123000`
- GREEN: `/hy-tmp/paean_blb_protocol_specs_green_20260704_123000`

Verification:

- `red.rc`: `1`, expected failure on old source because
  `evaluation_protocol` still used `list(self.action_ranges)` and
  `list(self.action_fixed)`.
- `green.rc`: `py_compile_rc=0`, `test_rc=0`.
- `green.log`: `tests.test_paean_blb_action_eval_static` passed.
