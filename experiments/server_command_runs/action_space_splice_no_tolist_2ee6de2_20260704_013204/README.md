# Action-Space Splice `arr.tolist()` Removal Evidence

Source commit: `2ee6de2` (`Avoid list materialization in action splicing`)

Optimization: `blb_stage2_rl/action_space.py` now splices per-step and
fusion-step action vectors by iterating the checked numpy array directly,
instead of materializing `arr.tolist()` for every splice. This removes a small
but hot Python-list allocation from Stage-2/Paean action construction while
preserving shape checks, offset mapping, and integer writes.

Server evidence:

- `rfr_action_space_splice_red_c80f0e5_20260704_013036/red_status.txt`:
  `red_rc=1`. The source guard failed against the old implementation because
  both splice helpers still contained `arr.tolist()`.
- `rfr_action_space_splice_green_c80f0e5_20260704_013129/green_status.txt`:
  `py_compile_rc=0`, `functional_rc=0`, `unittest_rc=1`. The compile and
  functional splice check passed, but the broader unittest target required
  fusion-map JSON fixtures that were not included in the temp Python-only
  package.
- `rfr_action_space_splice_green_focus_c80f0e5_20260704_013204/green_status.txt`:
  `py_compile_rc=0`, `source_guard_rc=0`, `functional_rc=0`. The functional
  log reports `splice_helpers_ok 0 23 23`, covering both step and fusion splice
  helpers without relying on generated fusion-map artifacts.

The evidence bundle excludes the temporary server source tree and keeps only
status files, logs, and source snapshots needed for audit.
