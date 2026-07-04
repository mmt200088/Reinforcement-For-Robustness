# Fusion Map Gate Filter Verification

Source commit: `248a0ec`

Server temp root: `/hy-tmp/rfr_fusion_map_gate_min_20260704_094103`

Purpose:

- Expose `scripts.report_fusion_count_map.iter_fusion_map_paths()` as the shared
  canonical fusion-map path iterator.
- Make the active `SERVER_COMMAND.md` phase2 map gate use that iterator instead
  of `glob("*.json")` plus ad hoc `_summary.json` filtering.
- Prevent post-build sidecars such as `map_summary.json` from being opened as
  fusion maps and failing with `KeyError: 'options'`.

Server verification:

- RED package: current HEAD old source plus new tests.
  - Command: `python3 -m unittest <2 new report_fusion_count_map tests> -v`
  - Result: expected failure (`AttributeError` for missing
    `iter_fusion_map_paths`, plus SERVER_COMMAND guard failure).
  - Log: `red.log`
- GREEN package: source commit `248a0ec` changes plus new tests.
  - Command: `python3 -m unittest <2 new report_fusion_count_map tests> -v`
  - Result: OK, 2 tests.
  - Log: `green.log`
- Wider GREEN:
  - Command: `python3 -m py_compile scripts/report_fusion_count_map.py
    tests/test_report_fusion_count_map.py && python3 -m unittest
    tests.test_report_fusion_count_map -v`
  - Result: OK, 21 tests.
  - Log: `green_full.log`
