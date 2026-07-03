# Action Registry K-Level Scan Evidence

Source commit: `9b78854`

This evidence covers a structured-artifact export optimization in
`scripts/blb_export_action_registry.py`: `_all_max_action_index()` now finds
the all-max truncation-K action index with one pass over `k_levels` instead of
first calling `max(k_levels)` and then copying the sequence with
`list(k_levels).index(...)`.

For duplicate maximum K values, the implementation still returns the first
maximum index, matching `list(...).index(max(...))`. Empty inputs still raise
`ValueError("max() arg is an empty sequence")`.

## Server Verification

- RED run directory:
  `/hy-tmp/rfr_registry_klevel_red_e2a9fd8_20260704_074716`
- GREEN run directory:
  `/hy-tmp/rfr_registry_klevel_green2_20260704_074904`

RED command:

```bash
python3 -m unittest tests.test_blb_export_action_registry_light.BLBExportActionRegistryLightTests.test_all_max_action_index_scans_k_levels_once_without_copy -v
```

RED result: failed as expected on the pre-change source because
`_all_max_action_index()` still contained
`list(k_levels).index(max(k_levels))`.

GREEN commands:

```bash
python3 -m py_compile scripts/blb_export_action_registry.py tests/test_blb_export_action_registry_light.py
python3 -m unittest tests.test_blb_export_action_registry_light -v
```

GREEN result: `PY_COMPILE_RC=0`, all three light registry tests passed, and
`TEST_RC=0`.

## Local Contents

- `red_unittest.log`: server RED focused unittest log.
- `green_validation.log`: server GREEN py-compile and light unittest log.
- `source_snapshot/blb_export_action_registry.py`: source snapshot from
  `9b78854`.
- `source_snapshot/test_blb_export_action_registry_light.py`: focused test
  snapshot from `9b78854`.
