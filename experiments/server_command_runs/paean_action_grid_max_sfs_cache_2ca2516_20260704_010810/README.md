# Paean Action Grid Max-SFS Cache Evidence

Source commit: `2ca2516` (`Cache Paean action grid max SFS loads`)

Base source for red test: `f486aec`

## Optimization

`Paean/action_grid.py` now caches `load_max_sfs(profile)` results by profile.
Slot-form final-eval action configs and fixed/range action-grid candidate
generation reuse the cached max-SF table instead of reparsing the same profile
table for every config in a batch.

## Red Gate

Directory:
`rfr_paean_max_sfs_cache_red_f486aec_20260704_010641/`

Command:

```bash
PYTHONPATH="$PWD" python -m unittest tests.test_paean_action_grid.PaeanActionGridTest.test_slot_config_loading_reuses_profile_max_sfs -v
```

Status: `red_rc=1`

Expected failure: two same-profile slot-form configs call `load_max_sfs()`
twice (`AssertionError: 2 != 1`).

## Green Gate

Directory:
`rfr_paean_max_sfs_cache_green_f486aec_20260704_010810/`

Commands:

```bash
PYTHONPATH="$PWD" python -m py_compile Paean/action_grid.py tests/test_paean_action_grid.py
PYTHONPATH="$PWD" python -m unittest tests.test_paean_action_grid -v
```

Status: `py_compile_rc=0`, `unittest_rc=0`

Result: 5 Paean action-grid tests passed, including the max-SF profile-cache
test.
