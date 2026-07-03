# Skeleton Profile Config Discovery Verification

Purpose: verify the `load_profile_configs()` discovery optimization in
`blb_stage2_rl/skeleton_stage_map.py`.

- Red commit: `3f5c04d`
- Green/source commit: `cb215bd`
- Server run roots:
  - `/hy-tmp/rfr_skeleton_profile_config_discovery_3f5c04d_20260703_213500`
  - `/hy-tmp/rfr_skeleton_profile_config_discovery_cb215bd_20260703_213800`

Checks:

- Red target unittest:
  `tests.test_blb_skeleton_stage_map.LoadProfileConfigsTest.test_skips_json_named_directories`
  failed with `IsADirectoryError` because the old `os.listdir()` discovery tried
  to parse a `.json`-named directory as JSON.
- Green `py_compile` for `blb_stage2_rl/skeleton_stage_map.py`,
  `tests/test_blb_skeleton_stage_map.py`, and `json_utils.py` returned `0`.
- Green target unittest returned `OK`.

Status files:

- `rfr_skeleton_profile_config_discovery_3f5c04d_20260703_213500/red_status.txt`
  records `red_rc=1`.
- `rfr_skeleton_profile_config_discovery_cb215bd_20260703_213800/green_status.txt`
  records `green_py_compile_rc=0` and `green_unittest_rc=0`.
