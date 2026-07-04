# BLB Manifest Stdout JSON Streaming

- Source commit: `362ea22`
- Remote RED package: `/hy-tmp/manifest_stdout_json_red_20260704_105032`
- Remote GREEN package: `/hy-tmp/manifest_stdout_json_green_20260704_105117`
- Scope: `scripts/blb_make_run_manifest.py` CLI stdout path summary.

## RED

Command:

```bash
python3 -m unittest tests.test_blb_make_run_manifest.BlbMakeRunManifestTest.test_main_streams_stdout_json_paths -v
```

Result: `red.rc=1`.

Expected failure: old source still used `print(json.dumps(paths, ...))`, which materialized the complete stdout JSON path summary before writing it.

## GREEN

Commands:

```bash
python3 -m py_compile scripts/blb_make_run_manifest.py
python3 -m unittest tests.test_blb_make_run_manifest -v
```

Result: `green.rc=0`.

Evidence: `green.log` shows `py_compile_rc=0`, `tests.test_blb_make_run_manifest` passing all 11 tests, and `final_rc=0`.
