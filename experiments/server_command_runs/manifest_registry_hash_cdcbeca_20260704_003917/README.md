# Manifest Registry Hash Streaming Evidence

Source commit: `cdcbeca` (`Stream manifest registry hash JSON`)

Base source for red test: `fc0fcb1`

## Optimization

`scripts/blb_make_run_manifest.py` now hashes parsed registry JSON with
`JSONEncoder.iterencode()` chunks instead of materializing one canonical
`json.dumps(...).encode(...)` byte string before sha256.

## Red Gate

Directory:
`rfr_manifest_registry_hash_red_fc0fcb1_20260704_003201/`

Command:

```bash
PYTHONPATH="$PWD" python -m unittest tests.test_blb_make_run_manifest.BlbMakeRunManifestTest.test_registry_hash_streams_parsed_json_without_json_dumps -v
```

Status: `red_rc=1`

Expected failure: patched `manifest.json.dumps` raises
`AssertionError: registry hash should stream JSON chunks`.

## Green Gate

Directory:
`rfr_manifest_registry_hash_green_fc0fcb1_20260704_003917/`

Commands:

```bash
PYTHONPATH="$PWD" python -m py_compile scripts/blb_make_run_manifest.py tests/test_blb_make_run_manifest.py
PYTHONPATH="$PWD" python -m unittest tests.test_blb_make_run_manifest -v
```

Status: `py_compile_rc=0`, `unittest_rc=0`

Result: 10 manifest tests passed, including the streaming registry hash test.
