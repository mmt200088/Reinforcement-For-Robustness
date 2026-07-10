# Stage-2 N-GPU Report Streaming Evidence

Source commit: `f3103d9`

Optimization:
- `scripts/stage2_ngpu_ab_compare.py` now streams report lines to stdout and
  the optional output file instead of materializing the complete report string
  in `main()`.
- `build_report()` remains available as a compatibility wrapper for callers
  that explicitly need a string.

Server:
- Host: `f1ac06029e4a`
- GPU inventory: one NVIDIA GeForce RTX 4090 with 24564 MiB
- Python: `3.10.19` from the `llm_ist` environment
- Git checkout: clean `jk_standard_rl` at `f3103d9`

Verification:
- RED implementation: `58e7a36`
- RED test/support files: `f3103d9`
- RED: `python -m unittest tests.test_stage2_ngpu_ab_compare -v`
  - `red/red.rc`: `1`
  - `Ran 12 tests`; only
    `test_main_streams_output_without_build_report_string` failed because the
    old `main()` called the patched `build_report()` function.
- GREEN: `python -m py_compile scripts/stage2_ngpu_ab_compare.py tests/test_stage2_ngpu_ab_compare.py`
  - `green/py_compile.rc`: `0`
- GREEN: `python -m unittest tests.test_stage2_ngpu_ab_compare -v`
  - `green/green.rc`: `0`
  - `Ran 12 tests`; all passed.
- Resource snapshot: `meta/server_resource_snapshot.rc`: `0`

Hardware scope:
- This evidence verifies the report-streaming behavior and source semantics.
- The replacement server has one visible GPU, so it cannot satisfy the
  separate 1GPU-vs-NGPU runtime/default-promotion gate.
