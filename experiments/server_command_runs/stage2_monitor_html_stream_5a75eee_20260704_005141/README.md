# Stage-2 Monitor HTML Streaming Evidence

Source commit: `5a75eee` (`Stream stage2 monitor HTML report rows`)

Base source for red test: `1db3745`

## Optimization

`scripts/stage2_first10k_monitor.py` now writes final monitor HTML report
rows incrementally. The large `reward_probe` and `gpu` nested sections are
encoded with `JSONEncoder.iterencode()` chunks and escaped directly into the
file handle instead of materializing full `json.dumps(..., indent=2)` strings
and a full table string before writing.

## Red Gate

Directory:
`rfr_stage2_report_html_stream_red_1db3745_20260704_004919/`

Command:

```bash
PYTHONPATH="$PWD" python -m unittest tests.test_stage2_first10k_monitor.Stage2First10kMonitorTest.test_write_report_streams_nested_json_without_json_dumps -v
```

Status: `red_rc=1`

Expected failure: patched `monitor.json.dumps` raises
`AssertionError: write_report should stream nested JSON chunks`.

## Green Gate

Directory:
`rfr_stage2_report_html_stream_green_1db3745_20260704_005141/`

Commands:

```bash
PYTHONPATH="$PWD" python -m py_compile scripts/stage2_first10k_monitor.py tests/test_stage2_first10k_monitor.py
PYTHONPATH="$PWD" python -m unittest tests.test_stage2_first10k_monitor -v
```

Status: `py_compile_rc=0`, `unittest_rc=0`

Result: 13 Stage-2 monitor tests passed, including the streaming HTML report
test.
