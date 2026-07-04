# Stage-1 Repeat Eval Stdout JSON Streaming

- Source commit: `5d248a0`
- Remote RED package: `/hy-tmp/stage1_stdout_json_red_20260704_103151`
- Remote GREEN package: `/hy-tmp/stage1_stdout_json_green_20260704_103647`
- Scope: `scripts/stage1_plaintext_repeat_eval.py` stdout summary output.

## RED

Command:

```bash
python3 -m unittest tests.test_stage1_eval_accel.Stage1GpuEvalScriptSourceTest.test_stage1_plaintext_repeat_eval_streams_stdout_json_summary -v
```

Result: `red.rc=1`.

Expected failure: old source still used `print(json.dumps(summary, indent=2, ensure_ascii=False))`, so stdout materialized the complete JSON string before writing it.

## GREEN

Commands:

```bash
python3 -m py_compile scripts/stage1_plaintext_repeat_eval.py
python3 -m unittest tests.test_stage1_eval_accel.Stage1GpuEvalScriptSourceTest -v
```

Result: `green.rc=0`.

Evidence: `green.log` shows `py_compile_rc=0`, three Stage-1 GPU eval source tests passing, and `final_rc=0`.
