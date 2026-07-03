# Stage-2 Monitor PPO Update Streaming Evidence

Source commit: `d0f543b`
Base red snapshot: `6dc415d`

This evidence captures the red/green server verification for streaming
`ppo_updates.jsonl` in `scripts/stage2_first10k_monitor.py`.

## Red

- Run directory: `rfr_stage2_monitor_stream_ppo_red_6dc415d_20260703_000051`
- Command: `PYTHONPATH="$PWD" python -m unittest tests.test_stage2_first10k_monitor.Stage2First10kMonitorTest.test_build_summary_streams_ppo_updates_without_full_read_jsonl -v`
- Status: `red_rc=1`
- Expected failure: old monitor code called `read_jsonl()` for
  `ppo_updates.jsonl`, so the new guard failed with
  `PPO updates should be streamed, not fully materialized`.

## Green

- Run directory: `rfr_stage2_monitor_stream_ppo_green_6dc415d_20260704_000540`
- Compile command: `PYTHONPATH="$PWD" python -m py_compile scripts/stage2_first10k_monitor.py tests/test_stage2_first10k_monitor.py`
- Test command: `PYTHONPATH="$PWD" python -m unittest tests.test_stage2_first10k_monitor tests.test_jsonl_utils -v`
- Status: `green_py_compile_rc=0`, `green_unittest_rc=0`
- Result: 28 tests passed.
