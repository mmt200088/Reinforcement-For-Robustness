# GLUE Action Config Shared JSON Reader Evidence

Source commit: `2ded3e7`

Purpose: verify that BLB GLUE action-config loading uses the shared streaming JSON reader and no longer materializes the whole config with `json.loads(open(...).read())`.

Server runs:
- Red: `/hy-tmp/rfr_glue_json_reader_red_cf0496e_20260704_040000`
- Green: `/hy-tmp/rfr_glue_json_reader_green_cf0496e_20260704_040930`

Green command coverage:
- `python -m py_compile generate_glue_submission.py`
- `python -m unittest tests.test_rl_data_points.RLDataPointWriterTest.test_json_artifact_scripts_use_shared_reader -v`
- Source guard confirming `payload = read_json_file(action_config_path, encoding="utf-8-sig")`
