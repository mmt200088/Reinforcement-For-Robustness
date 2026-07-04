# GPU Utilization Markdown Device Streaming

- Source commit: `9412e3b`
- Remote RED package: `/hy-tmp/gpu_markdown_device_stream_red_20260704_104046`
- Remote GREEN package: `/hy-tmp/gpu_markdown_device_stream_green_20260704_104315`
- Scope: `scripts/gpu_utilization_report.py` Markdown device-list rendering.

## RED

Command:

```bash
python3 -m unittest tests.test_gpu_utilization_report.GpuUtilizationReportTest.test_markdown_device_lists_do_not_materialize_iterables -v
```

Result: `red.rc=1`.

Expected failure: old `render_markdown()` called `list(summary.get("visible_devices")...)`, which materialized iterable device lists before joining them.

## GREEN

Commands:

```bash
python3 -m py_compile scripts/gpu_utilization_report.py
python3 -m unittest \
  tests.test_gpu_utilization_report.GpuUtilizationReportTest.test_markdown_device_lists_do_not_materialize_iterables \
  tests.test_gpu_utilization_report.GpuUtilizationReportTest.test_cli_writes_json_and_markdown_reports \
  tests.test_gpu_utilization_report.GpuUtilizationReportTest.test_summarizes_probe_devices_trials_and_idle_visible_devices \
  -v
```

Result: `green.rc=0`.

Evidence: `green.log` shows `py_compile_rc=0`, the new no-materialization test passing, representative CLI Markdown output still passing, and core probe-device summary output still passing.
