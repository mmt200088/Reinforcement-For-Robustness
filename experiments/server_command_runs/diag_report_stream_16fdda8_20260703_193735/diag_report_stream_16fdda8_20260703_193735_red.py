import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import mock

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "blb_stage2_rl"))
spec = importlib.util.spec_from_file_location("diag_red", root / "blb_stage2_rl" / "diagnostics.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as td:
    recorder = module.RLDiagnosticsRecorder(
        output_dir=str(Path(td) / "progress"),
        num_layers=12,
        num_action_slots=3,
    )
    recorder.record_episode(
        episode_stats=module.EpisodeStats(
            episode=0,
            total_reward=1.0,
            terminal_reward=0.5,
            per_step_sum=0.5,
            valid_steps=47,
            invalid_steps=0,
            steps_taken=47,
            total_bits=123,
            fusion_count=1,
            first_invalid_step=None,
            first_invalid_block=None,
            first_invalid_layer=None,
            early_terminated=False,
        ),
        full_action_vec=None,
        is_new_best=False,
        best_reward_so_far=1.0,
    )
    recorder._close_primary_jsonl()

    original_open = open

    def expect_full_document_write(path, fn):
        target = str(path)
        def fake_open(p, *args, **kwargs):
            if str(p) != target:
                return original_open(p, *args, **kwargs)
            handle = mock.MagicMock()
            handle.__enter__.return_value = handle
            handle.__exit__.return_value = None
            def guard(text):
                if isinstance(text, str) and text.count("\n") > 3:
                    raise AssertionError("full-document write observed")
            handle.write.side_effect = guard
            return handle
        try:
            with mock.patch("builtins.open", side_effect=fake_open), mock.patch.object(module.os, "replace"):
                fn()
        except AssertionError as exc:
            if "full-document write observed" in str(exc):
                return True
            raise
        return False

    summary_red = expect_full_document_write(recorder.summary_md_path + ".tmp", recorder._write_summary_md)
    html_red = expect_full_document_write(
        recorder.pareto_html_path + ".tmp",
        lambda: recorder._write_pareto_html([{"episode": 0, "total_reward": 1.0}]),
    )

if not (summary_red and html_red):
    raise SystemExit(f"expected both report writers to fail red; summary={summary_red} html={html_red}")
print("RED OK: old diagnostics report writers materialize full joined documents")
