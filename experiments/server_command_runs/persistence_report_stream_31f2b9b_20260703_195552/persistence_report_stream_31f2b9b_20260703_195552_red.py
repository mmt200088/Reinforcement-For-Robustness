import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import mock

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("persistence_red", root / "blb_stage2_rl" / "persistence.py")
persistence = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = persistence
spec.loader.exec_module(persistence)

with tempfile.TemporaryDirectory() as td:
    root_td = Path(td)
    report_paths = {
        str(root_td / "blb_stage2_best_action_full.md"),
        str(root_td / persistence.BLB_FINAL_REPORT_MD),
        str(root_td / persistence.BLB_ERROR_TXT),
    }
    original_open = open

    def fake_open(path, *args, **kwargs):
        path_str = str(path)
        if path_str not in report_paths:
            return original_open(path, *args, **kwargs)
        handle = mock.MagicMock()
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = None
        def reject_full_document_write(text):
            if not isinstance(text, str) or text.count("\n") <= 3:
                return
            if path_str.endswith(persistence.BLB_ERROR_TXT):
                if "Traceback:" in text and "Python:" in text:
                    raise AssertionError("persistence reports should stream lines")
                return
            raise AssertionError("persistence reports should stream lines")
        handle.write.side_effect = reject_full_document_write
        return handle

    caught = []
    with mock.patch("builtins.open", side_effect=fake_open):
        action_paths = persistence.write_action_description_files(
            td,
            {
                "profile": "mrpc",
                "num_layers": 1,
                "action_length": 1,
                "summary": {"record_count": 1},
                "records": [{
                    "global_index": 0,
                    "slot_label": "L00.B1.F.example",
                    "label": "L00.B1.F.example",
                    "location": "L00.B1",
                    "operation": "example",
                    "kind": "F",
                    "layer": 0,
                    "block": 1,
                    "distribution": "F",
                    "action_index": 1,
                    "effective": True,
                    "effective_value": 12,
                    "scaling_factor": 12,
                    "level_values": [8, 12],
                }],
            },
        )
        if action_paths["md"] == str(root_td / "blb_stage2_best_action_full.md"):
            caught.append("action_md_not_red")
        try:
            persistence.write_blb_final_report(
                td,
                run_basename="unit",
                profile="mrpc",
                total_episodes=10,
                completed_episodes=10,
                elapsed_sec=1.0,
                best_reward=1.0,
                best_breakdown={"terminal_priority": 3},
                best_action_vec=[1, 2, 3],
                baseline={"loss": 0.3},
                reward_weights={"cost": 1.0},
                episode_returns=[0.1, 0.2, 0.3],
                rescale_invoker_kind="unit",
            )
        except AssertionError:
            pass
        else:
            caught.append("final_report_not_red")
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            crash_path = persistence.dump_crash_report(td, exc=exc)
        if crash_path == str(root_td / persistence.BLB_ERROR_TXT):
            caught.append("crash_report_not_red")

if caught:
    raise SystemExit("expected old persistence writers to fail red: " + ",".join(caught))
print("RED OK: old persistence report writers materialize full joined documents")
