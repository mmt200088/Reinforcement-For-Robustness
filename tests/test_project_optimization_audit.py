from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts" / "project_optimization_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("project_optimization_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ProjectOptimizationAuditTest(unittest.TestCase):
    def test_build_project_audit_reports_flow_stage_file_presence(self):
        audit = _load_audit_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch(root / "llama_7B_LayerImportance.sh", "#!/usr/bin/env bash\n")
            _touch(root / "presets" / "mrpc-blb-stage2-rl.conf", "--stage2-rl-variant\nblb_v3\n")
            _touch(root / "layer_importance_evaluator.py", "")
            _touch(root / "stage1_rl" / "parallel_runner.py", "")
            _touch(root / "scripts" / "stage1_parallel_report.py", "")
            _touch(root / "blb_stage2_rl" / "parallel_runner.py", "")
            _touch(root / "blb_stage2_rl" / "probe_runner.py", "")
            _touch(root / "Rescale_optimizer" / "rescale_optimizer" / "replan_interface.py", "")
            _touch(root / "Paean" / "run_final_eval.py", "")
            _touch(root / "rl_data_points.py", "")

            report = audit.build_project_audit(root)

        stage_ids = [stage["id"] for stage in report["flow_stages"]]
        self.assertEqual(
            stage_ids,
            ["launcher", "stage1", "stage2", "rescale", "paean", "artifacts"],
        )
        launcher = report["flow_stages"][0]
        self.assertEqual(launcher["present_files"], 2)
        self.assertEqual(launcher["missing_files"], 2)
        self.assertTrue(launcher["files"][0]["present"])
        stage1 = report["flow_stages"][1]
        self.assertTrue(
            any(
                item["path"] == "scripts/stage1_parallel_report.py" and item["present"]
                for item in stage1["files"]
            )
        )
        self.assertEqual(report["summary"]["total_flow_stages"], 6)
        self.assertGreater(report["summary"]["missing_files"], 0)

    def test_artifact_summary_counts_known_runtime_evidence_files(self):
        audit = _load_audit_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "experiments" / "server_command_runs" / "run_a"
            _touch(run / "diagnostics" / "episodes.jsonl", '{"episode": 0}\n')
            _touch(run / "diagnostics" / "ppo_updates.jsonl", '{"update": 1}\n')
            _touch(run / "nvidia_smi.csv", "timestamp,index,utilization.gpu,memory.used\n")
            _touch(run / "status.json", "{}\n")
            _touch(run / "report.html", "<html></html>\n")

            report = audit.build_project_audit(root, artifact_roots=[run])

        artifacts = report["artifact_summary"]
        self.assertEqual(artifacts["roots_scanned"], 1)
        self.assertEqual(artifacts["counts"]["episodes_jsonl"], 1)
        self.assertEqual(artifacts["counts"]["ppo_updates_jsonl"], 1)
        self.assertEqual(artifacts["counts"]["nvidia_smi_csv"], 1)
        self.assertEqual(artifacts["counts"]["status_json"], 1)
        self.assertEqual(artifacts["counts"]["html_reports"], 1)
        self.assertEqual(artifacts["missing_evidence"], [])

    def test_cli_writes_json_and_markdown(self):
        audit = _load_audit_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch(root / "llama_7B_LayerImportance.sh", "#!/usr/bin/env bash\n")
            out_json = root / "audit.json"
            out_md = root / "audit.md"

            rc = audit.main([
                "--root",
                str(root),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ])

            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("flow_stages", data)
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Project Optimization Audit", markdown)
            self.assertIn("launcher", markdown)


if __name__ == "__main__":
    unittest.main()
