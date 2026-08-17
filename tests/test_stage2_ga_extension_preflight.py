from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from blb_stage2_rl import sequential_runner
from json_utils import stable_json_hash


class Stage2GAExtensionPreflightTests(unittest.TestCase):
    def test_failed_extension_attempt_archives_legacy_pending_strict_context(self):
        legacy_invocation = {
            "search_backend": "coinn_ga",
            "scientific_parameters": {
                "search_evaluation_budget": 45_664,
            },
        }
        requested_invocation = {
            "search_backend": "coinn_ga",
            "scientific_parameters": {
                "search_evaluation_budget": 11_464,
            },
        }
        resume_contract = {
            "evaluation_budget": 45_664,
            "requested_manifest": {
                "stage2_invocation": legacy_invocation,
            },
            "search_config": {
                "patience_generations": 5,
                "ga_generations": 800,
                "ga_maximum_evaluations": 45_664,
            },
        }
        manifest = {
            "status": "complete_least_violating",
            "communication_importance_ratio": 1.0,
            "resume_contract": resume_contract,
        }
        legacy_context = {
            "schema_version": "stage2_pending_strict_resume_context_v2",
            "invocation_contract": legacy_invocation,
            "resume_contract": resume_contract,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "search_coinn_ga")
            os.makedirs(output_dir)

            def write_json(name, payload):
                with open(
                        os.path.join(output_dir, name),
                        "w",
                        encoding="utf-8",
                        ) as handle:
                    json.dump(payload, handle)

            write_json("manifest.json", manifest)
            write_json("invocation.json", requested_invocation)
            write_json(
                "resume_result.pre_ga200_extension.json",
                {"legacy": "resume-result"},
            )
            write_json(
                "pending_strict_resume_context.json",
                legacy_context,
            )
            write_json(
                "ga200_extension_preflight.json",
                {
                    "schema_version": (
                        "stage2_ga_full_run_extension_preflight_v1"
                    ),
                    "legacy_invocation_hash": stable_json_hash(
                        legacy_invocation
                    ),
                    "requested_invocation_hash": stable_json_hash(
                        requested_invocation
                    ),
                    "legacy_resume_contract_hash": stable_json_hash(
                        resume_contract
                    ),
                    "legacy_status": "complete_least_violating",
                    "legacy_evaluation_count": 1_090,
                    "target_evaluation_count": 11_464,
                    "resume_result_archived": True,
                    "validated_at": "2026-08-17T19:40:19",
                },
            )

            legacy_result = SimpleNamespace(
                termination_reason="ga_no_incumbent_improvement",
                evaluation_count=1_090,
            )
            with (
                mock.patch.object(
                    sequential_runner,
                    "_build_search_invocation_contract",
                    return_value=requested_invocation,
                ),
                mock.patch(
                    "blb_stage2_rl.search_baseline_runner."
                    "_load_plain_completed_search_run",
                    return_value={"result": legacy_result},
                ),
                mock.patch(
                    "blb_stage2_rl.search_baseline_runner."
                    "_stage2_ga_full_run_invocation_extension_matches",
                    return_value=True,
                ),
                mock.patch(
                    "blb_stage2_rl.search_baseline_runner."
                    "_validate_ga_completion_proof",
                ),
            ):
                result = sequential_runner._preflight_completed_search_resume(
                    runner=SimpleNamespace(),
                    train_cfg=SimpleNamespace(search_backend="coinn_ga"),
                    fixed_gelu=[1, 2],
                    fixed_softmax=[6, 6],
                    fixed_label="stage1",
                    fixed_source="stage1_coinn_ga_result",
                    blb_progress_dir=tmpdir,
                )

            archived_context_path = os.path.join(
                output_dir,
                "pending_strict_resume_context.pre_ga200_extension.json",
            )
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(os.path.join(
                output_dir,
                "pending_strict_resume_context.json",
            )))
            with open(archived_context_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), legacy_context)


if __name__ == "__main__":
    unittest.main()
