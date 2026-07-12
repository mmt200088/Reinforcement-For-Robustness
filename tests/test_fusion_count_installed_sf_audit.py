import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


class NoisePoint:
    def __init__(self, sf, *, distribution="gaussian", N=8192):
        self.scaling_factor = sf
        self.distribution = distribution
        self.N = N


class FakeCfg:
    def __init__(self, *, base_sf=20, k=13):
        self.fresh = NoisePoint(base_sf)
        self.rescale = None
        self.powers = (NoisePoint(base_sf + 1), None, NoisePoint(base_sf + 3))
        self.output_truncation_k = k
        self.output_truncation_mode = "binary"
        self.rotation_after_rescale = False


class FakeHandler:
    def __init__(self):
        self.block2_cfg_per_layer = {}
        self.block4_cfg_per_layer = {}
        self.block5_cfg_per_layer = {}

    def get_active_blb_noise_layers(self):
        return {
            "block1": set(),
            "block2": set(self.block2_cfg_per_layer),
            "block3": set(),
            "block4": set(self.block4_cfg_per_layer),
            "block5": set(self.block5_cfg_per_layer),
            "first_input": set(),
        }


class FakeBridge:
    def __init__(self, handler, *, substitute_block4=False):
        self.handler = handler
        self.apply_count = 0
        self.substitute_block4 = substitute_block4

    def apply(self, *, block2_cfgs=None, block4_cfgs=None, block5_cfgs=None, **_kwargs):
        self.apply_count += 1
        self.handler.block2_cfg_per_layer = dict(block2_cfgs or {})
        self.handler.block4_cfg_per_layer = dict(block4_cfgs or {})
        self.handler.block5_cfg_per_layer = dict(block5_cfgs or {})
        if self.substitute_block4 and self.handler.block4_cfg_per_layer:
            layer = next(iter(self.handler.block4_cfg_per_layer))
            self.handler.block4_cfg_per_layer[layer] = FakeCfg(base_sf=99)


class InstalledSFAuditTest(unittest.TestCase):
    def test_serializer_accepts_only_authoritative_bridge_provenance(self):
        from scripts.fusion_count_installed_sf_audit import serialize_installed_cfgs

        cfg = FakeCfg(base_sf=21)
        rows = serialize_installed_cfgs(
            block2_cfgs={0: cfg},
            block4_cfgs={},
            block5_cfgs={},
            provenance="post_replan_bridge_apply",
        )

        by_point = {row["point"]: row for row in rows}
        self.assertEqual(by_point["fresh"]["scaling_factor"], 21)
        self.assertEqual(by_point["fresh"]["installation_state"], "installed")
        self.assertEqual(by_point["rescale"]["installation_state"], "not_installed")
        self.assertEqual(by_point["powers[0]"]["scaling_factor"], 22)
        self.assertEqual(by_point["powers[1]"]["installation_state"], "not_installed")
        self.assertEqual(by_point["output_truncation_k"]["truncation_k"], 13)
        self.assertTrue(all(row["provenance"] == "post_replan_bridge_apply" for row in rows))

        with self.assertRaisesRegex(ValueError, "authoritative"):
            serialize_installed_cfgs(
                block2_cfgs={0: cfg},
                block4_cfgs={},
                block5_cfgs={},
                provenance="map_option",
            )

    def test_capture_calls_original_then_verifies_handler_identity(self):
        from scripts.fusion_count_installed_sf_audit import InstalledConfigCapture

        handler = FakeHandler()
        bridge = FakeBridge(handler)
        b2 = {layer: FakeCfg(base_sf=20 + layer) for layer in range(12)}
        b4 = {layer: FakeCfg(base_sf=30 + layer) for layer in range(12)}
        b5 = {layer: FakeCfg(base_sf=40 + layer) for layer in range(12)}
        capture = InstalledConfigCapture(
            original_apply=bridge.apply,
            handler=handler,
            expected_layers=range(12),
        )

        capture.apply(block2_cfgs=b2, block4_cfgs=b4, block5_cfgs=b5)
        payload = capture.assert_complete()

        self.assertEqual(bridge.apply_count, 1)
        self.assertTrue(payload["handler_cfg_object_identity_match"])
        self.assertEqual(payload["handler_active_layers"]["block2"], list(range(12)))
        self.assertEqual(payload["handler_active_layers"]["block4"], list(range(12)))
        self.assertEqual(payload["handler_active_layers"]["block5"], list(range(12)))
        self.assertGreater(len(payload["installed_config_rows"]), 36)

    def test_capture_rejects_handler_cfg_substitution(self):
        from scripts.fusion_count_installed_sf_audit import InstalledConfigCapture

        handler = FakeHandler()
        bridge = FakeBridge(handler, substitute_block4=True)
        cfgs = {layer: FakeCfg(base_sf=20 + layer) for layer in range(12)}
        capture = InstalledConfigCapture(
            original_apply=bridge.apply,
            handler=handler,
            expected_layers=range(12),
        )

        with self.assertRaisesRegex(RuntimeError, "object identity"):
            capture.apply(block2_cfgs=cfgs, block4_cfgs=cfgs, block5_cfgs=cfgs)

    def test_validation_row_lookup_is_unshuffled_zero_to_407(self):
        from scripts.fusion_count_installed_sf_audit import build_validation_row_lookup

        source_rows = [{"idx": 1000 + row, "label": row % 2} for row in range(408)]
        lookup, labels = build_validation_row_lookup(source_rows)

        self.assertEqual(lookup[1000], 0)
        self.assertEqual(lookup[1407], 407)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[407], 1)
        self.assertEqual(sorted(lookup.values()), list(range(408)))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_validation_row_lookup([{"idx": 3, "label": 0}, {"idx": 3, "label": 1}])

    def test_prediction_aggregation_uses_only_validation_row_id(self):
        from scripts.fusion_count_installed_sf_audit import aggregate_prediction_rows

        lookup = {1373: 0, 18: 1}
        labels = {0: 1, 1: 0}
        rows = []
        for group in ("control", "all1"):
            for trial in range(2):
                rows.extend([
                    {
                        "group": group,
                        "run_seed": 10 + trial,
                        "trial_index": trial,
                        "dataset_idx": 1373,
                        "gold_label": 1,
                        "predicted_label": 1,
                        "correct": True,
                        "logits": [-0.2, 0.4],
                        "input_ids": [101, 7, 102],
                        "probe_position": 99,
                    },
                    {
                        "group": group,
                        "run_seed": 10 + trial,
                        "trial_index": trial,
                        "dataset_idx": 18,
                        "gold_label": 0,
                        "predicted_label": trial,
                        "correct": trial == 0,
                        "logits": [0.5 - trial, -0.5 + trial],
                        "input_ids": [101, 8, 102],
                        "probe_position": 3,
                    },
                ])

        aggregate = aggregate_prediction_rows(
            rows,
            row_lookup=lookup,
            labels=labels,
            expected_groups=("control", "all1"),
            expected_trials=2,
        )

        self.assertEqual([row["validation_row_id"] for row in aggregate], [0, 1])
        self.assertEqual(aggregate[0]["groups"]["all1"]["correct_count"], 2)
        self.assertEqual(aggregate[1]["groups"]["all1"]["correct_count"], 1)
        self.assertNotIn("dataset_idx", json.dumps(aggregate))
        self.assertNotIn("input_ids", json.dumps(aggregate))
        self.assertNotIn("probe_position", json.dumps(aggregate))

    def test_action_gate_requires_all1_boost_replan_and_k13(self):
        from scripts.fusion_count_installed_sf_audit import validate_allfusion1_result

        steps = []
        step_idx = 0
        for layer in range(12):
            for block in (2, 4, 5):
                steps.append({
                    "step_idx": step_idx,
                    "layer_idx": layer,
                    "block_idx": block,
                    "valid": True,
                    "fusion_count_replan": 1,
                    "boosted": True,
                    "k_value": 13,
                    "model_uses_replan_config": True,
                })
                step_idx += 1
        for _ in range(11):
            steps.append({
                "step_idx": step_idx,
                "layer_idx": 0,
                "block_idx": 1,
                "valid": True,
                "fusion_count_replan": 0,
                "boosted": False,
                "k_value": 13,
                "model_uses_replan_config": True,
            })
            step_idx += 1
        result = {
            "step_records": steps,
            "fusion_by_block": {"2": 12, "4": 12, "5": 12},
            "metrics": {"loss_mean": 0.3, "metric1_mean": 0.88, "metric2_mean": 0.87},
        }

        gate = validate_allfusion1_result(result)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["valid_step_count"], 47)

        steps[0]["boosted"] = False
        with self.assertRaisesRegex(ValueError, "boosted"):
            validate_allfusion1_result(result)

    def test_html_uses_only_validation_rows_and_installed_sf_provenance(self):
        from scripts.fusion_count_installed_sf_audit import render_audit_html

        payload = {
            "schema_version": "mrpc-allfusion1-installed-sf-audit-v1",
            "gate": {"passed": True},
            "capture": {
                "handler_cfg_object_identity_match": True,
                "installed_config_rows": [{
                    "layer": 0,
                    "block": "block4",
                    "point": "rescale",
                    "scaling_factor": None,
                    "truncation_k": None,
                    "installation_state": "fused_away",
                    "provenance": "post_replan_bridge_apply",
                }],
            },
            "validation_rows": [
                {"validation_row_id": 0, "gold_label": 1, "groups": {}},
                {"validation_row_id": 407, "gold_label": 0, "groups": {}},
            ],
            "protocol": {"gelu": [4] * 12, "softmax": [6] * 12, "k": 13},
        }

        html = render_audit_html(payload)
        self.assertIn("Validation row 0", html)
        self.assertIn("Validation row 407", html)
        self.assertIn("post_replan_bridge_apply", html)
        self.assertNotIn("dataset_idx", html)
        self.assertNotIn("input_ids", html)
        self.assertNotIn("probe_position", html)

        bad = json.loads(json.dumps(payload))
        bad["capture"]["installed_config_rows"][0]["provenance"] = "map_option"
        with self.assertRaisesRegex(ValueError, "authoritative"):
            render_audit_html(bad)

    def test_live_entrypoint_wraps_canonical_bridge_once_and_restores_it(self):
        from scripts.run_mrpc_allfusion1_installed_sf_audit import (
            execute_live_fixed_action_audit,
        )

        handler = FakeHandler()
        bridge = FakeBridge(handler)
        original_apply = bridge.apply
        seq_env = SimpleNamespace(base=SimpleNamespace(bridge=bridge))
        evaluator = SimpleNamespace(reversible_handler=handler)
        cfgs = {layer: FakeCfg(base_sf=20 + layer) for layer in range(12)}

        class FakeRLPath:
            def __init__(self):
                self.run_count = 0

            def _build_evaluator(self, args, *, stage1_gelu, stage1_softmax):
                self.build_args = (args, list(stage1_gelu), list(stage1_softmax))
                return evaluator

            def _build_seq_env(self, args, ev, *, stage1_gelu, stage1_softmax):
                self.seq_args = (args, ev, list(stage1_gelu), list(stage1_softmax))
                return seq_env, {"loss_mean": 0.2}

            def _run_group_canonical(self, current_seq_env, cfg, *, seed):
                self.run_count += 1
                self.run_args = (current_seq_env, cfg, seed)
                current_seq_env.base.bridge.apply(
                    block2_cfgs=cfgs,
                    block4_cfgs=cfgs,
                    block5_cfgs=cfgs,
                )
                steps = []
                for step_idx, (layer, block) in enumerate(
                    (layer, block) for layer in range(12) for block in (2, 4, 5)
                ):
                    steps.append({
                        "step_idx": step_idx,
                        "layer_idx": layer,
                        "block_idx": block,
                        "valid": True,
                        "fusion_count_replan": 1,
                        "boosted": True,
                        "k_value": 13,
                        "model_uses_replan_config": True,
                        "replan_application": {},
                    })
                for offset in range(11):
                    steps.append({
                        "step_idx": 36 + offset,
                        "layer_idx": offset + 1,
                        "block_idx": 1,
                        "valid": True,
                        "fusion_count_replan": 0,
                        "boosted": False,
                        "k_value": 13,
                        "model_uses_replan_config": True,
                        "replan_application": {},
                    })
                return {
                    "step_records": steps,
                    "fusion_by_block": {"2": 12, "4": 12, "5": 12},
                    "metrics": {
                        "loss_mean": 0.3,
                        "metric1_mean": 0.88,
                        "metric2_mean": 0.87,
                    },
                }

        fake_rlpath = FakeRLPath()
        result = execute_live_fixed_action_audit(
            SimpleNamespace(seed=77),
            {"name": "all1"},
            stage1_gelu=[4] * 12,
            stage1_softmax=[6] * 12,
            rlpath_module=fake_rlpath,
        )

        self.assertEqual(fake_rlpath.run_count, 1)
        self.assertEqual(bridge.apply_count, 1)
        self.assertEqual(bridge.apply, original_apply)
        self.assertTrue(result["gate"]["passed"])
        self.assertTrue(result["capture"]["handler_cfg_object_identity_match"])

    def test_live_entrypoint_restores_bridge_after_failure(self):
        from scripts.run_mrpc_allfusion1_installed_sf_audit import (
            execute_live_fixed_action_audit,
        )

        handler = FakeHandler()
        bridge = FakeBridge(handler)
        original_apply = bridge.apply
        seq_env = SimpleNamespace(base=SimpleNamespace(bridge=bridge))
        evaluator = SimpleNamespace(reversible_handler=handler)

        class FailingRLPath:
            @staticmethod
            def _build_evaluator(args, *, stage1_gelu, stage1_softmax):
                return evaluator

            @staticmethod
            def _build_seq_env(args, ev, *, stage1_gelu, stage1_softmax):
                return seq_env, {}

            @staticmethod
            def _run_group_canonical(current_seq_env, cfg, *, seed):
                raise RuntimeError("canonical failure")

        with self.assertRaisesRegex(RuntimeError, "canonical failure"):
            execute_live_fixed_action_audit(
                SimpleNamespace(seed=77),
                {"name": "all1"},
                stage1_gelu=[4] * 12,
                stage1_softmax=[6] * 12,
                rlpath_module=FailingRLPath,
            )

        self.assertEqual(bridge.apply, original_apply)


if __name__ == "__main__":
    unittest.main()
