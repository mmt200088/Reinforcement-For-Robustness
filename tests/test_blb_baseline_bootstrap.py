import hashlib
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

from rfr.preparation.rescale.baseline_bootstrap import (
    StaticSkeletonsBaseline,
    StaticSkeletonsLayerBlock,
    load_static_skeletons_baseline,
    static_skeletons_baseline_to_action,
)


class BLBBaselineBootstrapTests(unittest.TestCase):
    def test_stage2_profile_resolution_binds_dataset_model_and_depth(self):
        from rfr.preparation.rescale.baseline_bootstrap import (
            resolve_stage2_model_type,
            resolve_stage2_profile,
        )

        self.assertEqual(
            resolve_stage2_model_type("", num_layers=12),
            "bert-base",
        )
        self.assertEqual(
            resolve_stage2_model_type("", num_layers=24),
            "bert-large",
        )
        self.assertEqual(
            resolve_stage2_model_type("bert_large", num_layers=24),
            "bert-large",
        )
        self.assertEqual(
            resolve_stage2_profile("mrpc", model_type="bert-base", num_layers=12),
            "mrpc",
        )
        self.assertEqual(
            resolve_stage2_profile("mrpc", model_type="bert-large", num_layers=24),
            "mrpc_large",
        )
        self.assertEqual(
            resolve_stage2_profile(
                "mrpc_large", model_type="bert-large", num_layers=24,
            ),
            "mrpc_large",
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            resolve_stage2_profile(
                "mrpc", model_type="bert-large", num_layers=12,
            )
        with self.assertRaisesRegex(ValueError, "12 or 24"):
            resolve_stage2_profile(
                "mrpc", model_type="custom", num_layers=18,
            )

    def test_calibrated_action_context_wraps_static_baseline_with_provenance(self):
        import rfr.preparation.rescale.baseline_bootstrap as bootstrap

        self.assertTrue(
            hasattr(bootstrap, "load_calibrated_stage2_action_context"),
            "shared calibrated action-context loader is missing",
        )
        load_context = bootstrap.load_calibrated_stage2_action_context

        with tempfile.TemporaryDirectory() as td:
            archive = pathlib.Path(td) / "static_skeletons_mrpc.json"
            archive.write_bytes(b'{"fixture":true}\n')
            baseline = StaticSkeletonsBaseline(
                dataset="mrpc",
                num_layers=2,
                gelu_per_layer=[4, 4],
                softmax_per_layer=[6, 6],
                archive_path=str(archive),
            )
            action_vec = object()
            max_sfs = object()
            cost_stats = object()
            diagnostics = {"calibrated": True}

            with mock.patch.object(
                bootstrap,
                "load_static_skeletons_baseline",
                return_value=baseline,
            ) as load_baseline, mock.patch.object(
                bootstrap,
                "static_skeletons_baseline_to_action",
                return_value=(action_vec, max_sfs, cost_stats, diagnostics),
            ) as convert:
                context = load_context(
                    rescale_optimizer_root="/repo/configs/preparation/rescale",
                    dataset="mrpc",
                    num_layers=2,
                    gelu_per_layer=[4, 4],
                    softmax_per_layer=[6, 6],
                    snap_sf_to_noise_table=False,
                )
            bootstrap.validate_calibrated_stage2_action_context(
                context,
                dataset="mrpc",
                num_layers=2,
                gelu_per_layer=[4, 4],
                softmax_per_layer=[6, 6],
                snap_sf_to_noise_table=False,
            )
            archive.write_bytes(b'{"fixture":false}\n')
            with self.assertRaisesRegex(ValueError, "archive_sha256"):
                bootstrap.validate_calibrated_stage2_action_context(
                    context,
                    dataset="mrpc",
                    num_layers=2,
                    gelu_per_layer=[4, 4],
                    softmax_per_layer=[6, 6],
                    snap_sf_to_noise_table=False,
                )

        load_baseline.assert_called_once_with(
            rescale_optimizer_root="/repo/configs/preparation/rescale",
            dataset="mrpc",
            num_layers=2,
            gelu_per_layer=(4, 4),
            softmax_per_layer=(6, 6),
        )
        convert.assert_called_once_with(
            baseline,
            snap_sf_to_noise_table=False,
        )
        self.assertIs(context.baseline, baseline)
        self.assertIs(context.baseline_action_vec, action_vec)
        self.assertIs(context.max_sfs, max_sfs)
        self.assertIs(context.cost_stats, cost_stats)
        self.assertIs(context.diagnostics, diagnostics)
        self.assertEqual(context.provenance["dataset"], "mrpc")
        self.assertEqual(context.provenance["gelu_per_layer"], [4, 4])
        self.assertEqual(context.provenance["softmax_per_layer"], [6, 6])
        self.assertEqual(
            context.provenance["archive_sha256"],
            hashlib.sha256(b'{"fixture":true}\n').hexdigest(),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch required for calibrated action decode",
    )
    def test_calibrated_action_context_matches_mrpc_block3_static_baseline(self):
        from rfr.search.common.action_space import action_vector_to_cfgs
        from rfr.preparation.rescale.baseline_bootstrap import (
            load_calibrated_stage2_action_context,
        )

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        context = load_calibrated_stage2_action_context(
            rescale_optimizer_root="configs/preparation/rescale",
            dataset="mrpc",
            num_layers=num_layers,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
            snap_sf_to_noise_table=False,
        )
        decoded = action_vector_to_cfgs(
            action_vec=context.baseline_action_vec,
            max_sfs=context.max_sfs,
            num_layers=num_layers,
            gelu_degree=gelu,
            attn_degree=softmax,
        )
        cfg = decoded.block3_cfgs[0]

        self.assertEqual(cfg.x_fresh.scaling_factor, 31)
        self.assertEqual(cfg.inv_2n_encode.scaling_factor, 15)
        self.assertEqual(
            [entry.scaling_factor for entry in cfg.square_rescales],
            [35] * 6,
        )

    def test_static_skeletons_mixed_degree_baseline_decodes_per_layer_sfs(self):
        from rfr.search.common.action_space import describe_action_vector

        gelu = [1, 4]
        softmax = [2, 5]
        baseline = load_static_skeletons_baseline(
            "configs/preparation/rescale",
            "mrpc",
            num_layers=2,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
        )
        self.assertEqual(baseline.aggregate_valid_block_count, 9)
        self.assertIn((3, 0), baseline.per_block_layer)
        self.assertIn((3, 1), baseline.per_block_layer)
        action_vec, max_sfs, stats, _diag = static_skeletons_baseline_to_action(
            baseline,
            snap_sf_to_noise_table=False,
        )
        desc = describe_action_vector(
            action_vec,
            max_sfs=max_sfs,
            num_layers=2,
            gelu_degree=gelu,
            attn_degree=softmax,
            profile="mrpc",
        )
        values = {
            (int(r["layer"]), int(r["block_index"]), str(r["field"])): int(r["value"])
            for r in desc["records"]
            if (
                isinstance(r.get("block_index"), int)
                and r.get("kind") != "K"
                and r.get("value") is not None
            )
        }

        for (block_idx, layer_idx), layer_block in baseline.per_block_layer.items():
            for field_name, expected_sf in layer_block.field_baseline_sfs.items():
                self.assertEqual(
                    values[(layer_idx, block_idx, field_name)],
                    int(expected_sf),
                    (layer_idx, block_idx, layer_block.graph_key, field_name),
                )
        self.assertNotIn((0, 1, "gelu_out_sf"), values)
        layer0_block1_k = next(
            r for r in desc["records"]
            if (
                r.get("layer") == 0
                and r.get("block_index") == 1
                and r.get("field") == "output_truncation_k"
            )
        )
        self.assertTrue(layer0_block1_k["effective"])
        self.assertEqual(layer0_block1_k["value"], 13)
        self.assertAlmostEqual(stats.avg_k, 13.0)
        inactive_rescale = [
            r for r in desc["records"]
            if (
                isinstance(r.get("block_index"), int)
                and r.get("kind") == "R"
                and r.get("value") is None
                and r.get("action_index") == 0
            )
        ]
        self.assertGreater(len(inactive_rescale), 0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch required for action-vector decode",
    )
    def test_static_skeletons_block4_wo_rescale_uses_rl_field_name(self):
        from rfr.search.common.action_space import describe_action_vector

        baseline = StaticSkeletonsBaseline(
            dataset="mrpc",
            num_layers=1,
            gelu_per_layer=[4],
            softmax_per_layer=[4],
            archive_path="<unit-test>",
        )
        baseline.per_block_layer[(4, 0)] = StaticSkeletonsLayerBlock(
            block_idx=4,
            layer_idx=0,
            graph_key="block4_mrpc",
            field_baseline_sfs={"wo_rescale_sf": 31},
            field_kind_in_ro={"wo_rescale_sf": "rescale"},
            total_bits=287,
            fusion_count=0,
        )
        baseline.aggregate_total_bits = 287
        baseline.aggregate_fusion_count = 0
        baseline.aggregate_valid_block_count = 1

        action_vec, max_sfs, _stats, _diag = static_skeletons_baseline_to_action(
            baseline,
            snap_sf_to_noise_table=False,
        )
        desc = describe_action_vector(
            action_vec,
            max_sfs=max_sfs,
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[4],
            profile="mrpc",
        )

        record = next(
            r for r in desc["records"]
            if (
                r.get("layer") == 0
                and r.get("block_index") == 4
                and r.get("field") == "wo_rescale_sf"
            )
        )
        self.assertEqual(record["value"], 31)
        self.assertNotEqual(record["action_index"], 0)

    def test_runner_has_no_static_skeletons_baseline_fallback(self):
        runner_path = pathlib.Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "sequential_runner.py"
        source = runner_path.read_text(encoding="utf-8")

        self.assertIn("load_calibrated_stage2_action_context", source)
        self.assertIn("validate_calibrated_stage2_action_context", source)
        self.assertNotIn("estimate_all_max_action", source)
        self.assertIn("snap_sf_to_noise_table=False", source)
        self.assertIn("baseline_action_vec = np.asarray(", source)
        self.assertIn("calibrated_action_context.baseline_action_vec", source)
        self.assertNotIn("fallback 到 load_max_sfs", source)


if __name__ == "__main__":
    unittest.main()
