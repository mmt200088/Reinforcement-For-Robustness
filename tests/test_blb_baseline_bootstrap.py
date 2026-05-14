import json
import pathlib
import tempfile
import unittest

from blb_stage2_rl.baseline_bootstrap import (
    BaselineHandoverError,
    RESPONSE_SCHEMA_V1,
    StaticSkeletonsBaseline,
    StaticSkeletonsLayerBlock,
    baseline_response_to_cost_stats,
    handover_paths,
    load_static_skeletons_baseline,
    read_baseline_response,
    static_skeletons_baseline_to_action,
    validate_response_against_request,
    write_baseline_request,
)


class BLBBaselineBootstrapTests(unittest.TestCase):
    def _write_response(
            self,
            repo_root,
            dataset,
            request_id,
            gelu,
            softmax,
            *,
            include_layer0_block1=True,
            skip_pairs=frozenset(),
            ):
        paths = handover_paths(repo_root, dataset)
        results = []
        total_bits = 0
        total_fusion = 0
        for layer in range(len(gelu)):
            for block in (1, 2, 3, 4, 5):
                key = (block, layer)
                if key in skip_pairs:
                    continue
                if key == (1, 0) and not include_layer0_block1:
                    continue
                if block == 3:
                    graph_key = f"block3_exp_n{softmax[layer]}"
                elif block == 5:
                    graph_key = f"block5_n{gelu[layer]}"
                else:
                    graph_key = f"block{block}_{dataset}"
                bits = 1000 + layer * 10 + block
                fusion = block % 2
                total_bits += bits
                total_fusion += fusion
                results.append({
                    "config_name": f"{graph_key}_L{layer}",
                    "graph_key": graph_key,
                    "block": block,
                    "layer": layer,
                    "success": True,
                    "skeleton": [0, 1, 2],
                    "t_baseline": [1, 2, 3],
                    "q_bits_baseline": [60, 50, 40],
                    "modulus_chain": {"total_bits": bits, "q_bits": [60, 50, 40]},
                    "fusion_count": fusion,
                    "invalid_chain": None,
                    "cut_point_sf": [{"node": "ctpt", "sf": 16}],
                    "effective_rotations": [],
                    "error_message": "",
                })

        payload = {
            "schema": RESPONSE_SCHEMA_V1,
            "request_id": request_id,
            "dataset": dataset,
            "model": "bert-base",
            "num_layers": len(gelu),
            "ok": True,
            "error": None,
            "results": results,
            "aggregate": {
                "total_bits_sum": total_bits,
                "total_fusion_count": total_fusion,
                "valid_block_count": len(results),
                "invalid_block_count": 0,
            },
            "warnings": [],
        }
        with open(paths["response_path"], "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return total_bits, total_fusion

    def test_round_trip_response_validation_and_cost_stats(self):
        with tempfile.TemporaryDirectory() as repo_root:
            dataset = "mrpc"
            request_id = "unit-test-request"
            gelu = [1, 4]
            softmax = [2, 6]
            request_path = write_baseline_request(
                repo_root,
                dataset,
                gelu,
                softmax,
                model="bert-base",
                request_id=request_id,
            )
            total_bits, total_fusion = self._write_response(
                repo_root, dataset, request_id, gelu, softmax,
            )

            result = read_baseline_response(
                repo_root, dataset, expected_request_id=request_id,
            )

            self.assertTrue(result.ok)
            self.assertEqual(len(result.results), 10)
            self.assertEqual(result.by_block_layer[(3, 1)].graph_key, "block3_exp_n6")
            self.assertEqual(result.by_block_layer[(5, 0)].graph_key, "block5_n1")
            self.assertEqual(result.aggregate_total_bits_sum, total_bits)
            self.assertEqual(result.aggregate_total_fusion_count, total_fusion)
            self.assertEqual(validate_response_against_request(request_path, result), [])

            stats = baseline_response_to_cost_stats(result, baseline_avg_k=11.0)
            self.assertEqual(stats.total_bits_sum, total_bits)
            self.assertEqual(stats.total_fusion_count, total_fusion)
            self.assertEqual(stats.avg_k, 11.0)

            with self.assertRaises(BaselineHandoverError):
                read_baseline_response(
                    repo_root, dataset, expected_request_id="wrong-request",
                )

    def test_canonical_response_may_omit_layer0_block1(self):
        with tempfile.TemporaryDirectory() as repo_root:
            dataset = "mrpc"
            request_id = "unit-test-request-no-l0-b1"
            gelu = [1, 4]
            softmax = [2, 6]
            request_path = write_baseline_request(
                repo_root,
                dataset,
                gelu,
                softmax,
                model="bert-base",
                request_id=request_id,
            )
            total_bits, total_fusion = self._write_response(
                repo_root,
                dataset,
                request_id,
                gelu,
                softmax,
                include_layer0_block1=False,
            )

            result = read_baseline_response(
                repo_root, dataset, expected_request_id=request_id,
            )

            self.assertEqual(len(result.results), 9)
            self.assertNotIn((1, 0), result.by_block_layer)
            self.assertIn((2, 0), result.by_block_layer)
            self.assertEqual(result.aggregate_valid_block_count, 9)
            self.assertEqual(result.aggregate_invalid_block_count, 0)
            self.assertEqual(result.aggregate_total_bits_sum, total_bits)
            self.assertEqual(result.aggregate_total_fusion_count, total_fusion)
            self.assertEqual(validate_response_against_request(request_path, result), [])

    def test_rejects_missing_required_non_layer0_block1_entry(self):
        with tempfile.TemporaryDirectory() as repo_root:
            dataset = "mrpc"
            request_id = "unit-test-request-missing-required"
            gelu = [1, 4]
            softmax = [2, 6]
            write_baseline_request(
                repo_root,
                dataset,
                gelu,
                softmax,
                model="bert-base",
                request_id=request_id,
            )
            self._write_response(
                repo_root,
                dataset,
                request_id,
                gelu,
                softmax,
                include_layer0_block1=False,
                skip_pairs={(2, 1)},
            )

            with self.assertRaises(BaselineHandoverError):
                read_baseline_response(
                    repo_root, dataset, expected_request_id=request_id,
                )

    def test_rejects_invalid_stage1_degree(self):
        with tempfile.TemporaryDirectory() as repo_root:
            with self.assertRaises(ValueError):
                write_baseline_request(
                    repo_root,
                    "mrpc",
                    [3],
                    [2],
                    model="bert-base",
                    request_id="bad-gelu",
                )
            with self.assertRaises(ValueError):
                write_baseline_request(
                    repo_root,
                    "mrpc",
                    [1],
                    [7],
                    model="bert-base",
                    request_id="bad-softmax",
                )

    def test_static_skeletons_mixed_degree_baseline_decodes_per_layer_sfs(self):
        from blb_stage2_rl.action_space import describe_action_vector

        gelu = [1, 4]
        softmax = [2, 5]
        baseline = load_static_skeletons_baseline(
            "Rescale_optimizer",
            "mrpc",
            num_layers=2,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
        )
        action_vec, max_sfs, _stats, _diag = static_skeletons_baseline_to_action(
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

    def test_static_skeletons_block4_wo_rescale_uses_rl_field_name(self):
        from blb_stage2_rl.action_space import describe_action_vector

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
        runner_path = pathlib.Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "runner.py"
        source = runner_path.read_text(encoding="utf-8")

        self.assertIn("load_static_skeletons_baseline", source)
        self.assertNotIn("estimate_all_max_action", source)
        self.assertIn("snap_sf_to_noise_table=False", source)
        self.assertIn("baseline_action_vec = np.asarray(_ss_action_vec", source)
        self.assertNotIn("fallback 到 load_max_sfs", source)


if __name__ == "__main__":
    unittest.main()
