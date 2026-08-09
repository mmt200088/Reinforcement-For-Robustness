import ast
from collections.abc import Mapping
import copy
import importlib.util
import inspect
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_evaluator_method(name, **runtime_globals):
    path = _REPO / "layer_importance_evaluator.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluator_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LayerImportanceEvaluator"
    )
    method = next(
        (node for node in evaluator_class.body
         if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if method is None:
        raise AssertionError(f"LayerImportanceEvaluator.{name} is missing")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(runtime_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _load_paean_method(name, **runtime_globals):
    path = _REPO / "Paean" / "blb_action_eval.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BLBActionFinalEvaluationModule"
    )
    method = next(
        (node for node in module_class.body
         if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if method is None:
        raise AssertionError(f"BLBActionFinalEvaluationModule.{name} is missing")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(runtime_globals)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class Stage2FinalEvalHandoffTest(unittest.TestCase):
    def test_in_memory_fusion_handoff_rejects_missing_reloadable_group(self):
        build_handoff = _load_evaluator_method(
            "_build_stage2_final_eval_handoff",
            copy=copy,
            Mapping=Mapping,
            np=np,
        )
        evaluator = types.SimpleNamespace(dataset_key="mrpc")
        search_result = {
            "blb_v3_best_action_vec": [3, 1, 4],
            "blb_v3_profile": "mrpc",
            "blb_v3_fusion_count_action": True,
        }

        with self.assertRaisesRegex(ValueError, "reloadable group"):
            build_handoff(evaluator, search_result)

    def test_in_memory_layerwise_handoff_rejects_missing_fusion_flag(self):
        build_handoff = _load_evaluator_method(
            "_build_stage2_final_eval_handoff",
            copy=copy,
            Mapping=Mapping,
            np=np,
        )
        evaluator = types.SimpleNamespace(dataset_key="mrpc")
        search_result = {
            "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
            "blb_v3_best_action_vec": [3, 1, 4],
            "blb_v3_profile": "mrpc",
            "blb_v3_best_action_group": {
                "option_by_step": {"0": 1},
                "boosted_overrides": [],
            },
        }

        with self.assertRaisesRegex(ValueError, "fusion flag"):
            build_handoff(evaluator, search_result)

    def test_in_memory_layerwise_handoff_rejects_missing_action(self):
        build_handoff = _load_evaluator_method(
            "_build_stage2_final_eval_handoff",
            copy=copy,
            Mapping=Mapping,
            np=np,
        )
        evaluator = types.SimpleNamespace(dataset_key="mrpc")
        search_result = {
            "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
            "blb_v3_profile": "mrpc",
            "blb_v3_fusion_count_action": True,
            "blb_v3_best_action_group": {
                "option_by_step": {"0": 1},
                "boosted_overrides": [],
            },
        }

        with self.assertRaisesRegex(ValueError, "action"):
            build_handoff(evaluator, search_result)

    def test_checkpoint_loader_skips_incomplete_layerwise_final_for_valid_live(self):
        final_name = "blb_stage2_rl_checkpoint_final.pt"
        live_name = "blb_stage2_rl_checkpoint_live.pt"
        loader = _load_evaluator_method(
            "_load_prior_rl_search_results",
            copy=copy,
            Mapping=Mapping,
            np=np,
            os=os,
        )
        valid_group = {
            "option_by_step": {"0": 1},
            "boosted_overrides": [{
                "block_idx": 5,
                "layer_idx": 0,
                "field_values": {"gelu_out_sf": 53},
            }],
        }
        checkpoints = {
            final_name: {
                "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
                "best_action": [1, 1],
                "profile": "mrpc",
                "blb_v3_fusion_count_action": True,
            },
            live_name: {
                "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
                "best_action": [9, 8],
                "profile": "mrpc",
                "blb_v3_fusion_count_action": True,
                "blb_v3_best_action_group": valid_group,
            },
        }
        fake_torch = types.SimpleNamespace(
            load=lambda path, **_kwargs: checkpoints[pathlib.Path(path).name],
        )
        loader.__globals__["torch"] = fake_torch
        noise_module = types.ModuleType("noise_rl_module_v2")
        noise_module.STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
        noise_module.NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
        runner_module = types.ModuleType("blb_stage2_rl.runner")
        runner_module.BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = final_name
        runner_module.BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = live_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_dir = pathlib.Path(tmp_dir) / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            (progress_dir / final_name).write_bytes(b"invalid-contract")
            (progress_dir / live_name).write_bytes(b"valid-contract")
            evaluator = types.SimpleNamespace(
                resume_run_dir=tmp_dir,
                run_output_dir=None,
                stage2_rl_variant="blb_v3",
                dataset_key="mrpc",
                log=lambda _message: None,
                _get_max_noise_configuration=lambda: {
                    "input_noise_scaling_factors": np.asarray([60], dtype=int),
                },
            )
            with mock.patch.dict(sys.modules, {
                "noise_rl_module_v2": noise_module,
                "blb_stage2_rl.runner": runner_module,
            }):
                _stage1, stage2 = loader(evaluator)

        np.testing.assert_array_equal(stage2["blb_v3_best_action_vec"], [9, 8])
        self.assertIs(stage2["blb_v3_fusion_count_action"], True)
        self.assertEqual(stage2["blb_v3_best_action_group"], valid_group)

    def test_checkpoint_loader_rejects_incomplete_comparator_contract(self):
        final_name = "blb_stage2_rl_checkpoint_final.pt"
        live_name = "blb_stage2_rl_checkpoint_live.pt"
        loader = _load_evaluator_method(
            "_load_prior_rl_search_results",
            copy=copy,
            Mapping=Mapping,
            np=np,
            os=os,
        )
        checkpoint = {
            "rl_variant": "blb_v3_layerwise_search_bo_rf",
            "search_backend": "bo_rf",
            "best_action": [9, 8],
            "profile": "mrpc",
            "blb_v3_fusion_count_action": True,
            "blb_v3_best_action_group": {
                "option_by_step": {"0": 1},
                "policy_actions": [[1, 0]],
                "boosted_overrides": [],
            },
        }
        loader.__globals__["torch"] = types.SimpleNamespace(
            load=lambda _path, **_kwargs: checkpoint,
        )
        noise_module = types.ModuleType("noise_rl_module_v2")
        noise_module.STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
        noise_module.NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
        runner_module = types.ModuleType("blb_stage2_rl.runner")
        runner_module.BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = final_name
        runner_module.BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = live_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_dir = pathlib.Path(tmp_dir) / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            (progress_dir / final_name).write_bytes(b"incomplete-comparator")
            evaluator = types.SimpleNamespace(
                resume_run_dir=tmp_dir,
                run_output_dir=None,
                stage2_rl_variant="blb_v3",
                blb_v3_search_backend="bo_rf",
                dataset_key="mrpc",
                log=lambda _message: None,
                _get_max_noise_configuration=lambda: {
                    "input_noise_scaling_factors": np.asarray([60], dtype=int),
                },
            )
            with mock.patch.dict(sys.modules, {
                "noise_rl_module_v2": noise_module,
                "blb_stage2_rl.runner": runner_module,
            }):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "completed comparator checkpoint",
                ):
                    loader(evaluator)

    def test_checkpoint_loader_preserves_completed_comparator_contract(self):
        final_name = "blb_stage2_rl_checkpoint_final.pt"
        live_name = "blb_stage2_rl_checkpoint_live.pt"
        loader = _load_evaluator_method(
            "_load_prior_rl_search_results",
            copy=copy,
            Mapping=Mapping,
            np=np,
            os=os,
        )
        checkpoint = {
            "status": "completed",
            "strict_feasible": True,
            "rl_variant": "blb_v3_layerwise_search_bo_rf",
            "search_backend": "bo_rf",
            "final_config_fingerprint": "e" * 64,
            "best_action": [9, 8],
            "profile": "mrpc",
            "blb_v3_fusion_count_action": True,
            "blb_v3_best_action_group": {
                "option_by_step": {"0": 1},
                "policy_actions": [[1, 0]],
                "boosted_overrides": [],
            },
        }
        loader.__globals__["torch"] = types.SimpleNamespace(
            load=lambda _path, **_kwargs: checkpoint,
        )
        noise_module = types.ModuleType("noise_rl_module_v2")
        noise_module.STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
        noise_module.NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
        runner_module = types.ModuleType("blb_stage2_rl.runner")
        runner_module.BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = final_name
        runner_module.BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = live_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_dir = pathlib.Path(tmp_dir) / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            (progress_dir / final_name).write_bytes(b"completed-comparator")
            evaluator = types.SimpleNamespace(
                resume_run_dir=tmp_dir,
                run_output_dir=None,
                stage2_rl_variant="blb_v3",
                blb_v3_search_backend="bo_rf",
                dataset_key="mrpc",
                log=lambda _message: None,
                _get_max_noise_configuration=lambda: {
                    "input_noise_scaling_factors": np.asarray([60], dtype=int),
                },
            )
            with mock.patch.dict(sys.modules, {
                "noise_rl_module_v2": noise_module,
                "blb_stage2_rl.runner": runner_module,
            }):
                _stage1, stage2 = loader(evaluator)

        self.assertEqual(stage2["status"], "completed")
        self.assertIs(stage2["strict_feasible"], True)
        self.assertEqual(stage2["search_backend"], "bo_rf")
        self.assertEqual(
            stage2["rl_variant"],
            "blb_v3_layerwise_search_bo_rf",
        )
        self.assertEqual(stage2["final_config_fingerprint"], "e" * 64)

    def test_checkpoint_loader_raises_when_blb_checkpoint_cannot_be_loaded(self):
        final_name = "blb_stage2_rl_checkpoint_final.pt"
        live_name = "blb_stage2_rl_checkpoint_live.pt"
        loader = _load_evaluator_method(
            "_load_prior_rl_search_results",
            copy=copy,
            Mapping=Mapping,
            np=np,
            os=os,
        )

        def fail_load(_path, **_kwargs):
            raise RuntimeError("corrupt checkpoint bytes")

        loader.__globals__["torch"] = types.SimpleNamespace(load=fail_load)
        noise_module = types.ModuleType("noise_rl_module_v2")
        noise_module.STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
        noise_module.NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
        runner_module = types.ModuleType("blb_stage2_rl.runner")
        runner_module.BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = final_name
        runner_module.BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = live_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_dir = pathlib.Path(tmp_dir) / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            (progress_dir / final_name).write_bytes(b"corrupt")
            evaluator = types.SimpleNamespace(
                resume_run_dir=tmp_dir,
                run_output_dir=None,
                stage2_rl_variant="blb_v3",
                dataset_key="mrpc",
                log=lambda _message: None,
                _get_max_noise_configuration=lambda: {
                    "input_noise_scaling_factors": np.asarray([60], dtype=int),
                },
            )
            with mock.patch.dict(sys.modules, {
                "noise_rl_module_v2": noise_module,
                "blb_stage2_rl.runner": runner_module,
            }):
                with self.assertRaisesRegex(RuntimeError, "corrupt checkpoint bytes"):
                    loader(evaluator)

    def test_checkpoint_loader_rejects_layerwise_group_without_exact_overrides(self):
        final_name = "blb_stage2_rl_checkpoint_final.pt"
        live_name = "blb_stage2_rl_checkpoint_live.pt"
        loader = _load_evaluator_method(
            "_load_prior_rl_search_results",
            copy=copy,
            Mapping=Mapping,
            np=np,
            os=os,
        )
        checkpoint = {
            "rl_variant": "blb_v3_layerwise_robust_gtrxl_v1",
            "best_action": [4, 2],
            "profile": "mrpc",
            "blb_v3_fusion_count_action": True,
            "blb_v3_best_action_group": {"option_by_step": {"0": 1}},
        }
        loader.__globals__["torch"] = types.SimpleNamespace(
            load=lambda _path, **_kwargs: checkpoint,
        )
        noise_module = types.ModuleType("noise_rl_module_v2")
        noise_module.STAGE1_CHECKPOINT_FILENAME = "stage1_rl_checkpoint.pt"
        noise_module.NOISE_STAGE_CHECKPOINT_FILENAME = "noise_rl_checkpoint.pt"
        runner_module = types.ModuleType("blb_stage2_rl.runner")
        runner_module.BLB_STAGE2_FINAL_CHECKPOINT_FILENAME = final_name
        runner_module.BLB_STAGE2_LIVE_CHECKPOINT_FILENAME = live_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_dir = pathlib.Path(tmp_dir) / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            (progress_dir / final_name).write_bytes(b"malformed-group")
            evaluator = types.SimpleNamespace(
                resume_run_dir=tmp_dir,
                run_output_dir=None,
                stage2_rl_variant="blb_v3",
                dataset_key="mrpc",
                log=lambda _message: None,
                _get_max_noise_configuration=lambda: {
                    "input_noise_scaling_factors": np.asarray([60], dtype=int),
                },
            )
            with mock.patch.dict(sys.modules, {
                "noise_rl_module_v2": noise_module,
                "blb_stage2_rl.runner": runner_module,
            }):
                with self.assertRaisesRegex(RuntimeError, "reloadable group"):
                    loader(evaluator)


class PersistedBoostedOverrideDecodeTest(unittest.TestCase):
    def test_persisted_override_wins_when_fusion_map_value_differs(self):
        map_value = 31
        persisted_value = 47
        option = types.SimpleNamespace(
            option_id=1,
            boosted=True,
            explicit_field_values={
                "softmax_out_fresh_sf": map_value,
                "output_truncation_k": 13,
            },
            slots={"softmax_out_fresh_sf": 29, "output_truncation_k": 13},
        )
        graph = types.SimpleNamespace(k_slot_index=1, options=[option])
        fusion_map = types.SimpleNamespace(graphs={"block4": graph})
        fusion_map_module = types.ModuleType("blb_stage2_rl.fusion_count_map")
        fusion_map_module.FusionCountMap = types.SimpleNamespace(
            load=lambda _profile: fusion_map,
        )
        step = types.SimpleNamespace(
            graph_key_suffix="block4",
            step_idx=0,
            full_vec_offsets=(0, 1),
            includes_first_input=False,
            slot_field_names=("softmax_out_fresh_sf", "output_truncation_k"),
            layer_idx=0,
            block_idx=4,
        )
        decoded = types.SimpleNamespace(
            per_layer_field_values=[{}],
            block4_cfgs=[None],
        )

        def build_cfg(_block_idx, _layer_idx, field_values, **_kwargs):
            return types.SimpleNamespace(
                softmax_out_fresh=types.SimpleNamespace(
                    scaling_factor=int(field_values["softmax_out_fresh_sf"]),
                ),
                output_truncation_k=int(field_values["output_truncation_k"]),
            )

        decode = _load_paean_method(
            "_decode_fusion_count_fixed_action",
            ActionDecodeResult=object,
            Any=object,
            Mapping=Mapping,
            _block_default_N=lambda *_args, **_kwargs: 0,
            _decode_block_field_values=lambda *_args, **_kwargs: {
                "softmax_out_fresh_sf": 20,
                "output_truncation_k": 11,
            },
            action_vector_to_cfgs=lambda **_kwargs: decoded,
            build_block_cfg_from_field_values=build_cfg,
            np=np,
            step_schedule=lambda *_args, **_kwargs: [step],
            validate_action_vector=lambda raw, _num_layers: np.asarray(raw),
        )
        metadata = {
            "schema_version": "fusion_count_fixed_action_v1",
            "group": {
                "option_by_step": {"0": 1},
                "boosted_overrides": [{
                    "block_idx": 4,
                    "layer_idx": 0,
                    "field_values": {
                        "softmax_out_fresh_sf": persisted_value,
                        "output_truncation_k": 11,
                    },
                }],
            },
        }

        with mock.patch.dict(sys.modules, {
            "blb_stage2_rl.fusion_count_map": fusion_map_module,
        }):
            result = decode(
                object(),
                action_vec=[0, 2],
                metadata=metadata,
                max_sfs=object(),
                num_layers=1,
                gelu=[4],
                softmax=[6],
                profile="mrpc",
            )

        self.assertNotEqual(map_value, persisted_value)
        self.assertEqual(
            result.block4_cfgs[0].softmax_out_fresh.scaling_factor,
            persisted_value,
        )
        self.assertEqual(result.block4_cfgs[0].output_truncation_k, 11)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch required for Paean.blb_action_eval import",
)
class FusionCountFixedActionDecodeTest(unittest.TestCase):
    def test_per_step_fusion_option_replay_preserves_rl_selected_k(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from blb_stage2_rl.action_space import (
            K_LEVELS,
            load_max_sfs,
            make_all_max_action_vector,
            step_schedule,
        )
        from blb_stage2_rl.fusion_count_map import FusionCountMap

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        fusion_map = FusionCountMap.load("mrpc")
        schedule = step_schedule(
            num_layers,
            profile="mrpc",
            attn_degree_per_layer=softmax,
            gelu_degree_per_layer=gelu,
        )
        step = next(s for s in schedule if s.layer_idx == 0 and s.block_idx == 4)
        option_id = 1
        k_index = 2  # K_LEVELS[2] == 11 under the legacy-compatible table.

        action_vec = make_all_max_action_vector(num_layers)
        block_vec = fusion_map.expand(step.graph_key_suffix, option_id, k_index)
        for offset, value in zip(step.full_vec_offsets, block_vec.tolist()):
            action_vec[int(offset)] = int(value)

        metadata = {
            "schema_version": "fusion_count_fixed_action_v1",
            "group": {
                "option_by_step": {str(step.step_idx): option_id},
            },
        }

        module = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        decoded = module._decode_fusion_count_fixed_action(
            action_vec=action_vec,
            metadata=metadata,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=num_layers,
            gelu=gelu,
            softmax=softmax,
            profile="mrpc",
        )

        cfg = decoded.block4_cfgs[0]
        self.assertEqual(cfg.output_truncation_k, int(K_LEVELS[k_index]))
        # A boosted option must replay its map-owned explicit values instead of
        # the pre-boost action-index grid. Keep this assertion map-version aware:
        # precision-boost rebuilds may legitimately change the explicit SFs.
        option = fusion_map.options(step.graph_key_suffix)[option_id]
        self.assertTrue(option.boosted)
        self.assertIsNotNone(option.explicit_field_values)
        self.assertEqual(
            cfg.softmax_out_fresh.scaling_factor,
            option.explicit_field_values["softmax_out_fresh_sf"],
        )
        self.assertEqual(
            cfg.softmax_out_mask_encode.scaling_factor,
            option.explicit_field_values["softmax_out_mask_sf"],
        )

    def test_selected_vs_random_summary_keeps_existing_statistics(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        module = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        selected = [{
            "name": "selected",
            "loss": 1.1,
            "loss_std": 0.01,
            "p": 0.80,
            "p_std": 0.02,
            "s": 0.70,
            "s_std": 0.03,
            "total_bits_sum": 44,
            "total_fusion_count": 3,
            "avg_truncation_k": 12.0,
        }]
        random_results = [
            {"loss": 1.2, "loss_std": 0.04, "p": 0.70, "p_std": 0.05, "s": 0.65, "s_std": 0.06},
            {"loss": 1.0, "loss_std": 0.02, "p": 0.85, "p_std": 0.01, "s": 0.72, "s_std": 0.04},
            {"loss": 1.3, "loss_std": 0.03, "p": 0.78, "p_std": 0.03, "s": 0.75, "s_std": 0.02},
        ]

        summary = module._summarize_selected_vs_random(
            selected_results=selected,
            random_results=random_results,
            num_metrics=2,
        )

        self.assertEqual(summary["random_count"], 3)
        self.assertEqual(summary["random_stats"]["loss_mean"]["n"], 3)
        self.assertAlmostEqual(summary["random_stats"]["loss_mean"]["mean"], np.mean([1.2, 1.0, 1.3]))
        self.assertAlmostEqual(summary["random_stats"]["loss_mean"]["std"], np.std([1.2, 1.0, 1.3]))
        self.assertAlmostEqual(summary["random_stats"]["metric1_mean"]["max"], 0.85)
        ranks = summary["anchor_rank_vs_random"]
        self.assertEqual(ranks["metric1_higher_better"]["rank_better_than_selected"], 2)
        self.assertEqual(ranks["loss_lower_better"]["rank_better_than_selected"], 2)
        self.assertEqual(ranks["metric2_higher_better"]["rank_better_than_selected"], 1)

    def test_selected_vs_random_summary_streams_random_rows_once(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._summarize_selected_vs_random)

        self.assertNotIn("np.asarray([float(r.get(key, 0.0)) for r in rows]", source)
        self.assertNotIn("metric_rows = [", source)
        self.assertNotIn("loss_rows = [", source)
        self.assertNotIn("metric2_rows = [", source)

    def test_results_plot_scans_candidate_rows_once(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._save_results_plot)

        self.assertNotIn('np.asarray([float(r["loss"]) for r in candidate_results]', source)
        self.assertNotIn("np.asarray([float(r.get(\"loss_std\", 0.0)) for r in candidate_results]", source)
        self.assertNotIn('np.asarray([float(r["p"]) for r in candidate_results]', source)
        self.assertNotIn("np.asarray([float(r.get(\"p_std\", 0.0)) for r in candidate_results]", source)
        self.assertNotIn('np.asarray([float(r["total_bits_sum"]) for r in candidate_results]', source)
        self.assertNotIn('np.asarray([float(r["time_ms"]) for r in candidate_results]', source)

    def test_scatter_plot_scans_result_rows_once_per_group(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._save_scatter_plot)

        self.assertNotIn("def _xs_ys", source)
        self.assertNotIn('[float(r.get("p", 0.0)) for r in rows]', source)
        self.assertNotIn('[float(r.get("p_std", 0.0)) for r in rows]', source)
        self.assertNotIn('[float(r.get("s", 0.0)) for r in random_results]', source)
        self.assertNotIn('[float(r.get("s_std", 0.0)) for r in random_results]', source)
        self.assertNotIn('[float(r.get("s", 0.0)) for r in selected_results]', source)
        self.assertNotIn('[float(r.get("s_std", 0.0)) for r in selected_results]', source)

    def test_full_noise_markdown_table_streams_entries_without_copy(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._full_noise_config_markdown_table)

        self.assertNotIn("entries = list(", source)

    def test_fusion_fixed_action_decode_avoids_step_copy_wrappers(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._decode_fusion_count_fixed_action)

        self.assertNotIn("dict(raw_option_by_graph", source)
        self.assertNotIn("dict(raw_option_by_step", source)
        self.assertNotIn("base_arr[list(block_offsets)]", source)
        self.assertNotIn("dict(option_fields).items()", source)


if __name__ == "__main__":
    unittest.main()
