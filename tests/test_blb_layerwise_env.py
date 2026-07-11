"""Torch-free behavior tests for the 12-step Stage-2 layerwise environment."""

from __future__ import annotations

import dataclasses
import importlib.machinery
import importlib.util
import json
import pathlib
import sys
import types
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BLB_DIR = REPO_ROOT / "blb_stage2_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class LayerwiseEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pkg_name = "_blb_layerwise_env_test_pkg"
        for name in list(sys.modules):
            if name == pkg_name or name.startswith(f"{pkg_name}."):
                del sys.modules[name]
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(BLB_DIR)]
        sys.modules[pkg_name] = pkg

        def load(name, path):
            loader = importlib.machinery.SourceFileLoader(name, str(path))
            spec = importlib.util.spec_from_loader(name, loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loader.exec_module(module)
            return module

        cls.fcm = load(f"{pkg_name}.fusion_count_map", BLB_DIR / "fusion_count_map.py")
        cls.layerwise = load(f"{pkg_name}.layerwise_action", BLB_DIR / "layerwise_action.py")

        action_space = types.ModuleType(f"{pkg_name}.action_space")

        def all_max(num_layers):
            vector = np.full(num_layers * 73 + 1, 14, dtype=int)
            k_index = cls.layerwise.K_LEVELS.index(13)
            block_starts = (0, 9, 32, 40, 57)
            block_widths = (9, 23, 8, 17, 16)
            for layer_idx in range(num_layers):
                for start, width in zip(block_starts, block_widths):
                    vector[layer_idx * 73 + start + width - 1] = k_index
            vector[-1] = 4
            return vector

        action_space.make_all_max_action_vector = all_max
        sys.modules[action_space.__name__] = action_space

        @dataclasses.dataclass
        class RuntimeResult:
            block_cfg: object
            optimizer_output: object
            valid: bool
            total_bits: int
            fusion_count: int
            invalid_chain: object
            bridge_error: object
            config_name: str
            graph_key: str
            optimizer_wall_seconds: float
            bridge_error_type: object = None
            boosted_field_values: object = None
            replan_application: dict = dataclasses.field(default_factory=dict)
            optimizer_cfg_overrides: list = dataclasses.field(default_factory=list)

        cls.RuntimeResult = RuntimeResult
        sequential = types.ModuleType(f"{pkg_name}.sequential_env")
        sequential.BlockRuntimeResult = RuntimeResult
        sequential.evaluate_block_from_full_vector = lambda **kwargs: None
        sys.modules[sequential.__name__] = sequential

        cls.mod = load(f"{pkg_name}.layerwise_env", BLB_DIR / "layerwise_env.py")
        cls.fusion_map = cls.fcm.FusionCountMap.load("mrpc", root=str(BLB_DIR))

    def setUp(self):
        self.base = self._FakeBase()
        self.runtime_calls = []
        self.invalid_key = None

        def evaluate(**kwargs):
            self.runtime_calls.append({
                **kwargs,
                "full_vec": np.asarray(kwargs["full_vec"], dtype=int).copy(),
                "boosted_field_values": (
                    dict(kwargs["boosted_field_values"])
                    if kwargs.get("boosted_field_values") else None
                ),
            })
            key = (int(kwargs["layer_idx"]), int(kwargs["block_idx"]))
            valid = key != self.invalid_key
            return self.RuntimeResult(
                block_cfg=types.SimpleNamespace(key=key),
                optimizer_output=types.SimpleNamespace(valid=valid),
                valid=valid,
                total_bits=100 + key[1] if valid else 0,
                fusion_count=1 if key[1] == 4 and valid else 0,
                invalid_chain=None if valid else {"reason": "invalid test chain"},
                bridge_error=None,
                config_name=f"{kwargs['graph_key']}_L{key[0]}",
                graph_key=kwargs["graph_key"],
                optimizer_wall_seconds=0.01,
                boosted_field_values=kwargs.get("boosted_field_values"),
                replan_application={"model_uses_replan_config": valid},
                optimizer_cfg_overrides=[{"key": key}] if valid else [],
            )

        self.mod.evaluate_block_from_full_vector = evaluate
        self.env = self.mod.BLBStage2LayerwiseEnv(
            base_env=self.base, fusion_map=self.fusion_map,
        )

    class _FakeBase:
        def __init__(self):
            self.num_layers = 12
            self.env_cfg = types.SimpleNamespace(profile="mrpc")
            self.gelu_degree = [4] * 12
            self.attn_degree = [6] * 12
            self.gelu_degree_state = 4
            self.attn_degree_state = 6
            self.reset_seeds = []
            self.step_calls = []
            self.terminal_info = {"priority": 3}

        def reset(self, *, seed=None):
            self.reset_seeds.append(seed)
            return np.asarray([99.0], dtype=np.float32)

        def step(self, action, **kwargs):
            self.step_calls.append((np.asarray(action, dtype=int).copy(), kwargs))
            return np.asarray([999.0], dtype=np.float32), 7.25, True, self.terminal_info

    def test_reset_has_stable_geometry_and_owned_state(self):
        observation = self.env.reset(seed=17)

        self.assertEqual(self.base.reset_seeds, [17])
        self.assertEqual(self.env.horizon, 12)
        self.assertEqual(self.env.max_step_dim, 6)
        self.assertEqual(observation.shape, (self.env.state_dim,))
        self.assertEqual(self.env.current_spec().layer_idx, 0)
        self.assertEqual(len(self.env.schedule), 12)
        schedule = self.env.schedule
        schedule.clear()
        self.assertEqual(len(self.env.schedule), 12)
        exposed = self.env.pending_full_vector
        exposed[0] = -1
        self.assertNotEqual(self.env.pending_full_vector[0], -1)

    def test_evaluates_four_then_five_blocks_and_records_one_row(self):
        self.env.reset()
        next_obs, reward, done, info = self.env.step([1, 5, 0, 1, 2, 3])

        self.assertEqual([call["block_idx"] for call in self.runtime_calls], [2, 3, 4, 5])
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["layer_summary"]["active_block_count"], 4)
        self.assertEqual(len(self.env.layer_summaries), 1)
        self.assertEqual(next_obs.shape, (self.env.state_dim,))
        self.assertEqual(len(self.base.step_calls), 0)

        self.runtime_calls.clear()
        self.env.step([0, 0, 0, 0, 0, 0])
        self.assertEqual([call["block_idx"] for call in self.runtime_calls], [1, 2, 3, 4, 5])
        self.assertEqual(len(self.env.layer_summaries), 2)

    def test_terminal_calls_base_once_with_full_cost_and_handoff(self):
        self.env.reset()
        actions = []
        for layer_idx in range(12):
            action = [layer_idx % 2, 0, 1, 2, 3, 4]
            actions.append(action[:])
            obs, reward, done, info = self.env.step(action)
            if layer_idx < 11:
                self.assertEqual((reward, done), (0.0, False))
                self.assertEqual(len(self.base.step_calls), 0)

        self.assertTrue(done)
        self.assertEqual(reward, 7.25)
        self.assertEqual(obs.shape, (self.env.state_dim,))
        self.assertEqual(len(self.base.step_calls), 1)
        terminal_vector, kwargs = self.base.step_calls[0]
        self.assertEqual(kwargs["external_cost_score"], info["variable_cost"]["normalized"])
        self.assertEqual(kwargs["external_cost_rank"], info["variable_cost"]["normalized"])
        self.assertEqual(info["external_cost_score"], info["variable_cost"]["normalized"])
        self.assertEqual(info["external_cost_rank"], info["variable_cost"]["normalized"])
        self.assertGreaterEqual(info["external_cost_score"], 0.0)
        self.assertLessEqual(info["external_cost_score"], 1.0)
        expected_boosted_rows = [
            {
                "block_idx": int(block_idx),
                "layer_idx": int(layer_idx),
                "field_values": dict(field_values),
            }
            for (block_idx, layer_idx), field_values in sorted(
                kwargs["boosted_overrides"].items(),
                key=lambda item: (int(item[0][1]), int(item[0][0])),
            )
        ]
        self.assertEqual(info["boosted_overrides"], expected_boosted_rows)
        self.assertEqual(info["terminal_info"], {"priority": 3})
        self.assertEqual(info["terminal_reward"], 7.25)
        self.assertEqual(info["policy_actions"], actions)
        self.assertEqual(len(info["decoded_actions"]), 12)
        self.assertEqual(len(info["layer_summaries"]), 12)
        self.assertEqual(len(info["block4_fusion_choices"]), 12)
        self.assertEqual(len(info["k_choices"]), 59)
        self.assertEqual(len(info["fusion_option_ids"]), 12)
        self.assertTrue(info["boosted_overrides"])
        self.assertIsInstance(info["pending_full_vector"], list)
        self.assertEqual(info["pending_full_vector"], terminal_vector.tolist())
        json.dumps(info)

        # Block3 keeps baseline SF indices but carries the policy's K index.
        self.assertTrue(np.all(terminal_vector[32:39] == 14))
        self.assertEqual(terminal_vector[39], 2)

    def test_invalid_block_is_aggregated_without_early_termination(self):
        self.invalid_key = (0, 2)
        self.env.reset()
        _obs, reward, done, info = self.env.step([0, 0, 0, 0, 0, 0])

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertFalse(info["layer_summary"]["all_valid"])
        self.assertEqual(info["layer_summary"]["blocks"][0]["invalid_chain"], {
            "reason": "invalid test chain",
        })
        for _ in range(11):
            _obs, reward, done, info = self.env.step([0, 0, 0, 0, 0, 0])
        self.assertTrue(done)
        self.assertEqual(reward, 7.25)
        self.assertEqual(len(self.base.step_calls), 1)

    def test_rejects_misuse_and_owns_actions_and_terminal_results(self):
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.step([0] * 6)
        self.env.reset()
        action = [1, 0, 0, 0, 0, 0]
        self.env.step(action)
        action[0] = 0
        self.assertEqual(self.env.action_history[0][0], 1)
        history = self.env.action_history
        history[0][0] = 0
        self.assertEqual(self.env.action_history[0][0], 1)
        with self.assertRaises(ValueError):
            self.env.step([0] * 5)
        for _ in range(11):
            _obs, _reward, done, info = self.env.step([0] * 6)
        self.assertTrue(done)
        info["policy_actions"][0][0] = 0
        self.assertEqual(self.env.action_history[0][0], 1)
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            self.env.step([0] * 6)

    def test_reset_after_terminal_starts_a_fresh_episode(self):
        self.env.reset(seed=1)
        for _ in range(12):
            _obs, _reward, done, _info = self.env.step([0] * 6)
        self.assertTrue(done)

        obs = self.env.reset(seed=2)

        self.assertEqual(self.base.reset_seeds, [1, 2])
        self.assertEqual(obs.shape, (self.env.state_dim,))
        self.assertEqual(self.env.current_spec().layer_idx, 0)
        self.assertEqual(self.env.action_history, [])
        self.assertEqual(self.env.layer_summaries, [])
        _obs, reward, done, info = self.env.step([0] * 6)
        self.assertEqual((reward, done), (0.0, False))
        self.assertEqual(info["layer_idx"], 0)

    def test_nonuniform_degrees_select_block3_and_block5_graphs_per_layer(self):
        self.base.gelu_degree = [4, 2] + [1] * 10
        self.base.attn_degree = [6, 4] + [2] * 10
        self.env.reset()

        self.env.step([0] * 6)
        layer0_graphs = {
            int(call["block_idx"]): str(call["graph_key"])
            for call in self.runtime_calls
        }
        self.runtime_calls.clear()
        self.env.step([0] * 6)
        layer1_graphs = {
            int(call["block_idx"]): str(call["graph_key"])
            for call in self.runtime_calls
        }

        self.assertEqual(layer0_graphs[3], "block3_exp_n6")
        self.assertEqual(layer0_graphs[5], "block5_n4")
        self.assertEqual(layer1_graphs[3], "block3_exp_n4")
        self.assertEqual(layer1_graphs[5], "block5_n2")

    def test_terminal_info_snapshot_is_bounded_json_safe_and_runtime_info_is_private(self):
        @dataclasses.dataclass
        class Metrics:
            loss_mean: object
            metric1_mean: object
            loss_trials: object
            trial_seeds: object

        @dataclasses.dataclass
        class Breakdown:
            reward: object
            priority: object
            invalid: object
            cost_score: object
            acc_barrier_sat: object

        class HeavyDecoded:
            def __init__(self):
                self.large_cfgs = [object()] * 1000

        runtime_info = {
            "metrics": Metrics(
                loss_mean=np.float32(0.25),
                metric1_mean=np.float64(0.875),
                loss_trials=np.asarray([0.2, 0.3], dtype=np.float32),
                trial_seeds=(np.int64(11), np.int64(12)),
            ),
            "reward_breakdown": Breakdown(
                reward=np.float64(1.75),
                priority=np.int64(3),
                invalid=np.bool_(False),
                cost_score=np.float32(0.75),
                acc_barrier_sat=np.float64(2.5),
            ),
            "action_hash": "abc123",
            "invalid": np.bool_(False),
            "eval_failed": False,
            "forward_ran": True,
            "probe_diagnostics": {
                "wall_seconds": np.float64(1.25),
                "devices": ("cuda:0", "cuda:1"),
                "per_worker_trial_counts": np.asarray([1, 1], dtype=np.int64),
            },
            "timing": {"cost_eval_wall_seconds": np.float32(0.125)},
            "fusion_count_b2": np.int64(12),
            "fusion_count_b4": np.int64(6),
            "fusion_action_steps": [{"layer_idx": np.int64(0), "option_id": np.int64(1)}],
            "decoded": HeavyDecoded(),
        }
        self.base.terminal_info = runtime_info
        self.env.reset()
        for _ in range(12):
            _obs, _reward, _done, info = self.env.step([0] * 6)

        snapshot = info["terminal_info"]
        self.assertIs(self.env.runtime_terminal_info, runtime_info)
        with self.assertRaises(AttributeError):
            self.env.runtime_terminal_info = {}
        self.assertNotIn("decoded", snapshot)
        self.assertEqual(snapshot["metrics"]["loss_mean"], 0.25)
        np.testing.assert_allclose(snapshot["metrics"]["loss_trials"], [0.2, 0.3])
        self.assertEqual(snapshot["metrics"]["trial_seeds"], [11, 12])
        self.assertEqual(snapshot["reward_breakdown"]["priority"], 3)
        self.assertEqual(snapshot["reward_breakdown"]["cost_score"], 0.75)
        self.assertEqual(snapshot["probe_diagnostics"]["per_worker_trial_counts"], [1, 1])
        self.assertEqual(snapshot["fusion_count_b2"], 12)
        json.dumps(info)

        self.env.reset(seed=99)
        self.assertIsNone(self.env.runtime_terminal_info)


class LayerwiseRealHelperIntegrationTest(unittest.TestCase):
    """Exercise the real shared block runtime through the layerwise env."""

    @classmethod
    def setUpClass(cls):
        pkg_name = "_blb_layerwise_real_helper_test_pkg"
        for name in list(sys.modules):
            if name == pkg_name or name.startswith(f"{pkg_name}."):
                del sys.modules[name]
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(BLB_DIR)]
        sys.modules[pkg_name] = pkg
        cls.events = []

        def load(name, path):
            loader = importlib.machinery.SourceFileLoader(name, str(path))
            spec = importlib.util.spec_from_loader(name, loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loader.exec_module(module)
            return module

        cls.fcm = load(f"{pkg_name}.fusion_count_map", BLB_DIR / "fusion_count_map.py")
        cls.layerwise_action = load(
            f"{pkg_name}.layerwise_action", BLB_DIR / "layerwise_action.py",
        )

        action_space = types.ModuleType(f"{pkg_name}.action_space")
        action_space._BLOCK_SPECS = {
            block_idx: types.SimpleNamespace(
                fields=tuple((f"block{block_idx}_field_{slot_idx}",) for slot_idx in range(width)),
            )
            for block_idx, width in {1: 9, 2: 23, 3: 8, 4: 17, 5: 16}.items()
        }
        action_space._block_default_N = lambda block_idx, **kwargs: 1000 + int(block_idx)
        action_space._degree_for_layer = (
            lambda value, layer_idx, num_layers, **kwargs:
            int(value[layer_idx] if isinstance(value, (list, tuple)) else value)
        )
        action_space.K_LEVELS = cls.layerwise_action.K_LEVELS
        action_space.BlockStepSpec = object

        def all_max(num_layers):
            vector = np.full(int(num_layers) * 73 + 1, 14, dtype=int)
            k_index = cls.layerwise_action.K_LEVELS.index(13)
            for layer_idx in range(int(num_layers)):
                for start, width in ((0, 9), (9, 23), (32, 8), (40, 17), (57, 16)):
                    vector[layer_idx * 73 + start + width - 1] = k_index
            vector[-1] = 4
            return vector

        action_space.make_all_max_action_vector = all_max

        def decode(vector, max_sfs, **kwargs):
            layer_idx, block_idx = kwargs["only"]
            cls.events.append(("decode", int(layer_idx), int(block_idx)))
            cfg = types.SimpleNamespace(
                marker=f"decoded-b{int(block_idx)}",
                layer_idx=int(layer_idx),
                block_idx=int(block_idx),
            )
            return types.SimpleNamespace(
                cfgs_dict=lambda: {f"block{int(block_idx)}": {int(layer_idx): cfg}},
            )

        def build(block_idx, layer_idx, values, **kwargs):
            cls.events.append(("build", int(layer_idx), int(block_idx), dict(values)))
            return types.SimpleNamespace(
                marker=f"boosted-b{int(block_idx)}",
                layer_idx=int(layer_idx),
                block_idx=int(block_idx),
                values=dict(values),
            )

        action_space.action_vector_to_cfgs = decode
        action_space.build_block_cfg_from_field_values = build
        action_space.fusion_step_schedule = lambda *args, **kwargs: []
        action_space.horizon_for_num_layers = lambda layers: int(layers)
        action_space.resolve_fusion_map_option_id = lambda spec, index: int(index)
        action_space.splice_fusion_step_into_full_vec = lambda *args: None
        action_space.splice_step_action_into_full_vec = lambda *args: None
        action_space.step_schedule = lambda *args, **kwargs: []
        action_space.step_schedule_max_dim = lambda layers: 2
        sys.modules[action_space.__name__] = action_space

        env_module = types.ModuleType(f"{pkg_name}.env")
        env_module.BLBStage2Env = object
        sys.modules[env_module.__name__] = env_module
        fusion_cost = types.ModuleType(f"{pkg_name}.fusion_cost")
        fusion_cost.BlockChoice = object
        fusion_cost.compute_fusion_cost_saving = lambda *args, **kwargs: None
        sys.modules[fusion_cost.__name__] = fusion_cost
        reward = types.ModuleType(f"{pkg_name}.reward")
        reward.FUSION_COST_BUDGET_FRACTION = 0.5
        reward.FUSION_COST_W = 0.5
        reward.FUSION_SATURATION_TAU = 0.0
        reward.TRUNC_COST_W = 0.5
        reward.stage1_dense_cost_reward = lambda *args, **kwargs: 0.0
        sys.modules[reward.__name__] = reward
        optimizer = types.ModuleType(f"{pkg_name}.optimizer_cost")

        def apply_optimizer(**kwargs):
            config_name = next(iter(kwargs["opt_outputs"]))
            cfg = next(iter(next(iter(kwargs["cfgs_dict"].values())).values()))
            cls.events.append(("apply", config_name, cfg.marker))
            cfg.marker = f"{cfg.marker}:applied"
            overrides = [{"cfg_attr": "marker", "new_value": cfg.marker}]
            return {
                "model_uses_replan_config": True,
                "optimizer_cfg_overrides": {config_name: overrides},
            }

        optimizer.apply_optimizer_outputs_to_cfgs = apply_optimizer
        sys.modules[optimizer.__name__] = optimizer

        cls.sequential = load(
            f"{pkg_name}.sequential_env", BLB_DIR / "sequential_env.py",
        )
        cls.layerwise_env = load(
            f"{pkg_name}.layerwise_env", BLB_DIR / "layerwise_env.py",
        )
        cls.fusion_map = cls._fusion_map()

    @classmethod
    def _fusion_map(cls):
        def option(option_id, fusion_count, width, marker, *, boosted=False):
            payload = {
                "option_id": option_id,
                "fusion_count": fusion_count,
                "tie_index": 0,
                "total_variance": 1.0,
                "total_bits": 100,
                "slots": {},
                "action_indices": [marker] * width,
            }
            if boosted:
                payload["boosted"] = True
                payload["explicit_field_values"] = {
                    "boosted_sf": 61,
                    "output_truncation_k": 13,
                }
            return payload

        def graph(key, width, options):
            return {
                "graph_key": key,
                "k_slot_index": width - 1,
                "block_num_slots": width,
                "options": options,
            }

        return cls.fcm.FusionCountMap.from_payload({
            "profile": "mrpc",
            "graphs": {
                "block1_mrpc": graph("block1_mrpc", 9, [option(0, 0, 9, 14)]),
                "block2_mrpc": graph("block2_mrpc", 23, [
                    option(0, 0, 23, 14), option(1, 1, 23, 9),
                ]),
                "block4": graph("block4", 17, [
                    option(0, 0, 17, 14), option(1, 1, 17, 8),
                ]),
                "block5_n4": graph("block5_n4", 16, [
                    option(0, 0, 16, 14), option(1, 1, 16, 7, boosted=True),
                ]),
            },
        })

    def test_layerwise_ordinary_block2_and_boosted_block5_use_real_helper_chain(self):
        self.events.clear()

        class Bridge:
            invoker = types.SimpleNamespace(baselines={})

            def evaluate(inner_self, **kwargs):
                cfg = kwargs["cfg"]
                self.events.append(("bridge", kwargs["config_name"], cfg.marker))
                return types.SimpleNamespace(
                    valid=True,
                    total_bits=100 + int(cfg.block_idx),
                    fusion_count=1 if int(cfg.block_idx) in (2, 5) else 0,
                    invalid_chain=None,
                )

        base = types.SimpleNamespace(
            num_layers=12,
            max_sfs=object(),
            gelu_degree=[4] * 12,
            attn_degree=[6] * 12,
            gelu_degree_state=4,
            attn_degree_state=6,
            env_cfg=types.SimpleNamespace(profile="mrpc", rotation_name_map={}),
            rescale_bridge=Bridge(),
            reset=lambda **kwargs: np.zeros(1, dtype=np.float32),
            step=lambda *args, **kwargs: self.fail("terminal base step must not run at layer 0"),
        )
        env = self.layerwise_env.BLBStage2LayerwiseEnv(
            base_env=base, fusion_map=self.fusion_map,
        )
        self.assertIs(
            self.layerwise_env.evaluate_block_from_full_vector,
            self.sequential.evaluate_block_from_full_vector,
        )

        env.reset(seed=17)
        _obs, reward, done, info = env.step([0, 99, 0, 0, 0, 0])

        self.assertEqual((reward, done), (0.0, False))
        self.assertEqual([row["block_idx"] for row in info["layer_summary"]["blocks"]], [2, 3, 4, 5])
        self.assertIn(("decode", 0, 2), self.events)
        self.assertNotIn(("build", 0, 2), [event[:3] for event in self.events])
        self.assertIn(("bridge", "block2_mrpc_L0", "decoded-b2"), self.events)
        self.assertIn(("apply", "block2_mrpc_L0", "decoded-b2"), self.events)
        self.assertIn(("decode", 0, 5), self.events)
        self.assertIn(("build", 0, 5), [event[:3] for event in self.events])
        self.assertIn(("bridge", "block5_n4_L0", "boosted-b5"), self.events)
        self.assertIn(("apply", "block5_n4_L0", "boosted-b5"), self.events)
        self.assertEqual(len([event for event in self.events if event[0] == "apply"]), 4)


if __name__ == "__main__":
    unittest.main()
