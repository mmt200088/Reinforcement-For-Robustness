"""Torch-free behavior tests for the 12-step Stage-2 layerwise environment."""

from __future__ import annotations

import dataclasses
import importlib.machinery
import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BLB_DIR = REPO_ROOT / "blb_stage2_rl"


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

        def reset(self, *, seed=None):
            self.reset_seeds.append(seed)
            return np.asarray([99.0], dtype=np.float32)

        def step(self, action, **kwargs):
            self.step_calls.append((np.asarray(action, dtype=int).copy(), kwargs))
            return np.asarray([999.0], dtype=np.float32), 7.25, True, {"priority": 3}

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
        self.assertEqual(kwargs["boosted_overrides"], info["boosted_overrides"])
        self.assertEqual(info["terminal_info"], {"priority": 3})
        self.assertEqual(info["terminal_reward"], 7.25)
        self.assertEqual(info["policy_actions"], actions)
        self.assertEqual(len(info["decoded_actions"]), 12)
        self.assertEqual(len(info["layer_summaries"]), 12)
        self.assertEqual(len(info["block4_fusion_choices"]), 12)
        self.assertEqual(len(info["k_choices"]), 59)
        self.assertEqual(len(info["fusion_option_ids"]), 12)
        self.assertTrue(info["boosted_overrides"])

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


if __name__ == "__main__":
    unittest.main()
