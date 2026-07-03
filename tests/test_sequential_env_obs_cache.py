import unittest
from types import SimpleNamespace

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - local lightweight env
    BLBStage2SequentialEnv = None  # type: ignore[assignment]
    SequentialEnvConfig = None  # type: ignore[assignment]
else:
    from blb_stage2_rl.sequential_env import BLBStage2SequentialEnv, SequentialEnvConfig


@unittest.skipIf(BLBStage2SequentialEnv is None, "torch not available")
class SequentialEnvObsCacheTest(unittest.TestCase):
    def _env(self) -> BLBStage2SequentialEnv:
        env = BLBStage2SequentialEnv.__new__(BLBStage2SequentialEnv)
        env.cfg = SequentialEnvConfig()
        env.num_layers = 12
        env.horizon = 4
        env._max_step_dim = 2
        env.base = SimpleNamespace(attn_degree_state=4, gelu_degree_state=4)
        env._schedule = [
            SimpleNamespace(block_idx=2, layer_idx=0),
            SimpleNamespace(block_idx=4, layer_idx=0),
            SimpleNamespace(block_idx=5, layer_idx=0),
            SimpleNamespace(block_idx=1, layer_idx=1),
        ]
        env._step_idx = 0
        env._prev_actions = []
        env._prev_signals = []
        env._prev_actions_obs_buf = np.zeros(
            (env.horizon, env._max_step_dim), dtype=np.float32,
        )
        env._prev_signals_obs_buf = np.zeros((env.horizon, 3), dtype=np.float32)
        return env

    def _legacy_history_tail(self, env: BLBStage2SequentialEnv) -> np.ndarray:
        actions = np.zeros((env.horizon, env._max_step_dim), dtype=np.float32)
        for i, action in enumerate(env._prev_actions):
            actions[i, : len(action)] = np.asarray(action, dtype=np.float32) / 8.0
        signals = np.zeros((env.horizon, 3), dtype=np.float32)
        for i, signal in enumerate(env._prev_signals):
            signals[i, 0] = float(signal["valid"])
            signals[i, 1] = float(signal["total_bits"]) / 1000.0
            signals[i, 2] = float(signal["fusion_count"]) / 10.0
        return np.concatenate([actions.reshape(-1), signals.reshape(-1)])

    def test_incremental_obs_buffers_match_legacy_history_encoding(self):
        env = self._env()
        env._record_step(
            env.current_spec(),
            np.asarray([1, 3], dtype=np.int64),
            valid=True,
            total_bits=1234,
            fusion_count=2,
        )
        env._record_step(
            env.current_spec(),
            np.asarray([0, 5], dtype=np.int64),
            valid=False,
            total_bits=777,
            fusion_count=0,
            error="invalid",
        )

        cached_tail = np.concatenate([
            env._prev_actions_obs_buf.reshape(-1),
            env._prev_signals_obs_buf.reshape(-1),
        ])
        np.testing.assert_array_equal(cached_tail, self._legacy_history_tail(env))

        obs = env._build_obs()
        expected_width = env._compute_obs_width()
        self.assertEqual(obs.shape, (expected_width,))
        np.testing.assert_array_equal(obs[-cached_tail.size:], cached_tail)


if __name__ == "__main__":
    unittest.main()
