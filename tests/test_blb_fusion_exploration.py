"""ADR-012 exploration floor + borderline retest — torch-gated tests.

The epsilon mixture (``BLBStage2SequentialPolicy._action_dist``) and the
env-level borderline retest both need torch, so these run in the server
contract gate and skip on torch-less local boxes. The pure scheduling /
reward-shape parts of ADR-012 are covered torch-free in
``test_blb_fusion_curriculum.py`` / ``test_blb_fusion_reward.py``.
"""
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

STATE_DIM = 32


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class ExplorationEpsilonMixtureTest(unittest.TestCase):
    def _policy(self, eps):
        from blb_stage2_rl.sequential_policy import (
            BLBStage2SequentialPolicy,
            SequentialPolicyConfig,
        )
        cfg = SequentialPolicyConfig(
            state_dim=STATE_DIM, max_step_dim=2, max_num_levels=6, horizon=8,
        )
        torch.manual_seed(42)
        pol = BLBStage2SequentialPolicy(cfg)
        pol.eval()
        if eps is not None:
            pol.set_slot_exploration_epsilon(list(eps))
        return pol

    @staticmethod
    def _inputs():
        obs = torch.zeros(1, STATE_DIM)
        slot_mask = torch.ones(1, 2)
        levels = torch.tensor([[2, 6]])
        return obs, slot_mask, levels

    def test_option_one_floor_probability(self):
        """With eps on slot0, P(option 1) ~= eps/2 even for collapsed logits."""
        pol = self._policy(eps=(0.5, 0.0))
        # saturate slot-0 logits toward option 0 via the warmstart prior at a
        # huge gain — without the mixture option1 would be ~impossible.
        pol.apply_preferred_per_step_bias([0, 0], gain=50.0)
        obs, slot_mask, levels = self._inputs()
        torch.manual_seed(0)
        trials = 2000
        n1 = 0
        with torch.no_grad():
            for _ in range(trials):
                a, _, _ = pol.sample_action(
                    obs, slot_mask, levels, deterministic=False,
                )
                n1 += int(a[0, 0].item() == 1)
        # mixture: P(opt1) = (1-eps)*~0 + eps*(1/2) = 0.25
        self.assertGreater(n1 / trials, 0.18)
        self.assertLess(n1 / trials, 0.32)

    def test_zero_eps_matches_unset(self):
        """eps=0 must reproduce the pre-ADR-012 distribution exactly."""
        pol_a = self._policy(eps=None)
        pol_b = self._policy(eps=(0.0, 0.0))   # same seed -> same weights
        obs, slot_mask, levels = self._inputs()
        act = torch.tensor([[1, 3]])
        with torch.no_grad():
            lp_a, ent_a, _ = pol_a.evaluate_action(obs, act, slot_mask, levels)
            lp_b, ent_b, _ = pol_b.evaluate_action(obs, act, slot_mask, levels)
        self.assertAlmostEqual(float(lp_a.item()), float(lp_b.item()), places=6)
        self.assertAlmostEqual(float(ent_a.item()), float(ent_b.item()), places=6)

    def test_sample_and_evaluate_logprob_consistent_under_eps(self):
        """PPO soundness: sampled log_prob == evaluate_action log_prob."""
        pol = self._policy(eps=(0.3, 0.1))
        obs, slot_mask, levels = self._inputs()
        torch.manual_seed(3)
        with torch.no_grad():
            for _ in range(50):
                a, lp_s, _ = pol.sample_action(
                    obs, slot_mask, levels, deterministic=False,
                )
                lp_e, _, _ = pol.evaluate_action(obs, a, slot_mask, levels)
                self.assertAlmostEqual(
                    float(lp_s.item()), float(lp_e.item()), places=4,
                )

    def test_mask_respected_by_uniform_component(self):
        """Masked levels must keep probability 0 even under the mixture."""
        pol = self._policy(eps=(0.9, 0.9))
        obs, slot_mask, levels = self._inputs()
        mask = torch.zeros(1, 2, 6, dtype=torch.bool)
        mask[0, 0, 0] = True            # option slot: ONLY level 0 allowed
        mask[0, 1, :3] = True           # K slot: levels 0..2 allowed
        torch.manual_seed(11)
        with torch.no_grad():
            for _ in range(300):
                a, _, _ = pol.sample_action(
                    obs, slot_mask, levels, deterministic=False,
                    action_level_mask=mask,
                )
                self.assertEqual(int(a[0, 0].item()), 0)
                self.assertLess(int(a[0, 1].item()), 3)

    def test_entropy_floor_positive_under_eps(self):
        """Even saturated logits keep entropy > 0 under the mixture."""
        pol = self._policy(eps=(0.05, 0.02))
        pol.apply_preferred_per_step_bias([0, 3], gain=50.0)
        obs, slot_mask, levels = self._inputs()
        act = torch.tensor([[0, 3]])
        with torch.no_grad():
            _, ent, _ = pol.evaluate_action(
                obs, act, slot_mask, levels, baseline_prior_scale=1.0,
            )
        self.assertGreater(float(ent.item()), 1e-3)


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class BorderlineRetestUnitTest(unittest.TestCase):
    def test_deficit_norm_helper(self):
        from blb_stage2_rl.env import BLBStage2Env
        from blb_stage2_rl.reward import EpisodeMetrics, RewardWeights

        env = BLBStage2Env.__new__(BLBStage2Env)   # no heavy init
        env.reward_weights = RewardWeights(
            baseline_metric1=0.8672, baseline_metric2=0.8672,
        )
        env.acc_threshold = 0.858
        env.acc_threshold_m2 = 0.858

        def m(m1):
            return EpisodeMetrics(
                loss_mean=0.3, loss_std=0.002, metric1_mean=m1, metric2_mean=m1,
            )

        self.assertEqual(env._acc_worst_deficit_norm(m(0.8672)), 0.0)
        d_small = env._acc_worst_deficit_norm(m(0.8570))
        d_big = env._acc_worst_deficit_norm(m(0.8490))
        self.assertGreater(d_small, 0.0)
        self.assertGreater(d_big, d_small)
        self.assertLess(d_small, 1.0)

    def test_retest_salt_gives_disjoint_trial_seed_stream(self):
        seed = 123456789
        salted = (seed ^ 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        self.assertNotEqual(seed, salted)
        knuth = 2654435761
        first = {(seed ^ (t * knuth)) & 0x7FFFFFFFFFFFFFFF for t in range(5)}
        retest = {(salted ^ (t * knuth)) & 0x7FFFFFFFFFFFFFFF for t in range(10)}
        self.assertFalse(first & retest)


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class WorkersPerDeviceTest(unittest.TestCase):
    """2026-06-12 workers-per-device: assignment expansion + config default."""

    def test_expansion_interleaved(self):
        from blb_stage2_rl.parallel_runner import expand_device_ids_for_workers
        self.assertEqual(
            expand_device_ids_for_workers([0, 1, 2, 3, 4], 2),
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        )
        # wpd=1 must be the identity (bit-for-bit pre-change behavior)
        self.assertEqual(expand_device_ids_for_workers([0, 1], 1), [0, 1])
        self.assertEqual(expand_device_ids_for_workers([3], 4), [3, 3, 3, 3])
        # worker 0 stays on device_ids[0] for any wpd
        for wpd in (1, 2, 3):
            self.assertEqual(expand_device_ids_for_workers([2, 0, 1], wpd)[0], 2)

    def test_chunk_assignment_covers_all_episodes_any_worker_count(self):
        from blb_stage2_rl.seed_utils import assign_global_episodes
        for n, w in ((60, 10), (60, 5), (60, 7), (300, 10), (1, 10)):
            chunks = assign_global_episodes(n, w)
            flat = [g for c in chunks for g in c]
            self.assertEqual(sorted(flat), list(range(n)), (n, w))
            # contiguous chunks in worker order (global-order reassembly)
            self.assertEqual(flat, sorted(flat), (n, w))

    def test_config_default_is_one_worker_per_device(self):
        from blb_stage2_rl.runner import BLBStage2TrainConfig
        self.assertEqual(BLBStage2TrainConfig().stage2_workers_per_device, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
