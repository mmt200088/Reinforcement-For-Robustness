import unittest


class BLBWarmstartResumeTests(unittest.TestCase):
    def test_absolute_episode_still_anchors_after_resume_before_anchor_end(self):
        from blb_stage2_rl.runner import _effective_warmstart_anchor_episodes

        self.assertEqual(
            _effective_warmstart_anchor_episodes(
                configured_anchor_episodes=200,
                rollout_size=120,
                total_episodes=1000,
                start_episode=0,
                disable_warmstart_on_resume=False,
            ),
            200,
        )
        self.assertEqual(
            _effective_warmstart_anchor_episodes(
                configured_anchor_episodes=200,
                rollout_size=120,
                total_episodes=1000,
                start_episode=50,
                disable_warmstart_on_resume=False,
            ),
            200,
        )

    def test_anchor_is_not_extended_after_absolute_episode_passes_anchor_end(self):
        from blb_stage2_rl.runner import _warmstart_action_mode

        kwargs = {
            "anchor_episodes": 200,
            "cost_probe_count": 0,
            "neighbor_sampling": False,
            "has_mutable_neighbors": False,
            "neighbor_ramp_episodes": 0,
        }

        self.assertEqual(_warmstart_action_mode(episode_index=50, **kwargs), ("anchor", -1))
        self.assertEqual(_warmstart_action_mode(episode_index=250, **kwargs), ("policy", -1))

    def test_explicit_disable_warmstart_on_resume_turns_anchor_off(self):
        from blb_stage2_rl.runner import _effective_warmstart_anchor_episodes

        self.assertEqual(
            _effective_warmstart_anchor_episodes(
                configured_anchor_episodes=200,
                rollout_size=120,
                total_episodes=1000,
                start_episode=50,
                disable_warmstart_on_resume=True,
            ),
            0,
        )
