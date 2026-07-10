import unittest

from tests.source_inspection_utils import source_text


class PaeanBLBActionEvalStaticTest(unittest.TestCase):
    def test_plot_rendering_is_opt_out_and_defaults_enabled(self):
        text = source_text("Paean/blb_action_eval.py")
        run_start = text.index("def run(")
        run_end = text.index("def _resolve_base_action", run_start)
        run_source = text[run_start:run_end]

        self.assertIn('"RFR_PAEAN_RENDER_PLOTS", "1"', text)
        self.assertIn("if self._render_plots_enabled():", run_source)
        self.assertIn("plot_path = None", run_source)
        self.assertIn("scatter_path = None", run_source)

    def test_batch_candidates_reset_independent_process_seed_before_eval(self):
        text = source_text("Paean/blb_action_eval.py")
        loop_start = text.index("for idx, candidate in enumerate(selected_candidates")
        loop_end = text.index("# ---- Generate cost-matched random candidates", loop_start)
        loop = text[loop_start:loop_end]

        self.assertIn(
            "self._restore_isolated_candidate_rng_state(\n"
            "                candidate.metadata, isolated_candidate_rng_state,\n"
            "            )",
            loop,
        )
        self.assertLess(
            loop.index("self._restore_isolated_candidate_rng_state("),
            loop.index("self._evaluate_action_candidate("),
        )
        self.assertIn("def _capture_isolated_candidate_rng_state(", text)
        self.assertIn("def _restore_isolated_candidate_rng_state(", text)
        self.assertIn("random.setstate(state[\"python\"])", text)
        self.assertIn("np.random.set_state(state[\"numpy\"])", text)
        self.assertIn("torch.random.set_rng_state(state[\"torch_cpu\"])", text)
        self.assertIn("torch.cuda.set_rng_state_all(state[\"torch_cuda\"])", text)
        self.assertIn("from function_handler import reseed_noise_rng", text)
        self.assertIn("reseed_noise_rng(self.random_seed)", text)
        self.assertIn("finally:", loop)
        self.assertIn("reseed_noise_rng(None)", loop)

    def test_evaluation_protocol_reuses_action_spec_tuples_until_json_conversion(self):
        text = source_text("Paean/blb_action_eval.py")
        self.assertIn('"action_ranges": self.action_ranges,', text)
        self.assertIn('"action_fixed": self.action_fixed,', text)
        self.assertNotIn('"action_ranges": list(self.action_ranges),', text)
        self.assertNotIn('"action_fixed": list(self.action_fixed),', text)


if __name__ == "__main__":
    unittest.main()
