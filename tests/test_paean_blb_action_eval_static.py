import unittest

from tests.source_inspection_utils import source_text


class PaeanBLBActionEvalStaticTest(unittest.TestCase):
    def test_plot_rendering_is_opt_out_and_defaults_enabled(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        run_start = text.index("def run(")
        run_end = text.index("def _resolve_base_action", run_start)
        run_source = text[run_start:run_end]

        self.assertIn('"RFR_PAEAN_RENDER_PLOTS", "1"', text)
        self.assertIn("if self._render_plots_enabled():", run_source)
        self.assertIn("plot_path = None", run_source)
        self.assertIn("scatter_path = None", run_source)

    def test_batch_candidates_reset_independent_process_seed_before_eval(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        loop_start = text.index("for idx, candidate in enumerate(selected_candidates")
        loop_end = text.index("cost_match_diagnostics:", loop_start)
        loop = text[loop_start:loop_end]
        lifecycle_start = text.index(
            "def _evaluate_candidate_with_seed_lifecycle("
        )
        lifecycle_end = text.index(
            "def _validate_stage2_final_eval_handoff(", lifecycle_start
        )
        lifecycle = text[lifecycle_start:lifecycle_end]

        self.assertIn(
            "self._evaluate_candidate_with_seed_lifecycle(", loop
        )
        self.assertLess(
            loop.index("self._evaluate_candidate_with_seed_lifecycle("),
            loop.index("self._evaluate_action_candidate("),
        )
        self.assertIn("def _capture_isolated_candidate_rng_state(", text)
        self.assertIn("def _restore_isolated_candidate_rng_state(", text)
        self.assertIn("random.setstate(state[\"python\"])", text)
        self.assertIn("np.random.set_state(state[\"numpy\"])", text)
        self.assertIn("torch.random.set_rng_state(state[\"torch_cpu\"])", text)
        self.assertIn("torch.cuda.set_rng_state_all(state[\"torch_cuda\"])", text)
        self.assertIn("from rfr.search.runtime.model_handler import reseed_noise_rng", text)
        self.assertIn(
            "isolate_random_seed=(final_eval_handoff is not None)",
            text,
        )
        self.assertIn("reseed_noise_rng(self.random_seed)", text)
        self.assertNotIn("formal_noise_seed_authority", lifecycle)
        self.assertNotIn("paean_candidate_noise_seed_authority(", lifecycle)
        self.assertIn("finally:", lifecycle)
        self.assertIn("reseed_noise_rng(None)", lifecycle)

    def test_random_candidates_reset_seed_before_eval_and_restore_entropy(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        setup_start = text.index(
            "random_candidates, cost_match_diagnostics = ("
        )
        loop_start = text.index(
            "for idx, candidate in enumerate(random_candidates, start=1):"
        )
        loop_end = text.index(
            "results = selected_results + random_results", loop_start
        )
        setup = text[setup_start:loop_start]
        loop = text[loop_start:loop_end]

        self.assertIn('"isolate_random_seed": True', setup)
        self.assertNotIn("formal_noise_seed_authority", setup)
        self.assertNotIn("paean_candidate_noise_seed_authority(", setup)
        self.assertIn(
            "self._evaluate_candidate_with_seed_lifecycle(",
            loop,
        )
        self.assertLess(
            loop.index("self._evaluate_candidate_with_seed_lifecycle("),
            loop.index("self._evaluate_action_candidate("),
        )

    def test_comparator_forward_accepts_configured_common_random_seed(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        run_start = text.index("def run(")
        run_end = text.index("def _resolve_base_action", run_start)
        run_source = text[run_start:run_end]

        self.assertNotIn(
            "formal comparator final-eval requires random seed 42",
            run_source,
        )
        self.assertIn("self._build_rescale_bridge(", run_source)
        self.assertIn("self._evaluate_action_candidate(", run_source)
        self.assertIn("reseed_noise_rng(self.random_seed)", text)

    def test_evaluation_protocol_random_groups_uses_actual_result_count(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        self.assertIn(
            '"random_groups": ("enabled" if len(candidate_results) > 1 else "disabled"),',
            text,
        )
        self.assertNotIn(
            '"random_groups": "enabled" if self.random_enabled else "disabled",',
            text,
        )

    def test_evaluation_protocol_reuses_action_spec_tuples_until_json_conversion(self):
        text = source_text("src/rfr/evaluation/action_eval.py")
        self.assertIn('"action_ranges": self.action_ranges,', text)
        self.assertIn('"action_fixed": self.action_fixed,', text)
        self.assertNotIn('"action_ranges": list(self.action_ranges),', text)
        self.assertNotIn('"action_fixed": list(self.action_fixed),', text)

    def test_final_eval_uses_one_calibrated_context_for_every_decode_surface(self):
        text = source_text("src/rfr/evaluation/action_eval.py")

        self.assertIn("load_calibrated_stage2_action_context", text)
        self.assertNotIn("def _load_max_sfs(", text)
        self.assertNotIn("cache[key] = load_max_sfs(key)", text)
        self.assertIn("max_sfs=action_context.max_sfs", text)
        self.assertIn("action_context.provenance", text)

if __name__ == "__main__":
    unittest.main()
