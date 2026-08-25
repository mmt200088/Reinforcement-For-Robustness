from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Stage1SearchProducerWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        start = source.index(
            "        if (\n"
            "                not self.skip_stage1_rl\n"
            '                and self.blb_v3_search_backend != "ppo"\n'
            "        ):"
        )
        end = source.index(
            "        if (\n"
            "                not self.skip_stage1_rl\n"
            '                and self.blb_v3_search_backend == "ppo"\n'
            "        ):",
            start,
        )
        cls.block = source[start:end]

    def test_stage1_search_reopens_ordinary_completed_result_before_selection(self):
        self.assertIn(
            "from rfr.search.comparators.common.stage1_runner import (\n"
            "                Stage1SearchGracefulStop,\n"
            "                build_stage1_search_accounting,\n"
            "                load_completed_search_result,\n"
            "                run_stage1_search,\n"
            "            )",
            self.block,
        )
        reopen_call = "stage1_comparator_result = load_completed_search_result("
        self.assertIn(reopen_call, self.block)
        self.assertIn("stage1_output_dir\n            )", self.block)
        run_index = self.block.index("in_memory_stage1_result = run_stage1_search(")
        reopen_index = self.block.index(reopen_call)
        selected_index = self.block.index("selected_stage1 = stage1_comparator_result.best")
        self.assertLess(run_index, reopen_index)
        self.assertLess(reopen_index, selected_index)

    def test_smoke_validates_canonical_config_before_capping_real_evaluations(self):
        factory_index = self.block.index("stage1_search_config = stage1_comparator_search_config(backend)")
        validation_index = self.block.index("validate_stage1_comparator_setup(", factory_index)
        smoke_index = self.block.index("if self.comparator_smoke:", validation_index)
        replace_index = self.block.index("stage1_search_config = replace(", smoke_index)
        run_index = self.block.index("in_memory_stage1_result = run_stage1_search(", replace_index)

        self.assertLess(factory_index, validation_index)
        self.assertLess(validation_index, smoke_index)
        self.assertLess(smoke_index, replace_index)
        self.assertLess(replace_index, run_index)
        self.assertIn("from dataclasses import replace", self.block)
        self.assertIn("evaluation_cap=1", self.block)
        self.assertIn(
            '"comparator_smoke": bool(self.comparator_smoke)',
            self.block,
        )
        self.assertIn(
            'not self.comparator_smoke\n                    and backend == "greedy"',
            self.block,
        )
        self.assertIn(
            'not self.comparator_smoke\n                    and backend == "coinn_ga"',
            self.block,
        )

    def test_stage1_comparator_installs_candidate_boundary_graceful_stop(self):
        self.assertIn("Stage1SearchGracefulStop", self.block)
        self.assertIn("NOISE_STAGE_STOP_FLAG_FILENAME", self.block)
        self.assertIn("install_graceful_stop_handler", self.block)
        self.assertIn("is_graceful_stop_requested", self.block)
        self.assertIn(
            "stop_requested=stage1_comparator_stop_requested",
            self.block,
        )
        self.assertIn('"stopped_by": "graceful_stop"', self.block)
        self.assertIn("uninstall_graceful_stop_handler()", self.block)

    def test_stage1_selection_uses_plain_binding_and_locator(self):
        self.assertIn("stage1_selection_binding = {", self.block)
        for field in (
            '"backend": backend',
            '"action": list(selected_stage1.action)',
            '"gelu_degrees": list(selected_stage1.gelu_degrees)',
            '"softmax_degrees": list(selected_stage1.softmax_degrees)',
            '"num_layers": int(self.total_layers)',
        ):
            self.assertIn(field, self.block)
        self.assertIn(
            'stage1_result_path = os.path.join(\n                stage1_output_dir, "result.json",\n            )',
            self.block,
        )
        self.assertIn('"selection_binding": stage1_selection_binding', self.block)
        self.assertIn('"feasible": bool(selected_stage1.feasible)', self.block)
        self.assertIn('"result_sha256": stage1_result_sha256', self.block)
        self.assertIn('"selection_hash": stable_json_hash(', self.block)

    def test_stage1_provenance_module_is_removed(self):
        self.assertFalse((ROOT / "stage1_rl" / "provenance.py").exists())

    def test_stage1_producer_has_no_authority_or_provenance_protocol(self):
        for removed in (
            "stage1_rl.provenance",
            "canonical_stage1_result_path",
            "canonical_formal_stage1_search_config",
            "load_completed_stage1_search_authority",
            "stage1_completed_authority",
            "formal_run_identity",
            "formal_stage1_contract",
            "selection_provenance",
            "authenticated sealed completion",
        ):
            self.assertNotIn(removed, self.block)


if __name__ == "__main__":
    unittest.main()
