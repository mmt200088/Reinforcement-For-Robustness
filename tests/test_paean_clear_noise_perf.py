from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "Paean" / "blb_action_eval.py"


class PaeanClearNoisePerfTests(unittest.TestCase):
    def _method_source(self, name: str, next_name: str) -> str:
        source = SOURCE.read_text(encoding="utf-8")
        return source.split(f"    def {name}", 1)[1].split(f"    def {next_name}", 1)[0]

    def test_clear_legacy_noise_reuses_layer_indices_list(self):
        body = self._method_source("_clear_legacy_noise", "_clear_all_noise")

        self.assertIn("layer_indices = list(range(ev.total_layers))", body)
        self.assertIn("layer_indices=layer_indices", body)
        self.assertEqual(body.count("list(range(ev.total_layers))"), 1)

    def test_clear_all_noise_reuses_layer_indices_list(self):
        body = self._method_source("_clear_all_noise", "_save_results_markdown")

        self.assertIn("layer_indices = list(range(ev.total_layers))", body)
        self.assertIn("layer_indices=layer_indices", body)
        self.assertEqual(body.count("list(range(ev.total_layers))"), 1)


if __name__ == "__main__":
    unittest.main()
