import importlib.util
from pathlib import Path
import statistics
import sys
import types
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "scripts" / "stage1_approx_reuse_benchmark.py"


def _load_benchmark_module():
    fake_torch = types.ModuleType("torch")
    fake_transformers = types.ModuleType("transformers")
    fake_function_handler = types.ModuleType("function_handler")
    fake_transformers.BertConfig = object
    fake_transformers.BertForSequenceClassification = object
    fake_function_handler.ReversibleLayerHandler = object
    original = {
        name: sys.modules.get(name)
        for name in ("torch", "transformers", "function_handler")
    }
    sys.modules.update(
        {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "function_handler": fake_function_handler,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "stage1_approx_reuse_benchmark", BENCHMARK_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in original.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class Stage1ApproxReuseBenchmarkTest(unittest.TestCase):
    def test_summarize_timings_computes_means_without_intermediate_ms_lists(self):
        benchmark = _load_benchmark_module()

        self.assertTrue(hasattr(benchmark, "_summarize_timings"))
        with mock.patch.object(
            statistics,
            "mean",
            side_effect=AssertionError("timing summary should avoid statistics.mean"),
        ):
            summary = benchmark._summarize_timings(
                fast_install=[0.001, 0.003],
                fast_fwd=[0.009, 0.007],
                slow_install=[0.004, 0.006],
                slow_fwd=[0.016, 0.014],
            )

        self.assertEqual(summary["num_episodes_timed"], 2)
        self.assertAlmostEqual(summary["reuse_on"]["install_ms_mean"], 2.0)
        self.assertAlmostEqual(summary["reuse_on"]["forward_ms_mean"], 8.0)
        self.assertAlmostEqual(summary["reuse_on"]["total_ms_mean"], 10.0)
        self.assertAlmostEqual(summary["reuse_off"]["install_ms_mean"], 5.0)
        self.assertAlmostEqual(summary["reuse_off"]["forward_ms_mean"], 15.0)
        self.assertAlmostEqual(summary["reuse_off"]["total_ms_mean"], 20.0)
        self.assertAlmostEqual(summary["episode_speedup"], 2.0)
        self.assertAlmostEqual(summary["install_speedup"], 2.5)

    def test_parser_reuses_static_model_type_choices(self):
        source = BENCHMARK_PATH.read_text(encoding="utf-8")
        main_region = source[
            source.index("def main():"):
            source.index("\n\nif __name__ == \"__main__\":")
        ]

        self.assertIn("_MODEL_TYPES = tuple(_DIMS)", source)
        self.assertIn('choices=_MODEL_TYPES', main_region)
        self.assertNotIn("choices=list(_DIMS)", main_region)


if __name__ == "__main__":
    unittest.main()
