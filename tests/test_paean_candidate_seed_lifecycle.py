from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
import types
import unittest

_REPO = Path(__file__).resolve().parents[1]
_PAEAN_PATH = _REPO / "src/rfr/evaluation/action_eval.py"


def _load_paean_method(name, **runtime_globals):
    source = _PAEAN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PAEAN_PATH))
    module_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BLBActionFinalEvaluationModule"
    )
    method = next(
        (node for node in module_class.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if method is None:
        raise AssertionError(f"BLBActionFinalEvaluationModule.{name} is missing")
    future = ast.parse("from __future__ import annotations\n").body[0]
    module = ast.Module(body=[future, method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(runtime_globals)
    exec(compile(module, str(_PAEAN_PATH), "exec"), namespace)
    return namespace[name]


class PaeanCandidateSeedLifecycleTest(unittest.TestCase):
    def _runner_and_events(self, *, seed=7):
        events = []

        def reseed_noise_rng(value):
            events.append(value)

        method = _load_paean_method(
            "_evaluate_candidate_with_seed_lifecycle",
            Mapping=Mapping,
            reseed_noise_rng=reseed_noise_rng,
        )
        runner = types.SimpleNamespace(random_seed=seed)

        def restore(_self, metadata, state):
            self.assertEqual(state, {"captured": True})
            self.assertEqual(metadata, {"isolate_random_seed": True})
            reseed_noise_rng(seed)

        runner._restore_isolated_candidate_rng_state = types.MethodType(restore, runner)
        runner.evaluate_with_seed_lifecycle = types.MethodType(method, runner)
        return runner, events

    @staticmethod
    def _ordinary_metadata():
        return {"isolate_random_seed": True}

    def test_selected_and_random_candidates_share_common_random_numbers(self):
        runner, events = self._runner_and_events(seed=7)
        evaluations = []

        for name in ("selected", "random"):
            result = runner.evaluate_with_seed_lifecycle(
                metadata=self._ordinary_metadata(),
                isolated_candidate_rng_state={"captured": True},
                evaluate=lambda name=name: evaluations.append(name) or name,
            )
            self.assertEqual(result, name)

        self.assertEqual(evaluations, ["selected", "random"])
        self.assertEqual(events, [7, None, 7, None])

    def test_candidate_exception_still_restores_unseeded_noise_rng(self):
        runner, events = self._runner_and_events(seed=9)

        def fail_candidate():
            raise RuntimeError("injected candidate failure")

        with self.assertRaisesRegex(RuntimeError, "candidate failure"):
            runner.evaluate_with_seed_lifecycle(
                metadata=self._ordinary_metadata(),
                isolated_candidate_rng_state={"captured": True},
                evaluate=fail_candidate,
            )

        self.assertEqual(events, [9, None])

    def test_selected_and_random_loops_use_the_shared_seed_lifecycle(self):
        text = _PAEAN_PATH.read_text(encoding="utf-8")
        selected_start = text.index("for idx, candidate in enumerate(selected_candidates")
        selected_end = text.index("cost_match_diagnostics:", selected_start)
        random_start = text.index("for idx, candidate in enumerate(random_candidates", selected_end)
        random_end = text.index("results = selected_results + random_results", random_start)

        for loop in (
            text[selected_start:selected_end],
            text[random_start:random_end],
        ):
            self.assertIn("self._evaluate_candidate_with_seed_lifecycle(", loop)
            self.assertNotIn("formal_noise_seed_authority", loop)


if __name__ == "__main__":
    unittest.main()
