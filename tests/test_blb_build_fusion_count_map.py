import importlib.util
import unittest
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILDER_PATH = (
    _REPO_ROOT / "src" / "rfr" / "preparation" / "fusion" / "build_map.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location("blb_build_fusion_count_map_test", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FusionCountMapBuilderTests(unittest.TestCase):
    def test_golden_shard_iterator_uses_streaming_pool_results(self):
        builder = _load_builder()
        calls = []

        class FakePool:
            def __init__(self, processes):
                self.processes = processes

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, _fn, _payloads):
                raise AssertionError("pool.map batches all shard results")

            def imap_unordered(self, fn, payloads):
                calls.append(("imap_unordered", self.processes))
                for payload in payloads:
                    yield fn(payload)

        class FakeContext:
            def Pool(self, processes):
                return FakePool(processes)

        def fake_worker(payload):
            return int(payload["valid"]), []

        payloads = [{"valid": 2}, {"valid": 3}]
        with (
            mock.patch.object(builder.mp, "get_context", return_value=FakeContext()),
            mock.patch.object(builder, "_enumerate_shard_worker", side_effect=fake_worker),
        ):
            got = list(builder._iter_golden_shard_results(payloads, num_shards=2))

        self.assertEqual(got, [(2, []), (3, [])])
        self.assertEqual(calls, [("imap_unordered", 2)])

    def test_merge_golden_shard_results_consumes_single_pass_iterator(self):
        builder = _load_builder()

        class SinglePassResults:
            def __init__(self):
                self.used = False

            def __iter__(self):
                if self.used:
                    raise AssertionError("iterator consumed more than once")
                self.used = True
                yield (2, [((1, 2), 0, 10, 1.5, "a"), ((2, 3), 1, 9, 2.5, "b")])
                yield (1, [((3, 4), 2, 8, 3.5, "c")])

        evaluated, total = builder._merge_golden_shard_results(SinglePassResults())

        self.assertEqual(total, 3)
        self.assertEqual([ec.action_indices for ec in evaluated], [(1, 2), (2, 3), (3, 4)])
        self.assertEqual([ec.fusion_count for ec in evaluated], [0, 1, 2])

    def test_merge_golden_shard_results_drops_dominated_rows(self):
        builder = _load_builder()

        evaluated, total = builder._merge_golden_shard_results(iter([
            (2, [
                ((1, 2), 0, 10, 1.0, "best-f0"),
                ((9, 9), 0, 1, 2.0, "dominated-f0"),
            ]),
            (1, [
                ((3, 4), 1, 8, 0.5, "best-f1"),
            ]),
        ]))

        self.assertEqual(total, 3)
        self.assertEqual(
            {ec.action_indices for ec in evaluated},
            {(1, 2), (3, 4)},
        )
        self.assertNotIn((9, 9), {ec.action_indices for ec in evaluated})


if __name__ == "__main__":
    unittest.main()
