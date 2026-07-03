import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import rescale_optimizer_bridge as bridge_mod
    from rescale_optimizer_bridge import RescaleOptimizerBridge
except ModuleNotFoundError as exc:  # local macOS test env may be torch-free
    if exc.name != "torch":
        raise
    bridge_mod = None
    RescaleOptimizerBridge = None


def _point(sf):
    return SimpleNamespace(scaling_factor=int(sf))


class CountingInvoker:
    def __init__(self):
        self.calls = 0

    def __call__(self, config_name, payload):
        self.calls += 1
        return {
            "valid": True,
            "fusion_count": 1,
            "result": {
                "valid": True,
                "invalid_chain": None,
                "chain": {"total_bits": 123},
            },
            "seen_config_name": str(config_name),
            "seen_payload": dict(payload),
        }


@unittest.skipIf(RescaleOptimizerBridge is None, "torch unavailable")
class RescaleOptimizerBridgeCacheTest(unittest.TestCase):
    def test_identical_optimizer_request_is_cached_across_layer_suffixes(self):
        invoker = CountingInvoker()
        bridge = RescaleOptimizerBridge(
            invoker=invoker,
            auto_t_new_from_cfg=False,
            cache_max_entries=8,
        )
        cfg = SimpleNamespace(
            wffn2_encode=_point(40),
            mean_inv_d_encode=_point(40),
            var_inv_d_encode=_point(40),
        )

        first = bridge.evaluate(
            config_name="block1_mrpc_L1",
            block_name="block1",
            cfg=cfg,
        )
        second = bridge.evaluate(
            config_name="block1_mrpc_L7",
            block_name="block1",
            cfg=cfg,
        )

        self.assertEqual(invoker.calls, 1)
        self.assertEqual(first.config_name, "block1_mrpc_L1")
        self.assertEqual(second.config_name, "block1_mrpc_L7")
        self.assertFalse(first.raw["_optimizer_cache_hit"])
        self.assertTrue(second.raw["_optimizer_cache_hit"])
        self.assertEqual(second.total_bits, 123)


@unittest.skipIf(bridge_mod is None, "torch unavailable")
class BaselineArchiveCacheTest(unittest.TestCase):
    def test_load_baseline_archive_reuses_parse_and_returns_fresh_lists(self):
        payload = {
            "results": [
                {
                    "success": True,
                    "config_name": "block1_mrpc",
                    "skeleton": [1, 2, 3],
                    "cut_point_sf": [
                        {"i": 1, "sf": 30},
                        {"i": 2, "sf_post": 31},
                        {"i": 3, "sf": 32},
                    ],
                    "modulus_chain": {"drop_order": [60, 50, 40, 30]},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/static_skeletons_mrpc.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            cache = getattr(bridge_mod, "_BASELINE_ARCHIVE_CACHE", None)
            if cache is not None:
                cache.clear()

            original_load = bridge_mod.json.load
            load_calls = []

            def counting_load(handle):
                load_calls.append(handle.name)
                return original_load(handle)

            with mock.patch.object(bridge_mod.json, "load", side_effect=counting_load):
                first = bridge_mod.load_baseline_archive(path)
                second = bridge_mod.load_baseline_archive(path)

        self.assertEqual(load_calls, [path])
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first["block1_mrpc"][0], second["block1_mrpc"][0])
        first["block1_mrpc"][0].append(999)
        self.assertEqual(second["block1_mrpc"][0], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
