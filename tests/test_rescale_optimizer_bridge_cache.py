import unittest
from types import SimpleNamespace

try:
    from rescale_optimizer_bridge import RescaleOptimizerBridge
except ModuleNotFoundError as exc:  # local macOS test env may be torch-free
    if exc.name != "torch":
        raise
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


if __name__ == "__main__":
    unittest.main()
