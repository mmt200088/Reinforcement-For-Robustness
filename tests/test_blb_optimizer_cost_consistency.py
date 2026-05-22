import json
import unittest
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _jsonable(vars(value))
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


class BLBOptimizerCostConsistencyTests(unittest.TestCase):
    def _noise_point(self, sf):
        return SimpleNamespace(scaling_factor=int(sf))

    def _baseline_context(self, num_layers=2):
        from blb_stage2_rl.action_space import (
            build_optimizer_requests,
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )

        baseline = make_all_max_action_vector(num_layers=num_layers)
        max_sfs = load_max_sfs("mrpc")
        desc = describe_action_vector(
            baseline,
            max_sfs=max_sfs,
            num_layers=num_layers,
            gelu_degree=[4] * num_layers,
            attn_degree=[4] * num_layers,
            profile="mrpc",
        )

        def requests_for(action):
            from blb_stage2_rl.action_space import action_vector_to_cfgs

            decoded = action_vector_to_cfgs(
                action,
                max_sfs,
                num_layers=num_layers,
                gelu_degree=[4] * num_layers,
                attn_degree=[4] * num_layers,
            )
            requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())
            signature = {
                name: {"block": block, "cfg": _jsonable(cfg)}
                for name, (block, cfg) in sorted(requests.items())
            }
            return requests, json.dumps(signature, sort_keys=True, ensure_ascii=True)

        return baseline, desc, requests_for

    def _mutated(self, baseline, record):
        action = baseline.copy()
        slot = int(record["global_index"])
        width = int(record["num_levels"])
        action[slot] = 0 if int(baseline[slot]) != 0 else min(1, width - 1)
        return action

    def test_inactive_slots_do_not_change_optimizer_requests(self):
        baseline, desc, requests_for = self._baseline_context(num_layers=2)

        baseline_requests, baseline_sig = requests_for(baseline)
        self.assertNotIn("block1_mrpc_L0", baseline_requests)
        self.assertFalse(any("first_input" in key for key in baseline_requests))

        l0b1 = next(
            r for r in desc["records"]
            if int(r.get("layer", -1)) == 0 and r.get("block") == "block1"
        )
        first_input = next(r for r in desc["records"] if r.get("block") == "first_input")
        effective = next(
            r for r in desc["records"]
            if r.get("effective", True) and int(r.get("num_levels", 1)) > 1 and r.get("kind") != "K"
        )

        for record in (l0b1, first_input):
            _, mutated_sig = requests_for(self._mutated(baseline, record))
            self.assertEqual(mutated_sig, baseline_sig)

        _, effective_sig = requests_for(self._mutated(baseline, effective))
        self.assertNotEqual(effective_sig, baseline_sig)

    def test_effective_action_hash_ignores_inactive_compat_slots_only(self):
        from blb_stage2_rl.candidate_store import effective_action_hash, raw_action_hash

        baseline, desc, _requests_for = self._baseline_context(num_layers=2)
        l0b1 = next(
            r for r in desc["records"]
            if int(r.get("layer", -1)) == 0 and r.get("block") == "block1"
        )
        first_input = next(r for r in desc["records"] if r.get("block") == "first_input")
        effective = next(
            r for r in desc["records"]
            if r.get("effective", True) and int(r.get("num_levels", 1)) > 1 and r.get("kind") != "K"
        )

        baseline_eff = effective_action_hash(baseline, desc, baseline)
        for record in (l0b1, first_input):
            mutated = self._mutated(baseline, record)
            self.assertNotEqual(raw_action_hash(mutated), raw_action_hash(baseline))
            self.assertEqual(effective_action_hash(mutated, desc, baseline), baseline_eff)

        mutated_effective = self._mutated(baseline, effective)
        self.assertNotEqual(raw_action_hash(mutated_effective), raw_action_hash(baseline))
        self.assertNotEqual(effective_action_hash(mutated_effective, desc, baseline), baseline_eff)

    def test_direct_inprocess_replan_matches_compat_payload_path(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch unavailable for bridge import: {exc}")
        from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge

        cfg = SimpleNamespace(
            softmax_out_fresh=self._noise_point(18),
            softmax_out_mask_encode=self._noise_point(19),
            v_fresh=self._noise_point(20),
            v_mask_encode=self._noise_point(19),
            softmax_v_matmul_rescale=self._noise_point(18),
            softmax_v_mask_encode=self._noise_point(19),
            wo_encode=self._noise_point(19),
            ln_mean_inv_d_encode=self._noise_point(19),
            ln_mean_result_rescale=self._noise_point(18),
            ln_var_inv_d_encode=self._noise_point(19),
            ln_var_result_rescale=self._noise_point(18),
        )

        direct_invoker = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
            include=["block4"],
        )
        compat_invoker = InProcessInvoker.from_profile(
            rescale_optimizer_root="Rescale_optimizer",
            profile="mrpc",
            include=["block4"],
        )

        class CompatOnlyInvoker:
            baselines = compat_invoker.baselines

            def __call__(self, config_name, payload):
                return compat_invoker(config_name, payload)

        direct_bridge = RescaleOptimizerBridge(invoker=direct_invoker)
        compat_bridge = RescaleOptimizerBridge(invoker=CompatOnlyInvoker())

        direct = direct_bridge.evaluate(
            config_name="block4_L3",
            block_name="block4",
            cfg=cfg,
        )
        compat = compat_bridge.evaluate(
            config_name="block4_L3",
            block_name="block4",
            cfg=cfg,
        )

        self.assertEqual(direct.valid, compat.valid)
        self.assertEqual(direct.total_bits, compat.total_bits)
        self.assertEqual(direct.fusion_count, compat.fusion_count)
        self.assertEqual(direct.invalid_chain, compat.invalid_chain)
        for key in ("delta_overrides", "new_compact_config", "result", "t_new"):
            self.assertEqual(_jsonable(direct.raw.get(key)), _jsonable(compat.raw.get(key)))

    def test_bridge_prefers_direct_replan_variables_when_available(self):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"torch unavailable for bridge import: {exc}")
        from rescale_optimizer_bridge import RescaleOptimizerBridge

        class DirectOnlyInvoker:
            baselines = {
                "block4": ([0, 2, 5, 7, 8], [18, 18, 18, 18], [60, 50, 40])
            }

            def __init__(self):
                self.calls = []

            def __call__(self, _config_name, _payload):
                raise AssertionError("compat payload path should not be used")

            def replan_variables(self, config_name, *, t_new=None, delta_overrides=None):
                self.calls.append((config_name, list(t_new), dict(delta_overrides or {})))
                return {
                    "fusion_count": 0,
                    "valid": True,
                    "t_new": list(t_new),
                    "delta_overrides": dict(delta_overrides or {}),
                    "result": {
                        "valid": True,
                        "chain": {"total_bits": 123},
                        "invalid_chain": None,
                    },
                }

        cfg = SimpleNamespace(
            softmax_out_fresh=self._noise_point(18),
            softmax_out_mask_encode=self._noise_point(19),
            v_fresh=self._noise_point(20),
            v_mask_encode=self._noise_point(19),
            softmax_v_matmul_rescale=self._noise_point(18),
            softmax_v_mask_encode=self._noise_point(19),
            wo_encode=self._noise_point(19),
            ln_mean_inv_d_encode=self._noise_point(19),
            ln_mean_result_rescale=self._noise_point(18),
            ln_var_inv_d_encode=self._noise_point(19),
            ln_var_result_rescale=self._noise_point(18),
        )
        invoker = DirectOnlyInvoker()
        bridge = RescaleOptimizerBridge(invoker=invoker)

        out = bridge.evaluate(config_name="block4_L0", block_name="block4", cfg=cfg)

        self.assertTrue(out.valid)
        self.assertEqual(out.total_bits, 123)
        self.assertEqual(invoker.calls[0][0], "block4")
        self.assertEqual(invoker.calls[0][1], [18, 18, 18, 18])
        self.assertIn("ctct_rot_softmax_mul_v", invoker.calls[0][2])


if __name__ == "__main__":
    unittest.main()
