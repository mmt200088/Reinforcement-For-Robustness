import json
import unittest
from dataclasses import asdict, is_dataclass


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


if __name__ == "__main__":
    unittest.main()
