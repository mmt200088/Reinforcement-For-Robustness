import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ACTION_GRID_PATH = ROOT / "Paean" / "action_grid.py"


def _load_action_grid_module():
    stub = types.ModuleType("blb_stage2_rl.action_space")
    stub.K_LEVELS = (8, 9, 11, 13, 10, 12)
    stub.NUM_LEVELS_PER_DIM_BY_BLOCK_KIND = {"K": len(stub.K_LEVELS), "x": 5}
    stub.action_dims_for_config = lambda *args, **kwargs: []
    stub.action_vector_to_cfgs = lambda *args, **kwargs: None
    stub.build_optimizer_requests = lambda *args, **kwargs: {}
    stub.layer_dims = lambda *args, **kwargs: 0
    stub.load_max_sfs = lambda *args, **kwargs: {}
    stub.make_all_max_action_vector = lambda *args, **kwargs: []
    stub.make_all_min_action_vector = lambda *args, **kwargs: []
    stub.per_layer_field_offsets = lambda *args, **kwargs: {}
    stub.sf_from = lambda idx, max_sf, levels: int(max_sf) - 2 * (int(levels) - 1 - int(idx))
    stub.sum_truncation_k_in_action = lambda *args, **kwargs: 0

    name = "paean_action_grid_under_test"
    spec = importlib.util.spec_from_file_location(name, ACTION_GRID_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    previous = sys.modules.get("blb_stage2_rl.action_space")
    sys.modules["blb_stage2_rl.action_space"] = stub
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        if previous is None:
            sys.modules.pop("blb_stage2_rl.action_space", None)
        else:
            sys.modules["blb_stage2_rl.action_space"] = previous
    return module


class PaeanActionGridTest(unittest.TestCase):
    def test_k_value_to_action_index_uses_precomputed_lookup(self):
        action_grid = _load_action_grid_module()
        expected_idx = action_grid.K_LEVELS.index(11)

        with mock.patch("builtins.list", side_effect=AssertionError("K lookup should not allocate a list")):
            idx = action_grid._value_to_action_index(
                value=11,
                block_idx=3,
                field_name="output_truncation_k",
                kind="K",
                max_sfs={},
            )

        self.assertEqual(idx, expected_idx)

    def test_scaling_factor_value_lookup_reuses_choice_table(self):
        action_grid = _load_action_grid_module()
        calls = 0

        def counting_sf_from(idx, max_sf, levels):
            nonlocal calls
            calls += 1
            return int(max_sf) - 2 * (int(levels) - 1 - int(idx))

        class MaxSfs:
            def get(self, block_idx, field_name):
                self.last_key = (int(block_idx), str(field_name))
                return 30

        action_grid.sf_from = counting_sf_from
        max_sfs = MaxSfs()

        first = action_grid._value_to_action_index(
            value=28,
            block_idx=2,
            field_name="wffn1_sf",
            kind="x",
            max_sfs=max_sfs,
        )
        second = action_grid._value_to_action_index(
            value=30,
            block_idx=2,
            field_name="wffn1_sf",
            kind="x",
            max_sfs=max_sfs,
        )

        self.assertEqual((first, second), (3, 4))
        self.assertEqual(calls, 5)

    def test_selector_slots_are_cached_across_repeated_sets(self):
        action_grid = _load_action_grid_module()
        calls = 0

        def counting_offsets():
            nonlocal calls
            calls += 1
            return [
                (1, "mean_rescale_sf", "R"),
                (2, "wffn1_sf", "x"),
            ]

        class MaxSfs:
            def get(self, _block_idx, _field_name):
                return 30

        action_grid.per_layer_field_offsets = counting_offsets
        vec = [0] * 24

        action_grid._set_selector_value(vec, 12, MaxSfs(), "block2.wffn1", 30)
        action_grid._set_selector_value(vec, 12, MaxSfs(), "block2.wffn1", 28)

        self.assertEqual(calls, 1)
        self.assertEqual(vec[1], 3)
        self.assertEqual(vec[23], 3)

    def test_cost_matched_sampling_reuses_parsed_fixed_specs(self):
        action_grid = _load_action_grid_module()
        action_grid.action_dims_for_config = lambda _num_layers: [5]
        action_grid.sum_truncation_k_in_action = lambda _vec, _num_layers: 1
        action_grid.per_layer_field_offsets = lambda: [(2, "wffn1_sf", "x")]

        class MaxSfs:
            def get(self, _block_idx, _field_name):
                return 30

        original_parse = action_grid.parse_action_spec
        parse_calls = 0

        def counting_parse(spec):
            nonlocal parse_calls
            parse_calls += 1
            return original_parse(spec)

        action_grid.parse_action_spec = counting_parse

        candidates, diagnostics = action_grid.build_cost_matched_random_action_candidates(
            num_layers=1,
            profile="mrpc",
            selected_action_vec=[0],
            selected_total_bits=0,
            selected_total_fusion=0,
            selected_sum_k=0,
            bridge=None,
            max_sfs=MaxSfs(),
            gelu_degree=[4],
            attn_degree=[6],
            seed=7,
            count=1,
            max_attempts=5,
            fixed_specs=("block2.wffn1=30",),
        )

        self.assertEqual(candidates, [])
        self.assertEqual(diagnostics.avg_k_prefilter_skipped, 5)
        self.assertEqual(parse_calls, 1)

    def test_slot_config_loading_reuses_profile_max_sfs(self):
        action_grid = _load_action_grid_module()
        load_calls = 0

        def counting_load_max_sfs(profile):
            nonlocal load_calls
            load_calls += 1
            return {"profile": profile}

        action_grid.load_max_sfs = counting_load_max_sfs

        action_space_stub = types.ModuleType("blb_stage2_rl.action_space")
        action_space_stub.load_max_sfs = counting_load_max_sfs
        action_io_stub = types.ModuleType("blb_stage2_rl.action_io")

        def slots_payload_to_action_vec(payload, *, max_sfs, num_layers, gelu_degree, attn_degree):
            self.assertEqual(max_sfs, {"profile": "mrpc"})
            self.assertEqual(int(num_layers), 1)
            return [0], []

        action_io_stub.slots_payload_to_action_vec = slots_payload_to_action_vec

        previous_action_space = sys.modules.get("blb_stage2_rl.action_space")
        previous_action_io = sys.modules.get("blb_stage2_rl.action_io")
        sys.modules["blb_stage2_rl.action_space"] = action_space_stub
        sys.modules["blb_stage2_rl.action_io"] = action_io_stub
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                payload = {
                    "schema_version": "blb_v3_slots_human_v1",
                    "num_layers": 1,
                    "profile": "mrpc",
                    "gelu_degree": [4],
                    "attn_degree": [6],
                    "slots": [{"label": "L00.B2.W.wffn1", "scaling_factor": 30}],
                }
                first_path = root / "first.json"
                second_path = root / "second.json"
                first_path.write_text(json.dumps(payload), encoding="utf-8")
                second_path.write_text(json.dumps(payload), encoding="utf-8")

                action_grid.load_action_grid_config(str(first_path))
                action_grid.load_action_grid_config(str(second_path))
        finally:
            if previous_action_space is None:
                sys.modules.pop("blb_stage2_rl.action_space", None)
            else:
                sys.modules["blb_stage2_rl.action_space"] = previous_action_space
            if previous_action_io is None:
                sys.modules.pop("blb_stage2_rl.action_io", None)
            else:
                sys.modules["blb_stage2_rl.action_io"] = previous_action_io

        self.assertEqual(load_calls, 1)


if __name__ == "__main__":
    unittest.main()
