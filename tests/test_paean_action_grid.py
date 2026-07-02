import importlib.util
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
