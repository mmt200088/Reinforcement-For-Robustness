"""Torch-free tests for blb_stage2_rl.osr.

Covers:
  * compute_fingerprint stability + change detection per field
  * fingerprints_equal helper
  * save/load JSON roundtrip preserves disabled_cells + pruned_combos
  * load_osr_results refuses mismatched fingerprint by default, accepts with override
  * OSRPrePruneMask.apply_per_slot disables only the recorded cells
  * OSRPrePruneMask.is_combo_pruned returns True only for blacklisted tuples
  * Block 3 is excluded from the default scan block list

The package-level ``blb_stage2_rl/__init__.py`` imports torch via runner.py,
so we sidestep it by loading ``osr.py`` directly with importlib (matches the
workaround test_blb_substage_assembly.py uses).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OSR_PATH = os.path.join(_REPO_ROOT, "blb_stage2_rl", "osr.py")
_OSR_MODULE_NAME = "blb_stage2_rl_osr_smoke"


def _load_osr_module():
    # On Python 3.9 the dataclass decorator looks up ``cls.__module__`` in
    # ``sys.modules`` to evaluate string annotations; we have to register the
    # standalone-loaded module before exec_module runs the @dataclass line.
    if _OSR_MODULE_NAME in sys.modules:
        return sys.modules[_OSR_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_OSR_MODULE_NAME, _OSR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_OSR_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[_OSR_MODULE_NAME]
        raise
    return module


def _make_fingerprint(osr, **overrides):
    defaults = dict(
        profile="mrpc",
        num_layers=12,
        stage1_gelu_per_layer=[4] * 12,
        stage1_softmax_per_layer=[4] * 12,
        k_trials=4,
        probe_size=256,
        tol_acc=0.005,
        tol_stab=0.005,
        baseline_action_vec=list(range(577)),
    )
    defaults.update(overrides)
    return osr.compute_fingerprint(**defaults)


class FingerprintTests(unittest.TestCase):
    def test_stable_when_inputs_equal(self):
        osr = _load_osr_module()
        a = _make_fingerprint(osr)
        b = _make_fingerprint(osr)
        self.assertEqual(a, b)
        self.assertTrue(osr.fingerprints_equal(a, b))

    def test_changes_when_any_field_changes(self):
        osr = _load_osr_module()
        base = _make_fingerprint(osr)
        for kwargs in (
            {"profile": "sst2"},
            {"num_layers": 24},
            {"stage1_gelu_per_layer": [5] * 12},
            {"stage1_softmax_per_layer": [5] * 12},
            {"k_trials": 5},
            {"probe_size": 512},
            {"tol_acc": 0.01},
            {"tol_stab": 0.01},
            {"baseline_action_vec": [0] * 577},
        ):
            other = _make_fingerprint(osr, **kwargs)
            self.assertNotEqual(base, other, f"flip not detected: {kwargs}")
            self.assertFalse(
                osr.fingerprints_equal(base, other),
                f"fingerprints_equal returned True for change {kwargs}",
            )

    def test_osr_version_included(self):
        osr = _load_osr_module()
        fp = _make_fingerprint(osr)
        self.assertEqual(fp["osr_version"], osr.OSR_VERSION)

    def test_baseline_action_vec_hash_streams_json_array_without_materialization(self):
        osr = _load_osr_module()
        fp = _make_fingerprint(osr, baseline_action_vec=[0, 1, 2, 3])
        expected = "sha256:" + hashlib.sha256(b"[0,1,2,3]").hexdigest()
        self.assertEqual(fp["baseline_action_vec_hash"], expected)

        with open(_OSR_PATH, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("def _baseline_action_vec_hash(", text)
        self.assertIn("digest.update(b\"[\")", text)
        self.assertNotIn("reshape(-1).tolist()", text)
        self.assertNotIn("json.dumps(bvec", text)


class ResultsRoundtripTests(unittest.TestCase):
    def test_save_load_preserves_all_fields(self):
        osr = _load_osr_module()
        fp = _make_fingerprint(osr)
        results = osr.OSRPrePruneResults(fingerprint=fp)
        results.add_disabled_cell(layer_idx=3, block_idx=2, slot_idx=4, level_idx=1)
        results.add_disabled_cell(layer_idx=3, block_idx=2, slot_idx=4, level_idx=2)
        results.add_disabled_cell(layer_idx=7, block_idx=5, slot_idx=0, level_idx=3)
        results.add_pruned_combo(layer_idx=5, block_idx=1, action_tuple=(1, 2, 3, 0, 0, 1, 1, 2, 3))
        results.add_pruned_combo(layer_idx=5, block_idx=1, action_tuple=(0, 0, 0, 0, 0, 0, 0, 0, 0))
        results.scan_summary["foo"] = "bar"

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "osr_results.json")
            osr.save_osr_results(path, results)
            loaded = osr.load_osr_results(path, expected_fingerprint=fp)
            self.assertEqual(loaded.fingerprint, fp)
            self.assertEqual(loaded.total_disabled_cells(), 3)
            self.assertEqual(loaded.total_pruned_combos(), 2)
            self.assertEqual(
                loaded.disabled_cells[(3, 2)],
                {(4, 1), (4, 2)},
            )
            self.assertEqual(
                loaded.pruned_combos[(5, 1)],
                {(1, 2, 3, 0, 0, 1, 1, 2, 3), (0, 0, 0, 0, 0, 0, 0, 0, 0)},
            )
            self.assertEqual(loaded.scan_summary.get("foo"), "bar")

    def test_load_refuses_fingerprint_mismatch_by_default(self):
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults(fingerprint=_make_fingerprint(osr))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "osr_results.json")
            osr.save_osr_results(path, results)
            stale = _make_fingerprint(osr, profile="cola")
            with self.assertRaises(ValueError):
                osr.load_osr_results(path, expected_fingerprint=stale)

    def test_load_allows_mismatch_with_override(self):
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults(fingerprint=_make_fingerprint(osr))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "osr_results.json")
            osr.save_osr_results(path, results)
            stale = _make_fingerprint(osr, profile="cola")
            loaded = osr.load_osr_results(
                path,
                expected_fingerprint=stale,
                allow_fingerprint_mismatch=True,
            )
            # Returns the on-disk fingerprint, not the stale "expected" one.
            self.assertEqual(loaded.fingerprint["profile"], "mrpc")

    def test_load_with_no_expected_fingerprint_skips_check(self):
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults(fingerprint=_make_fingerprint(osr))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "osr_results.json")
            osr.save_osr_results(path, results)
            # No expected_fingerprint → no validation, no error.
            loaded = osr.load_osr_results(path)
            self.assertEqual(loaded.fingerprint["profile"], "mrpc")


class PrePruneMaskTests(unittest.TestCase):
    def test_apply_per_slot_disables_recorded_cells(self):
        import numpy as np
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults()
        # Disable (slot=2, level=1) and (slot=4, level=0) at (layer=3, block=2)
        results.add_disabled_cell(3, 2, 2, 1)
        results.add_disabled_cell(3, 2, 4, 0)
        mask = osr.OSRPrePruneMask(results)
        # 6 slots × 6 levels, all allowed initially
        level_mask = np.ones((6, 6), dtype=bool)
        out = mask.apply_per_slot(layer_idx=3, block_idx=2,
                                   action_level_mask=level_mask)
        self.assertFalse(bool(out[2, 1]))
        self.assertFalse(bool(out[4, 0]))
        # Other cells stay allowed
        out_copy = out.copy()
        out_copy[2, 1] = True
        out_copy[4, 0] = True
        self.assertTrue(bool(out_copy.all()))

    def test_apply_per_slot_respects_protected_actions(self):
        import numpy as np
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults()
        results.add_disabled_cell(3, 2, 2, 1)
        mask = osr.OSRPrePruneMask(results)
        level_mask = np.ones((6, 6), dtype=bool)
        # baseline action puts slot 2 at level 1 — protect it
        out = mask.apply_per_slot(
            layer_idx=3, block_idx=2,
            action_level_mask=level_mask,
            protected_actions=[(0, 0, 1, 0, 0, 0)],
        )
        # Protected (slot 2, level 1) survives despite being in disabled_cells.
        self.assertTrue(bool(out[2, 1]))

    def test_apply_per_slot_unrelated_layer_is_noop(self):
        import numpy as np
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults()
        results.add_disabled_cell(3, 2, 2, 1)
        mask = osr.OSRPrePruneMask(results)
        level_mask = np.ones((6, 6), dtype=bool)
        # Apply for a (layer, block) the mask says nothing about.
        out = mask.apply_per_slot(layer_idx=7, block_idx=4,
                                   action_level_mask=level_mask)
        self.assertTrue(bool(out.all()))

    def test_apply_per_slot_keeps_at_least_one_level_per_slot(self):
        import numpy as np
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults()
        # Pathological: disable EVERY level for slot 1.
        for lvl in range(6):
            results.add_disabled_cell(1, 1, 1, lvl)
        mask = osr.OSRPrePruneMask(results)
        level_mask = np.ones((6, 6), dtype=bool)
        # baseline puts slot 1 at level 3 — should be re-enabled to keep the
        # slot from having zero allowed levels.
        out = mask.apply_per_slot(
            layer_idx=1, block_idx=1,
            action_level_mask=level_mask,
            protected_actions=[(0, 3, 0, 0, 0, 0)],
        )
        self.assertTrue(bool(out[1].any()), "slot 1 must keep ≥1 allowed level")
        self.assertTrue(bool(out[1, 3]), "protected level 3 must survive")

    def test_is_combo_pruned_matches_only_blacklisted_tuples(self):
        osr = _load_osr_module()
        results = osr.OSRPrePruneResults()
        bad = (1, 2, 3, 0, 4)
        good = (0, 0, 0, 0, 0)
        results.add_pruned_combo(layer_idx=5, block_idx=2, action_tuple=bad)
        mask = osr.OSRPrePruneMask(results)
        self.assertTrue(mask.is_combo_pruned(5, 2, bad))
        self.assertFalse(mask.is_combo_pruned(5, 2, good))
        # Unrelated (layer, block) keys: never pruned
        self.assertFalse(mask.is_combo_pruned(6, 2, bad))
        self.assertFalse(mask.is_combo_pruned(5, 4, bad))


class BlockExclusionTests(unittest.TestCase):
    def test_default_block_list_excludes_block_3(self):
        osr = _load_osr_module()
        self.assertNotIn(3, osr.DEFAULT_OSR_BLOCKS)
        self.assertEqual(set(osr.DEFAULT_OSR_BLOCKS), {1, 2, 4, 5})


if __name__ == "__main__":
    unittest.main()
