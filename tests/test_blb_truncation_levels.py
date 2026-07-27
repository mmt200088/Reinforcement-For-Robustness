from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from blb_stage2_rl import truncation_levels


_MISSING = object()


@contextmanager
def _stubbed_action_space():
    package = importlib.import_module("blb_stage2_rl")
    module_name = "blb_stage2_rl.action_space"
    module_before = sys.modules.get(module_name, _MISSING)
    attribute_before = package.__dict__.get("action_space", _MISSING)

    bridge = types.ModuleType("blb_rl_bridge")
    for name in (
        "Block1ActionSpec",
        "Block2ActionSpec",
        "Block3ActionSpec",
        "Block4ActionSpec",
        "Block5ActionSpec",
        "build_block1_cfg_from_action",
        "build_block2_cfg_from_action",
        "build_block3_cfg_from_action",
        "build_block4_cfg_from_action",
        "build_block5_cfg_from_action",
    ):
        setattr(bridge, name, type(name, (), {}))
    handler = types.ModuleType("function_handler")
    handler.NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N = {}
    for name in (
        "Block1NoiseConfig",
        "Block2NoiseConfig",
        "Block3NoiseConfig",
        "Block4NoiseConfig",
        "Block5NoiseConfig",
    ):
        setattr(handler, name, type(name, (), {}))

    dependency_names = ("blb_rl_bridge", "function_handler")
    dependencies_before = {
        name: sys.modules.get(name, _MISSING)
        for name in dependency_names
    }
    sys.modules["blb_rl_bridge"] = bridge
    sys.modules["function_handler"] = handler
    sys.modules.pop(module_name, None)
    package.__dict__.pop("action_space", None)
    try:
        yield importlib.import_module(module_name)
    finally:
        if module_before is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module_before
        if attribute_before is _MISSING:
            package.__dict__.pop("action_space", None)
        else:
            package.action_space = attribute_before
        for name, module in dependencies_before.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TruncationLevelsTest(unittest.TestCase):
    def test_default_domain_preserves_legacy_indices_and_adds_k6_k7(self):
        self.assertEqual(
            truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT,
            (8, 9, 11, 13, 10, 12, 6, 7),
        )
        self.assertEqual(
            truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT[:6],
            (8, 9, 11, 13, 10, 12),
        )
        self.assertEqual(
            truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT[6:],
            (6, 7),
        )
        self.assertEqual(
            (truncation_levels.K_MIN_BITS, truncation_levels.K_MAX_BITS),
            (6, 13),
        )
        self.assertEqual(
            truncation_levels.SUPPORTED_K_VALUES,
            frozenset(range(6, 14)),
        )

    def test_load_k_levels_defaults_and_preserves_override_order(self):
        self.assertEqual(
            truncation_levels.load_k_levels({}),
            truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT,
        )
        self.assertEqual(
            truncation_levels.load_k_levels(
                {"BLB_TRUNCATION_K_LEVELS": ""}
            ),
            truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT,
        )
        self.assertEqual(
            truncation_levels.load_k_levels(
                {"BLB_TRUNCATION_K_LEVELS": "13, 6, 8, 7"}
            ),
            (13, 6, 8, 7),
        )

    def test_load_k_levels_rejects_duplicate_non_integer_and_empty_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            truncation_levels.load_k_levels(
                {"BLB_TRUNCATION_K_LEVELS": "8,9,8"}
            )
        with self.assertRaisesRegex(ValueError, "integers"):
            truncation_levels.load_k_levels(
                {"BLB_TRUNCATION_K_LEVELS": "8,nope,13"}
            )
        for raw in ("8,,13", "8, ,13"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    truncation_levels.load_k_levels(
                        {"BLB_TRUNCATION_K_LEVELS": raw}
                    )

    def test_validate_exact_k_domain_accepts_any_exact_order(self):
        reordered = (13, 6, 12, 7, 11, 8, 10, 9)
        self.assertEqual(
            truncation_levels.validate_exact_k_domain(reordered),
            reordered,
        )

    def test_validate_exact_k_domain_rejects_missing_foreign_and_duplicates(self):
        invalid_domains = (
            (6, 7, 8, 9, 10, 11, 12),
            (6, 7, 8, 9, 10, 11, 12, 14),
            (6, 7, 8, 9, 10, 11, 12, 12),
        )
        for levels in invalid_domains:
            with self.subTest(levels=levels):
                with self.assertRaisesRegex(ValueError, "each supported K value"):
                    truncation_levels.validate_exact_k_domain(levels)

    def test_checkpoint_k_domain_contract_carries_exact_ordered_levels(self):
        contract = truncation_levels.checkpoint_k_domain_contract()
        self.assertEqual(
            contract,
            {
                "schema_version": "stage2_truncation_k_domain_v1",
                "k_levels": list(truncation_levels.K_LEVELS),
            },
        )
        checkpoint = {
            truncation_levels.CHECKPOINT_K_DOMAIN_KEY: contract,
        }
        self.assertEqual(
            truncation_levels.validate_checkpoint_k_domain(checkpoint),
            truncation_levels.K_LEVELS,
        )

    def test_checkpoint_k_domain_rejects_missing_and_old_six_level_contracts(self):
        old_six = {
            "schema_version": "stage2_truncation_k_domain_v1",
            "k_levels": [8, 9, 11, 13, 10, 12],
        }
        reordered_eight = {
            "schema_version": "stage2_truncation_k_domain_v1",
            "k_levels": [13, 12, 11, 10, 9, 8, 7, 6],
        }
        for checkpoint in (
            {},
            {truncation_levels.CHECKPOINT_K_DOMAIN_KEY: old_six},
            {truncation_levels.CHECKPOINT_K_DOMAIN_KEY: reordered_eight},
        ):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaisesRegex(RuntimeError, "fresh run"):
                    truncation_levels.validate_checkpoint_k_domain(
                        checkpoint,
                        context="legacy Stage-2 checkpoint",
                    )

    def test_baseline_k_index_prefers_k13_and_falls_back_to_maximum(self):
        self.assertEqual(
            truncation_levels.baseline_k_index(
                truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT
            ),
            3,
        )
        self.assertEqual(
            truncation_levels.baseline_k_index((8, 10, 7, 12)),
            3,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            truncation_levels.baseline_k_index(())

    def test_consumers_share_the_canonical_k_levels(self):
        with _stubbed_action_space() as action_space:
            layerwise_action = importlib.import_module(
                "blb_stage2_rl.layerwise_action"
            )

            self.assertEqual(action_space.K_LEVELS, truncation_levels.K_LEVELS)
            self.assertEqual(layerwise_action.K_LEVELS, truncation_levels.K_LEVELS)
            self.assertEqual(action_space.LEVELS_K, truncation_levels.LEVELS_K)

    def test_top_level_import_works_with_only_stage2_package_on_pythonpath(self):
        stage2_dir = Path(__file__).resolve().parents[1] / "blb_stage2_rl"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stage2_dir)
        env.pop("BLB_TRUNCATION_K_LEVELS", None)
        script = "\n".join(
            (
                "import json",
                "import layerwise_action",
                "import truncation_levels",
                "print(json.dumps({",
                '    "levels": list(truncation_levels.K_LEVELS),',
                '    "layerwise_levels": list(layerwise_action.K_LEVELS),',
                "}))",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=temp_dir,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        expected = list(truncation_levels.DEFAULT_K_LEVELS_LEGACY_COMPAT)
        self.assertEqual(payload["levels"], expected)
        self.assertEqual(payload["layerwise_levels"], expected)

    def test_action_space_stub_restores_global_module_state(self):
        package = importlib.import_module("blb_stage2_rl")
        module_name = "blb_stage2_rl.action_space"
        original_module = sys.modules.get(module_name, _MISSING)
        original_attribute = package.__dict__.get("action_space", _MISSING)
        try:
            sys.modules.pop(module_name, None)
            package.__dict__.pop("action_space", None)
            with _stubbed_action_space() as action_space:
                self.assertIs(sys.modules[module_name], action_space)
                self.assertIs(package.action_space, action_space)
            self.assertNotIn(module_name, sys.modules)
            self.assertNotIn("action_space", package.__dict__)

            previous_module = types.ModuleType(module_name)
            previous_attribute = object()
            sys.modules[module_name] = previous_module
            package.action_space = previous_attribute
            with _stubbed_action_space() as action_space:
                self.assertIs(sys.modules[module_name], action_space)
                self.assertIs(package.action_space, action_space)
            self.assertIs(sys.modules[module_name], previous_module)
            self.assertIs(package.action_space, previous_attribute)
        finally:
            if original_module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module
            if original_attribute is _MISSING:
                package.__dict__.pop("action_space", None)
            else:
                package.action_space = original_attribute

    def test_unknown_block_baseline_uses_maximum_configured_k(self):
        with _stubbed_action_space() as action_space:
            with mock.patch.object(action_space, "K_LEVELS", (13, 14)):
                self.assertEqual(action_space._baseline_k_index_for_block(999), 1)


if __name__ == "__main__":
    unittest.main()
