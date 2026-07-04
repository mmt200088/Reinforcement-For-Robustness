import importlib.util
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Rescale_optimizer" / "scripts" / "gen_replan_actions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_replan_actions_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["gen_replan_actions_for_test"] = module
    spec.loader.exec_module(module)
    return module


class GenReplanActionsTest(unittest.TestCase):
    def test_format_actions_file_streams_delta_overrides_without_items_list(self):
        mod = _load_module()
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        format_region = source[
            source.index("def _format_actions_file("):
            source.index("\n\ndef main(")
        ]

        self.assertNotIn("items = list(delta_overrides.items())", format_region)
        rendered = mod._format_actions_file(
            "block_test",
            [30, 31],
            {"mul_a": 2, "mul_b": "x2"},
        )
        self.assertEqual(
            rendered,
            "{\n"
            '  "config_name": "block_test",\n'
            '  "notes": "Auto-generated from configs/static_skeletons.json. '
            'Edit t_new / delta_overrides as needed.",\n'
            '  "t_new": [30, 31],\n'
            '  "delta_overrides": {\n'
            '    "mul_a": 2,\n'
            '    "mul_b": "x2"\n'
            "  }\n"
            "}\n",
        )


if __name__ == "__main__":
    unittest.main()
