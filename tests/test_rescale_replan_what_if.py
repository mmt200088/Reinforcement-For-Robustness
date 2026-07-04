import importlib.util
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Rescale_optimizer" / "scripts" / "replan_what_if.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("replan_what_if_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["replan_what_if_for_test"] = module
    spec.loader.exec_module(module)
    return module


class ReplanWhatIfJsonTest(unittest.TestCase):
    def test_compact_json_streams_dict_items_without_full_list_copy(self):
        mod = _load_module()
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        dumps_region = source[
            source.index("def _dumps_compact_json("):
            source.index("\n\ndef _build_argparser(")
        ]

        self.assertNotIn("items = list(obj.items())", dumps_region)
        rendered = mod._dumps_compact_json(
            {
                "alpha": {"x": 1},
                "beta": {"y": 2},
                "gamma": [1, 2, 3],
            }
        )
        self.assertEqual(
            rendered,
            "{\n"
            '  "alpha": {"x": 1},\n'
            '  "beta": {"y": 2},\n'
            '  "gamma": [1, 2, 3]\n'
            "}",
        )


if __name__ == "__main__":
    unittest.main()
