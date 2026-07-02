import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Rescale_optimizer" / "scripts" / "update_noise_tables_from_csv.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_noise_tables_from_csv_for_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UpdateNoiseTablesDiscoveryTest(unittest.TestCase):
    def test_discovers_configs_without_path_glob(self):
        mod = _load_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "zeta.json").write_text("{}", encoding="utf-8")
            (root / "alpha.json").write_text("{}", encoding="utf-8")
            (root / "static_skeletons.json").write_text("{}", encoding="utf-8")
            (root / "static_skeletons_mrpc.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "nested.json").mkdir()

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("noise-table config discovery should not use Path.glob"),
            ):
                configs = mod._discover_config_paths(root)

        self.assertEqual([p.name for p in configs], ["alpha.json", "zeta.json"])


if __name__ == "__main__":
    unittest.main()
