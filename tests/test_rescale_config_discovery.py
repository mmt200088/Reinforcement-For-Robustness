import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RescaleConfigDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_run_configs = _load_script_module(
            "batch_run_configs_for_test",
            "Rescale_optimizer/scripts/batch_run_configs.py",
        )

    def test_batch_discovery_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "zeta.json").write_text("{}", encoding="utf-8")
            (root / "alpha.json").write_text("{}", encoding="utf-8")
            (root / "static_skeletons.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "nested.json").mkdir()

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("config discovery should not use Path.glob"),
            ):
                configs = self.batch_run_configs._discover_configs(root, explicit=None)

        self.assertEqual([p.name for p in configs], ["alpha.json", "zeta.json"])

    def test_batch_archive_writer_streams_output_file(self):
        source = (REPO_ROOT / "Rescale_optimizer/scripts/batch_run_configs.py").read_text(encoding="utf-8")
        main_region = source[source.index("def main("):]

        self.assertIn("def _write_doc(", source)
        self.assertIn("_write_doc(f,", main_region)
        self.assertIn('out_path.open("w", encoding="utf-8")', main_region)
        self.assertNotIn("_format_doc(entries", main_region)
        self.assertNotIn("write_text(out_text", main_region)

if __name__ == "__main__":
    unittest.main()
