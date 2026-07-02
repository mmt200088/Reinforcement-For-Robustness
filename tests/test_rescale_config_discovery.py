from contextlib import redirect_stdout
import importlib.util
import io
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
        cls.check_compress_headroom = _load_script_module(
            "check_compress_headroom_for_test",
            "Rescale_optimizer/scripts/check_compress_headroom.py",
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

    def test_headroom_main_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "zeta.json").write_text("{}", encoding="utf-8")
            (root / "alpha.json").write_text("{}", encoding="utf-8")
            (root / "static_skeletons.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "nested.json").mkdir()

            seen = []

            def fake_diagnose(path):
                seen.append(Path(path).name)
                return {"config": Path(path).stem, "rows": []}

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("config discovery should not use Path.glob"),
            ), mock.patch.object(
                self.check_compress_headroom,
                "diagnose",
                side_effect=fake_diagnose,
            ), redirect_stdout(io.StringIO()):
                rc = self.check_compress_headroom.main(["--configs-dir", str(root)])

        self.assertEqual(rc, 0)
        self.assertEqual(seen, ["alpha.json", "zeta.json"])


if __name__ == "__main__":
    unittest.main()
