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
    def test_load_csv_avoids_per_row_dict_reader(self):
        mod = _load_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "noise.csv"
            path.write_text(
                "N,scale_bits,B_enc,B_fresh,B_rs\n"
                "8192,12,0.0,1.25,2.5\n"
                "16384,13,0.0,3.75,4.5\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                mod.csv,
                "DictReader",
                side_effect=AssertionError("noise CSV parsing should not allocate one dict per row"),
            ):
                table = mod.load_csv(path)

        self.assertEqual(table[(8192, 12)], (1.25, 2.5))
        self.assertEqual(table[(16384, 13)], (3.75, 4.5))

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

    def test_fmt_dict_block_streams_items_without_full_list_copy(self):
        mod = _load_module()
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        fmt_region = source[
            source.index("def fmt_dict_block("):
            source.index("\n\ndef replace_block(")
        ]

        self.assertNotIn("items = list(d.items())", fmt_region)
        rendered = mod.fmt_dict_block(
            {
                12: 1.0,
                13: 2.0,
                14: 3.0,
                15: 4.0,
                16: 5.0,
            },
            "    ",
        )
        self.assertEqual(
            rendered,
            '    "12": 1.000000e+00, "13": 2.000000e+00, '
            '"14": 3.000000e+00, "15": 4.000000e+00,\n'
            '    "16": 5.000000e+00',
        )


if __name__ == "__main__":
    unittest.main()
