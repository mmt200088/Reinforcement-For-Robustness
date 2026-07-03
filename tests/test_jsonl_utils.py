import gzip
import pathlib
import tempfile
import unittest
from unittest import mock

from tests.source_inspection_utils import function_names, source_text
from jsonl_utils import (
    count_jsonl_with_required_fields,
    iter_jsonl,
    iter_jsonl_records,
    read_jsonl_fields,
    read_jsonl_xy,
    read_jsonl,
    resolve_jsonl_path,
    write_jsonl_rows,
)


class JsonlUtilsTest(unittest.TestCase):
    def test_iter_jsonl_skips_blank_bad_and_non_dict_rows_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\n\nbad\n[1, 2]\n{"b": 2}\n', encoding="utf-8")

            rows = list(iter_jsonl(path))

        self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    def test_iter_jsonl_can_raise_with_path_and_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\nbad\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"rows\.jsonl:2: invalid JSON"):
                list(iter_jsonl(path, errors="raise"))

    def test_iter_jsonl_can_yield_non_dict_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")

            rows = list(iter_jsonl(path, dict_only=False))

        self.assertEqual(rows, [{"a": 1}, [1, 2]])

    def test_iter_jsonl_records_preserves_line_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('\n{"a": 1}\n[1, 2]\n', encoding="utf-8")

            rows = list(iter_jsonl_records(path, dict_only=False))

        self.assertEqual(rows, [(2, {"a": 1}), (3, [1, 2])])

    def test_read_jsonl_missing_ok(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "missing.jsonl"

            self.assertEqual(read_jsonl(path, missing_ok=True), [])
            with self.assertRaises(FileNotFoundError):
                read_jsonl(path)

    def test_iter_jsonl_can_fallback_to_gzip_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            with gzip.open(str(path) + ".gz", "wt", encoding="utf-8") as handle:
                handle.write('{"a": 1}\n')

            rows = list(iter_jsonl(path, gzip_fallback=True))
            resolved_name = resolve_jsonl_path(path, gzip_fallback=True).name

        self.assertEqual(rows, [{"a": 1}])
        self.assertEqual(resolved_name, "rows.jsonl.gz")

    def test_read_jsonl_fields_passes_unstripped_lines_to_json_loader(self):
        import jsonl_utils

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"total_reward": 1.25, "unused": "x"}\n', encoding="utf-8")
            seen = []
            original_loads = jsonl_utils.json.loads

            def recording_loads(value):
                seen.append(value)
                return original_loads(value)

            with mock.patch.object(jsonl_utils.json, "loads", recording_loads):
                rows = read_jsonl_fields(path, fields=("total_reward",))

        self.assertEqual(rows, [{"total_reward": 1.25}])
        self.assertTrue(seen[0].endswith("\n"))

    def test_read_jsonl_fields_and_xy_skip_whitespace_lines(self):
        import jsonl_utils

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text(
                "   \n"
                '{"total_reward": 1.25, "unused": "x"}'
                "\n\t\n",
                encoding="utf-8",
            )
            original_loads = jsonl_utils.json.loads
            seen = []

            def guarded_loads(value):
                seen.append(value)
                return original_loads(value)

            with mock.patch.object(jsonl_utils.json, "loads", guarded_loads):
                rows = read_jsonl_fields(path, fields=("total_reward",))
                xs, ys = read_jsonl_xy(path, "total_reward", "total_reward")

        self.assertEqual(rows, [{"total_reward": 1.25}])
        self.assertEqual(xs, [1.25])
        self.assertEqual(ys, [1.25])
        self.assertFalse(any(value.isspace() for value in seen))

    def test_read_jsonl_xy_projects_points_without_row_dicts(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "top_candidates.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"total_bits": 10, "total_reward": 1.5, "large_debug": "x"}',
                        "{bad-json",
                        '{"total_bits": 12, "total_reward": 1.75, "large_debug": "y"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            xs, ys = read_jsonl_xy(path, "total_bits", "total_reward")

        self.assertEqual(xs, [10.0, 12.0])
        self.assertEqual(ys, [1.5, 1.75])

    def test_count_jsonl_with_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1, "b": 2}\n[1]\n{"a": 3}\n', encoding="utf-8")

            count, failures = count_jsonl_with_required_fields(path, ("a", "b"), label="rows")

        self.assertEqual(count, 3)
        self.assertIn("rows:2 is not a JSON object", failures)
        self.assertIn("rows missing required fields in 1 rows (line 3: b)", failures)

    def test_write_jsonl_rows_creates_parent_and_normalizes_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "nested" / "rows.jsonl"

            written = write_jsonl_rows(
                path,
                [{"b": 2, "a": pathlib.Path("x")}, {"c": 3}],
                sort_keys=True,
            )
            text = path.read_text(encoding="utf-8")

        self.assertEqual(written, path)
        self.assertEqual(
            text,
            '{"a": "x", "b": 2}\n{"c": 3}\n',
        )

    def test_write_jsonl_rows_streams_rows_to_file_handle(self):
        import jsonl_utils

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "nested" / "rows.jsonl"

            with mock.patch.object(
                jsonl_utils.json,
                "dumps",
                side_effect=AssertionError("write_jsonl_rows should stream via json.dump"),
            ):
                write_jsonl_rows(path, [{"a": pathlib.Path("x"), "b": 2}])

            text = path.read_text(encoding="utf-8")

        self.assertEqual(text, '{"a": "x", "b": 2}\n')

    def test_write_jsonl_rows_reuses_encoder_without_json_dump_calls(self):
        import jsonl_utils

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "nested" / "rows.jsonl"

            with mock.patch.object(
                jsonl_utils.json,
                "dump",
                side_effect=AssertionError("write_jsonl_rows should reuse one JSONEncoder"),
            ):
                write_jsonl_rows(
                    path,
                    [{"b": 2, "a": pathlib.Path("x")}, {"c": 3}],
                    sort_keys=True,
                )

            text = path.read_text(encoding="utf-8")

        self.assertEqual(text, '{"a": "x", "b": 2}\n{"c": 3}\n')


class JsonlUtilsStaticGuardTest(unittest.TestCase):
    def test_known_report_scripts_use_shared_jsonl_reader(self):
        expected = {
            "scripts/stage2_first10k_monitor.py": "from jsonl_utils import read_jsonl",
            "scripts/stage2_reward_probe_scaling_report.py": "from jsonl_utils import iter_jsonl",
            "scripts/gpu_utilization_report.py": "from jsonl_utils import iter_jsonl",
            "scripts/blb_fusion_ab_compare.py": "from jsonl_utils import iter_jsonl",
            "scripts/blb_regen_stage2_outputs.py": "from jsonl_utils import iter_jsonl",
            "blb_stage2_rl/candidate_store.py": "from jsonl_utils import iter_jsonl",
            "scripts/verify_stage2_persistent_outputs.py": (
                "from jsonl_utils import count_jsonl_with_required_fields"
            ),
            "tools/paper_figures.py": "from jsonl_utils import read_jsonl_fields, read_jsonl_xy",
            "tools/experiments_log.py": "from jsonl_utils import iter_jsonl",
        }
        forbidden = {"_read_jsonl", "_open_jsonl", "_count_jsonl", "_count_jsonl_with_required_fields"}
        for rel_path, needle in expected.items():
            with self.subTest(path=rel_path):
                text = source_text(rel_path)
                self.assertIn(needle, text)
                self.assertFalse(forbidden & function_names(rel_path))

    def test_finite_jsonl_artifact_script_uses_shared_writer(self):
        rel_path = "scripts/blb_f0_scan_feasible_domain.py"
        text = source_text(rel_path)

        self.assertIn("from jsonl_utils import write_jsonl_rows", text)
        self.assertNotIn("def _write_jsonl(", text)


if __name__ == "__main__":
    unittest.main()
