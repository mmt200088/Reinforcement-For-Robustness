import gzip
import pathlib
import tempfile
import unittest
from unittest import mock

from tests.source_inspection_utils import function_names, source_text
from rfr.common.jsonl_utils import (
    count_jsonl_with_required_fields,
    iter_jsonl,
    iter_jsonl_records,
    read_jsonl_fields,
    read_jsonl_float_field,
    read_jsonl_xy,
    read_jsonl,
    resolve_jsonl_path,
    write_jsonl_rows,
    recover_jsonl_file,
)


class JsonlUtilsTest(unittest.TestCase):
    def test_recover_jsonl_file_repairs_only_torn_tail(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "events.jsonl"
            path.write_bytes(b'{"episode": 1}\n{"episode": 2')

            recovered_size = recover_jsonl_file(path)

            self.assertEqual(path.read_bytes(), b'{"episode": 1}\n')
            self.assertEqual(recovered_size, path.stat().st_size)

    def test_recover_jsonl_file_rolls_back_to_committed_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "events.jsonl"
            first = b'{"episode": 1}\n'
            path.write_bytes(first + b'{"episode": 2}\n')

            recovered_size = recover_jsonl_file(path, committed_size=len(first))

            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(recovered_size, len(first))

    def test_recover_jsonl_file_rejects_non_boundary_commit(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "events.jsonl"
            path.write_bytes(b'{"episode": 1}\n')

            with self.assertRaisesRegex(ValueError, "JSONL boundary"):
                recover_jsonl_file(path, committed_size=5)

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

    def test_iter_jsonl_records_resolves_path_once(self):
        from rfr.common import jsonl_utils

        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\n', encoding="utf-8")
            original_resolve = jsonl_utils.resolve_jsonl_path
            calls = []

            def resolve_once(path_arg, *, gzip_fallback=False):
                calls.append(path_arg)
                if len(calls) > 1:
                    raise AssertionError("iter_jsonl_records should resolve path once")
                return original_resolve(path_arg, gzip_fallback=gzip_fallback)

            jsonl_utils.resolve_jsonl_path = resolve_once
            try:
                rows = list(jsonl_utils.iter_jsonl_records(path))
            finally:
                jsonl_utils.resolve_jsonl_path = original_resolve

        self.assertEqual(rows, [(1, {"a": 1})])

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
        from rfr.common import jsonl_utils

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
        from rfr.common import jsonl_utils

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

    def test_read_jsonl_float_field_projects_one_numeric_column(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "episodes.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"total_reward": 1.5, "large_debug": "x"}',
                        "{bad-json",
                        '{"total_reward": 1.75, "large_debug": "y"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rewards = read_jsonl_float_field(path, "total_reward")

        self.assertEqual(rewards, [1.5, 1.75])

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
        from rfr.common import jsonl_utils

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
        from rfr.common import jsonl_utils

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


if __name__ == "__main__":
    unittest.main()
