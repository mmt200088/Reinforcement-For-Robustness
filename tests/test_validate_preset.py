from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import validate_preset


class IterOnlyPresetFile:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        raise AssertionError("preset parsing should stream lines instead of read().splitlines()")

    def __iter__(self):
        return iter([
            "--stage1-rl-devices\n",
            "0,1,2,3\n",
            "# comment\n",
            "--stage2-rl-episodes 60000\n",
        ])


class ValidatePresetTest(unittest.TestCase):
    def test_extract_preset_flags_streams_lines_without_read(self):
        with (
            mock.patch.object(validate_preset.os.path, "isfile", return_value=True),
            mock.patch("builtins.open", return_value=IterOnlyPresetFile()),
        ):
            flags = validate_preset.extract_preset_flags("preset.conf")

        self.assertEqual(
            flags,
            [
                (1, "--stage1-rl-devices", "0,1,2,3"),
                (4, "--stage2-rl-episodes", "60000"),
            ],
        )

    def test_extract_python_argparse_flags_includes_multiline_aliases(self):
        source = "﻿" + """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--simple")
parser.add_argument(
    "--action-range",
    "--range",
    dest="action_ranges",
    action="append",
)
"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.py"
            path.write_text(source, encoding="utf-8")

            flags = validate_preset.extract_python_argparse_flags(str(path))
            repeatable = (
                validate_preset.extract_python_argparse_repeatable_flags(
                    str(path)
                )
            )

        self.assertEqual(flags, {"--simple", "--action-range", "--range"})
        self.assertEqual(repeatable, {"--action-range", "--range"})

    def test_repeatable_flags_are_not_reported_as_duplicates(self):
        entries = [
            (1, "--range", "block1.truncation=6,7"),
            (2, "--range", "block2.truncation=8,9"),
        ]
        with mock.patch.object(
                validate_preset,
                "extract_preset_flags",
                return_value=entries,
        ):
            problems = validate_preset.validate_preset(
                "unused.conf",
                {"--range"},
                repeatable_flags={"--range"},
            )

        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
