from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rfr.cli import evaluation_config as paean_config
from rfr.common.runtime_error_reporter import format_command as shared_format_command


class PaeanConfigTest(unittest.TestCase):
    def test_read_preset_args_streams_lines_without_read_text(self):
        with tempfile.TemporaryDirectory() as td:
            preset_dir = Path(td)
            preset = preset_dir / "stream.conf"
            preset.write_text(
                "--dataset mrpc\n"
                "--repeat\n"
                "5\n"
                "# comment\n"
                "--action-range block3.truncation=8,9\n",
                encoding="utf-8",
            )

            original_read_text = Path.read_text

            def fail_read_text(path, *_args, **_kwargs):
                if Path(path) == preset:
                    raise AssertionError("preset args should be streamed")
                return original_read_text(path, *_args, **_kwargs)

            with mock.patch.object(Path, "read_text", fail_read_text):
                args = paean_config._read_preset_args("stream", preset_dir=preset_dir)

        self.assertEqual(
            args,
            [
                "--dataset",
                "mrpc",
                "--repeat",
                "5",
                "--action-range",
                "block3.truncation=8,9",
            ],
        )

    def test_list_presets_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            preset_dir = Path(td)
            (preset_dir / "zeta.conf").write_text("--repeat 1\n", encoding="utf-8")
            (preset_dir / "alpha.conf").write_text("--repeat 2\n", encoding="utf-8")
            (preset_dir / "notes.txt").write_text("", encoding="utf-8")
            (preset_dir / "nested.conf").mkdir()

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("preset listing should not use Path.glob"),
            ):
                presets = paean_config.list_presets(preset_dir)

        self.assertEqual(presets, ["alpha", "zeta"])

    def test_format_command_reuses_runtime_error_reporter_helper(self):
        self.assertIs(paean_config.format_command, shared_format_command)
        self.assertEqual(
            paean_config.format_command(["python", "x y"]),
            "python 'x y'",
        )
        source = (
            Path(__file__).resolve().parents[1]
            / "src/rfr/cli/evaluation_config.py"
        ).read_text(
            encoding="utf-8",
        )
        self.assertIn("from rfr.common.runtime_error_reporter import format_command", source)
        self.assertNotIn("def format_command(", source)

    def test_checked_in_presets_parse_with_the_production_cli(self):
        for name in ("default", "mrpc-final-eval-only"):
            with self.subTest(name=name):
                settings = paean_config.parse_final_eval_settings(
                    ["--preset", name, "--dry-run"]
                )
                self.assertTrue(settings.dry_run)


if __name__ == "__main__":
    unittest.main()
