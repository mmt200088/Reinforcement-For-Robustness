from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Paean import config as paean_config


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


if __name__ == "__main__":
    unittest.main()
