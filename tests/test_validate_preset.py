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


if __name__ == "__main__":
    unittest.main()
