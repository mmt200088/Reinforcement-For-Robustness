import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import blb_verify_noise_install as verifier


class BLBVerifyNoiseInstallTest(unittest.TestCase):
    def setUp(self):
        verifier._NOISE_VARIANCE_TABLE_CACHE = None

    def tearDown(self):
        verifier._NOISE_VARIANCE_TABLE_CACHE = None

    def test_noise_variance_table_is_cached_after_first_source_parse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "function_handler.py").write_text(
                """
_NOISE_STD_RAW = {
    16384: {
        30: (0.25, 0.5, 1.0),
        31: (0.125, 0.25, 0.5),
    },
}
""",
                encoding="utf-8",
            )
            original_read_text = Path.read_text
            reads = []

            def counting_read_text(path, *args, **kwargs):
                if path.name == "function_handler.py":
                    reads.append(path)
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(verifier, "REPO_ROOT", root):
                with mock.patch.object(Path, "read_text", counting_read_text):
                    first = verifier.load_noise_variance_table()
                    second = verifier.load_noise_variance_table()

        self.assertIs(first, second)
        self.assertEqual(len(reads), 1)
        self.assertEqual(first[16384][30]["encoding"], 0.25**2)
        self.assertEqual(first[16384][30]["fresh"], 0.5**2)
        self.assertEqual(first[16384][30]["rescale"], 1.0)
        self.assertEqual(first[16384][30]["rotation"], 1.0)

    def test_lookup_variance_uses_cached_table_shape(self):
        table = {
            16384: {
                30: {
                    "encoding": 0.1,
                    "fresh": 0.2,
                    "rescale": 0.3,
                    "rotation": 0.3,
                }
            }
        }

        self.assertEqual(
            verifier.lookup_variance(table, N=16384, sf=30, distribution="fresh"),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()

