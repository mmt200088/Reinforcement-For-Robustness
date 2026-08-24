from pathlib import Path
import unittest

from device_utils import (
    normalize_logical_device_token,
    parse_device_ids,
    parse_logical_device_spec,
    split_device_spec_tokens,
)


class DeviceUtilsTest(unittest.TestCase):
    def test_parse_device_ids_accepts_cli_and_fire_forms(self):
        self.assertEqual(parse_device_ids(None), [])
        self.assertEqual(parse_device_ids(""), [])
        self.assertEqual(parse_device_ids("0,1"), [0, 1])
        self.assertEqual(parse_device_ids(" 0 , 1 "), [0, 1])
        self.assertEqual(parse_device_ids("(0, 1)"), [0, 1])
        self.assertEqual(parse_device_ids("[0, 1]"), [0, 1])
        self.assertEqual(parse_device_ids((0, 1, 2)), [0, 1, 2])
        self.assertEqual(parse_device_ids([0, 1]), [0, 1])
        self.assertEqual(parse_device_ids(0), [0])

    def test_parse_device_ids_rejects_bool_and_garbage(self):
        with self.assertRaises(ValueError):
            parse_device_ids(True)
        with self.assertRaises(ValueError):
            parse_device_ids([0, False])
        with self.assertRaises(ValueError):
            parse_device_ids("0,abc")

    def test_split_device_spec_tokens_preserves_physical_tokens(self):
        self.assertEqual(split_device_spec_tokens("0, GPU-abcd "), ["0", "GPU-abcd"])
        self.assertEqual(split_device_spec_tokens("(0, 1)"), ["0", "1"])
        self.assertEqual(split_device_spec_tokens(["0", " GPU-abcd "]), ["0", "GPU-abcd"])
        self.assertEqual(
            split_device_spec_tokens("disabled", disabled_tokens={"disabled"}),
            [],
        )

    def test_parse_logical_device_spec_normalizes_diagnostic_names(self):
        self.assertEqual(normalize_logical_device_token("0"), "cuda:0")
        self.assertEqual(normalize_logical_device_token(" CUDA:1 "), "cuda:1")
        self.assertEqual(normalize_logical_device_token("cpu"), "cpu")
        self.assertEqual(normalize_logical_device_token("none"), "")
        self.assertEqual(parse_logical_device_spec("0,cuda:1,cpu"), ["cuda:0", "cuda:1", "cpu"])
        self.assertEqual(parse_logical_device_spec("0;1", allow_semicolon=True), ["cuda:0", "cuda:1"])
        self.assertEqual(
            parse_logical_device_spec("disabled", disabled_tokens={"disabled"}),
            [],
        )

    def test_diagnostic_scripts_use_shared_logical_device_parser(self):
        repo = Path(__file__).resolve().parents[1]
        expected = {
            "scripts/gpu_utilization_report.py": "from device_utils import normalize_logical_device_token, parse_logical_device_spec",
        }
        for rel, needle in expected.items():
            text = (repo / rel).read_text(encoding="utf-8")
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
