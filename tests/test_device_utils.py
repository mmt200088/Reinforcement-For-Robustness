import unittest

from device_utils import parse_device_ids


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


if __name__ == "__main__":
    unittest.main()
