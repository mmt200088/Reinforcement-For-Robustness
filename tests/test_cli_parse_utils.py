import pathlib
import unittest

from rfr.common.cli_parse_utils import (
    parse_bool_flag,
    parse_broadcast_int_vector,
    parse_degree_config,
    parse_exact_json_int_list,
    parse_float_list_text,
    parse_int_list_text,
    parse_json_int_list,
    parse_noise_config,
    parse_optional_int_list,
    parse_optional_positive_float,
    parse_optional_positive_int,
    parse_positive_int,
    parse_stage1_episode_limit,
    parse_stage2_episode_limit,
    split_int_tokens,
)


class CliParseUtilsTest(unittest.TestCase):
    def test_split_and_parse_int_list_text(self):
        self.assertEqual(split_int_tokens("1, 2;3"), ["1", "2", "3"])
        self.assertEqual(parse_int_list_text("1, 2;3"), [1, 2, 3])
        self.assertEqual(split_int_tokens("1, 2;3", allow_semicolon=False), ["1", "2;3"])
        with self.assertRaises(ValueError):
            parse_int_list_text("1, 2;3", allow_semicolon=False)
        self.assertEqual(parse_float_list_text("0.1, .2;3"), [0.1, 0.2, 3.0])
        with self.assertRaises(ValueError):
            parse_float_list_text("0.1;0.2", allow_semicolon=False)

    def test_optional_int_list(self):
        self.assertIsNone(parse_optional_int_list(None))
        self.assertIsNone(parse_optional_int_list(""))
        self.assertEqual(parse_optional_int_list("4; 5"), [4, 5])

    def test_json_int_list_with_default_and_errors(self):
        self.assertEqual(parse_json_int_list("", default=[1, 2], name="x"), [1, 2])
        self.assertEqual(parse_json_int_list("[3, \"4\"]", default=[], name="x"), [3, 4])
        with self.assertRaises(SystemExit):
            parse_json_int_list("3", default=[], name="x")

    def test_exact_json_int_list(self):
        self.assertEqual(parse_exact_json_int_list("[1, \"2\"]", name="gelu", length=2), [1, 2])
        with self.assertRaisesRegex(ValueError, "gelu must be a JSON list with 3 entries"):
            parse_exact_json_int_list("[1, 2]", name="gelu", length=3)

    def test_broadcast_int_vector(self):
        self.assertEqual(parse_broadcast_int_vector(None, num_layers=3, default=4), [4, 4, 4])
        self.assertEqual(parse_broadcast_int_vector("2", num_layers=3, default=4), [2, 2, 2])
        self.assertEqual(parse_broadcast_int_vector("[1, 2, 3]", num_layers=3, default=4), [1, 2, 3])
        self.assertEqual(parse_broadcast_int_vector("1;2;3", num_layers=3, default=4), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "degree vector length 2 must be 1 or num_layers=3"):
            parse_broadcast_int_vector([1, 2], num_layers=3, default=4)

    def test_legacy_rl_cli_parsers(self):
        self.assertIsNone(parse_degree_config(None))
        self.assertIsNone(parse_degree_config(""))
        self.assertEqual(parse_degree_config([1, "2"]), [1, 2])
        self.assertEqual(parse_degree_config("[1, \"2\"]"), [1, 2])
        self.assertEqual(parse_degree_config("1, 2"), [1, 2])

        self.assertIsNone(parse_noise_config(""))
        self.assertEqual(parse_noise_config({"x": 1}), {"x": 1})
        self.assertEqual(parse_noise_config('{"x": 1}'), {"x": 1})

        self.assertTrue(parse_bool_flag("yes", "flag"))
        self.assertFalse(parse_bool_flag("off", "flag"))
        with self.assertRaisesRegex(ValueError, "Invalid boolean value for flag"):
            parse_bool_flag("maybe", "flag")

        self.assertEqual(parse_positive_int("3", "n"), 3)
        self.assertIsNone(parse_optional_positive_int("", "n"))
        self.assertEqual(parse_optional_positive_int("4", "n"), 4)
        with self.assertRaisesRegex(ValueError, "Invalid positive integer for n"):
            parse_positive_int("0", "n")

        self.assertEqual(parse_stage1_episode_limit("-1", "episodes"), -1)
        self.assertEqual(parse_stage2_episode_limit("0", "episodes"), 0)
        self.assertEqual(parse_stage2_episode_limit("120", "episodes"), 120)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            parse_stage2_episode_limit("-1", "episodes")
        self.assertIsNone(parse_optional_positive_float(None, "lr"))
        self.assertEqual(parse_optional_positive_float("0.5", "lr"), 0.5)
        with self.assertRaisesRegex(ValueError, "Invalid positive float for lr"):
            parse_optional_positive_float("0", "lr")

if __name__ == "__main__":
    unittest.main()
