import unittest

from scripts.blb_eval_action import _parse_degree_arg


class BLBEvalActionStage1ContextTests(unittest.TestCase):
    def test_parse_degree_vector_preserves_layerwise_values(self):
        self.assertEqual(
            _parse_degree_arg("[1, 1, 4]", num_layers=3, name="gelu"),
            [1, 1, 4],
        )

    def test_parse_degree_scalar_preserves_scalar(self):
        self.assertEqual(
            _parse_degree_arg("4", num_layers=12, name="gelu"),
            4,
        )

    def test_parse_degree_vector_rejects_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "gelu"):
            _parse_degree_arg("[1, 2]", num_layers=3, name="gelu")


if __name__ == "__main__":
    unittest.main()
