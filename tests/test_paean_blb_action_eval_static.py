import unittest

from tests.source_inspection_utils import source_text


class PaeanBLBActionEvalStaticTest(unittest.TestCase):
    def test_evaluation_protocol_reuses_action_spec_tuples_until_json_conversion(self):
        text = source_text("Paean/blb_action_eval.py")
        self.assertIn('"action_ranges": self.action_ranges,', text)
        self.assertIn('"action_fixed": self.action_fixed,', text)
        self.assertNotIn('"action_ranges": list(self.action_ranges),', text)
        self.assertNotIn('"action_fixed": list(self.action_fixed),', text)


if __name__ == "__main__":
    unittest.main()
