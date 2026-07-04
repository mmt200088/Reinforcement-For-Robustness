import unittest

from tests.source_inspection_utils import source_text


class PaeanRunFinalEvalStaticTest(unittest.TestCase):
    def test_build_command_dumps_action_specs_without_list_copy(self):
        text = source_text("Paean/run_final_eval.py")
        self.assertIn("json.dumps(settings.action_ranges)", text)
        self.assertIn("json.dumps(settings.action_fixed)", text)
        self.assertNotIn("json.dumps(list(settings.action_ranges))", text)
        self.assertNotIn("json.dumps(list(settings.action_fixed))", text)


if __name__ == "__main__":
    unittest.main()
