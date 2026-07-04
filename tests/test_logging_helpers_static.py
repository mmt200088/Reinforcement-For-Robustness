import unittest

from tests.source_inspection_utils import source_text


class LoggingHelpersStaticTest(unittest.TestCase):
    def test_json_formatter_reuses_encoder_per_record(self):
        text = source_text("blb_stage2_rl/logging_helpers.py")
        self.assertIn("_JSON_LOG_ENCODER = json.JSONEncoder(", text)
        self.assertIn("return _JSON_LOG_ENCODER.encode(payload)", text)
        self.assertNotIn("return json.dumps(payload, ensure_ascii=False, default=str)", text)


if __name__ == "__main__":
    unittest.main()
