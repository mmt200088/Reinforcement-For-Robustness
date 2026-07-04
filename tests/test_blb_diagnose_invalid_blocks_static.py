from pathlib import Path
import unittest


class BLBDiagnoseInvalidBlocksStaticTests(unittest.TestCase):
    def test_report_json_uses_shared_writer_without_json_dumps_string(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "blb_diagnose_invalid_blocks.py"
        source = source_path.read_text(encoding="utf-8")
        tree = compile(source, str(source_path), "exec")
        self.assertIsNotNone(tree)

        self.assertIn("from json_utils import read_json_file, write_json_file", source)
        self.assertIn("write_json_file(", source)
        self.assertIn('out_dir / "report.json"', source)
        self.assertNotIn('"report.json").write_text(', source)
        self.assertNotIn("json.dumps({", source)
