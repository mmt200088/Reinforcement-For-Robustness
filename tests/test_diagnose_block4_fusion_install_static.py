from pathlib import Path
import unittest


class DiagnoseBlock4FusionInstallStaticTests(unittest.TestCase):
    def test_main_streams_stdout_json_without_json_dumps_string(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_block4_fusion_install.py"
        source = source_path.read_text(encoding="utf-8")
        tree = compile(source, str(source_path), "exec")
        self.assertIsNotNone(tree)

        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(", main_source)
        self.assertIn("sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(", main_source)
