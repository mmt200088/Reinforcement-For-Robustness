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

    def test_main_reuses_rlpath_stage1_default_json_strings(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_block4_fusion_install.py"
        source = source_path.read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertIn("DEFAULT_STAGE1_GELU_JSON", source)
        self.assertIn("DEFAULT_STAGE1_SOFTMAX_JSON", source)
        self.assertIn("default=DEFAULT_STAGE1_GELU_JSON", main_source)
        self.assertIn("default=DEFAULT_STAGE1_SOFTMAX_JSON", main_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_GELU)", main_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_SOFTMAX)", main_source)
