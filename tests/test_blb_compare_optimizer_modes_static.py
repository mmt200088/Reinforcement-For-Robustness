from pathlib import Path
import unittest


class BLBCompareOptimizerModesStaticTests(unittest.TestCase):
    def test_main_streams_stdout_json_without_json_dumps_string(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "blb_compare_optimizer_modes.py"
        source = source_path.read_text(encoding="utf-8")
        tree = compile(source, str(source_path), "exec")
        self.assertIsNotNone(tree)

        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(", main_source)
        self.assertIn("sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(", main_source)

    def test_markdown_writer_streams_without_joined_write_text(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "blb_compare_optimizer_modes.py"
        source = source_path.read_text(encoding="utf-8")
        tree = compile(source, str(source_path), "exec")
        self.assertIsNotNone(tree)

        markdown_source = source[source.index("def _write_markdown("):source.index("def run_compare(")]

        self.assertIn(".open(", markdown_source)
        self.assertNotIn("write_text(", markdown_source)
        self.assertNotIn('"\\n".join(lines)', markdown_source)
