from pathlib import Path
import unittest


class BLBEvalActionStaticTests(unittest.TestCase):
    def test_main_streams_stdout_json_without_json_dumps_string(self):
        source_path = Path(__file__).resolve().parents[1] / "scripts" / "blb_eval_action.py"
        source = source_path.read_text(encoding="utf-8")
        tree = compile(source, str(source_path), "exec")
        self.assertIsNotNone(tree)

        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(record, sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(record", main_source)


if __name__ == "__main__":
    unittest.main()
