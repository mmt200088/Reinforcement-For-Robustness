import unittest

from runtime_error_reporter import (
    clear_error_summary,
    extract_cli_option,
    read_text_tail,
    write_error_summary,
)


class RuntimeErrorReporterTests(unittest.TestCase):
    def test_extract_cli_option_supports_space_and_equals_forms(self):
        argv = [
            "--dataset",
            "mrpc",
            "--output_dir",
            "run_a",
            "--noise-eval-repeat=5",
            "--output-dir=run_b",
        ]

        self.assertEqual(extract_cli_option(argv, ("output_dir",)), "run_a")
        self.assertEqual(extract_cli_option(argv, ("output-dir",)), "run_b")
        self.assertEqual(extract_cli_option(argv, ("missing",)), "")

    def test_write_error_summary_creates_logs_dir(self):
        with self.subTest():
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmpdir:
                from pathlib import Path

                run_dir = Path(tmpdir) / "sample_run"
                summary_path = write_error_summary(
                    str(run_dir),
                    program_name="rl_tune.py",
                    status="failed",
                    message="RuntimeError: CUDA out of memory",
                    argv=["python", "rl_tune.py", "--output_dir", str(run_dir)],
                    exit_code=1,
                    traceback_text="Traceback ...",
                )

                self.assertEqual(summary_path, run_dir / "logs" / "error_summary.txt")
                content = summary_path.read_text(encoding="utf-8")
                self.assertIn("[Program]\nrl_tune.py", content)
                self.assertIn("[Status]\nfailed", content)
                self.assertIn("CUDA out of memory", content)
                self.assertIn("Traceback ...", content)

    def test_clear_error_summary_removes_existing_file(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "sample_run"
            path = write_error_summary(
                str(run_dir),
                program_name="rl_tune.py",
                status="failed",
                message="boom",
            )
            self.assertTrue(path is not None and path.exists())

            clear_error_summary(str(run_dir))

            self.assertFalse(path.exists())

    def test_read_text_tail_returns_last_lines(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "output.log"
            log_path.write_text(
                "\n".join(f"line-{idx}" for idx in range(1, 8)),
                encoding="utf-8",
            )

            tail = read_text_tail(log_path, max_lines=3)

            self.assertEqual(tail, "line-5\nline-6\nline-7")


if __name__ == "__main__":
    unittest.main()
