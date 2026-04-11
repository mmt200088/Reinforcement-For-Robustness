import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


class CompareProcessGuardTests(unittest.TestCase):
    def test_start_child_passes_parent_death_preexec_fn_when_available(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        fake_process = SimpleNamespace(pid=12345, poll=lambda: None)
        fake_preexec = lambda: None

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "child_run"
            spec = compare_runner.ChildRunSpec(
                algorithm="rl",
                entrypoint="rl_tune.py",
                run_dir=run_dir,
                log_path=run_dir / "logs" / "output.log",
                command=["python", "rl_tune.py"],
                env_overrides={},
            )
            with patch.object(
                compare_runner,
                "_build_parent_death_preexec_fn",
                return_value=fake_preexec,
            ), patch(
                "subprocess.Popen",
                return_value=fake_process,
            ) as mock_popen:
                compare_runner.start_child(spec, extra_env={})

            self.assertIs(spec.process, fake_process)
            self.assertEqual(
                mock_popen.call_args.kwargs.get("preexec_fn"),
                fake_preexec,
            )


if __name__ == "__main__":
    unittest.main()
