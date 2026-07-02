import glob
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from tools import aggregate_seeds


@contextmanager
def _pushd(path: Path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _write_json(path: Path, payload: dict, mtime: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _make_seed_run(root: Path, run_tag: str, reward: float) -> None:
    persistent_dir = root / "Parting Chapter" / "persistent" / "bert" / "base" / "mrpc" / f"slug__{run_tag}"
    progress_dir = persistent_dir / "blb_stage2" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "blb_stage2_status.json").write_text(
        json.dumps({"completed_episodes": 10, "best": {"reward": reward}}),
        encoding="utf-8",
    )
    _write_json(
        persistent_dir / "final_eval" / f"blb_action_final_eval_results_{run_tag}.json",
        {"candidate_results": [{"loss": 1.0 - reward, "p": 0.8, "s": 0.7}]},
        mtime=100,
    )


class AggregateSeedsFinalEvalTest(unittest.TestCase):
    def test_read_final_eval_results_streams_latest_json_without_recursive_glob(self):
        with tempfile.TemporaryDirectory() as td:
            persistent_dir = Path(td)
            _write_json(
                persistent_dir / "final_eval" / "old" / "blb_action_final_eval_results_old.json",
                {"candidate_results": [{"loss": 2.0}]},
                mtime=100,
            )
            _write_json(
                persistent_dir / "nested" / "new" / "blb_action_final_eval_results_new.json",
                {"candidate_results": [{"loss": 1.0}]},
                mtime=200,
            )

            with mock.patch.object(
                glob,
                "glob",
                side_effect=AssertionError("recursive glob should not be used"),
            ):
                payload = aggregate_seeds._read_final_eval_results(str(persistent_dir))

        self.assertEqual(payload["candidate_results"][0]["loss"], 1.0)

    def test_main_aggregates_seed_rows_without_per_seed_persistent_lookup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_seed_run(root, "run_s1", reward=0.25)
            _make_seed_run(root, "run_s2", reward=0.5)

            output_dir = root / "out"
            seed_list = root / "seed_list.txt"
            seed_list.write_text("1 run_s1\n2 run_s2\n", encoding="utf-8")

            with _pushd(root), mock.patch.object(
                aggregate_seeds,
                "_find_persistent_dir",
                side_effect=AssertionError("main should reuse one persistent index"),
            ):
                rc = aggregate_seeds.main(
                    [
                        "--run-name",
                        "idx",
                        "--seed-list",
                        str(seed_list),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            rows = json.loads((output_dir / "seed_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual([row["status"] for row in rows], ["complete", "complete"])
        self.assertEqual([row["persistent_dir"].split("/")[-1] for row in rows], ["slug__run_s1", "slug__run_s2"])

    def test_persistent_dir_index_keeps_only_requested_run_tags(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_seed_run(root, "wanted", reward=0.25)
            _make_seed_run(root, "unrelated", reward=0.5)

            index = aggregate_seeds._build_persistent_dir_index(
                str(root / "Parting Chapter" / "persistent"),
                requested_run_tags={"wanted"},
            )

        self.assertEqual(list(index), ["wanted"])
        self.assertTrue(index["wanted"].endswith("slug__wanted"))


if __name__ == "__main__":
    unittest.main()
