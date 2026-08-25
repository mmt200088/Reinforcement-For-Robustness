"""torch-free 单测：config/run_layout.py（解耦后 RL 输出布局 SSOT）。

覆盖：combo 命名（空格）、扁平工作目录、record 序号扫描（含 combo 自身含数字的
sst2、以及多 combo 共存）、run-id 格式、完成标记往返、约束守卫、record 定位。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import run_layout as rl


class ComboAndPathTest(unittest.TestCase):
    def test_combo_name_spaces(self):
        self.assertEqual(rl.combo_name("bert-base", "mrpc"), "bert base mrpc")
        self.assertEqual(rl.combo_name("bert-large", "sst2"), "bert large sst2")
        self.assertEqual(rl.combo_name("bert-base", "rte"), "bert base rte")
        with self.assertRaises(ValueError):
            rl.combo_name("gpt-2", "mrpc")

    def test_combo_name_rejects_empty(self):
        with self.assertRaises(ValueError):
            rl.combo_name("", "mrpc")
        with self.assertRaises(ValueError):
            rl.combo_name("bert-base", "")

    def test_normalize_stage(self):
        for v in (1, "1", "stage1", "stage1-only", "stage1_only"):
            self.assertEqual(rl.normalize_stage(v), 1)
        for v in (2, "2", "stage2", "stage2-only"):
            self.assertEqual(rl.normalize_stage(v), 2)
        with self.assertRaises(ValueError):
            rl.normalize_stage("train")

    def test_flattened_working_dir(self):
        root = "Parting Chapter"
        self.assertEqual(
            rl.stage_working_dir("stage1-only", "bert-base", "mrpc", root=root),
            os.path.join(root, "stage1", "bert base mrpc"),
        )
        self.assertEqual(
            rl.stage_working_dir(2, "bert-large", "rte", root=root),
            os.path.join(root, "stage2", "bert large rte"),
        )

        self.assertNotIn(
            os.path.join("bert base mrpc", "stage1"),
            rl.stage_working_dir(1, "bert-base", "mrpc", root=root),
        )

    def test_record_root(self):
        root = "Parting Chapter"
        self.assertEqual(
            rl.stage_record_root(1, root=root), os.path.join(root, "stage1", "record")
        )

    def test_run_id_format(self):
        self.assertEqual(
            rl.run_id("bert-base", "rte", 1, "20260530"), "bert base rte 1 20260530"
        )

        self.assertEqual(
            rl.run_id("bert-base", "mrpc", 12, "20260531_141500"),
            "bert base mrpc 12 20260531",
        )


class RunNumberScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _mk_record(self, stage, name):
        d = os.path.join(rl.stage_record_root(stage, root=self.tmp), name)
        os.makedirs(d, exist_ok=True)
        return d

    def test_next_run_number_empty(self):
        self.assertEqual(rl.next_run_number(1, "bert-base", "rte", root=self.tmp), 1)

    def test_next_run_number_increments_per_combo(self):
        self._mk_record(1, "bert base rte 1 20260530")
        self.assertEqual(rl.next_run_number(1, "bert-base", "rte", root=self.tmp), 2)
        self._mk_record(1, "bert base rte 2 20260531")
        self.assertEqual(rl.next_run_number(1, "bert-base", "rte", root=self.tmp), 3)

    def test_combos_are_independent(self):
        self._mk_record(1, "bert base rte 1 20260530")
        self._mk_record(1, "bert base mrpc 1 20260530")

        self.assertEqual(rl.next_run_number(1, "bert-base", "rte", root=self.tmp), 2)
        self.assertEqual(rl.next_run_number(1, "bert-base", "mrpc", root=self.tmp), 2)

        self.assertEqual(rl.next_run_number(1, "bert-base", "sst2", root=self.tmp), 1)

    def test_sst2_number_in_combo_parses(self):

        self._mk_record(1, "bert base sst2 1 20260530")
        self._mk_record(1, "bert base sst2 2 20260531")
        self.assertEqual(rl.next_run_number(1, "bert-base", "sst2", root=self.tmp), 3)
        self.assertEqual(
            rl.existing_run_numbers(1, "bert-base", "sst2", root=self.tmp), [1, 2]
        )

        self.assertEqual(rl.next_run_number(1, "bert-large", "sst2", root=self.tmp), 1)

    def test_stage1_stage2_independent(self):
        self._mk_record(1, "bert base mrpc 1 20260530")
        self.assertEqual(rl.next_run_number(2, "bert-base", "mrpc", root=self.tmp), 1)

    def test_garbage_entries_ignored(self):
        self._mk_record(1, "bert base rte 1 20260530")
        self._mk_record(1, "not-a-run-id")
        self._mk_record(1, "bert base rte X 20260530")
        self._mk_record(1, "bert base rte 2 2026")
        self.assertEqual(rl.next_run_number(1, "bert-base", "rte", root=self.tmp), 2)

    def test_find_record_dir(self):
        self._mk_record(2, "bert base mrpc 1 20260530")
        d2 = self._mk_record(2, "bert base mrpc 2 20260531")

        self.assertEqual(
            rl.find_record_dir(2, "bert-base", "mrpc", root=self.tmp), d2
        )

        self.assertEqual(
            rl.find_record_dir(
                2, "bert-base", "mrpc", run_id_name="bert base mrpc 1 20260530",
                root=self.tmp,
            ),
            os.path.join(rl.stage_record_root(2, root=self.tmp), "bert base mrpc 1 20260530"),
        )

        self.assertIsNone(
            rl.find_record_dir(2, "bert-base", "rte", root=self.tmp)
        )

    def test_make_record_dir(self):
        self._mk_record(1, "bert base mrpc 1 20260530")
        rdir, rid, n = rl.make_record_dir(
            1, "bert-base", "mrpc", timestamp="20260601", root=self.tmp
        )
        self.assertEqual(n, 2)
        self.assertEqual(rid, "bert base mrpc 2 20260601")
        self.assertTrue(os.path.isdir(rdir))


class CompletedMarkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_marker_round_trip(self):
        wd = os.path.join(self.tmp, "stage1", "bert base mrpc")
        os.makedirs(wd, exist_ok=True)
        self.assertFalse(rl.is_completed(wd))
        rl.mark_completed(wd, {"episodes": 51000})
        self.assertTrue(rl.is_completed(wd))
        rl.clear_completed(wd)
        self.assertFalse(rl.is_completed(wd))


class ConstraintGuardTest(unittest.TestCase):
    def test_match_returns_none(self):
        persisted = {
            "stage1_accuracy_tolerance": 0.005,
            "stage2_limit_tolerance": 0.005,
            "stage2_stability_tolerance": 0.005,
        }
        current = {
            "stage1_accuracy_tolerance": "0.005",
            "stage2_limit_tolerance": 0.005,
            "stage2_stability_tolerance": 0.005,
        }
        self.assertIsNone(rl.constraint_mismatch(persisted, current))

    def test_mismatch_reported(self):
        persisted = {"stage1_accuracy_tolerance": 0.005}
        current = {"stage1_accuracy_tolerance": 0.01}
        msg = rl.constraint_mismatch(persisted, current)
        self.assertIsNotNone(msg)
        self.assertIn("stage1_accuracy_tolerance", msg)

    def test_missing_side_skipped(self):

        self.assertIsNone(rl.constraint_mismatch({}, {"stage1_accuracy_tolerance": 0.005}))
        self.assertIsNone(rl.constraint_mismatch({"stage2_limit_tolerance": 0.005}, {}))


class SnapshotHelperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_latest_record_dir_in_root(self):
        rroot = rl.stage_record_root(1, root=self.tmp)
        os.makedirs(os.path.join(rroot, "bert base mrpc 1 20260530"))
        os.makedirs(os.path.join(rroot, "bert base mrpc 2 20260531"))
        d = rl.latest_record_dir_in_root(rroot, "bert base mrpc")
        self.assertTrue(d.endswith("bert base mrpc 2 20260531"))
        d1 = rl.latest_record_dir_in_root(
            rroot, "bert base mrpc", run_id_name="bert base mrpc 1 20260530"
        )
        self.assertTrue(d1.endswith("bert base mrpc 1 20260530"))
        self.assertIsNone(rl.latest_record_dir_in_root(rroot, "bert base rte"))

    def test_next_run_number_in_root(self):
        rroot = rl.stage_record_root(2, root=self.tmp)
        self.assertEqual(rl.next_run_number_in_root(rroot, "bert base mrpc"), 1)
        os.makedirs(os.path.join(rroot, "bert base mrpc 1 20260530"))
        self.assertEqual(rl.next_run_number_in_root(rroot, "bert base mrpc"), 2)

    def test_snapshot_decoupled_record_writes_record_and_marks_completed(self):
        import datetime
        wd = rl.stage_working_dir(1, "bert-base", "mrpc", root=self.tmp)
        os.makedirs(wd, exist_ok=True)
        curve = os.path.join(wd, "ppo_training_curve.png")
        with open(curve, "w") as f:
            f.write("x")
        rdir, rid, n = rl.snapshot_decoupled_record(
            1, "bert base mrpc", wd,
            final_config={"gelu_degree_per_layer": [4, 2, 1]},
            final_eval={"metric1": 0.86},
            metadata={"stage": 1},
            curve_paths=[curve],
            report_md="# r",
            root=self.tmp,
        )
        self.assertEqual(n, 1)
        self.assertEqual(rid, "bert base mrpc 1 " + datetime.datetime.now().strftime("%Y%m%d"))
        for fn in ("final_config.json", "final_eval.json", "metadata.json", "report.md", "ppo_training_curve.png"):
            self.assertTrue(os.path.isfile(os.path.join(rdir, fn)), fn)
        self.assertTrue(rl.is_completed(wd))

        _, _, n2 = rl.snapshot_decoupled_record(
            1, "bert base mrpc", wd, final_config={}, final_eval={}, root=self.tmp,
        )
        self.assertEqual(n2, 2)


if __name__ == "__main__":
    unittest.main()
