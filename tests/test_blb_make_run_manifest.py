import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "blb_make_run_manifest.py"


def _load_manifest_module():
    pkg = types.ModuleType("blb_stage2_rl")
    action_space = types.ModuleType("blb_stage2_rl.action_space")
    action_space.action_dims_for_config = lambda num_layers: [2] * (int(num_layers) + 1)
    action_space.per_layer_field_offsets = lambda: [(1, "a", "F"), (2, "b", "W")]
    original_pkg = sys.modules.get("blb_stage2_rl")
    original_action_space = sys.modules.get("blb_stage2_rl.action_space")
    sys.modules["blb_stage2_rl"] = pkg
    sys.modules["blb_stage2_rl.action_space"] = action_space
    try:
        spec = importlib.util.spec_from_file_location("blb_make_run_manifest", MANIFEST_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if original_pkg is None:
            sys.modules.pop("blb_stage2_rl", None)
        else:
            sys.modules["blb_stage2_rl"] = original_pkg
        if original_action_space is None:
            sys.modules.pop("blb_stage2_rl.action_space", None)
        else:
            sys.modules["blb_stage2_rl.action_space"] = original_action_space


class BlbMakeRunManifestTest(unittest.TestCase):
    def test_run_git_bounds_subprocess_with_timeout(self):
        manifest = _load_manifest_module()
        calls = []

        def fake_check_output(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return b"ok\n"

        with mock.patch.object(manifest.subprocess, "check_output", fake_check_output):
            result = manifest._run_git(["status", "--short"])

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1].get("timeout"), 5)

    def test_build_manifest_marks_dirty_without_second_status_strip(self):
        manifest = _load_manifest_module()

        class NoStripStatus(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("manifest dirty check should not strip git status twice")

        args = argparse.Namespace(
            registry_path="",
            max_sfs_path="",
            rescale_optimizer_root="",
            stage1_config_path="",
            stage1_source="",
            model="bert-base",
            profile="mrpc",
            threshold_source="manual",
            dataset="mrpc",
            rescale_optimizer_mode="canonical",
            action_space_version="test",
            num_layers=12,
            decode_version="test",
            acc_limit=0.0,
            f1_limit=0.0,
            acc_std_limit=0.0,
            f1_std_limit=0.0,
            strict_z=1.0,
            mpc_truncation_cost_enabled=False,
        )

        def fake_run_git(git_args):
            if git_args[:2] == ["status", "--short"]:
                return NoStripStatus(" M run_manifest.json")
            if git_args and git_args[0] == "diff":
                return ""
            if git_args[:2] == ["rev-parse", "HEAD"]:
                return "abc123"
            return ""

        with mock.patch.object(manifest, "_run_git", side_effect=fake_run_git):
            result = manifest.build_manifest(args)

        self.assertTrue(result["git"]["dirty"])

    def test_build_manifest_reuses_per_layer_offsets(self):
        manifest = _load_manifest_module()
        calls = 0

        def counted_offsets():
            nonlocal calls
            calls += 1
            return [(1, "a", "F"), (2, "b", "W")]

        args = argparse.Namespace(
            registry_path="",
            max_sfs_path="",
            rescale_optimizer_root="",
            stage1_config_path="",
            stage1_source="",
            model="bert-base",
            profile="mrpc",
            threshold_source="manual",
            dataset="mrpc",
            rescale_optimizer_mode="canonical",
            action_space_version="test",
            num_layers=12,
            decode_version="test",
            acc_limit=0.0,
            f1_limit=0.0,
            acc_std_limit=0.0,
            f1_std_limit=0.0,
            strict_z=1.0,
            mpc_truncation_cost_enabled=False,
        )

        with (
            mock.patch.object(manifest, "per_layer_field_offsets", counted_offsets),
            mock.patch.object(manifest, "_run_git", return_value=""),
        ):
            result = manifest.build_manifest(args)

        self.assertEqual(calls, 1)
        self.assertEqual(result["action_space"]["slot_counts"], {"block1": 1, "block2": 1})
        self.assertEqual(result["action_space"]["per_layer_slot_count"], 2)

    def test_canonical_rescale_optimizer_hash_streams_file_contents(self):
        manifest = _load_manifest_module()
        original_read_bytes = Path.read_bytes

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "rescale_optimizer").mkdir()
            (root / "rescale_optimizer" / "solver.py").write_text("print('solver')\n", encoding="utf-8")
            (root / "configs" / "mrpc").mkdir(parents=True)
            (root / "configs" / "mrpc" / "block1_mrpc.json").write_text('{"graph": 1}\n', encoding="utf-8")

            def fail_read_bytes(_path):
                raise AssertionError("canonical hash should stream file bytes")

            try:
                Path.read_bytes = fail_read_bytes
                digest = manifest._canonical_rescale_optimizer_hash(str(root), "mrpc")
            finally:
                Path.read_bytes = original_read_bytes

        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")

    def test_canonical_rescale_optimizer_hash_streams_tree_without_global_sort(self):
        manifest = _load_manifest_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "rescale_optimizer").mkdir()
            (root / "rescale_optimizer" / "solver.py").write_text("print('solver')\n", encoding="utf-8")
            (root / "configs" / "mrpc").mkdir(parents=True)
            (root / "configs" / "mrpc" / "block1.json").write_text('{"graph": 1}\n', encoding="utf-8")

            with mock.patch(
                "builtins.sorted",
                side_effect=AssertionError("canonical hash should stream tree traversal"),
            ):
                digest = manifest._canonical_rescale_optimizer_hash(str(root), "mrpc")

        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")

    def test_canonical_rescale_optimizer_hash_preserves_global_path_order(self):
        manifest = _load_manifest_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, text in {
                "rescale_optimizer/z.py": "z\n",
                "rescale_optimizer/a.py": "a\n",
                "rescale_optimizer/ignore.txt": "ignored\n",
                "configs/mrpc/b.json": '{"b": 1}\n',
                "configs/mrpc/nested/c.json": '{"c": 2}\n',
                "replan_configs/mrpc/d.json": '{"d": 3}\n',
            }.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            expected = hashlib.sha256()
            for file_path in sorted(
                [
                    root / "configs" / "mrpc" / "b.json",
                    root / "configs" / "mrpc" / "nested" / "c.json",
                    root / "replan_configs" / "mrpc" / "d.json",
                    root / "rescale_optimizer" / "a.py",
                    root / "rescale_optimizer" / "z.py",
                ]
            ):
                expected.update(file_path.relative_to(root).as_posix().encode("utf-8"))
                expected.update(b"\0")
                with file_path.open("rb") as handle:
                    expected.update(handle.read())
                expected.update(b"\0")

            digest = manifest._canonical_rescale_optimizer_hash(str(root), "mrpc")

        self.assertEqual(digest, expected.hexdigest())

    def test_dir_sha256_streams_tree_without_global_sort(self):
        manifest = _load_manifest_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "a" / "one.py").write_text("one\n", encoding="utf-8")
            (root / "b").mkdir()
            (root / "b" / "two.json").write_text('{"two": 2}\n', encoding="utf-8")

            with mock.patch(
                "builtins.sorted",
                side_effect=AssertionError("directory hash should stream tree traversal"),
            ):
                digest = manifest._dir_sha256(root)

        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")

    def test_dir_sha256_preserves_global_path_order_and_skip_dirs(self):
        manifest = _load_manifest_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel, text in {
                "z/tail.py": "tail\n",
                "a/head.py": "head\n",
                ".git/ignored.py": "ignored\n",
                "__pycache__/ignored.py": "ignored\n",
                "m/mid.json": '{"mid": 1}\n',
            }.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            expected = hashlib.sha256()
            for file_path in sorted(
                [
                    root / "a" / "head.py",
                    root / "m" / "mid.json",
                    root / "z" / "tail.py",
                ]
            ):
                expected.update(file_path.relative_to(root).as_posix().encode("utf-8"))
                file_hash = manifest._file_sha256(file_path)
                expected.update(str(file_hash).encode("ascii"))

            digest = manifest._dir_sha256(root)

        self.assertEqual(digest, expected.hexdigest())

    def test_dir_sha256_prunes_skip_dirs_before_iterating_them(self):
        manifest = _load_manifest_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "keep.py").write_text("keep\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "large-object").write_text("ignored\n", encoding="utf-8")

            original_iterdir = Path.iterdir

            def guarded_iterdir(path):
                if Path(path).name == ".git":
                    raise AssertionError("skip directories should not be traversed")
                return original_iterdir(path)

            with mock.patch.object(Path, "iterdir", guarded_iterdir):
                digest = manifest._dir_sha256(root)

        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
