import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

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


if __name__ == "__main__":
    unittest.main()
