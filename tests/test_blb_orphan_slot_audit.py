import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import blb_orphan_slot_audit as audit


class OrphanSlotAuditTest(unittest.TestCase):
    def setUp(self):
        audit._AST_CACHE.clear()

    def tearDown(self):
        audit._AST_CACHE.clear()

    def test_slot_loader_reuses_function_handler_ast(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "function_handler.py").write_text(
                """
def make_block1_default_config(wffn2_sf, N):
    return Block1NoiseConfig(wffn2_encode=NoisePoint("fresh", int(wffn2_sf), int(N)))

def make_block2_default_config(wq_sf, N):
    return Block2NoiseConfig(wq_encode=NoisePoint("fresh", int(wq_sf), int(N)))
""",
                encoding="utf-8",
            )
            original_read_text = Path.read_text
            reads = []

            def counting_read_text(path, *args, **kwargs):
                if path.name == "function_handler.py":
                    reads.append(path)
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(audit, "REPO_ROOT", root):
                with mock.patch.object(Path, "read_text", counting_read_text):
                    self.assertEqual(audit.load_slot_to_cfg_field(1)["wffn2_sf"], ("wffn2_encode", "core"))
                    self.assertEqual(audit.load_slot_to_cfg_field(2)["wq_sf"], ("wq_encode", "core"))

        self.assertEqual(len(reads), 1)

    def test_bridge_loaders_share_rescale_bridge_ast(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "rescale_optimizer_bridge.py").write_text(
                """
def default_block1_cfg_to_delta(cfg):
    return {"ctpt_ffn2": int(cfg.wffn2_encode.scaling_factor)}

def default_block2_cfg_to_delta(cfg):
    return {"ctpt_wq_wk": int(cfg.wq_encode.scaling_factor)}

DEFAULT_CFG_TO_T_NEW_MAP = {
    "block1_mrpc": (_SkelEntry("wffn2_encode"),),
}
""",
                encoding="utf-8",
            )
            original_read_text = Path.read_text
            reads = []

            def counting_read_text(path, *args, **kwargs):
                if path.name == "rescale_optimizer_bridge.py":
                    reads.append(path)
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(audit, "REPO_ROOT", root):
                with mock.patch.object(Path, "read_text", counting_read_text):
                    self.assertEqual(
                        audit.load_cfg_field_to_graph_node(1)["ctpt_ffn2"],
                        ("wffn2_encode", "cfg_field"),
                    )
                    self.assertEqual(
                        audit.load_cfg_field_to_graph_node(2)["ctpt_wq_wk"],
                        ("wq_encode", "cfg_field"),
                    )
                    self.assertEqual(
                        audit.load_t_new_map()["block1_mrpc"],
                        [("wffn2_encode", None)],
                    )

        self.assertEqual(len(reads), 1)

