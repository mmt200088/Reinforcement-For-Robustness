from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import blb_orphan_slot_audit as audit


class OrphanSlotAuditTest(unittest.TestCase):
    def setUp(self):
        audit._AST_CACHE.clear()
        audit._GRAPH_CONFIG_NAMES_CACHE.clear()

    def tearDown(self):
        audit._AST_CACHE.clear()
        audit._GRAPH_CONFIG_NAMES_CACHE.clear()

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

    def test_graphs_for_block_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            names = [
                "block1_mrpc.json",
                "block2_mrpc.json",
                "block3_exp_n2.json",
                "block3_exp_n6.json",
                "block4.json",
                "block5_n1.json",
                "block5_n4.json",
                "_summary.json",
                "map_summary.json",
                "notes.txt",
            ]
            for name in names:
                (root / name).write_text("{}", encoding="utf-8")

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("graph discovery should not use Path.glob"),
            ):
                found = {
                    block_idx: [path.name for path in audit.graphs_for_block(block_idx, "mrpc", root)]
                    for block_idx in (1, 2, 3, 4, 5)
                }

        self.assertEqual(found[1], ["block1_mrpc.json"])
        self.assertEqual(found[2], ["block2_mrpc.json"])
        self.assertEqual(found[3], ["block3_exp_n2.json", "block3_exp_n6.json"])
        self.assertEqual(found[4], ["block4.json"])
        self.assertEqual(found[5], ["block5_n1.json", "block5_n4.json"])

    def test_graphs_for_block_reuses_directory_scan_across_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in (
                "block1_mrpc.json",
                "block2_mrpc.json",
                "block3_exp_n2.json",
                "block4.json",
                "block5_n1.json",
            ):
                (root / name).write_text("{}", encoding="utf-8")

            original_scandir = audit.os.scandir
            calls = []

            def counting_scandir(path):
                if Path(path) == root:
                    calls.append(Path(path))
                return original_scandir(path)

            with mock.patch.object(audit.os, "scandir", counting_scandir):
                found = {
                    block_idx: [path.name for path in audit.graphs_for_block(block_idx, "mrpc", root)]
                    for block_idx in (1, 2, 3, 4, 5)
                }

        self.assertEqual(found[1], ["block1_mrpc.json"])
        self.assertEqual(found[2], ["block2_mrpc.json"])
        self.assertEqual(found[3], ["block3_exp_n2.json"])
        self.assertEqual(found[4], ["block4.json"])
        self.assertEqual(found[5], ["block5_n1.json"])
        self.assertEqual(len(calls), 1)

    def test_main_streams_markdown_report_without_path_write_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "function_handler.py").write_text("", encoding="utf-8")
            (root / "rescale_optimizer_bridge.py").write_text(
                "DEFAULT_CFG_TO_T_NEW_MAP = {}\n",
                encoding="utf-8",
            )
            rescale_root = root / "Rescale_optimizer"
            configs_dir = rescale_root / "configs" / "mrpc"
            configs_dir.mkdir(parents=True)
            out_dir = root / "out"
            original_write_text = Path.write_text

            def reject_markdown_write_text(path, *args, **kwargs):
                if path.name == "audit_mrpc.md":
                    raise AssertionError("markdown report should stream through Path.open")
                return original_write_text(path, *args, **kwargs)

            with mock.patch.object(audit, "REPO_ROOT", root):
                with mock.patch.object(Path, "write_text", reject_markdown_write_text):
                    rc = audit.main([
                        "--profile",
                        "mrpc",
                        "--rescale-optimizer-root",
                        str(rescale_root),
                        "--out",
                        str(out_dir),
                    ])

            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "audit_mrpc.md").exists())
            self.assertTrue((out_dir / "audit_mrpc.json").exists())
