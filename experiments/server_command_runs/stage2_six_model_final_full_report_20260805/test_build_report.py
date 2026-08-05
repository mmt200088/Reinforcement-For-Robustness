#!/usr/bin/env python3
"""Focused regression tests for the six-model report builder."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_report.py")
SPEC = importlib.util.spec_from_file_location("six_model_report_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ArchiveStreamPathTest(unittest.TestCase):
    def test_legacy_archive_names_resolve_inside_streams_directory(self) -> None:
        commit = "66c3dbf29ec80eaa1cdcefec699827406e83f55a"
        root = "server_backups/20260731_bert_large_mrpc_checkpoint24720_final"

        episodes, ppo = BUILDER.archive_stream_paths(commit, root)

        self.assertEqual(
            episodes,
            f"{root}/streams/structured_episodes.jsonl.gz",
        )
        self.assertEqual(
            ppo,
            f"{root}/streams/structured_ppo_updates.jsonl.gz",
        )


if __name__ == "__main__":
    unittest.main()
