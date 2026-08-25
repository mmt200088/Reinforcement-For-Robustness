from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from rfr.preparation.data.protocol import (
    SUPPORTED_DATASETS,
    TRAIN_PROBE_SIZE,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ANALYSIS = ROOT / "Model_analysis"
if str(MODEL_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(MODEL_ANALYSIS))


plain = importlib.import_module("analyze_all_distribution_new")
approx = importlib.import_module("analyze_all_distribution_approx")
gelu = importlib.import_module("analyze_gelu_distribution")


EXPECTED_TASKS = (
    "mrpc",
    "rte",
    "sst2",
    "mrpc_large",
    "rte_large",
    "sst2_large",
)


def test_profile_registry_contains_only_six_supported_profiles():
    assert tuple(plain.TASK_REGISTRY) == EXPECTED_TASKS
    assert tuple(gelu.TASK_REGISTRY) == EXPECTED_TASKS
    assert approx.TASK_REGISTRY is plain.TASK_REGISTRY
    assert {
        config["dataset_config"] for config in plain.TASK_REGISTRY.values()
    } == set(SUPPORTED_DATASETS)


@pytest.mark.parametrize(
    "relative_path",
    (
        "Model_analysis/analyze_all_distribution_new.py",
        "Model_analysis/analyze_all_distribution_approx.py",
        "Model_analysis/analyze_gelu_distribution.py",
    ),
)
def test_profile_sources_have_no_unsupported_model_or_dataset_dispatch(relative_path):
    source = (ROOT / relative_path).read_text(encoding="utf-8").lower()

    for forbidden in (
        "automodelforcausallm",
        "_prepare_gpt2_data",
        "_install_gpt2_hooks",
        "wikitext",
        "gpt2_wt2",
        "gpt2m_wt2",
        "'cola'",
        "'stsb'",
        "'mnli'",
        "'qnli'",
        "'wnli'",
    ):
        assert forbidden not in source


def test_shared_profile_loader_uses_pinned_fixture_before_tokenization():
    source = Path(plain.__file__).read_text(encoding="utf-8")
    start = source.index("def _prepare_bert_data(")
    end = source.index("\ndef ", start + 5)
    method = source[start:end]

    assert "GLUE_DATASET_REVISION" in method
    assert "load_train_probe_fixture(" in method
    assert "resolve_glue_protocol_views(" in method
    assert ".train_probe" in method
    assert "GlueDataProtocolContext(" in method
    assert "context.as_payload()" in method
    assert "data['validation']" not in method
    assert "range(max_samples)" not in method
    assert "TRAIN_PROBE_SIZE" in method
    assert method.index("resolve_glue_protocol_views(") < method.index("def _tok(")


def test_all_profile_programs_persist_the_shared_protocol_payload():
    plain_source = Path(plain.__file__).read_text(encoding="utf-8")
    approx_source = Path(approx.__file__).read_text(encoding="utf-8")
    gelu_source = Path(gelu.__file__).read_text(encoding="utf-8")

    assert "write_profile_protocol(" in plain_source
    assert "write_profile_protocol(" in approx_source
    assert "write_profile_protocol(" in gelu_source
    assert "GLUE_DATASET_REVISION" in plain_source
