from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import time

import pytest

from glue_data_protocol import supported_profiles
from Paean import config as paean_config
from Paean import run_final_eval
from config import run_layout
import generate_glue_submission as submission
import rl_tune_general


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("mrpc", "rte", "sst2")
SUPPORTED_MODELS = ("bert-base", "bert-large")


def test_public_python_registries_match_shared_six_profile_matrix():
    assert paean_config.DATASET_CHOICES == SUPPORTED_DATASETS
    assert paean_config.MODEL_TYPE_CHOICES == SUPPORTED_MODELS
    assert tuple(submission.TASK_CONFIGS) == SUPPORTED_DATASETS
    assert tuple(submission.BERT_BASE_MODEL_NAMES) == SUPPORTED_DATASETS
    assert tuple(submission.BERT_LARGE_MODEL_NAMES) == SUPPORTED_DATASETS
    assert tuple(rl_tune_general.BASE_MODEL_BY_TYPE) == SUPPORTED_MODELS


@pytest.mark.parametrize("model_type,dataset", supported_profiles())
def test_all_six_profiles_resolve_consistently(model_type, dataset):
    expected = rl_tune_general.resolve_base_model(model_type, dataset)

    assert run_final_eval._base_model(model_type, dataset) == expected
    assert run_layout.combo_name(model_type, dataset)


@pytest.mark.parametrize(
    "model_type,dataset",
    (
        ("gpt-2", "mrpc"),
        ("bert-base", "cola"),
        ("bert-large", "qnli"),
    ),
)
def test_public_python_entrypoints_reject_unsupported_profiles(
    model_type, dataset
):
    with pytest.raises(ValueError):
        rl_tune_general.resolve_base_model(model_type, dataset)
    with pytest.raises(ValueError):
        run_final_eval._base_model(model_type, dataset)
    with pytest.raises(ValueError):
        run_layout.combo_name(model_type, dataset)


def test_general_rl_accepts_only_supported_tasks_with_one_model_family():
    assert rl_tune_general.validate_general_tasks(
        "bert-base", ["mrpc", "rte", "sst2"]
    ) == ("mrpc", "rte", "sst2")
    with pytest.raises(ValueError, match="unsupported profile"):
        rl_tune_general.validate_general_tasks(
            "bert-base", ["mrpc", "stsb"]
        )


@pytest.mark.parametrize(
    "args",
    (
        ("--dataset", "cola"),
        ("--model-type", "gpt-2"),
    ),
)
def test_launcher_rejects_unsupported_profile_before_python(args, tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    marker = tmp_path / "python-invoked"
    fake_python = fakebin / "python"
    fake_python.write_text(
        f"#!/usr/bin/env bash\ntouch {str(marker)!r}\nexit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_flock = fakebin / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        [
            "bash",
            "llama_7B_LayerImportance.sh",
            "run",
            "rl",
            *args,
            "--mode",
            "stage1-only",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    time.sleep(0.2)

    assert result.returncode != 0
    assert not marker.exists()
    assert "不支持" in result.stderr or "unsupported" in result.stderr.lower()


def test_active_entrypoint_sources_have_no_unsupported_dispatch_literals():
    paths = (
        "llama_7B_LayerImportance.sh",
        "Paean/config.py",
        "Paean/run_final_eval.py",
        "general_policy_module.py",
        "rl_tune_general.py",
        "generate_glue_submission.py",
    )
    forbidden = (
        "gpt-2",
        "gpt2",
        "cola",
        "stsb",
        "mnli",
        "qnli",
        "wnli",
        "qqp",
        "wikitext",
    )
    for relative_path in paths:
        source = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        for value in forbidden:
            assert value not in source, f"{relative_path}: {value}"


def test_general_and_submission_modules_remain_syntax_valid():
    for relative_path in (
        "Paean/config.py",
        "Paean/run_final_eval.py",
        "general_policy_module.py",
        "rl_tune_general.py",
        "generate_glue_submission.py",
    ):
        ast.parse(
            (ROOT / relative_path).read_text(encoding="utf-8").lstrip("\ufeff")
        )
