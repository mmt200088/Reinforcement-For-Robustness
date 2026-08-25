from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from rfr.preparation.data.protocol import supported_profiles
from Paean import config as paean_config
from Paean import run_final_eval
from rfr.common.config import run_layout


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("mrpc", "rte", "sst2")
SUPPORTED_MODELS = ("bert-base", "bert-large")


def test_public_python_registries_match_shared_six_profile_matrix():
    assert paean_config.DATASET_CHOICES == SUPPORTED_DATASETS
    assert paean_config.MODEL_TYPE_CHOICES == SUPPORTED_MODELS
    assert tuple(run_final_eval.BASE_MODEL_BY_TYPE) == SUPPORTED_MODELS


@pytest.mark.parametrize("model_type,dataset", supported_profiles())
def test_all_six_profiles_resolve_consistently(model_type, dataset):
    assert run_final_eval._base_model(model_type, dataset)
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
        run_final_eval._base_model(model_type, dataset)
    with pytest.raises(ValueError):
        run_layout.combo_name(model_type, dataset)


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
        "src/rfr/preparation/data/protocol.py",
        "rl_tune.py",
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


def test_production_entrypoints_remain_syntax_valid():
    for relative_path in (
        "Paean/config.py",
        "Paean/run_final_eval.py",
        "src/rfr/preparation/data/protocol.py",
        "rl_tune.py",
    ):
        ast.parse(
            (ROOT / relative_path).read_text(encoding="utf-8").lstrip("\ufeff")
        )


def test_active_config_inventory_contains_only_supported_profiles():
    glue_config = json.loads((ROOT / "glue_configs.json").read_text())
    assert tuple(key for key in glue_config if key != "_comment") == (
        "mrpc", "rte", "sst2"
    )

    approx_config = json.loads(
        (ROOT / "Model_analysis/configs/approx_per_dataset.json").read_text()
    )
    assert tuple(key for key in approx_config if key != "_comment") == (
        "mrpc", "rte", "sst2", "mrpc_large", "rte_large", "sst2_large"
    )

    rescale_profiles = tuple(sorted(
        path.name
        for path in (ROOT / "configs/preparation/rescale").iterdir()
        if path.is_dir()
    ))
    assert rescale_profiles == (
        "mrpc", "mrpc_large", "rte", "rte_large", "sst2", "sst2_large"
    )
