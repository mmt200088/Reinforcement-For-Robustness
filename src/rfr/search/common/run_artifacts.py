"""Shared run-directory layout and stage-status metadata; no search algorithm."""

from __future__ import annotations

from typing import Any, Dict, Optional
import os


NOISE_STAGE_STEP_INFO_FILE = "noise_ppo_step_info.txt"
NOISE_STAGE_TRAINING_CURVE_PATH = "noise_ppo_training_curve.png"
NOISE_STAGE_ENTROPY_CURVE_PATH = "noise_ppo_entropy_curve.png"
DEFAULT_STAGE1_SEARCH_LOG_FILE = "search.log"

SEARCH_LOG_HEADER = "=== PPO强化学习优化日志已启动（PPO RL Optimization Log Started） ==="
DEFAULT_STAGE1_STEP_INFO_FILE = "ppo_step_info.txt"
DEFAULT_STAGE1_TRAINING_CURVE_FILE = "ppo_training_curve.png"
DEFAULT_STAGE1_ENTROPY_CURVE_FILE = "ppo_entropy_curve.png"
DEFAULT_FINAL_EVAL_DIR = os.path.join("outputs", "rl", "evaluation")
DEFAULT_NOISE_PROGRESS_DIR = os.path.join("outputs", "rl", "stage2", "progress")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it doesn't exist (mkdir -p
    semantics). Idempotent. Used before atomic-writing JSON / pickle / etc."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def update_persistent_metadata_stage(
        run_output_dir: str,
        stage_key: str,
        status: str,
        extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Update stage-completion fields in a persistent run metadata file."""
    import json as _json
    import datetime as _dt
    meta_path = os.path.join(run_output_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = _json.load(f)
    except Exception:
        return
    stages = meta.setdefault("stage_status", {})
    stages[stage_key] = status
    if extra_fields:
        stage_detail = meta.setdefault("stage_detail", {})
        stage_detail.setdefault(stage_key, {}).update(extra_fields)
    meta["last_updated_at"] = _dt.datetime.now().isoformat()
    tmp_path = meta_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, meta_path)
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def resolve_run_output_layout(run_output_dir, flattened=False):
    s1_log = s2_log = DEFAULT_STAGE1_SEARCH_LOG_FILE
    run_output_dir = str(run_output_dir or "").strip()
    if not run_output_dir:
        return {
            "run_output_dir": "",
            "log_file": s1_log,
            "noise_log_file": s2_log,
            "stage1_step_info_file": DEFAULT_STAGE1_STEP_INFO_FILE,
            "stage1_training_curve_path": DEFAULT_STAGE1_TRAINING_CURVE_FILE,
            "stage1_entropy_curve_path": DEFAULT_STAGE1_ENTROPY_CURVE_FILE,
            "final_eval_dir": DEFAULT_FINAL_EVAL_DIR,
            "noise_step_info_file": NOISE_STAGE_STEP_INFO_FILE,
            "noise_training_curve_path": NOISE_STAGE_TRAINING_CURVE_PATH,
            "noise_entropy_curve_path": NOISE_STAGE_ENTROPY_CURVE_PATH,
            "noise_progress_dir": DEFAULT_NOISE_PROGRESS_DIR,
        }

    run_output_dir = os.path.normpath(run_output_dir)
    if flattened:


        stage1_dir = run_output_dir
        stage2_noise_dir = run_output_dir
    else:
        stage1_dir = os.path.join(run_output_dir, "stage1")
        stage2_noise_dir = os.path.join(run_output_dir, "stage2")
    stage2_noise_progress_dir = os.path.join(stage2_noise_dir, "progress")
    final_eval_dir = os.path.join(run_output_dir, "evaluation")

    layout = {
        "run_output_dir": run_output_dir,
        "log_file": os.path.join(stage1_dir, s1_log),
        "noise_log_file": os.path.join(stage2_noise_dir, s2_log),
        "stage1_step_info_file": os.path.join(stage1_dir, DEFAULT_STAGE1_STEP_INFO_FILE),
        "stage1_training_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_TRAINING_CURVE_FILE
        ),
        "stage1_entropy_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_ENTROPY_CURVE_FILE
        ),
        "final_eval_dir": final_eval_dir,
        "noise_step_info_file": os.path.join(stage2_noise_dir, NOISE_STAGE_STEP_INFO_FILE),
        "noise_training_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_TRAINING_CURVE_PATH
        ),
        "noise_entropy_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_ENTROPY_CURVE_PATH
        ),
        "noise_progress_dir": stage2_noise_progress_dir,
    }

    return layout
