"""BLB Stage 2 RL 训练侧的持久化辅助：状态板 / 训练曲线 / 报告 / 崩溃记录。

旧版 stage 2 RL（``noise_rl_module_v2``）在 ``rl_results/persistent/...`` 下
做了若干"训练之外"的小事：进度条、curve PNG、checkpoint 元数据、错误归档。
BLB Stage 2 RL 是最终版本，需要把这些项目操作类的输出补齐到新的持久化目录
``Parting Chapter/<run_basename>/stage2_noise/progress/``。

本模块提供四件事：

  1. ``BLBStatusBoard``      ── 训练期间持续刷新 ``blb_stage2_status.json`` 和
                                 ``blb_stage2_live_summary.md``，同时累积
                                 episode_returns / 训练曲线数据。
  2. ``write_training_curves`` ── 训练结束（或周期性）把曲线写成 PNG。matplotlib
                                  缺失时安全降级为只写 numpy/CSV。
  3. ``write_blb_final_report`` ── 训练结束时写一份中文 markdown 报告，
                                   汇总最优动作、reward 拆解、baseline 对比。
  4. ``dump_crash_report``       ── 异常崩溃时把 traceback + 最后状态写到
                                    ``blb_stage2_error.txt``。

设计要求：
  * 不引入硬依赖（matplotlib / pandas 缺失时不报错）。
  * 全部写盘动作 try/except 包住，失败只写日志，不打断训练。
  * 状态板 JSON 用原子写（先写到 ``*.tmp``，再 ``os.replace``），
    避免 live tail 读到一半被覆盖时拿到半截 JSON。
"""
from __future__ import annotations

import datetime as _dt
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO

from json_utils import to_jsonable as _to_jsonable
from training_curve_plot import save_stage1_style_training_curve


BLB_STATUS_FILENAME = "blb_stage2_status.json"
BLB_TRAINING_CURVE_PNG = "blb_stage2_training_curve.png"
BLB_TRAINING_CURVE_NPZ = "blb_stage2_training_curve.npz"
BLB_REWARD_PAPER_PNG = "blb_stage2_reward_paper.png"
BLB_REWARD_PAPER_PDF = "blb_stage2_reward_paper.pdf"
BLB_ENTROPY_CURVE_PNG = "blb_stage2_entropy_curve.png"
BLB_DIAGNOSTIC_CURVE_PNG = "blb_stage2_diagnostics_curve.png"
BLB_FINAL_REPORT_MD = "blb_stage2_report.md"
BLB_LIVE_SUMMARY_MD = "blb_stage2_live_summary.md"
BLB_SEARCH_LOG_TXT = "blb_stage2_search_log.txt"
BLB_ERROR_TXT = "blb_stage2_error.txt"
BLB_EPISODE_TRACE_CSV = "blb_stage2_episode_trace.csv"
_PLOT_RENDER_FALSE_VALUES = {"0", "false", "no", "off", "skip", "none"}
_TRACE_SCHEMA_CURRENT_PATHS: set[str] = set()

BLB_TRACE_FIELDNAMES = (
    "episode",
    "total_episodes",
    "ppo_update_count",
    "rollout_reward_mean",
    "rollout_reward_max",
    "rollout_reward_min",
    "rollout_metric1_mean",
    "rollout_metric2_mean",
    "rollout_metric1_min",
    "rollout_metric2_min",
    "rollout_loss_mean",
    "rollout_loss_std_mean",
    "rollout_loss_max",
    "best_reward",
    "priority1_count",
    "priority2_count",
    "priority3_count",
    "invalid_count",
    "apply_error_count",
    "eval_error_count",
    "last_error",
    "action_source",
    "anchor_count",
    "cost_probe_count",
    "action_source_anchor_count",
    "action_source_cost_probe_count",
    "action_source_neighbor_count",
    "action_source_policy_count",
    "action_mask_mode",
    "action_mask_hash",
    "action_bias_bonus",
    "mutated_slot_count",
    "mutated_effective_slot_count",
    "mutated_ineffective_slot_count",
    "mutated_slot_count_mean",
    "mutated_slot_count_max",
    "mutated_effective_slot_count_mean",
    "mutated_ineffective_slot_count_mean",
    "raw_entropy_by_kind",
    "masked_entropy_by_kind",
    "mutated_F_count",
    "mutated_W_count",
    "mutated_M_count",
    "mutated_S_count",
    "mutated_R_count",
    "mutated_K_count",
    "mutated_by_block",
    "policy_loss",
    "value_loss",
    "entropy",
    "clip_fraction",
    "n_samples",
)


def _atomic_json_dump(path: str, obj: Any) -> None:
    """原子写：``path.tmp`` → ``os.replace(path)``。失败抛 IOError。"""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".blb_status_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_text_dump(path: str, text: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".blb_live_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(text))
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _write_joined_lines_stream(fh: TextIO, lines: Iterable[str]) -> None:
    first = True
    for line in lines:
        if first:
            first = False
        else:
            fh.write("\n")
        fh.write(str(line))


def _stage2_plot_rendering_enabled(render_plots: Optional[bool]) -> bool:
    if render_plots is not None:
        return bool(render_plots)
    raw = os.environ.get("RFR_STAGE2_RENDER_PLOTS")
    if raw is None:
        return True
    return raw.strip().lower() not in _PLOT_RENDER_FALSE_VALUES


def _fmt_live_number(value: Any, *, signed: bool = False) -> str:
    try:
        val = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:+.6f}" if signed else f"{val:.6f}"


def _render_live_summary_markdown(state: Mapping[str, Any]) -> str:
    completed = int(state.get("completed_episodes", 0) or 0)
    total = int(state.get("total_episodes", 0) or 0)
    pct = (100.0 * completed / total) if total > 0 else 0.0
    recent = [float(x) for x in (state.get("recent_returns") or [])]
    recent_mean = sum(recent) / len(recent) if recent else None
    best = state.get("best") if isinstance(state.get("best"), Mapping) else {}
    last = state.get("last_breakdown") if isinstance(state.get("last_breakdown"), Mapping) else {}
    ppo = state.get("ppo_last_metrics") if isinstance(state.get("ppo_last_metrics"), Mapping) else {}

    lines = [
        "# BLB Stage-2 RL Live Summary",
        "",
        f"- Run: `{state.get('run_basename') or ''}`",
        f"- Profile: `{state.get('profile') or ''}`",
        f"- Phase: {state.get('phase') or 'n/a'}",
        f"- Updated at: {state.get('updated_at') or state.get('last_update') or 'n/a'}",
        f"- Elapsed seconds: {_fmt_live_number(state.get('elapsed_sec'))}",
        f"- Episode: {completed} / {total} ({pct:.2f}%)",
        f"- PPO updates: {int(state.get('ppo_update_count', 0) or 0)}",
        "",
        "## Reward",
        "",
        f"- Last reward: {_fmt_live_number(state.get('last_reward'), signed=True)}",
        f"- Recent reward mean: {_fmt_live_number(recent_mean, signed=True)}",
        f"- Best reward: {_fmt_live_number(best.get('reward'), signed=True)}",
        f"- Best episode: {best.get('episode') if best.get('episode') is not None else 'n/a'}",
        f"- Last priority: {state.get('last_priority') if state.get('last_priority') is not None else 'n/a'}",
        f"- Last invalid: {state.get('last_invalid') if state.get('last_invalid') is not None else 'n/a'}",
        "",
        "## Last Terminal Metrics",
        "",
    ]
    for key in (
        "terminal_loss_mean",
        "terminal_metric1_mean",
        "terminal_metric2_mean",
        "terminal_stab_violation",
        "fusion_count",
        "terminal_k_gain",
        "terminal_fusion_gain",
    ):
        if key in last:
            lines.append(f"- `{key}`: {last.get(key)}")

    lines.extend(["", "## Last PPO Update", ""])
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "clip_fraction",
        "window_mean_return",
        "window_mean_invalid",
        "approx_kl",
        "lr",
        "ent_coef",
    ):
        if key in ppo:
            lines.append(f"- `{key}`: {ppo.get(key)}")

    lines.extend([
        "",
        "## Key Artifacts",
        "",
        f"- Status JSON: `{BLB_STATUS_FILENAME}`",
        f"- Live summary: `{BLB_LIVE_SUMMARY_MD}`",
        "- Episodes: `diagnostics/episodes.jsonl`",
        "- PPO updates: `diagnostics/ppo_updates.jsonl`",
        "- Diagnostics summary: `diagnostics/diagnostics_summary.md`",
        "- Details batches: `details/`",
        f"- Training curve: `{BLB_TRAINING_CURVE_PNG}`",
        f"- Entropy curve: `{BLB_ENTROPY_CURVE_PNG}`",
        f"- Final report: `{BLB_FINAL_REPORT_MD}`",
        "",
    ])
    return "\n".join(lines)


def _migrate_trace_schema_if_needed(path: str, *, log_fn=None) -> None:
    """Keep live trace CSV readable when new rollout columns are added."""
    cache_key = os.path.abspath(path)
    if cache_key in _TRACE_SCHEMA_CURRENT_PATHS:
        return
    if (not os.path.isfile(path)) or os.path.getsize(path) == 0:
        return

    current_fields = list(BLB_TRACE_FIELDNAMES)
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                old_fields = next(reader)
            except StopIteration:
                return
            if old_fields == current_fields:
                _TRACE_SCHEMA_CURRENT_PATHS.add(cache_key)
                return

            old_index = {field: idx for idx, field in enumerate(old_fields)}
            anchor_idx = old_index.get("anchor_count")
            parent = os.path.dirname(path) or "."
            timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{path}.bak_schema_{timestamp}"
            fd, tmp_path = tempfile.mkstemp(prefix=".blb_trace_", suffix=".tmp", dir=parent)
            tmp_open = True
            try:
                shutil.copyfile(path, backup_path)
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as out_f:
                    tmp_open = False
                    writer = csv.DictWriter(out_f, fieldnames=current_fields)
                    writer.writeheader()
                    for raw in reader:
                        shifted_cost_probe_row = (
                            "cost_probe_count" not in old_index
                            and anchor_idx is not None
                            and len(raw) == len(old_fields) + 1
                        )
                        migrated: Dict[str, Any] = {}
                        for field in current_fields:
                            if shifted_cost_probe_row and field == "cost_probe_count":
                                src_idx = int(anchor_idx) + 1
                            elif (
                                shifted_cost_probe_row
                                and field in old_index
                                and old_index[field] > int(anchor_idx)
                            ):
                                src_idx = old_index[field] + 1
                            else:
                                src_idx = old_index.get(field)
                            migrated[field] = (
                                raw[src_idx] if src_idx is not None and src_idx < len(raw) else ""
                            )
                        if "cost_probe_count" not in old_index and not migrated.get("cost_probe_count"):
                            migrated["cost_probe_count"] = "0"
                        writer.writerow(migrated)
                os.replace(tmp_path, path)
                _TRACE_SCHEMA_CURRENT_PATHS.add(cache_key)
                if log_fn is not None:
                    log_fn(f"  [BLB trace] migrated CSV schema -> {path} (backup: {backup_path})")
            except Exception as exc:
                if tmp_open:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                if log_fn is not None:
                    log_fn(f"  [BLB trace][warning] failed to migrate {path}: {exc}")
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"  [BLB trace][warning] failed to inspect {path}: {exc}")


def append_blb_episode_trace_row(
        persistence_dir: str,
        row: Mapping[str, Any],
        *,
        log_fn=None,
        ) -> str:
    """Append one PPO-rollout diagnostic row to a stable CSV trace."""
    log = log_fn or (lambda _msg: None)
    os.makedirs(persistence_dir, exist_ok=True)
    path = os.path.join(persistence_dir, BLB_EPISODE_TRACE_CSV)
    cache_key = os.path.abspath(path)
    if cache_key not in _TRACE_SCHEMA_CURRENT_PATHS:
        _migrate_trace_schema_if_needed(path, log_fn=log)
    write_header = (not os.path.isfile(path)) or os.path.getsize(path) == 0
    safe_row = {
        key: _to_jsonable(row.get(key, ""))
        for key in BLB_TRACE_FIELDNAMES
    }
    try:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(BLB_TRACE_FIELDNAMES))
            if write_header:
                writer.writeheader()
            writer.writerow(safe_row)
        _TRACE_SCHEMA_CURRENT_PATHS.add(cache_key)
    except Exception as exc:
        try:
            log(f"  [BLB trace][warning] failed to write {path}: {exc}")
        except Exception:
            pass
    return path


def _records_to_slots_view(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Turn raw ``describe_action_vector`` records into the human SF/K view used by
    ``best_action_vec.json``. Mirrors ``action_io.action_vec_to_slots_list`` but
    works straight off the records dict so we don't have a runtime dep here."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        kind = str(rec.get("kind", ""))
        entry: Dict[str, Any] = {
            "label": str(rec.get("slot_label", "")),
            "global_index": int(rec.get("global_index", -1)),
            "layer": int(rec.get("layer", 0)),
            "block": rec.get("block_index"),
            "kind": kind,
            "field_name": str(rec.get("field", "")),
            "operation": str(rec.get("operation", "")),
            "location": str(rec.get("location", "")),
            "distribution": str(rec.get("distribution", "")),
            "action_index": int(rec.get("action_index", 0)),
            "level_values": list(rec.get("level_values") or []),
            "N": rec.get("N"),
            "max_sf": rec.get("max_sf"),
            "effective": bool(rec.get("effective", True)),
        }
        if kind == "K":
            entry["truncation_bits"] = rec.get("value")
        else:
            entry["scaling_factor"] = rec.get("value")
        if rec.get("note"):
            entry["note"] = str(rec.get("note"))
        out.append(entry)
    return out


def _slots_by_layer_grouped(slots: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Group flat slots into ``{layer_key: {block_key: {short_or_K: value}}}`` for
    compact display in the markdown and a friendlier JSON section."""
    layers: Dict[str, Dict[str, Any]] = {}
    first_input_value = None
    for row in slots:
        layer_idx = int(row.get("layer", 0))
        block_idx = row.get("block")
        kind = str(row.get("kind", ""))
        if block_idx is None:
            first_input_value = row.get("scaling_factor")
            continue
        label = str(row.get("label") or "")
        # short field: everything after L<n>.B<n>.<kind>.
        parts = label.split(".", 3)
        short = parts[-1] if len(parts) >= 4 else ""
        layer_key = f"L{layer_idx:02d}"
        block_key = f"B{int(block_idx)}"
        per_layer = layers.setdefault(layer_key, {})
        per_block = per_layer.setdefault(block_key, {})
        if kind == "K":
            per_block["K"] = row.get("truncation_bits")
        else:
            display_key = f"{kind}.{short}" if short else kind
            per_block[display_key] = row.get("scaling_factor")
    out: Dict[str, Any] = dict(sorted(layers.items()))
    if first_input_value is not None:
        out["first_input"] = first_input_value
    return out


def write_action_description_files(
        persistence_dir: str,
        description: Mapping[str, Any],
        *,
        label: str = "best",
        log_fn=None,
        ) -> Dict[str, str]:
    """Write a full readable BLB action description as JSON and Markdown.

    The JSON is structured for both humans and Paean:

      * top-level ``slots`` — flat list, each slot carries ``label``,
        ``scaling_factor`` / ``truncation_bits`` (the value humans care about),
        ``location``, ``operation``, ``kind``, etc. **Same schema as
        ``best_action_vec.json``** so Paean's ``--action-config`` accepts this
        file as-is.
      * top-level ``slots_by_layer`` — grouped compact view ``{"L05": {"B3":
        {"K": 10, "F.softmax_exp": 14, ...}, ...}, ...}`` for quick reading.
      * ``records`` — the legacy verbose per-slot records (kept for back-compat).
      * ``summary`` — counts (scaling-factor slots / truncation slots / ineffective).

    The Markdown table leads with the *decoded value* (SF / truncation_bits)
    so a reader can answer "what SF did L05.B3.softmax_exp get?" without
    looking at action indices.
    """
    log = log_fn or (lambda _msg: None)
    safe_label = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(label or "best")
    )
    os.makedirs(persistence_dir, exist_ok=True)
    json_path = os.path.join(persistence_dir, f"blb_stage2_{safe_label}_action_full.json")
    md_path = os.path.join(persistence_dir, f"blb_stage2_{safe_label}_action_full.md")
    out = {"json": json_path, "md": md_path}

    records = list((description or {}).get("records") or [])
    summary = dict((description or {}).get("summary") or {})
    slots_view = _records_to_slots_view(records)
    grouped = _slots_by_layer_grouped(slots_view)

    enriched: Dict[str, Any] = {
        "schema_version": "blb_v3_slots_human_v1",
        "label": safe_label,
        "profile": description.get("profile", ""),
        "num_layers": description.get("num_layers"),
        "action_length": description.get("action_length"),
        "slots": slots_view,                # human-readable primary view
        "slots_by_layer": grouped,          # compact grouped view
        "records": records,                 # legacy verbose records
        "summary": summary,
    }
    # Pass through any other top-level fields the caller set (e.g. action_vec).
    for k, v in (description or {}).items():
        if k not in enriched:
            enriched[k] = v

    try:
        _atomic_json_dump(json_path, _to_jsonable(enriched))
    except Exception as exc:
        log(f"  [BLB action][warning] failed to write {json_path}: {exc}")
        out["json"] = ""

    try:
        lines: List[str] = [
            f"# BLB Stage 2 action description: `{safe_label}`",
            "",
            f"- profile: `{description.get('profile', '')}`",
            f"- num_layers: `{description.get('num_layers', '')}`",
            f"- action_length: `{description.get('action_length', '')}`",
        ]
        if summary:
            lines.extend([
                f"- records: `{summary.get('record_count', len(records))}`",
                f"- scaling factor slots: `{summary.get('scaling_factor_count', '')}`",
                f"- truncation slots: `{summary.get('truncation_count', '')}`",
                f"- ineffective decoded slots: `{summary.get('ineffective_slot_count', '')}`",
            ])
        lines.extend([
            "",
            "**Slot label format**: `L{layer}.B{block}.{kind}[.{short_field}]` ",
            "(kind: F=fresh, W=weight encode, M=mask, S=scalar, R=rescale, K=trunc).",
            "",
            "Each row's primary value is the decoded **`scaling_factor`** "
            "(for F/W/M/S/R kinds) or **`truncation_bits`** (for K). "
            "`action_idx` is the policy-side index that produced that value "
            "and is included only for sanity-checking — humans should read "
            "the SF / truncation_bits columns.",
            "",
        ])

        # 1) Compact grouped view (the eye-friendly summary).
        lines.append("## 1. Per-layer / per-block 选择概览")
        lines.append("")
        if not grouped:
            lines.append("_(empty)_")
        else:
            lines.append("| 层 | block | 槽位 → 选择值 |")
            lines.append("|---|---|---|")
            for layer_key, layer_val in grouped.items():
                if layer_key == "first_input":
                    lines.append(f"| (legacy) first_input | – | `scaling_factor={layer_val}` |")
                    continue
                if not isinstance(layer_val, Mapping):
                    continue
                for block_key in sorted(layer_val.keys()):
                    block_val = layer_val[block_key]
                    if not isinstance(block_val, Mapping):
                        continue
                    cell_parts: List[str] = []
                    for slot_short, value in block_val.items():
                        if slot_short == "K":
                            cell_parts.append(f"K=**{value}**")
                        else:
                            disp = "off" if value is None else value
                            cell_parts.append(f"{slot_short}={disp}")
                    cell = ", ".join(cell_parts)
                    lines.append(f"| {layer_key} | {block_key} | {cell} |")
        lines.append("")

        # 2) Full per-slot detail table — lead with SF / K, keep action_idx as a side column.
        lines.append("## 2. 全槽位明细（按 global_index）")
        lines.append("")
        lines.append("| idx | slot | location | operation | dist | **value** | kind | action_idx | effective | N | max_sf | level_values | note |")
        lines.append("|---:|---|---|---|:---:|---:|:---:|---:|:---:|---:|---:|---|---|")
        for rec in records:
            note = str(rec.get("note", "")).replace("|", "\\|")
            location = str(rec.get("location", "")).replace("|", "\\|")
            operation = str(rec.get("operation", "")).replace("|", "\\|")
            slot_label = str(rec.get("slot_label", "")).replace("|", "\\|")
            max_sf = "" if rec.get("max_sf") is None else str(rec.get("max_sf"))
            value = rec.get("effective_value")
            if value is None and rec.get("effective") is False:
                value_display = "_off_"
            elif value is None:
                value_display = ""
            else:
                value_display = f"**{value}**"
            kind = str(rec.get("kind", ""))
            level_vals = rec.get("level_values") or []
            level_str = ",".join(str(v) for v in level_vals) if level_vals else ""
            lines.append(
                f"| {int(rec.get('global_index', -1))} | `{slot_label}` | `{location}` | `{operation}` | "
                f"`{rec.get('distribution', rec.get('kind', ''))}` | {value_display} | "
                f"`{kind}` | {int(rec.get('action_index', -1))} | {bool(rec.get('effective', True))} | "
                f"{rec.get('N', '')} | {max_sf} | `{level_str}` | {note} |"
            )
        lines.append("")
        lines.append(
            f"> JSON 配对文件：`{os.path.basename(json_path)}`。"
            "可以直接喂给 `Paean/run_final_eval.sh --action-config`。"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            _write_joined_lines_stream(f, lines)
    except Exception as exc:
        log(f"  [BLB action][warning] failed to write {md_path}: {exc}")
        out["md"] = ""
    return out


# ---------------------------------------------------------------------------
# 状态板：训练中持续刷新，让用户能 live tail
# ---------------------------------------------------------------------------
class BLBStatusBoard:
    """训练状态板。

    写出文件：
      * ``<persistence_dir>/blb_stage2_status.json``
        训练期间持续覆盖；包含训练阶段、episode/PPO 进度、最近 N 步统计、
        最优 reward / breakdown、baseline 信息。

    用法：
        board = BLBStatusBoard(persistence_dir, total_episodes=2000, profile="mrpc")
        board.set_phase("校准 baseline")
        board.set_baseline(baseline_dict)
        board.update_after_episode(ep, reward, breakdown_dict)
        board.update_after_ppo_update(update_count, ppo_metrics)
        board.set_best(best_reward, best_action_vec, best_breakdown_dict)
        board.set_phase("训练结束")
        board.flush()                # 显式 flush（最终落盘）
    """

    def __init__(
            self,
            persistence_dir: str,
            *,
            total_episodes: int,
            profile: str,
            run_basename: str = "",
            extra_meta: Optional[Mapping[str, Any]] = None,
            log_fn=None,
            ):
        self._dir = str(persistence_dir)
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, BLB_STATUS_FILENAME)
        self._log = log_fn or (lambda _msg: None)

        self._t0 = time.time()
        self._state: Dict[str, Any] = {
            "schema": "blb_stage2_status_v1",
            "started_at": _dt.datetime.now().isoformat(),
            "run_basename": str(run_basename),
            "profile": str(profile),
            "total_episodes": int(total_episodes),
            "completed_episodes": 0,
            "ppo_update_count": 0,
            "phase": "初始化",
            "elapsed_sec": 0.0,
            "last_update": _dt.datetime.now().isoformat(),
            "recent_returns": [],            # 最近 N 步 reward
            "best": {
                "reward": None,
                "episode": None,
                "action_vec": None,
                "breakdown": None,
            },
            "baseline": None,
            "ppo_last_metrics": None,
            "episode": 0,
            "best_reward": None,
            "best_episode": None,
            "last_reward": None,
            "last_priority": None,
            "last_invalid": None,
            "last_breakdown": None,
            "updated_at": None,
            "extra": dict(extra_meta or {}),
        }

    # ------------------------------------------------------------------
    # 设置 / 更新
    # ------------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        self._state["phase"] = str(phase)
        self.flush()

    def set_baseline(self, baseline: Any) -> None:
        self._state["baseline"] = _to_jsonable(baseline)
        self.flush()

    def update_after_episode(
            self,
            episode_idx_1based: int,
            reward: float,
            breakdown: Optional[Any] = None,
            *,
            keep_recent: int = 100,
            ) -> None:
        self._state["completed_episodes"] = int(episode_idx_1based)
        recent = list(self._state.get("recent_returns") or [])
        recent.append(float(reward))
        if len(recent) > int(keep_recent):
            recent = recent[-int(keep_recent):]
        self._state["recent_returns"] = recent
        last_breakdown = _to_jsonable(breakdown)
        self._state["last_reward"] = float(reward)
        self._state["last_breakdown"] = last_breakdown
        if isinstance(last_breakdown, Mapping):
            self._state["last_priority"] = last_breakdown.get("priority")
            self._state["last_invalid"] = bool(last_breakdown.get("invalid", False))
        # 不每步都 flush（频繁 IO 浪费），只在 PPO update 时 flush；这里只在内存更新

    def update_after_ppo_update(
            self,
            update_count: int,
            metrics: Mapping[str, Any],
            ) -> None:
        self._state["ppo_update_count"] = int(update_count)
        self._state["ppo_last_metrics"] = _to_jsonable(metrics)
        self.flush()

    def set_best(
            self,
            best_reward: float,
            best_action_vec: Optional[Sequence[int]] = None,
            best_breakdown: Optional[Any] = None,
            best_episode: Optional[int] = None,
            best_slots: Optional[Sequence[Mapping[str, Any]]] = None,
            best_slots_by_layer: Optional[Mapping[str, Any]] = None,
            ) -> None:
        """Update the ``best`` block of the status board.

        Args:
            best_slots: optional human-readable slot list (from
                ``action_io.action_vec_to_slots_list``). When given, surfaces
                in the status JSON's ``best.slots`` field so a ``tail -f`` /
                ``jq`` flow can inspect the current best without decoding the
                integer action vector. The raw ``action_vec`` is kept under
                ``best.action_vec`` for back-compat.
            best_slots_by_layer: optional grouped view (also from
                ``action_io.group_slots_by_layer_block``). Convenient for
                quick "which layer changed?" inspection.
        """
        self._state["best"] = {
            "reward": float(best_reward),
            "episode": int(best_episode) if best_episode is not None else None,
            "action_vec": (list(int(x) for x in best_action_vec) if best_action_vec is not None else None),
            "slots": _to_jsonable(best_slots) if best_slots is not None else None,
            "slots_by_layer": _to_jsonable(best_slots_by_layer) if best_slots_by_layer is not None else None,
            "breakdown": _to_jsonable(best_breakdown),
        }
        self.flush()

    def set_extra(self, key: str, value: Any) -> None:
        self._state.setdefault("extra", {})[str(key)] = _to_jsonable(value)
        self.flush()

    def mark_stopped(
            self,
            *,
            reason: str,
            completed_episodes: int,
            ) -> None:
        self._state["phase"] = f"已停止：{reason}"
        self._state["completed_episodes"] = int(completed_episodes)
        self._state["stopped_at"] = _dt.datetime.now().isoformat()
        self.flush()

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    def flush(self) -> None:
        try:
            self._state["elapsed_sec"] = round(time.time() - self._t0, 3)
            now = _dt.datetime.now().isoformat()
            self._state["last_update"] = now
            self._state["updated_at"] = now
            best = self._state.get("best") or {}
            self._state["episode"] = int(self._state.get("completed_episodes", 0))
            self._state["best_reward"] = best.get("reward")
            self._state["best_episode"] = best.get("episode")
            _atomic_json_dump(self._path, self._state)
            try:
                _atomic_text_dump(
                    os.path.join(self._dir, BLB_LIVE_SUMMARY_MD),
                    _render_live_summary_markdown(self._state),
                )
            except Exception as exc:
                try:
                    self._log(
                        f"  [BLB状态板][警告] 写 "
                        f"{os.path.join(self._dir, BLB_LIVE_SUMMARY_MD)} 失败：{exc}"
                    )
                except Exception:
                    pass
        except Exception as exc:
            try:
                self._log(f"  [BLB状态板][警告] 写 {self._path} 失败：{exc}")
            except Exception:
                pass

    @property
    def path(self) -> str:
        return self._path


# ---------------------------------------------------------------------------
# 训练曲线
# ---------------------------------------------------------------------------
def _float_array(values):
    """Materialize numeric curve values, preserving ndarray fast paths."""
    import numpy as _np

    if isinstance(values, _np.ndarray):
        return _np.asarray(values, dtype=float)
    return _np.asarray(list(values), dtype=float)


def _ema_smooth(values, window):
    """Symmetric exponential moving average. ``window`` controls decay.

    Used for the paper-style reward plot. Empty input → empty output.
    """
    import numpy as _np

    arr = _float_array(values)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr.copy()
    alpha = 2.0 / (float(window) + 1.0)
    out = _np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _seq_len(values) -> int:
    if values is None:
        return 0
    try:
        return len(values)
    except TypeError:
        return len(list(values))


def _moving_average(values, window):
    """Trailing simple moving average (matches Stage-1's ``Moving Avg (N)`` line).

    Returns ``(x_indices, ma_values)`` where ``x_indices`` are 1-based episode
    positions aligned to the trailing window end. Empty / too-short input →
    two empty arrays.
    """
    import numpy as _np

    arr = _float_array(values)
    n = arr.size
    w = int(max(1, window))
    if n == 0:
        return _np.array([], dtype=float), _np.array([], dtype=float)
    if w <= 1 or n < w:
        return _np.arange(1, n + 1), arr.copy()
    ma = _np.convolve(arr, _np.ones(w, dtype=float) / w, mode="valid")
    xs = _np.arange(w, n + 1)
    return xs, ma


def _stage1_style_panel(ax, raw, *, color, ma_color, ma_window, title,
                        ylabel, baseline=None, baseline_label="Baseline"):
    """Draw one Stage-1-style panel: raw (alpha) + Moving Avg + optional Baseline.

    Mirrors ``layer_importance_evaluator``'s 3-panel convention so Stage-2 curves
    are visually identical in style. ASCII titles only (matplotlib default font
    has no CJK glyphs; Chinese lives in the markdown report).
    """
    import numpy as _np

    arr = _float_array(raw)
    if arr.size == 0:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return
    xs = _np.arange(1, arr.size + 1)
    ax.plot(xs, arr, label="raw", alpha=0.4, color=color, linewidth=0.7)
    ma_x, ma_y = _moving_average(arr, ma_window)
    if ma_y.size:
        ax.plot(ma_x, ma_y, label=f"Moving Avg ({int(ma_window)})",
                linewidth=2.0, color=ma_color)
    if baseline is not None:
        try:
            ax.axhline(float(baseline), color="#666666", linestyle="--",
                       linewidth=1.0, alpha=0.8, label=baseline_label)
        except Exception:
            pass
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def write_training_curves(
        persistence_dir: str,
        *,
        episode_returns: Sequence[float],
        best_reward_curve: Optional[Sequence[float]] = None,
        ppo_loss_curve: Optional[Sequence[float]] = None,
        # Stage-1-parity per-episode series (all optional -> back-compat). When
        # provided, the main PNG mirrors Stage-1's Reward/Loss/metric1/metric2
        # panels exactly (raw + Moving Avg + Baseline). Stage-2-specific cost
        # diagnostics stay out of the main training curve.
        episode_losses: Optional[Sequence[float]] = None,
        episode_metric1s: Optional[Sequence[float]] = None,
        episode_metric2s: Optional[Sequence[float]] = None,
        episode_fusion_counts: Optional[Sequence[float]] = None,
        episode_avg_ks: Optional[Sequence[float]] = None,
        baselines: Optional[Mapping[str, float]] = None,
        metric1_name: str = "metric1",
        metric2_name: str = "metric2",
        # Entropy curve (separate PNG, mirrors Stage-1 ppo_entropy_curve.png).
        entropy_series: Optional[Sequence[float]] = None,
        entropy_episodes: Optional[Sequence[float]] = None,
        ma_window: Optional[int] = None,
        substage_boundaries: Optional[Sequence[int]] = None,
        substage_labels: Optional[Sequence[str]] = None,
        ema_window: int = 200,
        log_fn=None,
        render_plots: Optional[bool] = None,
        ) -> Dict[str, str]:
    """把训练曲线写成 PNG（matplotlib 可用时）+ NPZ（无脑兜底）。

    Emits (when the matching data is provided):
      * ``blb_stage2_training_curve.png`` — Stage-1 风格多联图：Reward / Loss /
        metric1 / metric2，每联 raw + Moving Avg + Baseline 参考线。只给
        ``episode_returns`` 时退化为单联 reward（向后兼容旧 legacy 调用方）。
      * ``blb_stage2_entropy_curve.png`` — 独立熵曲线（镜像 Stage-1
        ``ppo_entropy_curve.png``），需提供 ``entropy_series``。
      * ``blb_stage2_reward_paper.png`` (+ ``.pdf``) — 单联 paper-ready reward。

    Args:
        episode_losses / episode_metric1s / episode_metric2s: 每回合序列（与
            ``episode_returns`` 等长），主训练曲线按 Stage-1 版式绘制这些核心联。
        episode_fusion_counts / episode_avg_ks: 仍写入 NPZ，供诊断/离线报告使用，
            但不进入 ``blb_stage2_training_curve.png``。
        baselines: ``{"loss":..,"metric1":..,"metric2":..,"avg_k":..}`` 各联的
            baseline 参考线（fusion 的 baseline 恒为 0）。
        entropy_series / entropy_episodes: 每次 PPO 更新的策略熵 + 对应的
            completed-episode x 轴。
        ma_window: Moving Avg 窗口（None → 按总回合数自适应）。
        substage_boundaries / substage_labels / ema_window: paper 图用。

    Returns:
        ``{"png", "npz", "paper_png", "paper_pdf", "entropy_png"}``（缺省为 ""）。
    """
    log = log_fn or (lambda _msg: None)
    out = {"png": "", "npz": "", "paper_png": "", "paper_pdf": "", "entropy_png": ""}
    os.makedirs(persistence_dir, exist_ok=True)
    should_render_plots = _stage2_plot_rendering_enabled(render_plots)

    def _has(seq):
        return _seq_len(seq) > 0

    _bl = dict(baselines or {})
    n_ep = _seq_len(episode_returns)
    requested_ma_window = ma_window
    if ma_window is None:
        ma_window = max(10, n_ep // 200) if n_ep else 10

    # NPZ 总是写（最稳）
    try:
        import numpy as _np

        def _arr(seq):
            if seq is None:
                return _np.array([], dtype=float)
            arr = _float_array(seq)
            return arr if arr.size else _np.array([], dtype=float)

        npz_path = os.path.join(persistence_dir, BLB_TRAINING_CURVE_NPZ)
        _np.savez(
            npz_path,
            episode_returns=_arr(episode_returns),
            best_reward_curve=_arr(best_reward_curve),
            ppo_loss_curve=_arr(ppo_loss_curve),
            episode_losses=_arr(episode_losses),
            episode_metric1s=_arr(episode_metric1s),
            episode_metric2s=_arr(episode_metric2s),
            episode_fusion_counts=_arr(episode_fusion_counts),
            episode_avg_ks=_arr(episode_avg_ks),
            entropy_series=_arr(entropy_series),
            entropy_episodes=_arr(entropy_episodes),
        )
        out["npz"] = npz_path
    except Exception as exc:
        log(f"  [BLB曲线][警告] 写 NPZ 失败：{exc}")

    if not should_render_plots:
        log("  [BLB曲线][信息] 跳过 PNG/PDF 渲染（RFR_STAGE2_RENDER_PLOTS=0 或 render_plots=False）。")
        return out

    # ---- 主训练曲线（Stage-1 风格：Reward / Loss / metric1 / metric2）----
    # 标题/坐标统一用 ASCII：matplotlib 默认 DejaVu Sans 不含 CJK 字形，写中文会
    # 触发一堆 UserWarning 且 PNG 上变成方框。中文说明在 markdown 报告里给。
    try:
        png_path = os.path.join(persistence_dir, BLB_TRAINING_CURVE_PNG)
        if _has(episode_returns):
            out["png"] = save_stage1_style_training_curve(
                out_path=png_path,
                reward=episode_returns,
                loss=episode_losses if _has(episode_losses) else None,
                metric1=episode_metric1s if _has(episode_metric1s) else None,
                metric2=episode_metric2s if _has(episode_metric2s) else None,
                baseline_loss=_bl.get("loss"),
                baseline_metric1=_bl.get("metric1"),
                baseline_metric2=_bl.get("metric2"),
                metric1_name=metric1_name,
                metric2_name=metric2_name if _has(episode_metric2s) else None,
                title_suffix="",
                moving_average_window=(
                    int(requested_ma_window) if requested_ma_window is not None else 24
                ),
            )
        elif _has(ppo_loss_curve):
            out["png"] = save_stage1_style_training_curve(
                out_path=png_path,
                reward=ppo_loss_curve,
                title_suffix="",
                moving_average_window=(
                    int(requested_ma_window) if requested_ma_window is not None else 24
                ),
            )
    except Exception as exc:
        log(f"  [BLB曲线][信息] 跳过多联 PNG（matplotlib 不可用 / 渲染失败）：{exc}")

    # ---- 独立熵曲线（镜像 Stage-1 ppo_entropy_curve.png）----
    try:
        if _has(entropy_series):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            ent = _np.asarray(list(entropy_series), dtype=float)
            if _has(entropy_episodes) and _seq_len(entropy_episodes) == ent.size:
                ex = _np.asarray(list(entropy_episodes), dtype=float)
                xlabel = "Episode (at PPO update)"
            else:
                ex = _np.arange(1, ent.size + 1)
                xlabel = "PPO update"
            ent_png = os.path.join(persistence_dir, BLB_ENTROPY_CURVE_PNG)
            ent_w = int(max(2, min(int(ma_window), max(2, ent.size // 20))))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ex, ent, marker=".", markersize=2, linewidth=0.6,
                    color="#1f9e9e", alpha=0.75, label="Policy Entropy")
            _, ma_y = _moving_average(ent, ent_w)
            if ma_y.size:
                ax.plot(ex[ent.size - ma_y.size:], ma_y, linewidth=2.0,
                        color="darkgreen", label=f"Moving Avg ({ent_w})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Entropy")
            ax.set_title("PPO Training: Policy Entropy over Episodes")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            fig.savefig(ent_png, dpi=150)
            plt.close(fig)
            out["entropy_png"] = ent_png
    except Exception as exc:
        log(f"  [BLB曲线][信息] 跳过熵曲线 PNG：{exc}")

    # Paper-style single-panel plot. Convention: gray raw trace alpha=0.3,
    # bold EMA-smoothed foreground, optional best-so-far dashed line, and
    # vertical guides at sub-stage boundaries. Intended for direct inclusion
    # in publications; separate from the multi-panel diagnostic above.
    try:
        if episode_returns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            paper_png = os.path.join(persistence_dir, BLB_REWARD_PAPER_PNG)
            paper_pdf = os.path.join(persistence_dir, BLB_REWARD_PAPER_PDF)
            raw = _np.asarray(episode_returns, dtype=float)
            xs = _np.arange(1, raw.size + 1)
            smoothed = _ema_smooth(raw, int(max(2, ema_window)))

            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.plot(xs, raw, color="#888888", linewidth=0.5, alpha=0.35,
                    label="raw episode return")
            ax.plot(xs, smoothed, color="#1f77b4", linewidth=1.8,
                    label=f"EMA (window={int(ema_window)})")
            if best_reward_curve and len(best_reward_curve) == raw.size:
                ax.plot(
                    xs,
                    _np.asarray(best_reward_curve, dtype=float),
                    color="#ff7f0e",
                    linewidth=1.0,
                    linestyle="--",
                    label="best so far",
                )
            if substage_boundaries:
                ymin, ymax = ax.get_ylim()
                label_y = ymax - 0.05 * (ymax - ymin)
                labels = list(substage_labels or [])
                for i, b in enumerate(substage_boundaries):
                    if b is None or int(b) <= 1 or int(b) >= raw.size:
                        continue
                    ax.axvline(int(b), color="#666666", linewidth=0.7,
                               linestyle=":", alpha=0.7)
                    if i < len(labels):
                        ax.text(int(b), label_y, f" {labels[i]}",
                                fontsize=7, color="#444444",
                                rotation=90, va="top", ha="left")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episodic Return")
            ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
            ax.legend(loc="best", fontsize=8, frameon=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            fig.savefig(paper_png, dpi=200)
            try:
                fig.savefig(paper_pdf)
                out["paper_pdf"] = paper_pdf
            except Exception as exc:
                log(f"  [BLB曲线][信息] 跳过 paper PDF：{exc}")
            plt.close(fig)
            out["paper_png"] = paper_png
    except Exception as exc:
        log(f"  [BLB曲线][信息] 跳过 paper PNG：{exc}")
    return out


# ---------------------------------------------------------------------------
# 崩溃诊断曲线（ADR-014）：reward 分解 / fusion-vs-feasibility / 噪声 vs 余量
# ---------------------------------------------------------------------------
def write_diagnostic_curves(
        persistence_dir: str,
        *,
        priority: Optional[Sequence[int]] = None,
        fusion_count: Optional[Sequence[float]] = None,
        fusion_b2: Optional[Sequence[float]] = None,
        fusion_b4: Optional[Sequence[float]] = None,
        fusion_b5: Optional[Sequence[float]] = None,
        worst_signed_margin: Optional[Sequence[float]] = None,
        acc_barrier_sat: Optional[Sequence[float]] = None,
        acc_barrier_vio: Optional[Sequence[float]] = None,
        cost_score: Optional[Sequence[float]] = None,
        fusion_gain: Optional[Sequence[float]] = None,
        p3_metric_margin: Optional[Sequence[float]] = None,
        metric1_std: Optional[Sequence[float]] = None,
        rolling_window: int = 600,
        ma_window: Optional[int] = None,
        log_fn=None,
        render_plots: Optional[bool] = None,
        ) -> Dict[str, str]:
    """Stage-2 崩溃诊断多联图（``blb_stage2_diagnostics_curve.png``）。

    把「为什么崩」做成一眼可读的图，而不是从最终结果猜（4th-60k 痛点）：

      1. Priority mix (rolling) —— P1/P2/P3 占比。P3→0 = 崩溃。
      2. Fusion (rolling)      —— 总 fusion + per-block b2/b4/b5。失控时单调冲顶。
      3. Accuracy margin mu    —— ``worst_signed_margin`` raw + MA + 0 线。越界 = mu<0。
      4. Reward components      —— barrier_sat / barrier_vio / cost_score / p3_margin。
      5. Probe noise vs margin —— metric1_std vs |mu|。σ>余量 = barrier 失效根因。

    panels 1+2+3 一起读就是 smoking gun：fusion↑ → mu↓ → P3→0。所有入参可选，
    给哪条画哪条。``rolling_window`` 用于 priority/fusion 的滚动均值。
    """
    log = log_fn or (lambda _msg: None)
    out = {"diagnostics_png": ""}
    os.makedirs(persistence_dir, exist_ok=True)
    if not _stage2_plot_rendering_enabled(render_plots):
        log("  [诊断曲线][信息] 跳过 PNG 渲染（RFR_STAGE2_RENDER_PLOTS=0 或 render_plots=False）。")
        return out
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np

        def _arr(seq):
            if seq is None:
                return None
            values = list(seq)
            return _np.asarray(values, dtype=float) if values else None

        def _roll(seq):
            a = _arr(seq)
            if a is None:
                return None, None
            return _moving_average(a, int(max(1, min(rolling_window, a.size))))

        pri = _arr(priority)
        n = pri.size if pri is not None else (
            _arr(fusion_count).size if _arr(fusion_count) is not None else 0)
        if n == 0:
            log("  [诊断曲线][信息] 无数据，跳过。")
            return out
        if ma_window is None:
            ma_window = max(10, n // 200)

        panels = []  # list of (draw_fn, title)

        # Panel 1: priority mix (rolling fractions)
        if pri is not None:
            def _p1(ax):
                for val, color, lbl in ((3, "tab:green", "P3 (cost)"),
                                        (2, "tab:orange", "P2 (stab)"),
                                        (1, "tab:red", "P1 (acc)")):
                    ind = (pri == val).astype(float)
                    x, y = _moving_average(ind, int(max(1, min(rolling_window, ind.size))))
                    if y.size:
                        ax.plot(x, y, color=color, linewidth=1.5, label=lbl)
                ax.set_ylim(-0.02, 1.02)
                ax.set_ylabel("fraction")
            panels.append((_p1, f"Priority mix (rolling {rolling_window}) — P3->0 = collapse"))

        # Panel 2: fusion total + per-block (rolling)
        if _arr(fusion_count) is not None:
            def _p2(ax):
                for seq, color, lbl in ((fusion_count, "black", "fusion total"),
                                        (fusion_b2, "tab:blue", "b2"),
                                        (fusion_b4, "tab:red", "b4"),
                                        (fusion_b5, "tab:green", "b5")):
                    x, y = _roll(seq)
                    if y is not None and y.size:
                        lw = 1.8 if lbl == "fusion total" else 1.0
                        ax.plot(x, y, color=color, linewidth=lw, label=lbl)
                ax.set_ylabel("fused blocks")
            panels.append((_p2, f"Fusion (rolling {rolling_window}) — runaway if monotone to cap"))

        # Panel 3: accuracy margin mu (raw + MA + zero line)
        if _arr(worst_signed_margin) is not None:
            def _p3(ax):
                a = _arr(worst_signed_margin)
                xs = _np.arange(1, a.size + 1)
                ax.plot(xs, a, color="#888888", alpha=0.35, linewidth=0.6, label="mu (raw)")
                mx, my = _moving_average(a, ma_window)
                if my.size:
                    ax.plot(mx, my, color="tab:purple", linewidth=1.8, label=f"MA ({ma_window})")
                ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0, label="feasibility (mu=0)")
                ax.set_ylabel("worst signed margin")
            panels.append((_p3, "Accuracy margin mu (|baseline-thr| units) — mu<0 = P1"))

        # Panel 4: reward components (MA)
        if any(_arr(s) is not None for s in (acc_barrier_sat, acc_barrier_vio, cost_score, p3_metric_margin)):
            def _p4(ax):
                for seq, color, lbl in ((acc_barrier_sat, "tab:purple", "barrier_sat"),
                                        (acc_barrier_vio, "tab:red", "barrier_vio"),
                                        (cost_score, "tab:green", "cost_score"),
                                        (p3_metric_margin, "tab:orange", "p3_margin")):
                    a = _arr(seq)
                    if a is None:
                        continue
                    mx, my = _moving_average(a, ma_window)
                    if my.size:
                        ax.plot(mx, my, color=color, linewidth=1.3, label=lbl)
                ax.set_ylabel("reward component")
            panels.append((_p4, f"Reward components (MA {ma_window})"))

        # Panel 5: probe noise vs margin
        if _arr(metric1_std) is not None and _arr(worst_signed_margin) is not None:
            def _p5(ax):
                s = _arr(metric1_std)
                mu = _np.abs(_arr(worst_signed_margin))
                sx, sy = _moving_average(s, ma_window)
                mmx, mmy = _moving_average(mu, ma_window)
                if sy.size:
                    ax.plot(sx, sy, color="tab:red", linewidth=1.5, label="metric1_std (MA)")
                if mmy.size:
                    ax.plot(mmx, mmy, color="tab:blue", linewidth=1.5, label="|mu| (MA)")
                ax.set_ylabel("magnitude")
            panels.append((_p5, "Probe noise vs |margin| — std > margin => barrier noise-drowned"))

        if not panels:
            return out
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.7 * len(panels)), squeeze=False)
        for i, (draw, title) in enumerate(panels):
            ax = axes[i, 0]
            draw(ax)
            ax.set_xlabel("episode")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("BLB Stage-2 RL Collapse Diagnostics", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        png = os.path.join(persistence_dir, BLB_DIAGNOSTIC_CURVE_PNG)
        fig.savefig(png, dpi=150)
        plt.close(fig)
        out["diagnostics_png"] = png
    except Exception as exc:
        log(f"  [诊断曲线][信息] 跳过（matplotlib 不可用 / 渲染失败）：{exc}")
    return out


# ---------------------------------------------------------------------------
# 最终训练报告（中文 markdown）
# ---------------------------------------------------------------------------
def write_blb_final_report(
        persistence_dir: str,
        *,
        run_basename: str,
        profile: str,
        total_episodes: int,
        completed_episodes: int,
        elapsed_sec: float,
        best_reward: float,
        best_breakdown: Optional[Mapping[str, Any]],
        best_action_vec: Optional[Sequence[int]],
        baseline: Optional[Mapping[str, Any]],
        reward_weights: Optional[Mapping[str, Any]],
        episode_returns: Sequence[float],
        rescale_invoker_kind: str,
        any_invalid_baseline: bool = False,
        extra_lines: Sequence[str] = (),
        log_fn=None,
        best_slots: Optional[Sequence[Mapping[str, Any]]] = None,
        baseline_slots: Optional[Sequence[Mapping[str, Any]]] = None,
        slot_diff_vs_baseline: Optional[Sequence[Mapping[str, Any]]] = None,
        best_action_full_md_path: str = "",
        best_action_full_json_path: str = "",
        ) -> str:
    """落盘 ``blb_stage2_report.md``，返回路径。失败抛 IOError。"""
    log = log_fn or (lambda _msg: None)
    os.makedirs(persistence_dir, exist_ok=True)
    path = os.path.join(persistence_dir, BLB_FINAL_REPORT_MD)

    import numpy as _np
    rr = _np.asarray(list(episode_returns or []), dtype=float)
    rr_mean = float(rr.mean()) if rr.size > 0 else 0.0
    rr_max = float(rr.max()) if rr.size > 0 else 0.0
    rr_min = float(rr.min()) if rr.size > 0 else 0.0
    rr_std = float(rr.std(ddof=0)) if rr.size > 1 else 0.0

    lines: List[str] = []
    lines.append(f"# BLB Stage 2 RL 训练报告（最终版）")
    lines.append("")
    lines.append(f"- 运行名（run_basename）: `{run_basename}`")
    lines.append(f"- Profile（数据集）: `{profile}`")
    lines.append(f"- 生成时间: {_dt.datetime.now().isoformat()}")
    lines.append(f"- 训练时长: {elapsed_sec:.1f} 秒（约 {elapsed_sec/60:.1f} 分钟）")
    lines.append(f"- Episode 进度: {completed_episodes} / {total_episodes}")
    lines.append(f"- 模数链 invoker: `{rescale_invoker_kind}`")
    lines.append("")
    lines.append("## 1. Reward 概览")
    lines.append("")
    lines.append(f"- 最优 reward (best): **{best_reward:+.6f}**")
    lines.append(f"- 全程 episode reward 均值: {rr_mean:+.4f}")
    lines.append(f"- 全程 episode reward 最大值: {rr_max:+.4f}")
    lines.append(f"- 全程 episode reward 最小值: {rr_min:+.4f}")
    lines.append(f"- 全程 episode reward 标准差: {rr_std:.4f}")
    lines.append("")

    if best_breakdown:
        lines.append("## 2. 最优 reward 拆解")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|------|")
        for k, v in dict(best_breakdown).items():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    if baseline is not None:
        lines.append("## 3. Baseline（全 max action）对照")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|------|")
        for k, v in dict(baseline).items():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    if reward_weights is not None:
        lines.append("## 4. Reward 权重")
        lines.append("")
        lines.append("| 字段 | 值 |")
        lines.append("|------|------|")
        for k, v in dict(reward_weights).items():
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    if best_action_vec is not None:
        lines.append("## 5. 最优 action：选了什么 SF / K（人类视图）")
        lines.append("")
        lines.append(
            "完整的逐槽位明细在 "
            f"`{os.path.basename(best_action_full_md_path) or 'blb_stage2_best_action_full.md'}` "
            "（人类阅读）和 "
            f"`{os.path.basename(best_action_full_json_path) or 'blb_stage2_best_action_full.json'}` "
            "（可直接喂给 `Paean/run_final_eval.sh --action-config`）。下面只列出与 baseline 不同的槽位。"
        )
        lines.append("")
        # 5.a Per-block summary using best_slots if available.
        if best_slots:
            grouped = _slots_by_layer_grouped(best_slots)
            if grouped:
                lines.append("### 5.1 Best action · 按层 / block 选择概览")
                lines.append("")
                lines.append("| 层 | block | 槽位选择 |")
                lines.append("|---|---|---|")
                for layer_key, layer_val in grouped.items():
                    if layer_key == "first_input":
                        lines.append(f"| (legacy) first_input | – | `scaling_factor={layer_val}` |")
                        continue
                    if not isinstance(layer_val, Mapping):
                        continue
                    for block_key in sorted(layer_val.keys()):
                        block_val = layer_val[block_key]
                        if not isinstance(block_val, Mapping):
                            continue
                        cell_parts: List[str] = []
                        for slot_short, value in block_val.items():
                            if slot_short == "K":
                                cell_parts.append(f"K=**{value}**")
                            else:
                                disp = "off" if value is None else value
                                cell_parts.append(f"{slot_short}={disp}")
                        lines.append(f"| {layer_key} | {block_key} | {', '.join(cell_parts)} |")
                lines.append("")
        # 5.b Diff against baseline — the actionable view.
        if slot_diff_vs_baseline:
            sf_diffs = [d for d in slot_diff_vs_baseline if d.get("kind") != "K"]
            k_diffs = [d for d in slot_diff_vs_baseline if d.get("kind") == "K"]
            lines.append("### 5.2 Best vs baseline · 哪些槽位变了")
            lines.append("")
            lines.append(
                f"_共 {len(slot_diff_vs_baseline)} 个槽位发生变化"
                f"（{len(sf_diffs)} 个 SF + {len(k_diffs)} 个 K bits）_"
            )
            lines.append("")
            if k_diffs:
                lines.append("**截断 K bits 变化**：")
                lines.append("")
                lines.append("| 槽位 | baseline K | best K | Δ |")
                lines.append("|---|---:|---:|---:|")
                for d in sorted(k_diffs, key=lambda r: r.get("label", "")):
                    lines.append(
                        f"| `{d.get('label','')}` | {d.get('baseline_truncation_bits','?')} | "
                        f"{d.get('best_truncation_bits','?')} | "
                        f"{int(d.get('delta', 0)):+d} |"
                    )
                lines.append("")
            if sf_diffs:
                lines.append("**Scaling factor 变化**（前 25 条按 |Δ| 降序）：")
                lines.append("")
                lines.append("| 槽位 | kind | baseline SF | best SF | Δ |")
                lines.append("|---|:---:|---:|---:|---:|")
                def _abs_delta(r):
                    d_ = r.get("delta")
                    return -1 if d_ is None else abs(int(d_))
                for d in sorted(sf_diffs, key=_abs_delta, reverse=True)[:25]:
                    b_ = d.get("baseline_scaling_factor")
                    a_ = d.get("best_scaling_factor")
                    if b_ is None:
                        delta_s = "off→on"
                    elif a_ is None:
                        delta_s = "on→off"
                    else:
                        delta_s = f"{int(d.get('delta', 0)):+d}"
                    lines.append(
                        f"| `{d.get('label','')}` | `{d.get('kind','')}` | "
                        f"{'off' if b_ is None else b_} | "
                        f"{'off' if a_ is None else a_} | {delta_s} |"
                    )
                lines.append("")
        else:
            lines.append(
                "_（baseline diff 不可用 —— 如果想看 best vs baseline 的具体改动，"
                "请打开 `diagnostics/best_action_vec.json` 的 `diff_vs_baseline` 字段。）_"
            )
            lines.append("")
        # 5.c Original flat int vector hidden in a collapsible block (debugging only).
        lines.append("<details>")
        lines.append("<summary>调试用：原始 action_vec（整数索引）</summary>")
        lines.append("")
        lines.append(f"- 长度: {len(best_action_vec)}")
        lines.append("")
        lines.append("```")
        lines.append(", ".join(str(int(x)) for x in best_action_vec))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if any_invalid_baseline:
        lines.append("## ⚠️ baseline 链不合法警告")
        lines.append("")
        lines.append("Baseline 校准时 Rescale_optimizer 报告 `invalid_chain != None`。")
        lines.append("通常说明 graph / max_sfs JSON 与 BLB cfg 字段命名不对齐，"
                     "或者 baseline t 在某个 stage 越界。请检查 "
                     "`Rescale_optimizer/configs/<profile>/static_skeletons_<profile>.json`。")
        lines.append("")

    if extra_lines:
        lines.append("## 附加说明")
        lines.append("")
        for ln in extra_lines:
            lines.append(str(ln))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> 持久化目录：`Parting Chapter/<run>/stage2_noise/progress/`。"
                 "live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / "
                 "训练曲线（PNG + NPZ）/ 本报告 都在该目录下。")

    with open(path, "w", encoding="utf-8") as f:
        _write_joined_lines_stream(f, lines)
    return path


# ---------------------------------------------------------------------------
# 崩溃归档
# ---------------------------------------------------------------------------
def dump_crash_report(
        persistence_dir: str,
        *,
        exc: BaseException,
        last_state: Optional[Mapping[str, Any]] = None,
        log_fn=None,
        ) -> str:
    """异常时把 traceback + 最后状态写到 ``blb_stage2_error.txt``。返回路径。"""
    log = log_fn or (lambda _msg: None)
    os.makedirs(persistence_dir, exist_ok=True)
    path = os.path.join(persistence_dir, BLB_ERROR_TXT)
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"BLB Stage 2 RL 崩溃归档")
    lines.append(f"时间: {_dt.datetime.now().isoformat()}")
    lines.append(f"异常类型: {type(exc).__name__}")
    lines.append(f"异常信息: {exc!s}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Traceback:")
    lines.append("-" * 80)
    lines.append(traceback.format_exc())
    lines.append("-" * 80)
    if last_state:
        lines.append("")
        lines.append("最后状态快照:")
        lines.append("-" * 80)
        try:
            lines.append(json.dumps(_to_jsonable(last_state), ensure_ascii=False, indent=2, default=str))
        except Exception:
            lines.append(str(last_state))
        lines.append("-" * 80)
    lines.append("")
    lines.append(f"Python: {sys.version}")
    lines.append(f"Platform: {sys.platform}")
    lines.append(f"CWD: {os.getcwd()}")
    try:
        with open(path, "w", encoding="utf-8") as f:
            _write_joined_lines_stream(f, lines)
    except Exception as inner_exc:
        log(f"  [BLB崩溃归档][警告] 写 {path} 失败：{inner_exc}")
        return ""
    return path


# ---------------------------------------------------------------------------
# Step-detail rollover writer + reward-crash watcher
# (legacy noise_rl_module_v2 parity: stage2_noise/details/*.txt + warning.txt)
# ---------------------------------------------------------------------------
BLB_DETAILS_DIRNAME = "details"
BLB_DETAILS_FILENAME_FMT = "noise_ppo_step_info_{start}-{end}.txt"
BLB_WARNING_FILENAME = "warning.txt"


class BLBStepDetailsWriter:
    """Per-batch detail log writer matching legacy ``details/`` layout.

    Writes one file per ``batch_size`` episodes under
    ``<noise_root>/details/noise_ppo_step_info_<start>-<end>.txt``. Each file
    holds one record per episode: return, priority/invalid, key cost signals,
    and (optionally) a short slot-diff against the baseline. The full main
    log stays slim because per-episode error spam is routed here instead.
    """

    def __init__(
            self,
            noise_root: str,
            *,
            batch_size: int = 360,
            log_fn=None,
            ):
        self._root = str(noise_root)
        self._dir = os.path.join(self._root, BLB_DETAILS_DIRNAME)
        os.makedirs(self._dir, exist_ok=True)
        self._batch_size = max(1, int(batch_size))
        self._log = log_fn or (lambda _msg: None)
        self._buffer: List[str] = []
        self._batch_start: Optional[int] = None
        self._batch_end: Optional[int] = None

    def append_episode(
            self,
            *,
            episode: int,
            episode_return: float,
            priority: int,
            invalid: bool,
            error_text: str = "",
            opt_signals: Optional[Mapping[str, Any]] = None,
            slot_diff: str = "",
            extra_lines: Optional[Sequence[str]] = None,
            ) -> None:
        """Buffer one episode's record. Auto-flushes when batch fills."""
        ep = int(episode)
        if self._batch_start is None:
            self._batch_start = ((ep - 1) // self._batch_size) * self._batch_size + 1
            self._batch_end = self._batch_start + self._batch_size - 1
        prio_label = (
            "P1(acc)" if priority == 1 else
            "P2(stab)" if priority == 2 else
            "P3(cost)" if priority == 3 else
            f"P{priority}"
        )
        if invalid:
            prio_label = f"{prio_label}+optimizer_invalid"
        block = [
            f"── 回合 episode {ep} ── 回报 episode_return={float(episode_return):+.4f}  优先级 priority={prio_label}",
        ]
        if opt_signals:
            block.append(
                "  Rescale 信号: "
                f"total_bits={opt_signals.get('total_bits_sum', 0)}, "
                f"fusion_count={opt_signals.get('total_fusion_count', 0)}, "
                f"any_invalid={bool(opt_signals.get('any_invalid', False))}"
            )
        if slot_diff:
            block.append(f"  动作变化（vs baseline）: {slot_diff}")
        if error_text:
            txt = str(error_text).strip().replace("\n", " | ")
            if len(txt) > 480:
                txt = txt[:480] + " ..."
            block.append(f"  错误信息: {txt}")
        if extra_lines:
            for ln in extra_lines:
                block.append(f"  {ln}")
        block.append("")
        self._buffer.append("\n".join(block))
        if ep >= int(self._batch_end or 0):
            self.flush()

    def flush(self) -> Optional[str]:
        if not self._buffer or self._batch_start is None:
            return None
        path = os.path.join(
            self._dir,
            BLB_DETAILS_FILENAME_FMT.format(
                start=int(self._batch_start),
                end=int(self._batch_end),
            ),
        )
        try:
            mode = "a" if os.path.exists(path) else "w"
            with open(path, mode, encoding="utf-8") as f:
                if mode == "w":
                    f.write(
                        f"=== BLB Stage-2 RL 逐回合诊断 · 回合区间 "
                        f"{int(self._batch_start)}-{int(self._batch_end)} ===\n\n"
                    )
                f.write("\n".join(self._buffer))
                f.write("\n")
        except Exception as exc:
            try:
                self._log(f"  [BLB details][warning] 写 {path} 失败：{exc}")
            except Exception:
                pass
            self._buffer.clear()
            return None
        self._buffer.clear()
        # Roll batch window forward to the next interval
        self._batch_start = int(self._batch_end or 0) + 1
        self._batch_end = self._batch_start + self._batch_size - 1
        return path

    @property
    def current_batch_path(self) -> str:
        if self._batch_start is None or self._batch_end is None:
            return ""
        return os.path.join(
            self._dir,
            BLB_DETAILS_FILENAME_FMT.format(
                start=int(self._batch_start),
                end=int(self._batch_end),
            ),
        )


class BLBRewardCrashWatcher:
    """Watches PPO-rollout mean reward and emits drop warnings.

    Mirrors legacy ``warning.txt`` semantics: when a new rollout's mean reward
    is at least ``drop_threshold`` below the previous rollout (or a rolling
    baseline), write a warning entry pointing at the current details batch
    file. Useful for spotting policy collapse early.
    """

    def __init__(
            self,
            noise_root: str,
            *,
            drop_threshold: float = 0.3,
            log_fn=None,
            ):
        self._root = str(noise_root)
        self._path = os.path.join(self._root, BLB_WARNING_FILENAME)
        self._drop_threshold = float(drop_threshold)
        self._log = log_fn or (lambda _msg: None)
        self._prev_mean: Optional[float] = None
        self._count = 0

    def observe_rollout(
            self,
            *,
            rollout_mean: float,
            episode_start: int,
            episode_end: int,
            details_path: str = "",
            phase_label: str = "BLB Stage-2 RL（v3）",
            ) -> Optional[Dict[str, Any]]:
        try:
            mean = float(rollout_mean)
        except Exception:
            return None
        if not math.isfinite(mean):
            self._prev_mean = mean
            return None
        warning: Optional[Dict[str, Any]] = None
        prev = self._prev_mean
        if prev is not None and math.isfinite(prev):
            drop = float(prev) - mean
            if drop > self._drop_threshold:
                self._count += 1
                warning = {
                    "type": "rollout_reward_drop",
                    "drop": drop,
                    "prev_mean": float(prev),
                    "curr_mean": mean,
                    "episode_start": int(episode_start),
                    "episode_end": int(episode_end),
                    "details_path": str(details_path or ""),
                }
                self._append_warning(warning, phase_label=phase_label)
        self._prev_mean = mean
        return warning

    def _append_warning(self, warning: Mapping[str, Any], *, phase_label: str) -> None:
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            need_header = (not os.path.exists(self._path)) or os.path.getsize(self._path) == 0
            with open(self._path, "a", encoding="utf-8") as f:
                if need_header:
                    f.write("=== BLB Stage-2 RL 奖励暴跌警告（reward-drop warnings）===\n")
                    f.write(f"阶段（phase）: {phase_label}\n")
                    f.write(f"启动时间（first-write）: {_dt.datetime.now().isoformat()}\n\n")
                f.write(f"--- 警告 #{int(self._count)} ---\n")
                f.write("  类型: rollout reward drop\n")
                f.write(
                    "  跌幅（drop）: "
                    f"{float(warning.get('drop', 0.0)):.4f} "
                    f"(prev={float(warning.get('prev_mean', 0.0)):.4f}, "
                    f"curr={float(warning.get('curr_mean', 0.0)):.4f})\n"
                )
                f.write(
                    "  回合范围: "
                    f"{int(warning.get('episode_start', 0))}-"
                    f"{int(warning.get('episode_end', 0))}\n"
                )
                details = str(warning.get("details_path", ""))
                if details:
                    f.write(f"    -> {details}\n")
                f.write(f"  写入时间: {_dt.datetime.now().isoformat()}\n\n")
        except Exception as exc:
            try:
                self._log(f"  [BLB warning][warning] 写 warning.txt 失败：{exc}")
            except Exception:
                pass

    @property
    def warning_path(self) -> str:
        return self._path

    @property
    def total_count(self) -> int:
        return int(self._count)
