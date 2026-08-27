"""Atomic Stage-2 status, diagnostics, curve, and checkpoint artifacts."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from rfr.common.json_utils import to_jsonable as _to_jsonable
from rfr.evaluation.training_curve_plot import save_stage1_style_training_curve


BLB_STATUS_FILENAME = "blb_stage2_status.json"
BLB_TRAINING_CURVE_PNG = "blb_stage2_training_curve.png"
BLB_TRAINING_CURVE_NPZ = "blb_stage2_training_curve.npz"
BLB_REWARD_PAPER_PNG = "blb_stage2_reward_paper.png"
BLB_REWARD_PAPER_PDF = "blb_stage2_reward_paper.pdf"
BLB_ENTROPY_CURVE_PNG = "blb_stage2_entropy_curve.png"
BLB_LIVE_SUMMARY_MD = "blb_stage2_live_summary.md"
_PLOT_RENDER_FALSE_VALUES = {"0", "false", "no", "off", "skip", "none"}
_FLOAT_ARRAY_DIRECT_SEQUENCE_TYPES = (list, tuple, range)


def _atomic_json_dump(path: str, obj: Any) -> None:
    """Write JSON through a temporary file and atomically replace the destination."""
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


def _stage2_plot_rendering_enabled(render_plots: Optional[bool]) -> bool:
    if render_plots is not None:
        return bool(render_plots)
    raw = os.environ.get("RFR_STAGE2_RENDER_PLOTS")
    if raw is None:
        return False
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
    episode_progress = f"{completed} / {total} ({pct:.2f}%)"
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
        f"- Episode: {episode_progress}",
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
        "",
    ])
    return "\n".join(lines)


class BLBStatusBoard:
    """Persist the compact status snapshot for a Stage 2 run."""

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
        if int(total_episodes) <= 0:
            raise ValueError("total_episodes must be positive")
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
            "phase": "initializing",
            "elapsed_sec": 0.0,
            "last_update": _dt.datetime.now().isoformat(),
            "recent_returns": [],
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
                ``best.action_vec`` for exact replay.
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


def _float_array(values):
    """Materialize numeric curve values, preserving ndarray fast paths."""
    import numpy as _np

    if isinstance(values, _np.ndarray):
        return _np.asarray(values, dtype=float)
    if isinstance(values, _FLOAT_ARRAY_DIRECT_SEQUENCE_TYPES):
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
        return sum(1 for _ in values)


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


        episode_losses: Optional[Sequence[float]] = None,
        episode_metric1s: Optional[Sequence[float]] = None,
        episode_metric2s: Optional[Sequence[float]] = None,
        episode_fusion_counts: Optional[Sequence[float]] = None,
        episode_avg_ks: Optional[Sequence[float]] = None,
        baselines: Optional[Mapping[str, float]] = None,
        metric1_name: str = "metric1",
        metric2_name: str = "metric2",

        entropy_series: Optional[Sequence[float]] = None,
        entropy_episodes: Optional[Sequence[float]] = None,
        ma_window: Optional[int] = None,
        substage_boundaries: Optional[Sequence[int]] = None,
        substage_labels: Optional[Sequence[str]] = None,
        ema_window: int = 200,
        log_fn=None,
        render_plots: Optional[bool] = None,
        ) -> Dict[str, str]:
    """Write training curves as PNG when available and always preserve NPZ data."""
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
        log("  [BLB曲线][信息] PNG/PDF 渲染已延后；设置 RFR_STAGE2_RENDER_PLOTS=1 或 render_plots=True 可启用。")
        return out


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


    try:
        if _has(entropy_series):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            ent = _float_array(entropy_series)
            if _has(entropy_episodes) and _seq_len(entropy_episodes) == ent.size:
                ex = _float_array(entropy_episodes)
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
