"""RL 局部最优 / 健康检测（Stage-1 与 Stage-2 共用，torch-free）。

历史上 ``detect_rl_local_optimum`` 定义在 ``layer_importance_evaluator.py`` 里，
而那个文件 import 了 torch / transformers，重得很。为了让 **Stage-2** 的收尾
（``blb_stage2_rl.sequential_runner``）、离线再生器（``scripts/blb_regen_stage2_outputs.py``）
和单元测试都能在**没有 torch 的机器上**复用同一套判据，把这个纯 numpy 的函数挪到这里。

``layer_importance_evaluator`` 仍以 ``from rl_local_optimum import detect_rl_local_optimum``
重新导出它，所以 Stage-1 行为与历史完全一致（``noise_rl_module_v2`` 里
``from layer_importance_evaluator import detect_rl_local_optimum`` 也照常工作）。

本模块只依赖 numpy。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np


def _float_array(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False).reshape(-1)
    return np.asarray(list(values), dtype=float)


def _float_array_if_length(values, expected_len: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = _float_array(values)
    if arr.size != int(expected_len):
        return None
    return arr


def detect_rl_local_optimum(
    episode_returns,
    episode_entropies=None,
    best_score_history=None,
    action_history=None,
    window=200,
    entropy_collapse_threshold=0.05,
    plateau_cv_threshold=0.02,
    best_stuck_window=None,
    action_kl_threshold=0.5,
):
    """检测 RL 是否陷入局部最优；返回包含逐项判据 + 综合结论的 dict。

    Parameters
    ----------
    episode_returns : Sequence[float]
        每回合的 return / reward。必填。
    episode_entropies : Sequence[float], optional
        每回合（或每次 PPO 更新）的策略熵。强烈建议提供。
    best_score_history : Sequence[float], optional
        全局最优分数随回合的演进序列；若为 None，则用 cummax(episode_returns) 近似。
    action_history : Sequence[Sequence[int]], optional
        每回合的动作序列；若提供则计算多样性塌缩信号。
    window : int
        判定使用的滑窗大小（episode）。
    entropy_collapse_threshold : float
        熵塌缩阈值。低于该值视作熵塌缩。
    plateau_cv_threshold : float
        变异系数 (std/|mean|) 阈值；越小说明 reward 越平。
    best_stuck_window : int, optional
        best 不更新的窗口；默认 = 2 * window。
    action_kl_threshold : float
        动作分布 KL 阈值。

    Returns
    -------
    dict
        包含 keys:
          - signals: 各判据 bool
          - metrics: 各判据数值指标
          - likely_local_optimum: 综合判定
          - summary: 一行人类可读总结
    """
    returns = _float_array(episode_returns)
    n = returns.size
    out = {"signals": {}, "metrics": {}, "likely_local_optimum": False, "summary": ""}
    if n < max(20, window // 4):
        out["summary"] = f"样本不足（n={n}），无法判定。"
        return out

    w = int(min(window, n))
    recent_returns = returns[-w:]

    # ---- A. entropy collapse ----
    entropy_collapsed = False
    recent_entropy_mean = None
    ent_arr = None
    if episode_entropies is not None:
        ent_arr = _float_array(episode_entropies)
        if ent_arr.size == 0:
            ent_arr = None
    if ent_arr is not None:
        ew = int(min(w, ent_arr.size))
        recent_entropy_mean = float(np.mean(ent_arr[-ew:]))
        entropy_collapsed = recent_entropy_mean < entropy_collapse_threshold
    out["metrics"]["recent_entropy_mean"] = recent_entropy_mean
    out["signals"]["entropy_collapsed"] = bool(entropy_collapsed)

    # ---- B. reward plateau ----
    mean_r = float(np.mean(recent_returns))
    std_r = float(np.std(recent_returns))
    cv = std_r / (abs(mean_r) + 1e-8)
    reward_plateau = cv < plateau_cv_threshold
    out["metrics"]["recent_reward_mean"] = mean_r
    out["metrics"]["recent_reward_std"] = std_r
    out["metrics"]["recent_reward_cv"] = cv
    out["signals"]["reward_plateau"] = bool(reward_plateau)

    # ---- C. best stuck ----
    if best_score_history is None:
        best_curve = np.maximum.accumulate(returns)
    else:
        best_curve = _float_array(best_score_history)
    bw = int(best_stuck_window if best_stuck_window is not None else 2 * w)
    bw = min(bw, best_curve.size)
    best_stuck = False
    last_improve_gap = None
    if bw >= 2:
        recent_best = best_curve[-bw:]
        last_improve_gap = int(bw - 1 - int(np.argmax(recent_best)))
        best_stuck = (recent_best[-1] - recent_best[0]) <= 1e-9
    out["metrics"]["best_stuck_window"] = bw
    out["metrics"]["episodes_since_last_best_improve"] = last_improve_gap
    out["signals"]["best_stuck"] = bool(best_stuck)

    # ---- D. action diversity collapse (optional) ----
    diversity_collapsed = False
    action_kl = None
    if action_history is not None:
        action_rows = list(action_history)
    else:
        action_rows = []
    if len(action_rows) >= 2 * w:
        try:
            recent = np.asarray(action_rows[-w:]).reshape(-1)
            early = np.asarray(action_rows[:w]).reshape(-1)
            vals = np.unique(np.concatenate([recent, early]))
            def _hist(x):
                h = np.array([(x == v).sum() for v in vals], dtype=float)
                h = h / max(h.sum(), 1.0)
                return h + 1e-8
            p_recent = _hist(recent)
            p_early = _hist(early)
            action_kl = float(np.sum(p_recent * np.log(p_recent / p_early)))
            diversity_collapsed = (action_kl > action_kl_threshold and
                                   (recent_entropy_mean is not None and
                                    recent_entropy_mean < entropy_collapse_threshold * 2))
        except Exception:
            action_kl = None
    out["metrics"]["action_kl_recent_vs_early"] = action_kl
    out["signals"]["action_diversity_collapsed"] = bool(diversity_collapsed)

    # ---- 综合判定：A/B/C 中 ≥2 条成立 ----
    score = int(entropy_collapsed) + int(reward_plateau) + int(best_stuck)
    out["likely_local_optimum"] = score >= 2 or diversity_collapsed
    flags = []
    if entropy_collapsed:
        flags.append(f"entropy_collapse(H={recent_entropy_mean:.4f})")
    if reward_plateau:
        flags.append(f"reward_plateau(cv={cv:.4f})")
    if best_stuck:
        flags.append(f"best_stuck({last_improve_gap}ep_no_improve)")
    if diversity_collapsed:
        flags.append(f"action_kl={action_kl:.3f}")
    out["summary"] = (
        ("[LIKELY LOCAL-OPTIMUM] " if out["likely_local_optimum"] else "[OK] ")
        + (", ".join(flags) if flags else "no warning signals")
    )
    return out


def format_local_optimum_report(
    diag: dict,
    *,
    title: str = "RL",
    completed_episodes: Optional[int] = None,
    extra_lines: Optional[Sequence[str]] = None,
) -> str:
    """把 ``detect_rl_local_optimum`` 的 dict 渲染成 Stage-1 ``pruning_search_log.txt``
    同款文本版式（标题可换 Stage-1 / Stage-2）。"""
    lines = []
    lines.append(f"=== {title} 局部最优检测报告 ===")
    if completed_episodes is not None:
        lines.append(f"完成回合数: {int(completed_episodes)}")
    lines.append(f"判定: {diag.get('summary', '')}")
    lines.append("")
    lines.append("--- 各项判据信号 ---")
    for k, v in (diag.get("signals") or {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("--- 数值指标 ---")
    for k, v in (diag.get("metrics") or {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("--- 说明 ---")
    lines.append(
        "判定规则：A.熵塌缩 / B.reward 平台 / C.best 长期不更新 三条中"
    )
    lines.append(
        "≥2 条成立 → likely_local_optimum=True；或 D.动作分布塌缩单独成立。"
    )
    for ln in (extra_lines or []):
        lines.append(str(ln))
    return "\n".join(lines) + "\n"


def attribute_collapse(
    *,
    priority: Sequence[int],
    fusion_count: Optional[Sequence[float]] = None,
    worst_signed_margin: Optional[Sequence[float]] = None,
    window: int = 600,
    p3_floor: float = 0.1,
):
    """Locate collapse onset and classify HOT (over-fusion) vs COLD (no fusion).

    A Stage-2 hot collapse (the 3rd/4th 60k) shows rolling P3 falling to ~0 while
    fusion ran AWAY and the accuracy margin went negative; a cold collapse shows
    P3 fine-ish but fusion stuck at ~0. This pins WHEN it happened and WHY from
    the per-episode series, so the failure is read, not guessed. Returns a list
    of human-readable lines (Chinese) appended to the detection report.
    """
    lines: List[str] = ["", "--- 崩溃归因（collapse attribution）---"]
    pri = [int(p) for p in priority]
    n = len(pri)
    if n < 80:
        lines.append(f"  样本不足（n={n}），无法归因。")
        return lines
    # Real 60k runs use the full ``window`` (600); short runs scale down so a
    # mid-run collapse is still detectable (n//4 keeps several windows in view).
    w = int(min(window, max(20, n // 4)))
    p3 = np.array([1.0 if p == 3 else 0.0 for p in pri], dtype=float)
    onset = None
    for i in range(w, n + 1):
        if float(p3[i - w:i].mean()) < p3_floor:
            onset = i
            break
    if onset is None:
        lines.append(f"  未检测到崩溃（rolling{w} P3 始终 ≥ {p3_floor:.0%}）。")
        return lines
    fc = _float_array_if_length(fusion_count, n)
    mu = _float_array_if_length(worst_signed_margin, n)
    lines.append(f"  崩溃起点（rolling{w} P3 首次 < {p3_floor:.0%}）: episode≈{onset}")
    verdict = "未知"
    if fc is not None:
        early = float(fc[:w].mean())
        at_onset = float(fc[max(0, onset - w):onset].mean())
        final = float(fc[-w:].mean())
        lines.append(f"  fusion 均值: 早期 {early:.2f} → 起点 {at_onset:.2f} → 末段 {final:.2f}")
        if at_onset > early + 2.0 or final > early + 4.0:
            verdict = "HOT（过度融合 / over-fusion）"
        elif final <= early + 1.0 and early < 2.0:
            verdict = "COLD（几乎不融合 / no fusion）"
        else:
            verdict = "MIXED / 其他"
    if mu is not None:
        lines.append(f"  margin(mu) 均值: 早期 {float(mu[:w].mean()):.4f} → 起点 {float(mu[max(0, onset - w):onset].mean()):.4f} → 末段 {float(mu[-w:].mean()):.4f}")
    lines.append(f"  判定: {verdict}")
    return lines


def write_local_optimum_report(
    report_path: str,
    *,
    episode_returns,
    episode_entropies=None,
    best_score_history=None,
    completed_episodes: Optional[int] = None,
    window: Optional[int] = None,
    title: str = "RL",
    extra_lines: Optional[Sequence[str]] = None,
    priority: Optional[Sequence[int]] = None,
    fusion_count: Optional[Sequence[float]] = None,
    worst_signed_margin: Optional[Sequence[float]] = None,
    log_fn=None,
) -> str:
    """计算检测判据并写出 Stage-1 同款检测报告文件。返回写出的路径（失败返回 ""）。

    若提供 ``priority`` (+ 可选 ``fusion_count`` / ``worst_signed_margin``)，追加一段
    崩溃归因（起点 + HOT/COLD 判定）。best-effort：任何异常只记日志，不抛出。
    """
    log = log_fn or (lambda _msg: None)
    try:
        returns = _float_array(episode_returns)
        n = returns.size
        if window is None:
            window = max(50, int(max(1, n) * 0.1))
        diag = detect_rl_local_optimum(
            episode_returns=returns,
            episode_entropies=episode_entropies,
            best_score_history=best_score_history,
            action_history=None,
            window=window,
        )
        all_extra = list(extra_lines or [])
        if priority is not None:
            all_extra.extend(attribute_collapse(
                priority=priority,
                fusion_count=fusion_count,
                worst_signed_margin=worst_signed_margin,
            ))
        text = format_local_optimum_report(
            diag,
            title=title,
            completed_episodes=completed_episodes if completed_episodes is not None else n,
            extra_lines=all_extra,
        )
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"  [检测] 局部最优检测报告 → {report_path}")
        log(f"  [检测] {diag.get('summary', '')}")
        return report_path
    except Exception as exc:  # noqa: BLE001 - best-effort artifact
        log(f"  [检测][警告] 局部最优检测失败：{exc}")
        return ""
