"""BLB Stage 2 RL 训练侧的持久化辅助：状态板 / 训练曲线 / 报告 / 崩溃记录。

旧版 stage 2 RL（``noise_rl_module_v2``）在 ``rl_results/persistent/...`` 下
做了若干"训练之外"的小事：进度条、curve PNG、checkpoint 元数据、错误归档。
BLB Stage 2 RL 是最终版本，需要把这些项目操作类的输出补齐到新的持久化目录
``Parting Chapter/<run_basename>/blb_stage2/progress/``。

本模块提供四件事：

  1. ``BLBStatusBoard``      ── 训练期间持续刷新 ``blb_stage2_status.json``，
                                 同时累积 episode_returns / 训练曲线数据。
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
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


BLB_STATUS_FILENAME = "blb_stage2_status.json"
BLB_TRAINING_CURVE_PNG = "blb_stage2_training_curve.png"
BLB_TRAINING_CURVE_NPZ = "blb_stage2_training_curve.npz"
BLB_FINAL_REPORT_MD = "blb_stage2_report.md"
BLB_ERROR_TXT = "blb_stage2_error.txt"


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


def _to_jsonable(obj: Any) -> Any:
    """尽量把复杂对象（dataclass / numpy / dict / list）转成 JSON-able 形式。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # numpy scalar / array
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.generic):
            return obj.item()
    except Exception:
        pass
    # 兜底：用 str
    try:
        return str(obj)
    except Exception:
        return None


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
            ) -> None:
        self._state["best"] = {
            "reward": float(best_reward),
            "episode": int(best_episode) if best_episode is not None else None,
            "action_vec": (list(int(x) for x in best_action_vec) if best_action_vec is not None else None),
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
            self._state["last_update"] = _dt.datetime.now().isoformat()
            _atomic_json_dump(self._path, self._state)
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
def write_training_curves(
        persistence_dir: str,
        *,
        episode_returns: Sequence[float],
        best_reward_curve: Optional[Sequence[float]] = None,
        ppo_loss_curve: Optional[Sequence[float]] = None,
        log_fn=None,
        ) -> Dict[str, str]:
    """把训练曲线写成 PNG（matplotlib 可用时）+ NPZ（无脑兜底）。

    Returns:
        ``{"png": <png_path or "">, "npz": <npz_path or "">}``
    """
    log = log_fn or (lambda _msg: None)
    out = {"png": "", "npz": ""}
    os.makedirs(persistence_dir, exist_ok=True)

    # NPZ 总是写（最稳）
    try:
        import numpy as _np
        npz_path = os.path.join(persistence_dir, BLB_TRAINING_CURVE_NPZ)
        _np.savez(
            npz_path,
            episode_returns=_np.asarray(episode_returns, dtype=float),
            best_reward_curve=(_np.asarray(best_reward_curve, dtype=float) if best_reward_curve else _np.array([], dtype=float)),
            ppo_loss_curve=(_np.asarray(ppo_loss_curve, dtype=float) if ppo_loss_curve else _np.array([], dtype=float)),
        )
        out["npz"] = npz_path
    except Exception as exc:
        log(f"  [BLB曲线][警告] 写 NPZ 失败：{exc}")

    # PNG 视 matplotlib 是否可用
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png_path = os.path.join(persistence_dir, BLB_TRAINING_CURVE_PNG)
        n_panels = 1 + (1 if best_reward_curve else 0) + (1 if ppo_loss_curve else 0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), squeeze=False)
        ax_idx = 0
        # 标题/坐标统一用 ASCII：matplotlib 默认 DejaVu Sans 不含 CJK 字形，
        # 写中文会触发一堆 UserWarning 且 PNG 上变成方框。中文说明在 markdown
        # 报告里给。
        ax = axes[ax_idx, 0]
        if episode_returns:
            ax.plot(range(1, len(episode_returns) + 1), episode_returns, linewidth=0.8, label="episode return")
            ax.set_xlabel("episode")
            ax.set_ylabel("return")
            ax.set_title("BLB Stage 2 RL: per-episode reward")
            ax.grid(True, alpha=0.3)
            ax.legend()
        ax_idx += 1
        if best_reward_curve:
            ax = axes[ax_idx, 0]
            ax.plot(range(1, len(best_reward_curve) + 1), best_reward_curve, color="tab:orange", linewidth=1.0, label="best reward so far")
            ax.set_xlabel("episode")
            ax.set_ylabel("best reward")
            ax.set_title("Best reward over training")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_idx += 1
        if ppo_loss_curve:
            ax = axes[ax_idx, 0]
            ax.plot(range(1, len(ppo_loss_curve) + 1), ppo_loss_curve, color="tab:red", linewidth=1.0, label="policy_loss")
            ax.set_xlabel("PPO update")
            ax.set_ylabel("loss")
            ax.set_title("PPO policy loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_idx += 1
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        out["png"] = png_path
    except Exception as exc:
        log(f"  [BLB曲线][信息] 跳过 PNG（matplotlib 不可用 / 渲染失败）：{exc}")
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
        lines.append("## 5. 最优 action 向量")
        lines.append("")
        lines.append(f"- 长度: {len(best_action_vec)}")
        lines.append("")
        lines.append("```")
        lines.append(", ".join(str(int(x)) for x in best_action_vec))
        lines.append("```")
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
    lines.append("> 持久化目录：`Parting Chapter/<run>/blb_stage2/progress/`。"
                 "live checkpoint / final checkpoint / best_cfg.pkl / 状态板 / "
                 "训练曲线（PNG + NPZ）/ 本报告 都在该目录下。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
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
            f.write("\n".join(lines))
    except Exception as inner_exc:
        log(f"  [BLB崩溃归档][警告] 写 {path} 失败：{inner_exc}")
        return ""
    return path
