"""Smoke test for ``blb_stage2_rl/persistence.py``：状态板 / 曲线 / 报告 / 崩溃归档。

不依赖 torch 也不依赖 evaluator —— 只测 persistence 模块自己。

跑法：
    python tests/test_blb_persistence.py
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
import types
import traceback


def main() -> int:
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    sys.path.insert(0, ROOT)

    # 直接 import 文件，绕开 blb_stage2_rl/__init__.py（它会拉 runner → torch）
    import importlib.util
    pers_path = os.path.join(ROOT, "blb_stage2_rl", "persistence.py")
    spec = importlib.util.spec_from_file_location("blb_stage2_rl_persistence", pers_path)
    pers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pers
    spec.loader.exec_module(pers)
    BLBStatusBoard = pers.BLBStatusBoard
    write_training_curves = pers.write_training_curves
    write_blb_final_report = pers.write_blb_final_report
    dump_crash_report = pers.dump_crash_report

    tmp = tempfile.mkdtemp(prefix="blb_pers_test_")
    try:
        # 1) 状态板 init + 各 setter + flush
        board = BLBStatusBoard(
            tmp, total_episodes=10, profile="mrpc", run_basename="unit_test_run",
            extra_meta={"smoke": True},
        )
        board.set_phase("校准 baseline")
        board.set_baseline({"total_bits_sum": 2520, "total_fusion_count": 3, "avg_k": 13})
        board.set_phase("训练中")
        for i in range(10):
            board.update_after_episode(i + 1, reward=-0.5 + 0.1 * i)
        board.update_after_ppo_update(1, {"policy_loss": 0.42, "value_loss": 0.13, "entropy": 1.2})
        board.set_best(0.31, best_action_vec=[0, 1, 2, 3], best_breakdown={"reward": 0.31, "r_bits": 0.2})
        board.set_phase("已完成")
        board.flush()
        with open(board.path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["schema"] == "blb_stage2_status_v1"
        assert doc["completed_episodes"] == 10
        assert doc["best"]["reward"] == 0.31
        assert doc["best"]["action_vec"] == [0, 1, 2, 3]
        assert doc["baseline"]["total_bits_sum"] == 2520
        assert doc["ppo_last_metrics"]["policy_loss"] == 0.42
        print(f"[OK] 状态板 JSON 写入并字段完整：{board.path}")

        # 2) 训练曲线（NPZ 必写；PNG 可选）
        ep_returns = [-0.5 + 0.1 * i for i in range(10)]
        best_curve = []
        bb = -1.0
        for r in ep_returns:
            bb = max(bb, r)
            best_curve.append(bb)
        ploss = [0.42, 0.35]
        out = write_training_curves(
            tmp,
            episode_returns=ep_returns,
            best_reward_curve=best_curve,
            ppo_loss_curve=ploss,
        )
        assert out["npz"], "NPZ 必须要写"
        assert os.path.isfile(out["npz"])
        print(f"[OK] 训练曲线 NPZ：{out['npz']}  PNG：{out['png'] or '(matplotlib 未安装，已跳过)'}")

        # 3) 最终报告
        report_path = write_blb_final_report(
            tmp,
            run_basename="unit_test_run",
            profile="mrpc",
            total_episodes=10,
            completed_episodes=10,
            elapsed_sec=12.34,
            best_reward=0.31,
            best_breakdown={"reward": 0.31, "r_bits": 0.2, "r_fusion": 0.1, "r_k": 0.01},
            best_action_vec=[0, 1, 2, 3, 0, 1],
            baseline={"total_bits_sum": 2520, "total_fusion_count": 3, "avg_k": 13.0,
                      "loss_mean": 0.85, "metric1_mean": 0.78},
            reward_weights={"w_bits": 0.001, "w_fusion": 0.1, "w_k": 0.05,
                            "acc_threshold": 0.77, "stab_threshold": 0.15},
            episode_returns=ep_returns,
            rescale_invoker_kind="heuristic",
        )
        assert os.path.isfile(report_path)
        content = open(report_path, "r", encoding="utf-8").read()
        assert "BLB Stage 2 RL 训练报告" in content
        assert "最优 reward" in content
        assert "Baseline" in content
        assert "0.31" in content
        print(f"[OK] 最终 markdown 报告：{report_path} ({len(content)} chars)")

        # 4) 崩溃归档
        try:
            raise RuntimeError("人为触发的测试异常")
        except RuntimeError as exc:
            err_path = dump_crash_report(
                tmp,
                exc=exc,
                last_state={"completed_episodes": 5, "best_reward": 0.12, "phase": "训练循环崩溃"},
            )
        assert os.path.isfile(err_path)
        err = open(err_path, "r", encoding="utf-8").read()
        assert "BLB Stage 2 RL 崩溃归档" in err
        assert "RuntimeError" in err
        assert "Traceback" in err
        assert "completed_episodes" in err
        print(f"[OK] 崩溃归档：{err_path}")

        # 5) 列出生成的文件
        print("\n生成文件清单：")
        for f in sorted(os.listdir(tmp)):
            full = os.path.join(tmp, f)
            print(f"  {f}  ({os.path.getsize(full)} bytes)")

        print("\n" + "=" * 70)
        print(" persistence smoke test 全部通过")
        print("=" * 70)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
