"""BLB Stage 2 强化学习（加强版）。

本模块在 BLB 五个 block 与 first-input fresh 噪声候选点上做联合 RL 决策：
为每层每个噪声点选择 (scaling_factor, truncation_k, rotation_flags)，
在精度 / 稳定性硬约束下最小化部署侧的 CKKS + MPC 总开销。

与旧版 ``noise_rl_module_v2.NoiseRLModuleV2`` 的关系：
  * 完全独立的代码路径，不复用旧版 PPO / state / 单 N 噪声表。
  * 通过 ``BLBNoiseRLBridge`` 接 ``function_handler.ReversibleLayerHandler``
    的多 N BLB 噪声安装入口；BLB 与 legacy 的互斥校验由 handler 内部完成。
  * 通过 ``RescaleOptimizerBridge`` 接真实 ``Rescale_optimizer`` 子项目得到
    cost 信号；训练路径不再提供 heuristic/stub fallback，初始化失败会直接中止。
  * 顶层入口 ``BLBStage2RLRunner.run(...)`` 返回的 dict 与旧版兼容，
    使 ``LayerImportanceEvaluator.run_unified_final_eval`` 等下游消费保持不变。

CLI / Python 调用：
    from blb_stage2_rl.training import BLBStage2RLRunner
    runner = BLBStage2RLRunner(evaluator)
    result = runner.run(fixed_gelu, fixed_softmax, fixed_label, fixed_source)

详见 ``docs/BLB_stage2_rl_spec.md``。
"""

__all__ = ["BLBStage2RLRunner"]


def __getattr__(name):
    if name == "BLBStage2RLRunner":
        from .training import BLBStage2RLRunner

        return BLBStage2RLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
