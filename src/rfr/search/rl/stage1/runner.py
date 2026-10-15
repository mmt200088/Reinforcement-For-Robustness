"""Stage-1 baseline, search loop and selected-config handoff; no Stage-2 execution."""

from __future__ import annotations

from collections import deque
import time
import math
from typing import Mapping
import numpy as np
import torch
import torch.optim as optim
from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT
from rfr.search.runtime.control import (
    PROGRESS_BOX_PPO_INTERVAL as NOISE_RL_PROGRESS_BOX_PPO_INTERVAL,
    STOP_FLAG_FILENAME as NOISE_STAGE_STOP_FLAG_FILENAME,
    consume_stop_flag as consume_stop_flag_file,
    format_eta_finish as _fmt_eta_finish,
    graceful_stop_requested as is_graceful_stop_requested,
    install_graceful_stop_handler,
    log_box as _log_rounded_box,
    reset_graceful_stop_state,
    uninstall_graceful_stop_handler,
    write_warning_report as _write_warning_report,
)
from rfr.common.report_format_utils import format_elapsed as _fmt_elapsed
from rfr.common.report_format_utils import progress_bar as _progress_bar
from rfr.search.runtime.elastic_gpu import (
    ElasticGPUFailure,
    is_recoverable_gpu_failure,
    raise_if_elastic_gpu_restart_requested,
)
from rfr.search.common.data_points import (
    RLDataPointWriter,
    make_unique_run_id,
)
import os
import hashlib
from rfr.search.rl.stage1 import config as stage1_config
from rfr.search.rl.stage1 import training as stage1_training
from rfr.search.rl.stage1.policy import GTrXLStrategyNetwork
from rfr.search.rl.stage1.rollout import (
    RecurrentRolloutBuffer,
    _get_stage1_gelu_mask_templates,
    _stage1_prob_tensors_to_nested_lists,
    _stage1_scalar_tensors_to_float_list,
)
from rfr.search.rl.stage1.env import TransformerOptEnv
from rfr.search.common.run_artifacts import update_persistent_metadata_stage

def run_stage1(workflow):
    """Run Stage 1 and return its selected config and exact baseline vectors.

    Mutate the existing workflow/checkpoint state; never launch Stage 2 here.
    The normal-completion, interruption and comparator rules are unchanged.
    """
    workflow.log("\n" + "="*60)
    workflow.log("开始配置评估（STARTING CONFIGURATION EVALUATION）")
    if workflow.skip_stage1_rl:
        workflow.log("[信息] 第一阶段RL搜索已跳过（--skip-stage1-rl）。")
    workflow.log("="*60)

    base_gelu, base_softmax = workflow.get_stage1_exact_baseline_configuration()
    cost_ref_gelu, cost_ref_softmax = workflow.get_stage1_cost_reference_configuration()
    base_tot_c, base_g_c, base_s_c = workflow.get_simulated_cost(cost_ref_gelu, cost_ref_softmax)
    num_metrics = workflow.get_num_metrics()
    base_loss = base_p = base_s = None
    reward_reference_split = workflow.get_reward_reference_split_name()

    if workflow.skip_stage1_rl:
        workflow.log("\n--- 阶段1（Phase 1）: 已跳过（SKIPPED）（--skip-stage1-rl） ---")
        workflow.log("[信息] 基线建立已跳过，因其仅用于第一阶段RL搜索。")
        if workflow.run_output_dir:
            update_persistent_metadata_stage(
                workflow.run_output_dir, "stage1_search", "skipped")

    else:


        workflow.log(
            "\n--- 阶段1（Phase 1）: 建立基线 "
            "(Establishing Baseline on train_probe) ---"
        )
        if reward_reference_split != TRAIN_PROBE_SPLIT:
            raise RuntimeError(
                "Stage-1 baseline must use train_probe, got "
                f"{reward_reference_split!r}."
            )


        try:
            from rfr.search.runtime.probe_runner import (
                enable_cuda_reward_probe_fast_math,
            )
            enable_cuda_reward_probe_fast_math()
            if torch.cuda.is_available():
                workflow.log(
                    "  [stage1-eval] TF32 fast matmul enabled"
                    "（与 Stage-2 reward probe 同一设置）"
                )
        except Exception as _tf32_exc:
            workflow.log(f"  [stage1-eval] TF32 enable skipped: {_tf32_exc!r}")

        base_loss_val, base_p_val, base_s_val, base_time_val = workflow.stage1_evaluate(
            base_gelu,
            base_softmax,
            split=reward_reference_split,
        )

        workflow.log(f"基线指标（Baseline Metrics）（{reward_reference_split}）：")
        workflow.log(f"  {workflow._fmt_metrics(base_loss_val, base_p_val, base_s_val)}")
        workflow.log("  基线配置（Baseline config）: 原始明文 GELU/Softmax（未安装多项式替换）")
        workflow.log(
            f"  成本参考（Cost reference）: GELU=4 / Softmax=6, "
            f"仿真成本（Sim Cost）={base_tot_c:.2f} "
            f"(GELU成本G={base_g_c:.2f}, Softmax成本S={base_s_c:.2f})"
        )

        base_loss = base_loss_val
        base_p = base_p_val
        base_s = base_s_val


        limit_loss = base_loss * (1.0 + workflow.error_threshold)
        limit_p = base_p * (1.0 - workflow.correlation_drop_ratio)
        limit_s = base_s * (1.0 - workflow.correlation_drop_ratio)

        workflow.log("约束条件（Constraints）（基于 train_probe）：")
        workflow.log(f"  {workflow._fmt_constraints(limit_loss, limit_p, limit_s)}")


    episode_rewards = []
    episode_losses = []
    episode_metric1s = []
    episode_metric2s = []
    episode_entropies = []
    best_config = None
    search_best_config = None
    global_best_config = None
    best_reward = float("-inf")
    best_cost = float("inf")
    stage1_completed_episodes = 0
    stage1_comparator_result = None

    if (
            not workflow.skip_stage1_rl
            and workflow.blb_v3_search_backend != "ppo"
    ):
        from rfr.common.json_utils import read_json_file
        from rfr.search.comparators.common.stage1_core import (
            Stage1Constraints,
            stage1_comparator_search_config,
            validate_stage1_comparator_setup,
        )
        from rfr.search.comparators.common.stage1_runner import (
            Stage1SearchGracefulStop,
            build_stage1_search_accounting,
            load_completed_search_result,
            run_stage1_search,
        )
        backend = workflow.blb_v3_search_backend
        if int(num_metrics) != 2:
            raise RuntimeError(
                "MRPC Stage-1 comparator requires accuracy and weighted_f1"
            )
        stage1_constraints = Stage1Constraints.from_baseline(
            baseline_loss=float(base_loss),
            baseline_metrics=(float(base_p), float(base_s)),
            loss_relative_tolerance=0.001,
            metric_relative_tolerance=0.001,
            metric_names=("accuracy", "weighted_f1"),
        )
        if (
                not math.isclose(
                    stage1_constraints.loss_max,
                    float(limit_loss),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or tuple(stage1_constraints.metric_mins)
                != (float(limit_p), float(limit_s))
        ):
            raise RuntimeError(
                "Stage-1 comparator runtime thresholds do not match the "
                "0.1% MRPC limits"
            )
        stage1_search_config = stage1_comparator_search_config(
            backend,
            bo_no_improvement_patience=(
                workflow.comparator_bo_stage1_no_improvement
            ),
            greedy_no_improvement_rounds=(
                workflow.comparator_greedy_stage1_no_improvement_rounds
            ),
            ga_update_generations=(
                workflow.comparator_ga_stage1_generations
            ),
        )
        validate_stage1_comparator_setup(
            backend=backend,
            config=stage1_search_config,
            num_layers=int(workflow.total_layers),
            constraints=stage1_constraints,
        )
        stage1_output_dir = os.path.join(
            workflow.run_output_dir
            or os.path.dirname(workflow.stage1_step_info_file),
            "stage1_comparator",
            backend,
        )
        preload_path = os.path.join(stage1_output_dir, "checkpoint.json")
        if not os.path.isfile(preload_path):
            preload_path = None
        stage1_comparator_stop_flag_path = os.path.join(
            stage1_output_dir,
            NOISE_STAGE_STOP_FLAG_FILENAME,
        )
        reset_graceful_stop_state()
        consume_stop_flag_file(stage1_comparator_stop_flag_path)
        install_graceful_stop_handler(log_fn=workflow.log)

        def stage1_comparator_stop_requested():
            return is_graceful_stop_requested(
                stage1_comparator_stop_flag_path
            )

        workflow.log(
            f"\n--- 阶段2（Phase 2）: {backend} Stage-1 search "
            "(train_probe) ---"
        )
        workflow.log(
            "  [优雅停止] 可发送一次 SIGINT，或创建 "
            f"{stage1_comparator_stop_flag_path}；当前候选完成并落盘后退出。"
        )
        try:
            in_memory_stage1_result = run_stage1_search(
                backend=backend,
                evaluator=workflow,
                num_layers=int(workflow.total_layers),
                constraints=stage1_constraints,
                config=stage1_search_config,
                output_dir=stage1_output_dir,
                manifest={
                    "backend": backend,
                    "model": str(getattr(
                        getattr(workflow.model, "config", None),
                        "_name_or_path",
                        type(workflow.model).__name__,
                    )),
                    "dataset": str(workflow.data_path),
                    "split": TRAIN_PROBE_SPLIT,
                    "dataset_protocol_hash": workflow.dataset_protocol_hash,
                    "stage1_bound_into_stage2": not workflow.comparator_stage1_only,
                    "stage2_backend": (
                        None if workflow.comparator_stage1_only else backend
                    ),
                },
                preload_path=preload_path,
                stop_requested=stage1_comparator_stop_requested,
            )
        except Stage1SearchGracefulStop as stopped:
            consume_stop_flag_file(stage1_comparator_stop_flag_path)
            if workflow.run_output_dir:
                update_persistent_metadata_stage(
                    workflow.run_output_dir,
                    "stage1_search",
                    "in_progress",
                    extra_fields={
                        "backend": backend,
                        "stopped_by": "graceful_stop",
                        "completed_evaluations": int(
                            stopped.observation_count
                        ),
                        "checkpoint_path": preload_path or os.path.join(
                            stage1_output_dir, "checkpoint.json"
                        ),
                    },
                )
            workflow.log(
                "  [优雅停止] Stage-1 comparator checkpoint 已安全落盘；"
                "下次用相同参数、不带 --fresh 启动即可续跑。"
            )
            raise SystemExit(0)
        finally:
            uninstall_graceful_stop_handler()
        stage1_comparator_result = load_completed_search_result(
            stage1_output_dir
        )
        if (
                stage1_comparator_result.as_dict()
                != in_memory_stage1_result.as_dict()
        ):
            raise RuntimeError(
                "in-memory Stage-1 result does not match its completed "
                "ordinary artifacts"
            )
        selected_stage1 = stage1_comparator_result.best
        if not selected_stage1.valid:
            raise RuntimeError(
                "Stage-1 comparator produced no valid configuration"
            )
        if (
                backend == "greedy"
                and stage1_comparator_result.termination_reason
                != "consecutive_no_improvement_rounds"
        ):
            raise RuntimeError(
                "Stage-1 Greedy did not verify every 1-opt and 2-opt "
                "local optimum"
            )
        if (
                backend == "coinn_ga"
                and (
                    stage1_comparator_result.termination_reason
                    != "completed_generations"
                    or int(
                        stage1_comparator_result.config.ga_update_generations
                    ) != workflow.comparator_ga_stage1_generations
                )
        ):
            raise RuntimeError(
                "Stage-1 COINN-GA did not satisfy its configured full-generation "
                "contract"
            )

        stage1_result_path = os.path.join(
            stage1_output_dir, "result.json",
        )
        with open(stage1_result_path, "rb") as result_handle:
            stage1_result_sha256 = hashlib.sha256(
                result_handle.read()
            ).hexdigest()
        stage1_manifest = read_json_file(
            os.path.join(stage1_output_dir, "manifest.json")
        )
        if not isinstance(stage1_manifest, Mapping):
            raise RuntimeError("Stage-1 manifest must be a JSON object")
        best_config = {
            "gelu": np.asarray(
                selected_stage1.gelu_degrees, dtype=int,
            ),
            "softmax": np.asarray(
                selected_stage1.softmax_degrees, dtype=int,
            ),
            "feasible": bool(selected_stage1.feasible),
            "selection_status": (
                "feasible"
                if selected_stage1.feasible
                else "least_violating"
            ),
            "backend": backend,
            "dataset_protocol_hash": workflow.dataset_protocol_hash,
            "evaluation": selected_stage1.as_dict(),
            "result_path": stage1_result_path,
            "result_sha256": stage1_result_sha256,
            "search_accounting": build_stage1_search_accounting(
                result=stage1_comparator_result,
                manifest=stage1_manifest,
            ),
        }
        search_best_config = best_config
        global_best_config = best_config
        best_cost = float(selected_stage1.cost)
        stage1_completed_episodes = int(
            stage1_comparator_result.evaluation_count
        )
        if workflow.run_output_dir:
            update_persistent_metadata_stage(
                workflow.run_output_dir,
                "stage1_search",
                (
                    "completed"
                    if selected_stage1.feasible
                    else "completed_infeasible"
                ),
                extra_fields={
                    "backend": backend,
                    "feasible": bool(selected_stage1.feasible),
                    "evaluation_count": stage1_completed_episodes,
                    "result_path": stage1_result_path,
                },
            )


    if (
            not workflow.skip_stage1_rl
            and workflow.blb_v3_search_backend == "ppo"
    ):
        workflow.log("\n--- 阶段2（Phase 2）: PPO强化学习训练（PPO Reinforcement Learning Training） ---")

        step_info_details_dir = os.path.join(os.path.dirname(workflow.step_info_file), "details")
        os.makedirs(step_info_details_dir, exist_ok=True)
        step_info_chunk_file = [None]
        step_info_chunk_idx = [0]
        step_info_is_resuming = [False]


        step_info_chunk_anchor = [0]
        stage1_warning_file = os.path.join(os.path.dirname(workflow.step_info_file), "warning.txt")
        stage1_prev_avg_reward = [None]
        stage1_warnings = []

        def _get_stage1_chunk_filename(episode_1based):
            anchor = step_info_chunk_anchor[0]
            rel = episode_1based - anchor - 1
            chunk_start = anchor + (rel // stage1_config.STEP_INFO_CHUNK_SIZE) * stage1_config.STEP_INFO_CHUNK_SIZE + 1
            chunk_end = chunk_start + stage1_config.STEP_INFO_CHUNK_SIZE - 1
            return os.path.join(
                step_info_details_dir,
                f"ppo_step_info_{chunk_start}-{chunk_end}.txt",
            )

        def _open_stage1_chunk(episode_1based):
            target = _get_stage1_chunk_filename(episode_1based)
            anchor = step_info_chunk_anchor[0]
            new_idx = (episode_1based - anchor - 1) // stage1_config.STEP_INFO_CHUNK_SIZE
            if step_info_chunk_file[0] is not None and step_info_chunk_idx[0] == new_idx:
                return step_info_chunk_file[0]
            if step_info_chunk_file[0] is not None:
                step_info_chunk_file[0].close()
            chunk_start = anchor + new_idx * stage1_config.STEP_INFO_CHUNK_SIZE + 1
            chunk_end = chunk_start + stage1_config.STEP_INFO_CHUNK_SIZE - 1

            if step_info_is_resuming[0] and os.path.isfile(target):
                f = open(target, "a", encoding="utf-8")
                f.write(f"\n=== [续训 Resume] PPO StepInfo · 回合 {chunk_start}-{chunk_end} ===\n\n")
                step_info_is_resuming[0] = False
            else:
                f = open(target, "w", encoding="utf-8")
                f.write(f"=== PPO每步信息（StepInfo）回合 {chunk_start}-{chunk_end} ===\n\n")
                step_info_is_resuming[0] = False
            step_info_chunk_file[0] = f
            step_info_chunk_idx[0] = new_idx
            return f


        gtrxl_net = GTrXLStrategyNetwork(
            num_layers=workflow.total_layers,
            d_model=stage1_config.GTRXL_D_MODEL,
            n_heads=stage1_config.GTRXL_N_HEADS,
            n_gtrxl_layers=stage1_config.GTRXL_N_LAYERS,
            d_ff=stage1_config.GTRXL_D_FF,
            dropout=stage1_config.GTRXL_DROPOUT
        ).to(workflow.device)


        optimizer = optim.Adam(gtrxl_net.parameters(), lr=workflow.stage1_ppo_lr_initial)
        gtrxl_ppo_update_count = 0


        baseline_metrics = (base_loss, base_p, base_s)


        class RLEvaluatorWrapper:
            def __init__(wrapper_self, evaluator, split_name):
                wrapper_self.evaluator = evaluator
                wrapper_self.split_name = split_name

            def evaluate_model(wrapper_self, gelu_arr, softmax_arr):
                return wrapper_self.evaluator.stage1_evaluate(
                    gelu_arr,
                    softmax_arr,
                    split=wrapper_self.split_name,
                )

        online_reward_split = workflow.get_online_reward_split_name()
        proxy_base_loss, proxy_base_p, proxy_base_s, _ = workflow.stage1_evaluate(
            base_gelu,
            base_softmax,
            split=online_reward_split,
        )
        workflow.log(
            f"[信息] 使用 {online_reward_split} 进行在线奖励计算 "
            f"（约束保持在 {reward_reference_split} 上）"
        )
        rl_evaluator = RLEvaluatorWrapper(workflow, online_reward_split)

        env = TransformerOptEnv(
            workflow.total_layers,
            base_tot_c,
            baseline_metrics,
            rl_evaluator,
            constraint_limits={
                "loss": float(limit_loss),
                "metric1": float(limit_p),
                "metric2": float(limit_s),
            },
            num_metrics=workflow.get_num_metrics(),
        )
        env.prev_episode_metrics = {
            "loss": proxy_base_loss,
            "metric1": proxy_base_p,
            "metric2": proxy_base_s,
            "cost": base_tot_c,
        }

        _stage1_model_type = (
            "bert-large" if int(workflow.total_layers) == 24
            else "bert-base" if int(workflow.total_layers) == 12
            else f"layers-{int(workflow.total_layers)}"
        )
        _stage1_run_source = (
            workflow.run_output_dir
            or os.path.dirname(workflow.stage1_step_info_file)
            or f"{_stage1_model_type}-{workflow.data_path}"
        )
        try:
            _stage1_run_id_base = os.path.relpath(_stage1_run_source, os.getcwd())
        except ValueError:
            _stage1_run_id_base = str(_stage1_run_source)
        _stage1_resume_ckpt_path = workflow._get_stage1_resume_checkpoint_path()
        _stage1_resume_metadata = None
        if _stage1_resume_ckpt_path:
            _stage1_resume_metadata = torch.load(
                _stage1_resume_ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        _stage1_resume_run_id = (
            _stage1_resume_metadata.get("structured_run_id")
            if isinstance(_stage1_resume_metadata, dict)
            else None
        )
        _stage1_run_id = (
            str(_stage1_resume_run_id)
            if _stage1_resume_run_id
            else make_unique_run_id(_stage1_run_id_base)
        )
        stage1_data_writer = RLDataPointWriter(
            root_dir=os.path.join(
                os.path.dirname(workflow.stage1_step_info_file),
                "records",
            ),
            run_id=_stage1_run_id,
            stage="stage1",
            model_type=_stage1_model_type,
            dataset=str(workflow.data_path),
        )
        stage1_data_writer.write_manifest({
            "source_run_output_dir": workflow.run_output_dir,
            "source_data_run_id_base": _stage1_run_id_base,
            "stage1_step_info_file": workflow.stage1_step_info_file,
            "total_layers": int(workflow.total_layers),
            "search_algorithm": workflow.search_algorithm,
            "dataset_protocol_hash": workflow.dataset_protocol_hash,
            "rl_algo": "ppo",
            "stage1_episodes_requested": int(workflow.stage1_rl_episode_limit),
            "termination": "maximum_episodes",
            "ppo_update_interval": int(stage1_config.PPO_UPDATE_INTERVAL),
            "ppo_lr_initial": float(workflow.stage1_ppo_lr_initial),
            "stage1_rl_devices": workflow.stage1_rl_devices,
            "random_seed": int(getattr(workflow, "random_seed", 42)),
            "baseline": {
                "loss": float(base_loss),
                "metric1": float(base_p),
                "metric2": float(base_s),
                "cost_reference": float(base_tot_c),
                "gelu": np.asarray(base_gelu, dtype=int).tolist(),
                "softmax": np.asarray(base_softmax, dtype=int).tolist(),
            },
            "constraints": {
                "loss_max": float(limit_loss),
                "metric1_min": float(limit_p),
                "metric2_min": float(limit_s),
                "stage1_accuracy_tolerance": float(workflow.error_threshold),
                "stage1_metric_tolerance": float(workflow.correlation_drop_ratio),
            },
            "cost_reference": {
                "gelu": np.asarray(cost_ref_gelu, dtype=int).tolist(),
                "softmax": np.asarray(cost_ref_softmax, dtype=int).tolist(),
            },
            "action_space": {
                "gelu_map": stage1_config.GELU_MAP,
                "gelu_action_mask": stage1_config.STAGE1_GELU_ACTION_MASK.tolist(),
                "fixed_softmax_degree": int(stage1_config.FIXED_SOFTMAX_DEGREE),
            },
        })
        workflow.log(
            f"  [data-points] Stage-1 structured RL data → {stage1_data_writer.run_dir}"
        )


        _stage1_parallel_runner = stage1_training._build_stage1_parallel_runner(workflow, baseline_metrics=baseline_metrics,
            base_tot_c=base_tot_c,
            constraint_limits={
                "loss": float(limit_loss),
                "metric1": float(limit_p),
                "metric2": float(limit_s),
            },
            eval_split_name=online_reward_split,
            proxy_prev_metrics={
                "loss": proxy_base_loss,
                "metric1": proxy_base_p,
                "metric2": proxy_base_s,
                "cost": base_tot_c,
            },
        )
        if _stage1_parallel_runner is not None:
            workflow.log(
                f"  [multi-gpu] Stage-1 rollout enabled: "
                f"workers={_stage1_parallel_runner.num_workers} "
                f"devices={[str(w.device) for w in _stage1_parallel_runner.workers]} "
                f"episodes_per_worker={stage1_config.PPO_UPDATE_INTERVAL // _stage1_parallel_runner.num_workers}"
            )
        _stage1_parallel_stash = deque()
        _stage1_parallel_window_t0 = None
        _stage1_parallel_window_idx = None
        _stage1_parallel_collect_seconds = 0.0
        _stage1_parallel_model_forward_seconds = 0.0
        _stage1_parallel_model_forward_calls = 0
        _stage1_parallel_replay_seconds = 0.0
        _stage1_parallel_detail_seconds = 0.0

        buffer = RecurrentRolloutBuffer()


        best_reward = float('-inf')
        best_cost = float('inf')
        window_best_reward = float('-inf')
        window_best_cost = float('inf')
        window_best_config = None

        workflow.total_episodes = workflow.stage1_rl_episode_limit or stage1_config.PPO_MAX_EPISODES
        workflow._reset_runtime_ppo_state()


        from rfr.search.rl.stage1.checkpoint import (
            save_stage1_rl_checkpoint,
            load_stage1_rl_checkpoint,
            STAGE1_CHECKPOINT_FILENAME,
            recover_stage1_detail_files,
            stage1_detail_file_sizes,
        )
        stage1_checkpoint_path = os.path.join(
            os.path.dirname(workflow.stage1_step_info_file),
            STAGE1_CHECKPOINT_FILENAME,
        )

        stage1_stop_flag_path = os.path.join(
            os.path.dirname(workflow.stage1_step_info_file),
            NOISE_STAGE_STOP_FLAG_FILENAME,
        )
        reset_graceful_stop_state()
        consume_stop_flag_file(stage1_stop_flag_path)
        install_graceful_stop_handler(log_fn=workflow.log)
        workflow.log(
            f"  [优雅停止] 训练期间可按 Ctrl+C 或创建 {stage1_stop_flag_path} "
            f"触发安全停止（在下一回合边界保存 checkpoint 后退出）。"
        )
        stage1_resume_start_episode = 0
        _stage1_cuda_rng_role_registry = None
        if _stage1_resume_ckpt_path:
            _log_rounded_box(
                workflow.log,
                [
                    "断点续训（Resume Stage-1 from checkpoint）",
                    f"加载: {_stage1_resume_ckpt_path}",
                ],
            )


            ckpt = load_stage1_rl_checkpoint(
                _stage1_resume_ckpt_path,
                gtrxl_net,
                optimizer,
                device=workflow.device,
                expected_dataset_protocol_hash=workflow.dataset_protocol_hash,
            )
            _stage1_cuda_rng_role_registry = list(
                ckpt.get("cuda_rng_state_by_role") or ()
            )
            _checkpoint_run_id = ckpt.get("structured_run_id")
            if (
                _checkpoint_run_id is not None
                and str(_checkpoint_run_id) != stage1_data_writer.run_id
            ):
                raise RuntimeError(
                    "Stage-1 checkpoint structured run-id mismatch: "
                    f"{_checkpoint_run_id!r} != {stage1_data_writer.run_id!r}"
                )
            stage1_data_writer.recover_jsonl_files(
                ckpt.get("structured_jsonl_sizes")
            )
            recover_stage1_detail_files(
                step_info_details_dir,
                ckpt.get("detail_file_sizes"),
            )
            del _stage1_resume_metadata
            stage1_resume_start_episode = int(ckpt["completed_episodes"])
            gtrxl_ppo_update_count = int(ckpt["gtrxl_ppo_update_count"])
            episode_rewards = list(ckpt["episode_rewards"])
            episode_losses = list(ckpt["episode_losses"])
            episode_metric1s = list(ckpt["episode_metric1s"])
            episode_metric2s = list(ckpt["episode_metric2s"])
            episode_entropies = list(ckpt["episode_entropies"])
            best_reward = float(ckpt["best_reward"])
            best_cost = float(ckpt["best_cost"])
            best_config = ckpt.get("best_config")
            search_best_config = None
            global_best_config = None
            window_best_reward = float(ckpt.get("window_best_reward", float('-inf')))
            window_best_cost = float(ckpt.get("window_best_cost", float('inf')))
            window_best_config = ckpt.get("window_best_config")
            stage1_prev_avg_reward[0] = ckpt.get("stage1_prev_avg_reward")
            stage1_warnings = list(ckpt.get("stage1_warnings", []))
            _ev_rt = ckpt.get("ev_runtime_state", {})
            workflow.reward_history = deque(
                _ev_rt.get("reward_history", []),
                maxlen=stage1_config.RUNNING_REWARD_HISTORY_SIZE,
            )
            workflow.reward_mean = float(_ev_rt.get("reward_mean", 0.0))
            workflow.reward_std = float(_ev_rt.get("reward_std", 1.0))
            workflow._rebuild_reward_statistics_accumulators()
            workflow.current_episode = int(_ev_rt.get("current_episode", stage1_resume_start_episode))

            if "return_normalizer_mean" in _ev_rt:
                workflow.return_normalizer.mean = float(_ev_rt["return_normalizer_mean"])
                workflow.return_normalizer.var = float(_ev_rt["return_normalizer_var"])
                workflow.return_normalizer.count = float(_ev_rt["return_normalizer_count"])
            step_info_is_resuming[0] = True


            step_info_chunk_anchor[0] = stage1_resume_start_episode
            workflow.log(
                f"  已恢复至回合 {stage1_resume_start_episode}，"
                f"将从回合 {stage1_resume_start_episode + 1} "
                f"继续训练至 {workflow.stage1_rl_episodes}"
            )
            if (
                workflow.stage1_rl_episode_limit is not None
                and stage1_resume_start_episode >= workflow.stage1_rl_episode_limit
            ):
                workflow.log(
                    f"  ⚠ checkpoint 已完成 {stage1_resume_start_episode} 回合，"
                    f"目标回合数 {workflow.stage1_rl_episode_limit} 无需追加训练。"
                )

        _stage1_rl_t0 = time.time()
        stage1_completed_episodes = int(stage1_resume_start_episode)
        stage1_stop_reason = "max_episodes"
        _stage1_episode_iter = range(
            stage1_resume_start_episode, workflow.stage1_rl_episode_limit,
        )
        for episode in _stage1_episode_iter:

            current_lr, current_entropy = workflow.update_hyperparameters(optimizer, episode)


            _handled_via_parallel = False
            if _stage1_parallel_runner is not None:
                if not _stage1_parallel_stash:
                    _window_idx_for_runner = episode // stage1_config.PPO_UPDATE_INTERVAL
                    _remaining_total = workflow.stage1_rl_episode_limit - episode
                    _window_size = min(stage1_config.PPO_UPDATE_INTERVAL, _remaining_total)
                    _stage1_parallel_window_t0 = time.time()
                    _stage1_parallel_window_idx = _window_idx_for_runner
                    _stage1_parallel_replay_seconds = 0.0
                    _stage1_parallel_detail_seconds = 0.0
                    _stage1_parallel_collect_t0 = time.time()


                    _rollouts = _stage1_parallel_runner.run_window(
                        gtrxl_net=gtrxl_net,
                        total_episodes=_window_size,
                        window_idx=_window_idx_for_runner,
                        base_seed=int(getattr(workflow, "random_seed", 42)),
                    )
                    _stage1_parallel_collect_seconds = (
                        time.time() - _stage1_parallel_collect_t0
                    )
                    _stage1_parallel_stash.extend(_rollouts)
                    from rfr.search.rl.stage1.parallel_runner import format_diagnostics_line

                    if _stage1_parallel_runner.last_diagnostics is not None:
                        _stage1_parallel_model_forward_seconds = float(
                            _stage1_parallel_runner.last_diagnostics.model_forward_seconds
                        )
                        _stage1_parallel_model_forward_calls = int(
                            _stage1_parallel_runner.last_diagnostics.model_forward_calls
                        )
                        workflow.log(
                            "  " + format_diagnostics_line(
                                _stage1_parallel_runner.last_diagnostics
                            )
                        )
                    workflow.log(
                        f"  [stage1-rollout] window={_window_idx_for_runner} collected "
                        f"{len(_rollouts)} episodes across "
                        f"{_stage1_parallel_runner.num_workers} workers"
                    )
                    _shared_eval_cache = getattr(
                        workflow, "_stage1_worker_eval_cache", None
                    )
                    if _shared_eval_cache is not None:
                        workflow.log(
                            f"  [stage1-rollout] window={_window_idx_for_runner} "
                            + _shared_eval_cache.stats_line()
                        )
                rollout = _stage1_parallel_stash.popleft()
                _stage1_parallel_replay_t0 = time.time()

                buffer.start_episode()
                for _k in range(len(rollout.actions_g)):
                    buffer.add_step(
                        cont_feat=rollout.cont_features[_k],
                        layer_idx=rollout.layer_indices[_k],
                        prev_g=rollout.prev_g_actions[_k],
                        action_g=rollout.actions_g[_k],
                        logprob=rollout.logprobs[_k],
                        reward=rollout.rewards[_k],
                        value=rollout.values[_k],
                        done=rollout.dones[_k],
                        gelu_mask=rollout.gelu_masks[_k],
                    )
                buffer.end_episode()
                episode_reward = float(rollout.episode_reward)
                step_infos = [dict(_si) for _si in rollout.step_infos]
                for _so_idx, _si in enumerate(step_infos):
                    _si["step_global"] = episode * workflow.total_layers + _so_idx
                    _si["episode_id"] = episode
                    _si["current_lr"] = current_lr
                    _si["current_entropy_coef"] = current_entropy


                env.gelu_config = list(rollout.gelu_config)
                env.softmax_config = list(rollout.softmax_config)
                env.accumulated_cost = float(rollout.episode_cost)
                env.current_episode_metrics = (
                    dict(rollout.final_config_metrics)
                    if rollout.final_config_metrics is not None
                    else None
                )
                _stage1_parallel_replay_seconds += (
                    time.time() - _stage1_parallel_replay_t0
                )
                _handled_via_parallel = True

            if not _handled_via_parallel:

                state = env.reset()
                prev_g_idx = stage1_config.SOS_TOKEN_GELU


                seq_cont_feats = torch.empty(
                    (1, workflow.total_layers, stage1_config.POLICY_CONTINUOUS_DIM),
                    dtype=torch.float32,
                    device=workflow.device,
                )
                seq_layer_indices = torch.empty(
                    (1, workflow.total_layers), dtype=torch.long, device=workflow.device,
                )
                seq_prev_g = torch.empty(
                    (1, workflow.total_layers), dtype=torch.long, device=workflow.device,
                )
                seq_gelu_masks = torch.empty(
                    (1, workflow.total_layers, int(stage1_config.STAGE1_GELU_ACTION_MASK.size)),
                    dtype=torch.bool,
                    device=workflow.device,
                )
                gelu_mask_np, gelu_mask_t = _get_stage1_gelu_mask_templates(workflow.device)
                seq_gelu_masks.copy_(gelu_mask_t.view(1, 1, -1).expand_as(seq_gelu_masks))

                episode_reward = 0
                step_infos = []
                transition_records = []
                logprob_tensors = []
                value_tensors = []
                gelu_prob_tensors = []
                buffer.start_episode()

                for step in range(workflow.total_layers):


                    N = workflow.total_layers
                    layer_idx = int(np.argmax(state[0:N]))
                    cont_feat_np = env.get_policy_cont_features()
                    cont_feat_record = np.asarray(cont_feat_np, dtype=np.float32)

                    cont_feat_t = torch.as_tensor(
                        cont_feat_record, dtype=torch.float32, device=workflow.device
                    )


                    seq_cont_feats[0, step].copy_(cont_feat_t)
                    seq_layer_indices[0, step] = int(layer_idx)
                    seq_prev_g[0, step] = int(prev_g_idx)


                    full_cont = seq_cont_feats[:, : step + 1, :]
                    full_layer = seq_layer_indices[:, : step + 1]
                    full_prev_g = seq_prev_g[:, : step + 1]
                    full_gelu_mask = seq_gelu_masks[:, : step + 1, :]


                    with torch.no_grad():
                        gelu_action, logprob, value, gelu_probs =\
                            gtrxl_net.get_action_and_logprob(
                                full_cont, full_layer, full_prev_g,
                                return_probs=True, gelu_mask=full_gelu_mask
                            )


                    gelu_action_idx = int(gelu_action.item())
                    next_state, reward, done, info = env.step(gelu_action_idx)
                    logprob_tensors.append(logprob.detach())
                    value_tensors.append(value.detach())
                    gelu_prob_tensors.append(gelu_probs.detach())


                    step_info = {
                        'step_global': episode * workflow.total_layers + step,
                        'episode_id': episode,
                        'layer_index': info['layer_index'],
                        'state_vector': state.tolist(),
                        'gelu_action_index': gelu_action_idx,
                        'gelu_action_degree': int(stage1_config.GELU_MAP[gelu_action_idx]),
                        'step_reward': float(reward),
                        'done': bool(done),
                        'logprob': None,
                        'curr_gelu_degree': info['curr_gelu_degree'],
                        'curr_softmax_degree': info['curr_softmax_degree'],
                        'gelu_prob_dist': None,
                        'critic_value': None,
                        'accumulated_cost': info['accumulated_cost'],
                        'gelu_config': info['gelu_config'],
                        'softmax_config': info['softmax_config'],
                        'current_lr': current_lr,
                        'current_entropy_coef': current_entropy
                    }
                    step_infos.append(step_info)

                    transition_records.append(
                        (
                            cont_feat_record,
                            layer_idx,
                            prev_g_idx,
                            gelu_action_idx,
                            reward,
                            float(done),
                            gelu_mask_np,
                        )
                    )


                    prev_g_idx = gelu_action_idx

                    episode_reward += reward
                    state = next_state

                logprob_values = _stage1_scalar_tensors_to_float_list(logprob_tensors)
                critic_values = _stage1_scalar_tensors_to_float_list(value_tensors)
                gelu_prob_dists = _stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)
                for idx, step_info in enumerate(step_infos):
                    step_info["logprob"] = logprob_values[idx]
                    step_info["critic_value"] = critic_values[idx]
                    step_info["gelu_prob_dist"] = gelu_prob_dists[idx]


                for idx, (
                    cont_feat_record,
                    layer_idx,
                    prev_g_record,
                    action_g_record,
                    reward_record,
                    done_record,
                    gelu_mask_record,
                ) in enumerate(transition_records):
                    buffer.add_step(
                        cont_feat=cont_feat_record,
                        layer_idx=layer_idx,
                        prev_g=prev_g_record,
                        action_g=action_g_record,
                        logprob=logprob_values[idx],
                        reward=reward_record,
                        value=critic_values[idx],
                        done=done_record,
                        gelu_mask=gelu_mask_record,
                    )
                buffer.end_episode()
            episode_rewards.append(episode_reward)
            stage1_completed_episodes = episode + 1


            if hasattr(env, 'current_episode_metrics') and env.current_episode_metrics is not None:
                episode_losses.append(env.current_episode_metrics['loss'])
                episode_metric1s.append(env.current_episode_metrics['metric1'])
                episode_metric2s.append(env.current_episode_metrics['metric2'])
            else:
                episode_losses.append(base_loss)
                episode_metric1s.append(base_p)
                episode_metric2s.append(base_s)


            workflow.update_reward_statistics(episode_reward)

            _stage1_detail_t0 = time.time()
            chunk_f = _open_stage1_chunk(episode + 1)
            chunk_f.write(f"--- 回合（Episode） {episode + 1} (奖励Reward={episode_reward:.4f}) ---\n")
            for si in step_infos:
                workflow._write_step_info(si, chunk_f)
                chunk_f.write("\n")
                stage1_data_writer.write_step(si)
            chunk_f.flush()
            if _handled_via_parallel:
                _stage1_parallel_detail_seconds += (
                    time.time() - _stage1_detail_t0
                )


            final_config = {
                'gelu': np.array(env.gelu_config),
                'softmax': np.array(env.softmax_config),
                'cost': env.accumulated_cost,
                'reward': episode_reward,
                'dataset_protocol_hash': workflow.dataset_protocol_hash,
            }

            if episode_reward > window_best_reward:
                window_best_reward = episode_reward
                window_best_cost = env.accumulated_cost
                window_best_config = {
                    'gelu': np.array(env.gelu_config),
                    'softmax': np.array(env.softmax_config),
                    'cost': env.accumulated_cost,
                    'reward': episode_reward,
                    'dataset_protocol_hash': workflow.dataset_protocol_hash,
                }

            _stage1_is_new_best = bool(episode_reward > best_reward)
            if _stage1_is_new_best:
                best_reward = episode_reward
                best_cost = env.accumulated_cost
                best_config = final_config.copy()
                workflow.log(f"  回合（Episode） {episode+1}: 新最优！（New Best!） 奖励（Reward）={episode_reward:.4f}, 成本（Cost）={env.accumulated_cost:.2f}")
                workflow.log(f"    GELU配置: {env.gelu_config}")
                workflow.log(f"    Softmax配置: {env.softmax_config}")

            _episode_metrics = (
                dict(env.current_episode_metrics)
                if getattr(env, "current_episode_metrics", None) is not None
                else {
                    "loss": base_loss,
                    "metric1": base_p,
                    "metric2": base_s,
                    "cost": env.accumulated_cost,
                }
            )
            stage1_data_writer.write_episode({
                "episode": int(episode + 1),
                "episode_zero_based": int(episode),
                "split": online_reward_split,
                "dataset_protocol_hash": workflow.dataset_protocol_hash,
                "reward": float(episode_reward),
                "loss": float(_episode_metrics["loss"]),
                "metric1": float(_episode_metrics["metric1"]),
                "metric2": float(_episode_metrics["metric2"]),
                "cost": float(env.accumulated_cost),
                "gelu": np.asarray(env.gelu_config, dtype=int).tolist(),
                "softmax": np.asarray(env.softmax_config, dtype=int).tolist(),
                "current_lr": float(current_lr),
                "current_entropy_coef": float(current_entropy),
                "reward_mean": float(workflow.reward_mean),
                "reward_std": float(workflow.reward_std),
                "is_new_best": _stage1_is_new_best,
                "best_reward_so_far": float(best_reward),
                "best_cost_so_far": float(best_cost),
                "best_gelu_so_far": (
                    np.asarray(best_config["gelu"], dtype=int).tolist()
                    if best_config is not None else None
                ),
                "best_softmax_so_far": (
                    np.asarray(best_config["softmax"], dtype=int).tolist()
                    if best_config is not None else None
                ),
            })


            if (episode + 1) % stage1_config.PPO_UPDATE_INTERVAL == 0:
                _stage1_ppo_update_t0 = time.time()
                try:
                    policy_loss, value_loss, entropy = stage1_training.ppo_update_gtrxl(workflow, gtrxl_net, optimizer, buffer, workflow.device,
                        entropy_coef=current_entropy,
                        ppo_update_step=gtrxl_ppo_update_count
                    )
                except ElasticGPUFailure:
                    raise
                except Exception as exc:
                    if not is_recoverable_gpu_failure(exc):
                        raise
                    raise ElasticGPUFailure(
                        device=workflow.device,
                        role="learner-primary",
                        operation="stage1_ppo_update",
                        cause=exc,
                    ) from exc
                _stage1_ppo_update_seconds = time.time() - _stage1_ppo_update_t0
                gtrxl_ppo_update_count += 1
                buffer.clear()
                episode_entropies.append(entropy)
                _stage1_parallel_update_payload = None
                if (
                        _stage1_parallel_runner is not None
                        and _stage1_parallel_window_t0 is not None
                        and _stage1_parallel_window_idx is not None):
                    _diag = _stage1_parallel_runner.last_diagnostics
                    _stage1_parallel_total_seconds = (
                        time.time() - _stage1_parallel_window_t0
                    )
                    _stage1_parallel_window_episodes = (
                        int(sum(_diag.per_worker_episode_counts))
                        if _diag is not None
                        else stage1_config.PPO_UPDATE_INTERVAL
                    )
                    _stage1_parallel_known_seconds = (
                        _stage1_parallel_collect_seconds
                        + _stage1_parallel_replay_seconds
                        + _stage1_parallel_detail_seconds
                        + _stage1_ppo_update_seconds
                    )
                    _stage1_parallel_other_seconds = max(
                        0.0,
                        _stage1_parallel_total_seconds - _stage1_parallel_known_seconds,
                    )
                    _stage1_parallel_ep_per_hour = (
                        _stage1_parallel_window_episodes
                        / max(_stage1_parallel_total_seconds, 1e-9)
                        * 3600.0
                    )
                    _stage1_parallel_update_payload = {
                        "window": int(_stage1_parallel_window_idx),
                        "episodes": int(_stage1_parallel_window_episodes),
                        "total_seconds": float(_stage1_parallel_total_seconds),
                        "collect_seconds": float(_stage1_parallel_collect_seconds),
                        "model_forward_seconds": float(_stage1_parallel_model_forward_seconds),
                        "model_forward_calls": int(_stage1_parallel_model_forward_calls),
                        "replay_seconds": float(_stage1_parallel_replay_seconds),
                        "detail_seconds": float(_stage1_parallel_detail_seconds),
                        "report_write_seconds": float(_stage1_parallel_detail_seconds),
                        "ppo_update_seconds": float(_stage1_ppo_update_seconds),
                        "other_seconds": float(_stage1_parallel_other_seconds),
                        "throughput_ep_per_hour": float(_stage1_parallel_ep_per_hour),
                    }
                    if _diag is not None:
                        _stage1_parallel_update_payload["worker_seconds"] = list(_diag.per_worker_seconds)
                        _stage1_parallel_update_payload["worker_episode_counts"] = list(_diag.per_worker_episode_counts)
                        _stage1_parallel_update_payload["devices"] = list(_diag.devices)
                        _stage1_parallel_update_payload["speedup_vs_sequential"] = float(_diag.speedup_vs_sequential)
                    workflow.log(
                        "  [stage1-rollout-total] "
                        f"window={_stage1_parallel_window_idx} "
                        f"episodes={_stage1_parallel_window_episodes} "
                        f"total={_stage1_parallel_total_seconds:.3f}s "
                        f"collect={_stage1_parallel_collect_seconds:.3f}s "
                        f"model_forward={_stage1_parallel_model_forward_seconds:.3f}s "
                        f"replay={_stage1_parallel_replay_seconds:.3f}s "
                        f"detail={_stage1_parallel_detail_seconds:.3f}s "
                        f"report_write={_stage1_parallel_detail_seconds:.3f}s "
                        f"ppo_update={_stage1_ppo_update_seconds:.3f}s "
                        f"other={_stage1_parallel_other_seconds:.3f}s "
                        f"throughput={_stage1_parallel_ep_per_hour:.1f}ep/h"
                    )
                    _stage1_parallel_window_t0 = None
                    _stage1_parallel_window_idx = None
                    _stage1_parallel_collect_seconds = 0.0
                    _stage1_parallel_model_forward_seconds = 0.0
                    _stage1_parallel_model_forward_calls = 0
                    _stage1_parallel_replay_seconds = 0.0
                    _stage1_parallel_detail_seconds = 0.0

                avg_reward = np.mean(episode_rewards[-stage1_config.PPO_UPDATE_INTERVAL:])
                stage1_data_writer.write_ppo_update({
                    "update": int(gtrxl_ppo_update_count),
                    "episode": int(episode + 1),
                    "policy_loss": float(policy_loss),
                    "value_loss": float(value_loss),
                    "entropy": float(entropy),
                    "avg_reward": float(avg_reward),
                    "current_lr": float(optimizer.param_groups[0]["lr"]),
                    "current_entropy_coef": float(current_entropy),
                    "best_reward_so_far": float(best_reward),
                    "best_cost_so_far": float(best_cost),
                    "buffer_episodes": int(stage1_config.PPO_UPDATE_INTERVAL),
                    "parallel": _stage1_parallel_update_payload,
                })
                warmup_status = "warmup" if gtrxl_ppo_update_count <= stage1_config.GTRXL_WARMUP_STEPS else "normal"
                _log_rounded_box(
                    workflow.log,
                    [
                        f"回合（Episode） {episode + 1}",
                        f"平均奖励: {avg_reward:.4f}, 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}, 熵: {entropy:.4f}",
                        (
                            f"[GTrXL调度] LR: {optimizer.param_groups[0]['lr']:.6f}, "
                            f"熵系数: {current_entropy:.6f}, 更新次数: #{gtrxl_ppo_update_count} ({warmup_status})"
                        ),
                    ],
                )


                _stage1_reached_episode_cap = (
                    workflow.stage1_rl_episode_limit is not None
                    and episode + 1 >= workflow.stage1_rl_episode_limit
                )
                if (gtrxl_ppo_update_count % NOISE_RL_PROGRESS_BOX_PPO_INTERVAL == 0
                        or _stage1_reached_episode_cap):
                    _s1_elapsed = time.time() - _stage1_rl_t0
                    _s1_done = episode + 1 - stage1_resume_start_episode
                    _s1_avg_ep = _s1_elapsed / max(_s1_done, 1)
                    _s1_best_lines = []
                    if best_config is not None:
                        _s1_best_lines.append(
                            f"Reward-Best: {best_config.get('reward', 0):.4f}  "
                            f"成本: {best_config.get('cost', 0):.2f}"
                        )
                        _s1_best_lines.append(f"  GELU:    {list(best_config.get('gelu', []))}")
                        _s1_best_lines.append(f"  Softmax: {list(best_config.get('softmax', []))}")
                    else:
                        _s1_best_lines.append("Reward-Best: 尚未找到")
                    _s1_remain = workflow.stage1_rl_episode_limit - (episode + 1)
                    _s1_eta = _s1_avg_ep * _s1_remain
                    _s1_progress_title = (
                        f"Stage-1 RL 进度 · 回合 {episode + 1} / "
                        f"{workflow.stage1_rl_episode_limit}"
                    )
                    _s1_progress_lines = [
                        _progress_bar(episode + 1, workflow.stage1_rl_episode_limit),
                        *_s1_best_lines,
                        f"已用时: {_fmt_elapsed(_s1_elapsed)}  "
                        f"预计剩余: {_fmt_elapsed(_s1_eta)}  "
                        f"预计完成: {_fmt_eta_finish(_s1_eta)}  "
                        f"PPO 更新: {gtrxl_ppo_update_count} 次",
                    ]
                    _log_rounded_box(
                        workflow.log,
                        [
                            _s1_progress_title,
                            *_s1_progress_lines,
                        ],
                        indent="  ",
                    )

                if stage1_prev_avg_reward[0] is not None:
                    reward_drop = stage1_prev_avg_reward[0] - avg_reward
                    if reward_drop > stage1_config.REWARD_DROP_WARNING_THRESHOLD:
                        window_start_ep = episode + 1 - stage1_config.PPO_UPDATE_INTERVAL + 1
                        window_end_ep = episode + 1
                        involved_files = sorted(set(
                            _get_stage1_chunk_filename(e)
                            for e in range(window_start_ep, window_end_ep + 1)
                        ))
                        involved_basenames = [os.path.basename(fp) for fp in involved_files]
                        warn_msg = {
                            "type": "阶段1奖励骤降",
                            "window": gtrxl_ppo_update_count,
                            "prev_avg": float(stage1_prev_avg_reward[0]),
                            "curr_avg": float(avg_reward),
                            "drop": float(reward_drop),
                            "threshold": float(stage1_config.REWARD_DROP_WARNING_THRESHOLD),
                            "episode_range": (window_start_ep, window_end_ep),
                            "detail_files": involved_basenames,
                        }
                        stage1_warnings.append(warn_msg)
                        workflow.log(
                            f"  ⚠ 警告: 平均奖励下降 {reward_drop:.4f} "
                            f"(阈值={stage1_config.REWARD_DROP_WARNING_THRESHOLD}), "
                            f"涉及回合 {window_start_ep}-{window_end_ep}"
                        )
                stage1_prev_avg_reward[0] = avg_reward

                window_best_reward = float('-inf')
                window_best_cost = float('inf')
                window_best_config = None

                if (episode + 1) < workflow.stage1_rl_episode_limit:
                    env.prev_episode_metrics = {
                        "loss": proxy_base_loss,
                        "metric1": proxy_base_p,
                        "metric2": proxy_base_s,
                        "cost": base_tot_c,
                    }
                    env.current_episode_metrics = None


                if step_info_chunk_file[0] is not None:
                    step_info_chunk_file[0].flush()
                _stage1_structured_jsonl_sizes = (
                    stage1_data_writer.committed_jsonl_sizes()
                )
                _stage1_detail_file_sizes = stage1_detail_file_sizes(step_info_details_dir)
                save_stage1_rl_checkpoint(
                    path=stage1_checkpoint_path,
                    gtrxl_net=gtrxl_net,
                    optimizer=optimizer,
                    episode=episode,
                    gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                    episode_rewards=episode_rewards,
                    episode_losses=episode_losses,
                    episode_metric1s=episode_metric1s,
                    episode_metric2s=episode_metric2s,
                    episode_entropies=episode_entropies,
                    best_reward=best_reward,
                    best_cost=best_cost,
                    best_config=best_config,
                    search_best_config=search_best_config,
                    global_best_config=global_best_config,
                    window_best_reward=window_best_reward,
                    window_best_cost=window_best_cost,
                    window_best_config=window_best_config,
                    ev_runtime_state={
                        "reward_history": list(workflow.reward_history),
                        "reward_mean": float(workflow.reward_mean),
                        "reward_std": float(workflow.reward_std),
                        "current_episode": int(workflow.current_episode),
                        "return_normalizer_mean": float(workflow.return_normalizer.mean),
                        "return_normalizer_var": float(workflow.return_normalizer.var),
                        "return_normalizer_count": float(workflow.return_normalizer.count),
                    },
                    stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                    stage1_warnings=stage1_warnings,
                    dataset_protocol_hash=workflow.dataset_protocol_hash,
                    structured_run_id=stage1_data_writer.run_id,
                    structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                    detail_file_sizes=_stage1_detail_file_sizes,
                    cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
                )
                if not _stage1_reached_episode_cap:
                    if _stage1_parallel_runner is not None:
                        _deferred_gpu_failure = (
                            _stage1_parallel_runner.pop_deferred_gpu_failure()
                        )
                        if _deferred_gpu_failure is not None:
                            raise _deferred_gpu_failure
                    raise_if_elastic_gpu_restart_requested()
            if is_graceful_stop_requested(stage1_stop_flag_path):
                workflow.log(
                    f"\n  [优雅停止] 已检测到停止请求，正在保存 Stage-1 checkpoint "
                    f"(episode={episode + 1}) ..."
                )
                if step_info_chunk_file[0] is not None:
                    step_info_chunk_file[0].flush()
                _stage1_structured_jsonl_sizes = (
                    stage1_data_writer.committed_jsonl_sizes()
                )
                _stage1_detail_file_sizes = stage1_detail_file_sizes(
                    step_info_details_dir
                )
                save_stage1_rl_checkpoint(
                    path=stage1_checkpoint_path,
                    gtrxl_net=gtrxl_net,
                    optimizer=optimizer,
                    episode=episode,
                    gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                    episode_rewards=episode_rewards,
                    episode_losses=episode_losses,
                    episode_metric1s=episode_metric1s,
                    episode_metric2s=episode_metric2s,
                    episode_entropies=episode_entropies,
                    best_reward=best_reward,
                    best_cost=best_cost,
                    best_config=best_config,
                    search_best_config=search_best_config,
                    global_best_config=global_best_config,
                    window_best_reward=window_best_reward,
                    window_best_cost=window_best_cost,
                    window_best_config=window_best_config,
                    ev_runtime_state={
                        "reward_history": list(workflow.reward_history),
                        "reward_mean": float(workflow.reward_mean),
                        "reward_std": float(workflow.reward_std),
                        "current_episode": int(workflow.current_episode),
                        "return_normalizer_mean": float(workflow.return_normalizer.mean),
                        "return_normalizer_var": float(workflow.return_normalizer.var),
                        "return_normalizer_count": float(workflow.return_normalizer.count),
                    },
                    stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                    stage1_warnings=stage1_warnings,
                    dataset_protocol_hash=workflow.dataset_protocol_hash,
                    structured_run_id=stage1_data_writer.run_id,
                    structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                    detail_file_sizes=_stage1_detail_file_sizes,
                    cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
                )
                consume_stop_flag_file(stage1_stop_flag_path)
                stage1_data_writer.write_summary({
                    "status": "stopped",
                    "stop_reason": "graceful_stop",
                    "completed_episodes": int(episode + 1),
                    "best_reward": float(best_reward),
                    "best_cost": float(best_cost),
                    "best_config": best_config,
                })
                stage1_data_writer.close()
                if workflow.run_output_dir:
                    update_persistent_metadata_stage(
                        workflow.run_output_dir, "stage1_search", "in_progress",
                        extra_fields={
                            "completed_episodes": episode + 1,
                            "total_episodes": int(workflow.stage1_rl_episode_limit),
                            "stopped_by": "graceful_stop",
                        },
                    )
                workflow.log(
                    f"  [优雅停止] checkpoint 已写入 → {stage1_checkpoint_path}\n"
                    f"  下次用相同参数直接运行即可自动续训练（rl/ga 持久化目录），"
                    f"或使用 --resume-from 指向本 run 目录（general-rl）。"
                )
                uninstall_graceful_stop_handler()
                raise SystemExit(0)

        if step_info_chunk_file[0] is not None:
            step_info_chunk_file[0].close()
            step_info_chunk_file[0] = None

        if stage1_warnings:
            _write_warning_report(stage1_warning_file, stage1_warnings, stage_label="阶段1（Stage-1 RL）")
            workflow.log(f"  ⚠ 共检测到 {len(stage1_warnings)} 次奖励骤降警告，详见: {stage1_warning_file}")


        _s1_final_ep = max(stage1_completed_episodes - 1, stage1_resume_start_episode - 1)
        if stage1_completed_episodes > stage1_resume_start_episode:
            _stage1_structured_jsonl_sizes = (
                stage1_data_writer.committed_jsonl_sizes()
            )
            _stage1_detail_file_sizes = stage1_detail_file_sizes(
                step_info_details_dir
            )
            save_stage1_rl_checkpoint(
                path=stage1_checkpoint_path,
                gtrxl_net=gtrxl_net,
                optimizer=optimizer,
                episode=_s1_final_ep,
                gtrxl_ppo_update_count=gtrxl_ppo_update_count,
                episode_rewards=episode_rewards,
                episode_losses=episode_losses,
                episode_metric1s=episode_metric1s,
                episode_metric2s=episode_metric2s,
                episode_entropies=episode_entropies,
                best_reward=best_reward,
                best_cost=best_cost,
                best_config=best_config,
                search_best_config=search_best_config,
                global_best_config=global_best_config,
                window_best_reward=window_best_reward,
                window_best_cost=window_best_cost,
                window_best_config=window_best_config,
                ev_runtime_state={
                    "reward_history": list(workflow.reward_history),
                    "reward_mean": float(workflow.reward_mean),
                    "reward_std": float(workflow.reward_std),
                    "current_episode": int(workflow.current_episode),
                    "return_normalizer_mean": float(workflow.return_normalizer.mean),
                    "return_normalizer_var": float(workflow.return_normalizer.var),
                    "return_normalizer_count": float(workflow.return_normalizer.count),
                },
                stage1_prev_avg_reward=stage1_prev_avg_reward[0],
                stage1_warnings=stage1_warnings,
                dataset_protocol_hash=workflow.dataset_protocol_hash,
                structured_run_id=stage1_data_writer.run_id,
                structured_jsonl_sizes=_stage1_structured_jsonl_sizes,
                detail_file_sizes=_stage1_detail_file_sizes,
                cuda_rng_role_registry=_stage1_cuda_rng_role_registry,
            )
            workflow.log(f"  [完成] Stage-1 最终 checkpoint 已保存 → {stage1_checkpoint_path}")
        if workflow.run_output_dir:
            update_persistent_metadata_stage(
                workflow.run_output_dir, "stage1_search", "completed",
                extra_fields={
                    "episodes": int(stage1_completed_episodes),
                    "completed_episodes": int(stage1_completed_episodes),
                    "target_episodes": int(workflow.stage1_rl_episode_limit),
                    "stop_reason": stage1_stop_reason,
                    "best_reward": float(best_reward),
                    "best_cost": float(best_cost),
                },
            )
        stage1_data_writer.write_summary({
            "status": "completed",
            "stop_reason": stage1_stop_reason,
            "completed_episodes": int(stage1_completed_episodes),
            "target_episodes": int(workflow.stage1_rl_episode_limit),
            "ppo_updates": int(gtrxl_ppo_update_count),
            "best_reward": float(best_reward),
            "best_cost": float(best_cost),
            "best_config": best_config,
            "warnings": stage1_warnings,
            "episode_count": len(episode_rewards),
            "entropy_points": len(episode_entropies),
        })
        stage1_data_writer.close()


        try:
            _portable_path = os.path.join(
                os.path.dirname(workflow.stage1_step_info_file),
                "stage1_policy.pt",
            )
            torch.save(
                {
                    "version": 1,
                    "kind": "stage1_gtrxl_policy",
                    "net_state_dict": gtrxl_net.state_dict(),
                    "arch": {
                        "num_layers": int(workflow.total_layers),
                        "d_model": int(stage1_config.GTRXL_D_MODEL),
                        "n_heads": int(stage1_config.GTRXL_N_HEADS),
                        "n_gtrxl_layers": int(stage1_config.GTRXL_N_LAYERS),
                        "d_ff": int(stage1_config.GTRXL_D_FF),
                        "dropout": float(stage1_config.GTRXL_DROPOUT),
                    },
                    "metadata": {
                        "trained_episodes": int(stage1_completed_episodes),
                        "best_reward": float(best_reward),
                        "best_cost": float(best_cost),
                        "error_threshold": float(workflow.error_threshold),
                        "correlation_drop_ratio": float(workflow.correlation_drop_ratio),
                    },
                },
                _portable_path,
            )
            workflow.log(f"  Stage-1 policy saved to {_portable_path}")
        except Exception as _e:
            workflow.log(f"  [迁移][警告] portable policy 保存失败：{_e}")


        best_config, used_baseline = workflow._select_stage1_reward_best_config(
            best_config,
            best_reward,
            base_gelu,
            base_softmax,
            base_tot_c,
        )

        if used_baseline:
            workflow.log("\n未找到可行解，使用基线配置。")
        else:
            workflow.log(
                "\n[Stage-1] 最终配置使用原始 PPO reward-best；"
                "GELU/Softmax 为确定性替换，不再按 window 二次评估重排。"
            )

        workflow.log(f"\n--- PPO训练完成（PPO Training Completed） ---")
        workflow.log(f"已找到最优配置（通过RL）（Best Configuration Found by RL）：")
        workflow.log(f"  GELU配置: {best_config['gelu'].tolist()}")
        workflow.log(f"  Softmax配置: {best_config['softmax'].tolist()}")
        workflow.log(f"  成本（Cost）: {best_config['cost']:.2f}, 奖励（Reward）: {best_config['reward']:.4f}")


    if len(episode_rewards) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            episodes = np.arange(1, len(episode_rewards) + 1)
            rewards = np.array(episode_rewards, dtype=np.float32)
            losses = np.array(episode_losses, dtype=np.float32)
            metric1s = np.array(episode_metric1s, dtype=np.float32)
            metric2s = np.array(episode_metric2s, dtype=np.float32)


            metric_names_tuple = workflow.get_metric_names()
            _num_m = workflow.get_num_metrics()
            _m1_name = metric_names_tuple[0]
            _m2_name = metric_names_tuple[1] if _num_m > 1 else None


            window = min(max(5, stage1_config.PPO_UPDATE_INTERVAL // 5), 50)

            def compute_ma(data):
                if len(data) >= window:
                    kernel = np.ones(window, dtype=np.float32) / window
                    return np.convolve(data, kernel, mode="valid")
                return data

            rewards_ma = compute_ma(rewards)
            losses_ma = compute_ma(losses)
            metric1s_ma = compute_ma(metric1s)
            metric2s_ma = compute_ma(metric2s) if _num_m > 1 else None

            if len(rewards) >= window:
                episodes_ma = episodes[window - 1:]
            else:
                episodes_ma = episodes

            dataset_info = f" ({workflow.data_path})"
            val_guided_info = " [Train Probe]"

            if _num_m == 1:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')

                ax1 = axes[0]
                ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

                ax2 = axes[1]
                ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                ax2.legend()

                ax3 = axes[2]
                ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                ax3.legend()
            else:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')

                ax1 = axes[0, 0]
                ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

                ax2 = axes[0, 1]
                ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                ax2.legend()

                ax3 = axes[1, 0]
                ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                ax3.legend()

                ax4 = axes[1, 1]
                ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color='purple')
                ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkviolet')
                ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name)
                ax4.set_title(f"{_m2_name} (higher is better)"); ax4.grid(True, alpha=0.3)
                ax4.axhline(y=base_s, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                ax4.legend()

            plot_path = workflow.stage1_training_curve_path
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            workflow.log(f"PPO训练曲线已保存至（PPO training curves saved to）: {plot_path}")

            if episode_entropies:
                update_episodes = np.arange(stage1_config.PPO_UPDATE_INTERVAL, len(episode_rewards) + 1, stage1_config.PPO_UPDATE_INTERVAL)
                entropies = np.array(episode_entropies, dtype=np.float32)
                if len(update_episodes) == len(entropies):
                    fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                    ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color='teal', marker='o', markersize=3)
                    window_ent = min(5, max(1, len(entropies) // 5))
                    if len(entropies) >= window_ent:
                        kernel_ent = np.ones(window_ent, dtype=np.float32) / window_ent
                        ent_ma = np.convolve(entropies, kernel_ent, mode="valid")
                        ax_ent.plot(update_episodes[window_ent - 1:], ent_ma, label=f"Moving Avg ({window_ent})", linewidth=2, color='darkgreen')
                    ax_ent.set_xlabel("Episode (at PPO update)")
                    ax_ent.set_ylabel("Entropy")
                    ax_ent.set_title("PPO Training: Policy Entropy over Episodes")
                    ax_ent.grid(True, alpha=0.3)
                    ax_ent.legend()
                    plt.tight_layout()
                    entropy_plot_path = workflow.stage1_entropy_curve_path
                    plt.savefig(entropy_plot_path, dpi=150)
                    plt.close()
                    workflow.log(f"PPO熵曲线已保存至（PPO entropy curve saved to）: {entropy_plot_path}")
                else:
                    workflow.log(f"[警告] 熵曲线未绘制（Entropy curve not plotted）: update_episodes长度={len(update_episodes)}, entropies长度={len(entropies)}")
        except Exception as e:
            workflow.log(f"[警告] PPO训练曲线绘制失败（Failed to plot PPO training curves）: {e}")


    if best_config is not None and not workflow.skip_stage1_rl:
        workflow._write_stage1_best_config_file(
            best_config=best_config,
            completed_episodes=stage1_completed_episodes,
        )

    workflow.log("\n" + "="*60)
    workflow.log("最终评估报告（FINAL EVALUATION REPORT）（验证集）")
    workflow.log("="*60)



    return best_config, base_gelu, base_softmax
