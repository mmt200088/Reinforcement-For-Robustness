import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from function_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
)


# ---------------------------------------------------------------------------
# Stage-2-only hyperparameters from the PDF redesign.
# ---------------------------------------------------------------------------

NOISE_STAGE_GTRXL_D_MODEL = 256
NOISE_STAGE_GTRXL_N_HEADS = 8
NOISE_STAGE_GTRXL_N_LAYERS = 4
NOISE_STAGE_GTRXL_D_FF = 512
NOISE_STAGE_GTRXL_DROPOUT = 0.1

NOISE_STAGE_PPO_MAX_EPISODES = 80000
NOISE_STAGE_PPO_EPS_CLIP = 0.15
NOISE_STAGE_PPO_K_EPOCHS = 10
NOISE_STAGE_GTRXL_WARMUP_STEPS = 5000

NOISE_STAGE_MC_SAMPLES = 1

NOISE_REWARD_ESTIMATOR_HIDDEN_DIMS = (128, 64)
NOISE_REWARD_ESTIMATOR_LR = 1e-3
NOISE_REWARD_ESTIMATOR_REPLAY_CAPACITY = 4096
NOISE_REWARD_ESTIMATOR_WARMUP_EPISODES = 32
NOISE_REWARD_ESTIMATOR_BATCH_SIZE = 64
NOISE_REWARD_ESTIMATOR_EPOCHS = 4

NOISE_REWARD_BLEND_ALPHA_START = 0.2
NOISE_REWARD_BLEND_ALPHA_END = 0.8


class NoiseRLModule:
    """Standalone second-stage noise RL module.

    Follows the same pattern as FinalEvaluationModule: receives the evaluator
    and encapsulates all stage-2 noise RL logic (training, evaluation, plotting).
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def run(self, fixed_gelu, fixed_softmax, fixed_label, fixed_source):
        from layer_importance_evaluator import (
            INPUT_NOISE_SCALING_MAP,
            INPUT_NOISE_COST_MAP,
            INPUT_NOISE_SCALING_TO_NORM,
            WEIGHT_NOISE_COST_MAP,
            WEIGHT_NOISE_SCALING_TO_NORM,
            WQ_NOISE_SCALING_MAP,
            WK_NOISE_SCALING_MAP,
            WV_NOISE_SCALING_MAP,
            WO_NOISE_SCALING_MAP,
            WFFN1_NOISE_SCALING_MAP,
            WFFN2_NOISE_SCALING_MAP,
            NOISE_STAGE_NUM_ACTIONS,
            NOISE_STAGE_SOS_TOKENS,
            NOISE_STAGE_CONT_DIM,
            NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            NOISE_STAGE_ACTION_DIMS,
            NOISE_STAGE_TRAINING_CURVE_PATH,
            NOISE_STAGE_ENTROPY_CURVE_PATH,
            PPO_UPDATE_INTERVAL,
            PPO_VALUE_COEF,
            REWARD_THRESHOLD,
            REWARD_DENSE_SCALE,
            REWARD_COST_WEIGHT,
            REWARD_SAFETY_BONUS,
            REWARD_CLIP_MIN,
            REWARD_CLIP_MAX,
            REWARD_NORMALIZATION_SCALE,
            BUDGET_DEVIATION_SCALE,
            HISTORY_MASK_VALUE,
            DIFF_REWARD_SCALE_ACC,
            DIFF_REWARD_POWER,
            LOG_BARRIER_VIOLATION_SCALE,
            LOG_BARRIER_VIOLATION_STEEPNESS,
            LOG_BARRIER_SATISFACTION_SCALE,
            LSTM_POS_DIM,
            LSTM_PROJ_DIM,
            GTRXL_ENTROPY_LOWER_BOUND,
            GTRXL_MINI_BATCH_EPISODES,
            USE_VALIDATION_FOR_REWARD,
            VALUE_CLIP_RANGE,
            GTrXLBlock,
        )

        ev = self.evaluator
        ev.log("\n" + "=" * 60)
        ev.log("PHASE 5: SECOND-STAGE NOISE RL")
        ev.log(f"Fixed GELU/Softmax source={fixed_source}, label={fixed_label}")
        ev.log(f"Fixed GELU   : {np.asarray(fixed_gelu, dtype=int).tolist()}")
        ev.log(f"Fixed Softmax: {np.asarray(fixed_softmax, dtype=int).tolist()}")
        ev.log("=" * 60)

        fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        baseline_noise_config = ev._get_max_noise_configuration()

        reward_reference_split = ev.get_reward_reference_split_name()
        baseline_train = ev.evaluate_model_with_attention_noise(
            fixed_gelu,
            fixed_softmax,
            use_train=True,
            **baseline_noise_config,
        )
        baseline_val = ev.evaluate_model_with_attention_noise(
            fixed_gelu,
            fixed_softmax,
            split=reward_reference_split,
            **baseline_noise_config,
        )
        baseline_tot_c, baseline_breakdown = ev.get_noise_simulated_cost(**baseline_noise_config)

        ev.log("Noise-Stage Baseline Metrics (Training Set):")
        ev.log(f"  {ev._fmt_metrics(*baseline_train[:3])}")
        ev.log(f"  Noise Cost: {baseline_tot_c:.2f} | Breakdown={baseline_breakdown}")
        ev.log(f"Noise-Stage Baseline Metrics ({reward_reference_split} - used for reward):")
        ev.log(f"  {ev._fmt_metrics(*baseline_val[:3])}")

        if USE_VALIDATION_FOR_REWARD:
            base_loss, base_p, base_s = baseline_val[:3]
        else:
            base_loss, base_p, base_s = baseline_train[:3]

        limit_loss = base_loss + ev.error_threshold
        limit_p = base_p * (1.0 - ev.correlation_drop_ratio)
        limit_s = base_s * (1.0 - ev.correlation_drop_ratio)
        constraint_split_label = reward_reference_split if USE_VALIDATION_FOR_REWARD else "train"
        ev.log(f"Noise-Stage Constraints (based on {constraint_split_label}):")
        ev.log(f"  {ev._fmt_constraints(limit_loss, limit_p, limit_s)}")
        training_hparams = {
            "gtrxl_d_model": NOISE_STAGE_GTRXL_D_MODEL,
            "gtrxl_n_heads": NOISE_STAGE_GTRXL_N_HEADS,
            "gtrxl_n_layers": NOISE_STAGE_GTRXL_N_LAYERS,
            "gtrxl_d_ff": NOISE_STAGE_GTRXL_D_FF,
            "gtrxl_dropout": NOISE_STAGE_GTRXL_DROPOUT,
            "ppo_max_episodes": NOISE_STAGE_PPO_MAX_EPISODES,
            "ppo_update_interval": PPO_UPDATE_INTERVAL,
            "ppo_eps_clip": NOISE_STAGE_PPO_EPS_CLIP,
            "ppo_k_epochs": NOISE_STAGE_PPO_K_EPOCHS,
            "ppo_value_coef": PPO_VALUE_COEF,
            "gtrxl_warmup_steps": NOISE_STAGE_GTRXL_WARMUP_STEPS,
            "gtrxl_mini_batch_episodes": GTRXL_MINI_BATCH_EPISODES,
            "mc_samples": NOISE_STAGE_MC_SAMPLES,
            "reward_estimator_hidden_dims": list(NOISE_REWARD_ESTIMATOR_HIDDEN_DIMS),
            "reward_estimator_lr": NOISE_REWARD_ESTIMATOR_LR,
            "reward_estimator_replay_capacity": NOISE_REWARD_ESTIMATOR_REPLAY_CAPACITY,
            "reward_estimator_warmup_episodes": NOISE_REWARD_ESTIMATOR_WARMUP_EPISODES,
            "reward_estimator_batch_size": NOISE_REWARD_ESTIMATOR_BATCH_SIZE,
            "reward_estimator_epochs": NOISE_REWARD_ESTIMATOR_EPOCHS,
            "reward_blend_alpha_start": NOISE_REWARD_BLEND_ALPHA_START,
            "reward_blend_alpha_end": NOISE_REWARD_BLEND_ALPHA_END,
        }
        ev.log("Noise-Stage Training Hyperparameters:")
        ev.log(
            "  "
            f"GTrXL(d_model={NOISE_STAGE_GTRXL_D_MODEL}, heads={NOISE_STAGE_GTRXL_N_HEADS}, "
            f"layers={NOISE_STAGE_GTRXL_N_LAYERS}, d_ff={NOISE_STAGE_GTRXL_D_FF}, "
            f"dropout={NOISE_STAGE_GTRXL_DROPOUT})"
        )
        ev.log(
            "  "
            f"PPO(max_episodes={NOISE_STAGE_PPO_MAX_EPISODES}, eps_clip={NOISE_STAGE_PPO_EPS_CLIP}, "
            f"k_epochs={NOISE_STAGE_PPO_K_EPOCHS}, warmup_steps={NOISE_STAGE_GTRXL_WARMUP_STEPS})"
        )
        ev.log(
            "  "
            f"MC samples={NOISE_STAGE_MC_SAMPLES} | RewardEstimator(hidden={NOISE_REWARD_ESTIMATOR_HIDDEN_DIMS}, "
            f"lr={NOISE_REWARD_ESTIMATOR_LR}, replay={NOISE_REWARD_ESTIMATOR_REPLAY_CAPACITY}, "
            f"warmup={NOISE_REWARD_ESTIMATOR_WARMUP_EPISODES}, batch={NOISE_REWARD_ESTIMATOR_BATCH_SIZE}, "
            f"epochs={NOISE_REWARD_ESTIMATOR_EPOCHS})"
        )

        with open(ev.noise_step_info_file, "w", encoding="utf-8") as f:
            f.write("=== Noise PPO StepInfo 中间结果日志 ===\n")
            f.write("每步包含: step_global, episode_id, layer_index, state_vector, 7个动作的scaling factor, 7个动作概率分布, critic_value, accumulated_cost, 各类noise配置\n\n")

        original_total_episodes = getattr(ev, "total_episodes", NOISE_STAGE_PPO_MAX_EPISODES)
        ev.total_episodes = NOISE_STAGE_PPO_MAX_EPISODES
        ev._reset_runtime_ppo_state()
        noise_net = _NoiseGTrXLStrategyNetwork(
            num_layers=ev.total_layers,
            d_model=NOISE_STAGE_GTRXL_D_MODEL,
            n_heads=NOISE_STAGE_GTRXL_N_HEADS,
            n_gtrxl_layers=NOISE_STAGE_GTRXL_N_LAYERS,
            d_ff=NOISE_STAGE_GTRXL_D_FF,
            dropout=NOISE_STAGE_GTRXL_DROPOUT,
            gtrxl_block_cls=GTrXLBlock,
            lstm_pos_dim=LSTM_POS_DIM,
            lstm_proj_dim=LSTM_PROJ_DIM,
            noise_stage_num_actions=NOISE_STAGE_NUM_ACTIONS,
            noise_stage_sos_tokens=NOISE_STAGE_SOS_TOKENS,
            noise_stage_prev_action_embed_dim=NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            noise_stage_cont_dim=NOISE_STAGE_CONT_DIM,
            noise_stage_action_dims=NOISE_STAGE_ACTION_DIMS,
        ).to(ev.device)
        optimizer = optim.Adam(noise_net.parameters(), lr=ev.ppo_lr_initial)
        reward_estimator = _NoiseRewardEstimator(
            input_dim=NOISE_STAGE_NUM_ACTIONS * ev.total_layers + 1,
            hidden_dims=NOISE_REWARD_ESTIMATOR_HIDDEN_DIMS,
        ).to(ev.device)
        reward_estimator_optimizer = optim.Adam(
            reward_estimator.parameters(), lr=NOISE_REWARD_ESTIMATOR_LR
        )
        reward_replay = _NoiseRewardReplayBuffer(
            capacity=NOISE_REWARD_ESTIMATOR_REPLAY_CAPACITY
        )
        noise_ppo_update_count = 0
        reward_estimator_update_count = 0
        reward_estimator_last_loss = None

        class _NoiseRLEvaluatorWrapper:
            def __init__(wrapper_self, evaluator, fixed_gelu, fixed_softmax, split_name=None, use_train=None):
                wrapper_self.evaluator = evaluator
                wrapper_self.fixed_gelu = np.asarray(fixed_gelu, dtype=int)
                wrapper_self.fixed_softmax = np.asarray(fixed_softmax, dtype=int)
                if split_name is not None:
                    wrapper_self.split_name = split_name
                elif use_train is None:
                    wrapper_self.split_name = "train"
                else:
                    wrapper_self.split_name = "train" if use_train else "validation_full"

            def evaluate_noise_model(wrapper_self, **noise_kwargs):
                return wrapper_self.evaluator.evaluate_model_with_attention_noise(
                    wrapper_self.fixed_gelu,
                    wrapper_self.fixed_softmax,
                    split=wrapper_self.split_name,
                    **noise_kwargs,
                )

        rl_evaluator = _NoiseRLEvaluatorWrapper(
            ev,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            use_train=(not USE_VALIDATION_FOR_REWARD),
        )
        if USE_VALIDATION_FOR_REWARD:
            ev.refresh_validation_proxy(window_index=0, stage_label="Stage-2 Noise RL")
            online_reward_split = ev.get_online_reward_split_name()
            rl_evaluator.split_name = online_reward_split
            proxy_baseline = ev.evaluate_model_with_attention_noise(
                fixed_gelu,
                fixed_softmax,
                split=online_reward_split,
                **baseline_noise_config,
            )
            ev.log(
                f"[Info] Noise-stage online reward uses {online_reward_split} "
                f"(constraints stay on {reward_reference_split})"
            )
        else:
            online_reward_split = "train"
            proxy_baseline = baseline_train
        env = _NoiseOptEnv(
            ev.total_layers,
            baseline_tot_c,
            (base_loss, base_p, base_s),
            rl_evaluator,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            num_metrics=ev.get_num_metrics(),
            input_noise_allowed=INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            weight_noise_allowed=WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            input_noise_cost_map=INPUT_NOISE_COST_MAP,
            weight_noise_cost_map=WEIGHT_NOISE_COST_MAP,
            input_noise_scaling_map=INPUT_NOISE_SCALING_MAP,
            wq_noise_scaling_map=WQ_NOISE_SCALING_MAP,
            wk_noise_scaling_map=WK_NOISE_SCALING_MAP,
            wv_noise_scaling_map=WV_NOISE_SCALING_MAP,
            wo_noise_scaling_map=WO_NOISE_SCALING_MAP,
            wffn1_noise_scaling_map=WFFN1_NOISE_SCALING_MAP,
            wffn2_noise_scaling_map=WFFN2_NOISE_SCALING_MAP,
            input_noise_scaling_to_norm=INPUT_NOISE_SCALING_TO_NORM,
            weight_noise_scaling_to_norm=WEIGHT_NOISE_SCALING_TO_NORM,
            noise_stage_sos_tokens=NOISE_STAGE_SOS_TOKENS,
            noise_stage_num_actions=NOISE_STAGE_NUM_ACTIONS,
            history_mask_value=HISTORY_MASK_VALUE,
            reward_threshold=REWARD_THRESHOLD,
            reward_dense_scale=REWARD_DENSE_SCALE,
            reward_cost_weight=REWARD_COST_WEIGHT,
            reward_safety_bonus=REWARD_SAFETY_BONUS,
            reward_clip_min=REWARD_CLIP_MIN,
            reward_clip_max=REWARD_CLIP_MAX,
            reward_normalization_scale=REWARD_NORMALIZATION_SCALE,
            budget_deviation_scale=BUDGET_DEVIATION_SCALE,
            diff_reward_scale_acc=DIFF_REWARD_SCALE_ACC,
            diff_reward_power=DIFF_REWARD_POWER,
            log_barrier_violation_scale=LOG_BARRIER_VIOLATION_SCALE,
            log_barrier_violation_steepness=LOG_BARRIER_VIOLATION_STEEPNESS,
            log_barrier_satisfaction_scale=LOG_BARRIER_SATISFACTION_SCALE,
            mc_samples=NOISE_STAGE_MC_SAMPLES,
        )
        env.prev_episode_metrics = {
            "loss": float(proxy_baseline[0]),
            "metric1": float(proxy_baseline[1]),
            "metric2": float(proxy_baseline[2]),
            "cost": float(baseline_tot_c),
        }
        buffer = _NoiseRecurrentRolloutBuffer()

        episode_rewards = []
        episode_blended_rewards = []
        episode_losses = []
        episode_metric1s = []
        episode_metric2s = []
        episode_entropies = []
        reward_blend_alphas = []
        reward_prediction_abs_errors = []
        raw_final_rewards = []
        blended_final_rewards = []
        best_reward = float("-inf")
        best_cost = float("inf")
        best_noise_config = None
        window_best_reward = float("-inf")
        window_best_cost = float("inf")
        window_best_noise_config = None
        search_best_noise_config = None
        global_best_noise_config = None

        def confirm_noise_candidate(candidate_config, episode_idx, window_idx):
            nonlocal search_best_noise_config, global_best_noise_config

            if candidate_config is None or not ev.has_dataset_split("val_search_full"):
                return

            noise_kwargs = {
                key: value.copy()
                for key, value in candidate_config.items()
                if key.endswith("scaling_factors")
            }
            search_loss, search_p, search_s, _ = ev.evaluate_model_with_attention_noise(
                fixed_gelu,
                fixed_softmax,
                split="val_search_full",
                **noise_kwargs,
            )
            search_ok = ev._candidate_meets_constraints(
                search_loss,
                search_p,
                search_s,
                limit_loss,
                limit_p,
                limit_s,
            )

            confirmed_candidate = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in candidate_config.items()
            }
            confirmed_candidate.update({
                "proxy_reward": float(candidate_config.get("reward", 0.0)),
                "search_loss": float(search_loss),
                "search_metric1": float(search_p),
                "search_metric2": float(search_s),
                "search_ok": bool(search_ok),
                "confirmed_episode": int(episode_idx) + 1,
                "confirmed_window": int(window_idx) + 1,
            })

            if ev.has_dataset_split("val_holdout"):
                holdout_loss, holdout_p, holdout_s, _ = ev.evaluate_model_with_attention_noise(
                    fixed_gelu,
                    fixed_softmax,
                    split="val_holdout",
                    **noise_kwargs,
                )
                holdout_ok = ev._candidate_meets_constraints(
                    holdout_loss,
                    holdout_p,
                    holdout_s,
                    limit_loss,
                    limit_p,
                    limit_s,
                )
                confirmed_candidate.update({
                    "holdout_loss": float(holdout_loss),
                    "holdout_metric1": float(holdout_p),
                    "holdout_metric2": float(holdout_s),
                    "holdout_ok": bool(holdout_ok),
                })
            else:
                confirmed_candidate.update({
                    "holdout_loss": float(search_loss),
                    "holdout_metric1": float(search_p),
                    "holdout_metric2": float(search_s),
                    "holdout_ok": bool(search_ok),
                })

            ev.log(
                f"  Noise window {window_idx + 1} candidate confirmation: "
                f"proxy_reward={confirmed_candidate['proxy_reward']:.4f}, "
                f"search={ev._fmt_metrics(search_loss, search_p, search_s)}"
            )
            if ev.has_dataset_split("val_holdout"):
                ev.log(
                    "    Holdout: "
                    f"{ev._fmt_metrics(confirmed_candidate['holdout_loss'], confirmed_candidate['holdout_metric1'], confirmed_candidate['holdout_metric2'])}"
                )

            if search_ok and ev._is_better_confirmed_candidate(
                confirmed_candidate,
                search_best_noise_config,
                metric_prefix="search",
            ):
                search_best_noise_config = {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in confirmed_candidate.items()
                }
                ev.log(
                    f"    Noise Search-Best updated at episode {episode_idx + 1}: "
                    f"cost={search_best_noise_config['cost']:.2f}, "
                    f"proxy_reward={search_best_noise_config['proxy_reward']:.4f}"
                )

            if confirmed_candidate["holdout_ok"] and ev._is_better_confirmed_candidate(
                confirmed_candidate,
                global_best_noise_config,
                metric_prefix="holdout",
            ):
                global_best_noise_config = {
                    key: (value.copy() if isinstance(value, np.ndarray) else value)
                    for key, value in confirmed_candidate.items()
                }
                ev.log(
                    f"    Noise Global-Best updated at episode {episode_idx + 1}: "
                    f"cost={global_best_noise_config['cost']:.2f}, "
                    f"proxy_reward={global_best_noise_config['proxy_reward']:.4f}"
                )

        for episode in range(NOISE_STAGE_PPO_MAX_EPISODES):
            current_lr, current_entropy = ev.update_hyperparameters(optimizer, episode)
            state = env.reset()
            prev_actions = torch.tensor(
                [list(NOISE_STAGE_SOS_TOKENS)], dtype=torch.long, device=ev.device
            ).unsqueeze(0)
            seq_cont_feats = []
            seq_layer_indices = []
            seq_prev_actions = []
            step_infos = []
            episode_reward_raw = 0.0
            episode_reward_blended = 0.0
            episode_raw_final_reward = None
            episode_blended_final_reward = None
            episode_reward_estimator_pred = None
            episode_reward_blend_alpha = None
            episode_mc_eval = None
            buffer.start_episode()

            for step in range(ev.total_layers):
                layer_idx = env.current_layer
                cont_feat_np = env.get_continuous_features()
                cont_feat = torch.tensor(cont_feat_np, dtype=torch.float32, device=ev.device).view(1, 1, -1)
                layer_tensor = torch.tensor([[layer_idx]], dtype=torch.long, device=ev.device)

                seq_cont_feats.append(cont_feat)
                seq_layer_indices.append(layer_tensor)
                seq_prev_actions.append(prev_actions)

                cont_seq = torch.cat(seq_cont_feats, dim=1)
                layer_seq = torch.cat(seq_layer_indices, dim=1)
                prev_action_seq = torch.cat(seq_prev_actions, dim=1)

                actions, logprob, value, prob_list = noise_net.get_action_and_logprob(
                    cont_seq, layer_seq, prev_action_seq, return_probs=True
                )
                next_state, reward, done, info = env.step(*[a.item() for a in actions])
                reward_for_buffer = reward
                if done:
                    raw_final_reward = float(info.get("raw_final_reward", 0.0))
                    reward_feature = env.get_reward_estimator_features()
                    reward_replay.add(reward_feature, raw_final_reward)

                    reward_estimator_pred = None
                    reward_blend_alpha = 1.0
                    blended_final_reward = raw_final_reward
                    if (
                        reward_estimator_update_count > 0
                        and len(reward_replay) >= NOISE_REWARD_ESTIMATOR_WARMUP_EPISODES
                    ):
                        reward_estimator_pred = _predict_noise_reward_estimator(
                            reward_estimator, reward_feature, ev.device
                        )
                        reward_blend_alpha = _compute_noise_reward_blend_alpha(
                            episode,
                            NOISE_STAGE_PPO_MAX_EPISODES,
                            start_alpha=NOISE_REWARD_BLEND_ALPHA_START,
                            end_alpha=NOISE_REWARD_BLEND_ALPHA_END,
                        )
                        blended_final_reward = (
                            reward_blend_alpha * raw_final_reward
                            + (1.0 - reward_blend_alpha) * reward_estimator_pred
                        )
                        reward_blend_alphas.append(reward_blend_alpha)
                        reward_prediction_abs_errors.append(
                            abs(reward_estimator_pred - raw_final_reward)
                        )

                    reward_for_buffer = float(info["dense_reward"] + blended_final_reward)
                    info["raw_reward"] = raw_final_reward
                    info["blended_reward"] = blended_final_reward
                    info["reward_estimator_pred"] = reward_estimator_pred
                    info["reward_blend_alpha"] = reward_blend_alpha
                    info["reward_estimator_loss"] = reward_estimator_last_loss
                    info["buffer_step_reward"] = reward_for_buffer

                    episode_raw_final_reward = raw_final_reward
                    episode_blended_final_reward = blended_final_reward
                    episode_reward_estimator_pred = reward_estimator_pred
                    episode_reward_blend_alpha = reward_blend_alpha
                    episode_mc_eval = info.get("mc_eval")
                else:
                    info["raw_reward"] = None
                    info["blended_reward"] = None
                    info["reward_estimator_pred"] = None
                    info["reward_blend_alpha"] = None
                    info["reward_estimator_loss"] = reward_estimator_last_loss
                    info["buffer_step_reward"] = reward_for_buffer

                mc_eval = info.get("mc_eval") or {}
                step_info = {
                    "step_global": episode * ev.total_layers + step,
                    "episode_id": episode,
                    "layer_index": info["layer_index"],
                    "state_vector": state.tolist(),
                    "curr_input_noise_scaling_factor": info["curr_input_noise_scaling_factor"],
                    "curr_wq_noise_scaling_factor": info["curr_wq_noise_scaling_factor"],
                    "curr_wk_noise_scaling_factor": info["curr_wk_noise_scaling_factor"],
                    "curr_wv_noise_scaling_factor": info["curr_wv_noise_scaling_factor"],
                    "curr_wo_noise_scaling_factor": info["curr_wo_noise_scaling_factor"],
                    "curr_wffn1_noise_scaling_factor": info["curr_wffn1_noise_scaling_factor"],
                    "curr_wffn2_noise_scaling_factor": info["curr_wffn2_noise_scaling_factor"],
                    "x_prob_dist": prob_list[0].detach().cpu().numpy().tolist(),
                    "wq_prob_dist": prob_list[1].detach().cpu().numpy().tolist(),
                    "wk_prob_dist": prob_list[2].detach().cpu().numpy().tolist(),
                    "wv_prob_dist": prob_list[3].detach().cpu().numpy().tolist(),
                    "wo_prob_dist": prob_list[4].detach().cpu().numpy().tolist(),
                    "wffn1_prob_dist": prob_list[5].detach().cpu().numpy().tolist(),
                    "wffn2_prob_dist": prob_list[6].detach().cpu().numpy().tolist(),
                    "critic_value": value.item(),
                    "accumulated_cost": info["accumulated_cost"],
                    "input_noise_config": info["input_noise_config"],
                    "wq_noise_config": info["wq_noise_config"],
                    "wk_noise_config": info["wk_noise_config"],
                    "wv_noise_config": info["wv_noise_config"],
                    "wo_noise_config": info["wo_noise_config"],
                    "wffn1_noise_config": info["wffn1_noise_config"],
                    "wffn2_noise_config": info["wffn2_noise_config"],
                    "current_lr": current_lr,
                    "current_entropy_coef": current_entropy,
                    "mc_samples": mc_eval.get("num_samples"),
                    "mc_loss_mean": mc_eval.get("loss_mean"),
                    "mc_loss_std": mc_eval.get("loss_std"),
                    "mc_metric1_mean": mc_eval.get("metric1_mean"),
                    "mc_metric1_std": mc_eval.get("metric1_std"),
                    "mc_metric2_mean": mc_eval.get("metric2_mean"),
                    "mc_metric2_std": mc_eval.get("metric2_std"),
                    "raw_reward": info.get("raw_reward"),
                    "blended_reward": info.get("blended_reward"),
                    "reward_estimator_pred": info.get("reward_estimator_pred"),
                    "reward_estimator_loss": info.get("reward_estimator_loss"),
                    "reward_blend_alpha": info.get("reward_blend_alpha"),
                    "buffer_step_reward": info.get("buffer_step_reward"),
                }
                step_infos.append(step_info)

                buffer.add_step(
                    cont_feat=torch.tensor(cont_feat_np, dtype=torch.float32),
                    layer_idx=layer_idx,
                    prev_actions=prev_actions.squeeze(0).squeeze(0).detach().cpu(),
                    actions=actions.detach().cpu(),
                    logprob=logprob.detach().cpu(),
                    reward=reward_for_buffer,
                    value=value.detach().cpu(),
                    done=float(done),
                )

                prev_actions = actions.view(1, 1, -1).to(ev.device)
                episode_reward_raw += reward
                episode_reward_blended += reward_for_buffer
                state = next_state

            buffer.end_episode()
            episode_rewards.append(episode_reward_raw)
            episode_blended_rewards.append(episode_reward_blended)
            if env.current_episode_metrics is not None:
                episode_losses.append(env.current_episode_metrics["loss"])
                episode_metric1s.append(env.current_episode_metrics["metric1"])
                episode_metric2s.append(env.current_episode_metrics["metric2"])
            else:
                episode_losses.append(base_loss)
                episode_metric1s.append(base_p)
                episode_metric2s.append(base_s)

            if episode_raw_final_reward is not None:
                raw_final_rewards.append(episode_raw_final_reward)
                blended_final_rewards.append(
                    episode_blended_final_reward
                    if episode_blended_final_reward is not None
                    else episode_raw_final_reward
                )

            env.update_prev_metrics()
            ev.update_reward_statistics(episode_reward_raw)
            with open(ev.noise_step_info_file, "a", encoding="utf-8") as f:
                f.write(
                    f"--- Episode {episode + 1} "
                    f"(RawReward={episode_reward_raw:.4f}, PPOReward={episode_reward_blended:.4f}) ---\n"
                )
                for si in step_infos:
                    _write_noise_step_info(si, f)
                    f.write("\n")

            final_noise_config = {
                "input_noise_scaling_factors": np.array(env.input_noise_config, dtype=int),
                "wq_noise_scaling_factors": np.array(env.wq_noise_config, dtype=int),
                "wk_noise_scaling_factors": np.array(env.wk_noise_config, dtype=int),
                "wv_noise_scaling_factors": np.array(env.wv_noise_config, dtype=int),
                "wo_noise_scaling_factors": np.array(env.wo_noise_config, dtype=int),
                "wffn1_noise_scaling_factors": np.array(env.wffn1_noise_config, dtype=int),
                "wffn2_noise_scaling_factors": np.array(env.wffn2_noise_config, dtype=int),
                "cost": env.accumulated_cost,
                "reward": episode_reward_raw,
                "ppo_reward": episode_reward_blended,
                "raw_final_reward": episode_raw_final_reward,
                "blended_final_reward": episode_blended_final_reward,
                "reward_estimator_pred": episode_reward_estimator_pred,
                "reward_blend_alpha": episode_reward_blend_alpha,
                "mc_eval": dict(episode_mc_eval) if episode_mc_eval is not None else None,
                "reward_components": (
                    dict(env.last_reward_components)
                    if getattr(env, "last_reward_components", None) is not None
                    else None
                ),
            }

            if episode_reward_raw > window_best_reward or (
                episode_reward_raw == window_best_reward and env.accumulated_cost < window_best_cost
            ):
                window_best_reward = episode_reward_raw
                window_best_cost = env.accumulated_cost
                window_best_noise_config = {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else dict(value)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in final_noise_config.items()
                }

            if episode_reward_raw > best_reward or (
                episode_reward_raw == best_reward and env.accumulated_cost < best_cost
            ):
                best_reward = episode_reward_raw
                best_cost = env.accumulated_cost
                best_noise_config = {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else dict(value)
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in final_noise_config.items()
                }
                ev.log(
                    f"  Noise Episode {episode + 1}: New Best! "
                    f"RawReward={episode_reward_raw:.4f}, PPOReward={episode_reward_blended:.4f}, "
                    f"Cost={env.accumulated_cost:.2f}"
                )
                if episode_mc_eval is not None:
                    ev.log(
                        "    MC Eval: "
                        f"Loss={episode_mc_eval['loss_mean']:.4f}±{episode_mc_eval['loss_std']:.4f}, "
                        f"M1={episode_mc_eval['metric1_mean']:.4f}±{episode_mc_eval['metric1_std']:.4f}, "
                        f"M2={episode_mc_eval['metric2_mean']:.4f}±{episode_mc_eval['metric2_std']:.4f}"
                    )
                ev.log(f"    x     : {env.input_noise_config}")
                ev.log(f"    wq    : {env.wq_noise_config}")
                ev.log(f"    wk    : {env.wk_noise_config}")
                ev.log(f"    wv    : {env.wv_noise_config}")
                ev.log(f"    wo    : {env.wo_noise_config}")
                ev.log(f"    wffn1 : {env.wffn1_noise_config}")
                ev.log(f"    wffn2 : {env.wffn2_noise_config}")

            if (episode + 1) % PPO_UPDATE_INTERVAL == 0:
                policy_loss, value_loss, entropy = _ppo_update_noise_gtrxl(
                    ev, noise_net, optimizer, buffer, ev.device,
                    entropy_coef=current_entropy,
                    ppo_update_step=noise_ppo_update_count,
                    ppo_eps_clip=NOISE_STAGE_PPO_EPS_CLIP,
                    ppo_k_epochs=NOISE_STAGE_PPO_K_EPOCHS,
                    ppo_value_coef=PPO_VALUE_COEF,
                    gtrxl_warmup_steps=NOISE_STAGE_GTRXL_WARMUP_STEPS,
                    gtrxl_entropy_lower_bound=GTRXL_ENTROPY_LOWER_BOUND,
                    gtrxl_mini_batch_episodes=GTRXL_MINI_BATCH_EPISODES,
                    value_clip_range=VALUE_CLIP_RANGE,
                )
                noise_ppo_update_count += 1
                reward_estimator_loss = _train_noise_reward_estimator(
                    reward_estimator,
                    reward_estimator_optimizer,
                    reward_replay,
                    ev.device,
                    warmup_episodes=NOISE_REWARD_ESTIMATOR_WARMUP_EPISODES,
                    batch_size=NOISE_REWARD_ESTIMATOR_BATCH_SIZE,
                    epochs=NOISE_REWARD_ESTIMATOR_EPOCHS,
                )
                if reward_estimator_loss is not None:
                    reward_estimator_last_loss = reward_estimator_loss
                    reward_estimator_update_count += 1
                buffer.clear()
                episode_entropies.append(entropy)
                avg_reward = np.mean(episode_rewards[-PPO_UPDATE_INTERVAL:])
                avg_blended_reward = np.mean(episode_blended_rewards[-PPO_UPDATE_INTERVAL:])
                warmup_status = (
                    "warmup"
                    if noise_ppo_update_count <= NOISE_STAGE_GTRXL_WARMUP_STEPS
                    else "normal"
                )
                ev.log(
                    f"  Noise Episode {episode + 1}: Avg Raw Reward={avg_reward:.4f}, "
                    f"Avg PPO Reward={avg_blended_reward:.4f}, Policy Loss={policy_loss:.4f}, "
                    f"Value Loss={value_loss:.4f}, Entropy={entropy:.4f}"
                )
                ev.log(
                    f"    [Noise GTrXL Schedule] LR={optimizer.param_groups[0]['lr']:.6f}, "
                    f"Entropy Coef={current_entropy:.6f}, Update#{noise_ppo_update_count} ({warmup_status})"
                )
                ev.log(
                    "    [Noise Reward Estimator] "
                    f"Replay={len(reward_replay)}, Updates={reward_estimator_update_count}, "
                    f"LastLoss={reward_estimator_last_loss if reward_estimator_last_loss is not None else 'N/A'}"
                )
                confirm_noise_candidate(
                    window_best_noise_config,
                    episode_idx=episode,
                    window_idx=noise_ppo_update_count - 1,
                )
                window_best_reward = float("-inf")
                window_best_cost = float("inf")
                window_best_noise_config = None

                if USE_VALIDATION_FOR_REWARD and (episode + 1) < NOISE_STAGE_PPO_MAX_EPISODES:
                    next_window_idx = noise_ppo_update_count
                    ev.refresh_validation_proxy(
                        window_index=next_window_idx,
                        stage_label="Stage-2 Noise RL",
                    )
                    online_reward_split = ev.get_online_reward_split_name()
                    rl_evaluator.split_name = online_reward_split
                    proxy_baseline = ev.evaluate_model_with_attention_noise(
                        fixed_gelu,
                        fixed_softmax,
                        split=online_reward_split,
                        **baseline_noise_config,
                    )
                    env.prev_episode_metrics = {
                        "loss": float(proxy_baseline[0]),
                        "metric1": float(proxy_baseline[1]),
                        "metric2": float(proxy_baseline[2]),
                        "cost": float(baseline_tot_c),
                    }
                    env.current_episode_metrics = None

        if window_best_noise_config is not None:
            confirm_noise_candidate(
                window_best_noise_config,
                episode_idx=NOISE_STAGE_PPO_MAX_EPISODES - 1,
                window_idx=noise_ppo_update_count,
            )

        best_noise_config = None
        if global_best_noise_config is not None:
            best_noise_config = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in global_best_noise_config.items()
            }
        elif search_best_noise_config is not None:
            best_noise_config = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in search_best_noise_config.items()
            }

        if best_noise_config is not None:
            best_reward = max(best_reward, 0.0)

        if best_noise_config is None or best_reward < -50:
            ev.log("\nNo feasible noise-stage solution found, using max-scaling baseline configuration.")
            best_noise_config = {key: value.copy() for key, value in baseline_noise_config.items()}
            best_noise_config["cost"] = baseline_tot_c
            best_noise_config["reward"] = 0.0
            best_noise_config["ppo_reward"] = 0.0
            best_noise_config["raw_final_reward"] = 0.0
            best_noise_config["blended_final_reward"] = 0.0
            best_noise_config["reward_estimator_pred"] = None
            best_noise_config["reward_blend_alpha"] = None
            best_noise_config["mc_eval"] = None
            best_noise_config["reward_components"] = None

        ev.log("\n--- Noise PPO Training Completed ---")
        ev.log("Best Noise Configuration Found:")
        for key in (
            "input_noise_scaling_factors",
            "wq_noise_scaling_factors",
            "wk_noise_scaling_factors",
            "wv_noise_scaling_factors",
            "wo_noise_scaling_factors",
            "wffn1_noise_scaling_factors",
            "wffn2_noise_scaling_factors",
        ):
            ev.log(f"  {key}: {best_noise_config[key].tolist()}")
        ev.log(
            f"  Cost: {best_noise_config['cost']:.2f}, Reward: {best_noise_config['reward']:.4f}, "
            f"PPO Reward: {best_noise_config.get('ppo_reward', best_noise_config['reward']):.4f}"
        )

        _plot_noise_training_curves(
            ev, episode_rewards, episode_losses, episode_metric1s, episode_metric2s, episode_entropies,
            base_loss=base_loss, base_p=base_p, base_s=base_s,
            training_curve_path=NOISE_STAGE_TRAINING_CURVE_PATH,
            entropy_curve_path=NOISE_STAGE_ENTROPY_CURVE_PATH,
            ppo_update_interval=PPO_UPDATE_INTERVAL,
            use_validation=USE_VALIDATION_FOR_REWARD,
        )

        ev.total_episodes = original_total_episodes
        ev.apply_configuration(fixed_gelu, fixed_softmax)
        ev.clear_input_noise_configuration()
        ev.clear_weight_noise_configuration()

        reward_diagnostics = {
            "mc_samples": NOISE_STAGE_MC_SAMPLES,
            "reward_estimator_replay_size": len(reward_replay),
            "reward_estimator_updates": reward_estimator_update_count,
            "reward_estimator_last_loss": (
                float(reward_estimator_last_loss)
                if reward_estimator_last_loss is not None
                else None
            ),
            "blend_alpha_mean": (
                float(np.mean(reward_blend_alphas))
                if reward_blend_alphas
                else None
            ),
            "prediction_abs_error_mean": (
                float(np.mean(reward_prediction_abs_errors))
                if reward_prediction_abs_errors
                else None
            ),
            "raw_final_reward_mean": (
                float(np.mean(raw_final_rewards))
                if raw_final_rewards
                else None
            ),
            "blended_final_reward_mean": (
                float(np.mean(blended_final_rewards))
                if blended_final_rewards
                else None
            ),
        }

        return {
            "fixed_gelu": fixed_gelu.copy(),
            "fixed_softmax": fixed_softmax.copy(),
            "baseline_noise_config": {k: v.copy() for k, v in baseline_noise_config.items()},
            "baseline_tot_c": float(baseline_tot_c),
            "best_noise_config": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in best_noise_config.items()},
            "limit_loss": float(limit_loss),
            "limit_p": float(limit_p),
            "limit_s": float(limit_s),
            "training_hparams": training_hparams,
            "reward_diagnostics": reward_diagnostics,
        }


# ---------------------------------------------------------------------------
# Internal helpers (module-private)
# ---------------------------------------------------------------------------

class _NoiseRecurrentRolloutBuffer:
    """Rollout buffer for the second-stage 7-action noise RL."""

    def __init__(self):
        self.episodes = []
        self._current = None

    def start_episode(self):
        self._current = {
            "cont_features": [],
            "layer_indices": [],
            "prev_actions": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def add_step(self, cont_feat, layer_idx, prev_actions, actions, logprob, reward, value, done):
        self._current["cont_features"].append(cont_feat)
        self._current["layer_indices"].append(layer_idx)
        self._current["prev_actions"].append(prev_actions)
        self._current["actions"].append(actions)
        self._current["logprobs"].append(logprob)
        self._current["rewards"].append(reward)
        self._current["values"].append(value)
        self._current["dones"].append(done)

    def end_episode(self):
        self.episodes.append(self._current)
        self._current = None

    def clear(self):
        self.episodes.clear()

    @property
    def num_episodes(self):
        return len(self.episodes)

    def get_batch(self, device):
        cont_features = torch.stack([
            torch.stack(ep["cont_features"]) for ep in self.episodes
        ]).to(device)

        layer_indices = torch.stack([
            torch.tensor(ep["layer_indices"], dtype=torch.long) for ep in self.episodes
        ]).to(device)

        prev_actions = torch.stack([
            torch.stack(ep["prev_actions"]) for ep in self.episodes
        ]).to(device)

        actions = torch.stack([
            torch.stack(ep["actions"]) for ep in self.episodes
        ]).to(device)

        logprobs = torch.stack([
            torch.stack(ep["logprobs"]) for ep in self.episodes
        ]).to(device)

        rewards = torch.tensor([
            ep["rewards"] for ep in self.episodes
        ], dtype=torch.float32).to(device)

        values = torch.stack([
            torch.stack(ep["values"]) for ep in self.episodes
        ]).to(device)

        dones = torch.tensor([
            ep["dones"] for ep in self.episodes
        ], dtype=torch.float32).to(device)

        return cont_features, layer_indices, prev_actions, actions, logprobs, rewards, values, dones


class _NoiseRewardReplayBuffer:
    """Replay of episode-level terminal rewards for reward-estimator training."""

    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self.features = deque(maxlen=self.capacity)
        self.targets = deque(maxlen=self.capacity)

    def add(self, feature, target):
        self.features.append(np.asarray(feature, dtype=np.float32).copy())
        self.targets.append(float(target))

    def sample(self, batch_size):
        batch_size = min(int(batch_size), len(self.targets))
        indices = np.random.choice(len(self.targets), size=batch_size, replace=False)
        features = np.stack([self.features[idx] for idx in indices]).astype(np.float32)
        targets = np.array([self.targets[idx] for idx in indices], dtype=np.float32)
        return features, targets

    def __len__(self):
        return len(self.targets)


class _NoiseRewardEstimator(nn.Module):
    """Small MLP that smooths the terminal reward under noisy evaluation."""

    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                gain = np.sqrt(2) if module.out_features != 1 else 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _NoiseGTrXLStrategyNetwork(nn.Module):
    """Second-stage GTrXL actor-critic with 7 independent noise-action heads."""

    action_names = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")

    def __init__(self, num_layers=12, d_model=64,
                 n_heads=4, n_gtrxl_layers=3,
                 d_ff=128, dropout=0.1,
                 gtrxl_block_cls=None,
                 lstm_pos_dim=16, lstm_proj_dim=32,
                 noise_stage_num_actions=7,
                 noise_stage_sos_tokens=None,
                 noise_stage_prev_action_embed_dim=4,
                 noise_stage_cont_dim=6,
                 noise_stage_action_dims=None):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self._action_dims = noise_stage_action_dims
        self._sos_tokens = noise_stage_sos_tokens

        self.embed_layer_idx = nn.Embedding(num_layers, lstm_pos_dim)
        self.prev_action_embeddings = nn.ModuleList([
            nn.Embedding(noise_stage_sos_tokens[i] + 1, noise_stage_prev_action_embed_dim)
            for i in range(noise_stage_num_actions)
        ])
        self.fc_continuous = nn.Sequential(
            nn.Linear(noise_stage_cont_dim, lstm_proj_dim),
            nn.LayerNorm(lstm_proj_dim),
            nn.SiLU()
        )

        token_input_dim = (
            lstm_pos_dim +
            noise_stage_num_actions * noise_stage_prev_action_embed_dim +
            lstm_proj_dim
        )
        self.input_proj = nn.Identity() if token_input_dim == d_model else nn.Linear(token_input_dim, d_model)

        self.gtrxl_blocks = nn.ModuleList([
            gtrxl_block_cls(d_model, n_heads, d_ff, dropout)
            for _ in range(n_gtrxl_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        self.noise_heads = nn.ModuleDict({
            name: nn.Linear(64, noise_stage_action_dims[idx])
            for idx, name in enumerate(self.action_names)
        })

        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self._causal_mask_cache = {}
        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        for head in self.noise_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)

        if isinstance(self.input_proj, nn.Linear):
            nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0.0)

        for block in self.gtrxl_blocks:
            for p in block.attn.in_proj_weight.chunk(3):
                nn.init.orthogonal_(p)
            nn.init.orthogonal_(block.attn.out_proj.weight)
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def _get_causal_mask(self, seq_len, device):
        if seq_len not in self._causal_mask_cache or self._causal_mask_cache[seq_len].device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def _build_tokens(self, cont_features, layer_indices, prev_actions):
        emb_l = self.embed_layer_idx(layer_indices)
        prev_embs = [
            emb(prev_actions[:, :, idx])
            for idx, emb in enumerate(self.prev_action_embeddings)
        ]
        feat_c = self.fc_continuous(cont_features)
        token_input = torch.cat([emb_l, *prev_embs, feat_c], dim=-1)
        return self.input_proj(token_input)

    def forward(self, cont_features, layer_indices, prev_actions, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_actions)
        seq_len = tokens.size(1)
        causal_mask = self._get_causal_mask(seq_len, tokens.device)

        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        actor_feat = self.actor_head(x)
        logits = {name: self.noise_heads[name](actor_feat) for name in self.action_names}
        values = self.critic_head(x).squeeze(-1)
        return logits, values

    def get_action_and_logprob(self, cont_features, layer_indices, prev_actions, return_probs=False):
        logits_dict, values = self.forward(cont_features, layer_indices, prev_actions)
        value = values[:, -1].squeeze(0)

        actions = []
        probs = []
        logprob = torch.zeros((), dtype=torch.float32, device=cont_features.device)
        for name in self.action_names:
            logits = logits_dict[name][:, -1, :].squeeze(0)
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions.append(action)
            logprob = logprob + dist.log_prob(action)
            if return_probs:
                probs.append(torch.softmax(logits, dim=-1))

        actions_tensor = torch.stack(actions)
        if return_probs:
            return actions_tensor, logprob, value, probs
        return actions_tensor, logprob, value

    def evaluate_actions(self, cont_features, layer_indices, prev_actions, actions):
        logits_dict, values = self.forward(cont_features, layer_indices, prev_actions)
        logprobs = torch.zeros_like(values)
        entropy = torch.zeros_like(values)
        for idx, name in enumerate(self.action_names):
            dist = Categorical(logits=logits_dict[name])
            logprobs = logprobs + dist.log_prob(actions[:, :, idx])
            entropy = entropy + dist.entropy()
        return logprobs, entropy, values


class _NoiseOptEnv:
    """Second-stage RL environment over x/Wq/Wk/Wv/Wo/Wffn1/Wffn2 noise scaling factors."""

    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 fixed_gelu, fixed_softmax, constraint_limits=None, prev_metrics=None, num_metrics=2,
                 input_noise_allowed=None, weight_noise_allowed=None,
                 input_noise_cost_map=None, weight_noise_cost_map=None,
                 input_noise_scaling_map=None,
                 wq_noise_scaling_map=None, wk_noise_scaling_map=None,
                 wv_noise_scaling_map=None, wo_noise_scaling_map=None,
                 wffn1_noise_scaling_map=None, wffn2_noise_scaling_map=None,
                 input_noise_scaling_to_norm=None, weight_noise_scaling_to_norm=None,
                 noise_stage_sos_tokens=None, noise_stage_num_actions=7,
                 history_mask_value=0.0,
                 reward_threshold=0.01, reward_dense_scale=0.1,
                 reward_cost_weight=20.0, reward_safety_bonus=1.0,
                 reward_clip_min=-5.0, reward_clip_max=5.0,
                 reward_normalization_scale=20.0,
                 budget_deviation_scale=0.05,
                 diff_reward_scale_acc=50.0, diff_reward_power=0.5,
                 log_barrier_violation_scale=10.0,
                 log_barrier_violation_steepness=20.0,
                 log_barrier_satisfaction_scale=0.5,
                 mc_samples=5):
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        self.fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        self.fixed_softmax = np.asarray(fixed_softmax, dtype=int)
        self.num_metrics = num_metrics

        self._input_noise_allowed = input_noise_allowed
        self._weight_noise_allowed = weight_noise_allowed
        self._input_noise_cost_map = input_noise_cost_map
        self._weight_noise_cost_map = weight_noise_cost_map
        self._input_noise_scaling_map = input_noise_scaling_map
        self._wq_map = wq_noise_scaling_map
        self._wk_map = wk_noise_scaling_map
        self._wv_map = wv_noise_scaling_map
        self._wo_map = wo_noise_scaling_map
        self._wffn1_map = wffn1_noise_scaling_map
        self._wffn2_map = wffn2_noise_scaling_map
        self._input_noise_to_norm = input_noise_scaling_to_norm
        self._weight_noise_to_norm = weight_noise_scaling_to_norm
        self._sos_tokens = noise_stage_sos_tokens
        self._num_actions = noise_stage_num_actions
        self._history_mask = history_mask_value
        self._reward_threshold = reward_threshold
        self._reward_dense_scale = reward_dense_scale
        self._reward_cost_weight = reward_cost_weight
        self._reward_safety_bonus = reward_safety_bonus
        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max
        self._reward_norm_scale = reward_normalization_scale
        self._budget_dev_scale = budget_deviation_scale
        self._diff_reward_scale_acc = diff_reward_scale_acc
        self._diff_reward_power = diff_reward_power
        self._log_barrier_viol_scale = log_barrier_violation_scale
        self._log_barrier_viol_steep = log_barrier_violation_steepness
        self._log_barrier_sat_scale = log_barrier_satisfaction_scale
        self._mc_samples = max(1, int(mc_samples))

        if constraint_limits is None:
            self.constraint_limits = {
                "loss": self.baseline_loss * (1 + self._reward_threshold),
                "metric1": self.baseline_p * (1 - self._reward_threshold),
                "metric2": self.baseline_s * (1 - self._reward_threshold),
            }
        else:
            self.constraint_limits = constraint_limits

        if prev_metrics is None:
            self.prev_episode_metrics = {
                "loss": self.baseline_loss,
                "metric1": self.baseline_p,
                "metric2": self.baseline_s,
                "cost": self.baseline_cost,
            }
        else:
            self.prev_episode_metrics = prev_metrics

        self.max_cost_per_layer = (
            self._input_noise_cost_map[max(self._input_noise_allowed)] +
            6 * self._weight_noise_cost_map[max(self._weight_noise_allowed)]
        )
        self.expected_cost_per_layer = (
            np.mean([self._input_noise_cost_map[sf] for sf in self._input_noise_allowed]) +
            6 * np.mean([self._weight_noise_cost_map[sf] for sf in self._weight_noise_allowed])
        )
        self.current_episode_metrics = None
        self.last_reward_components = None
        self.last_mc_eval = None
        self.reset()

    def reset(self):
        self.current_layer = 0
        self.accumulated_cost = 0.0
        self.current_episode_metrics = None
        self.last_reward_components = None
        self.last_mc_eval = None
        self.input_noise_config = []
        self.wq_noise_config = []
        self.wk_noise_config = []
        self.wv_noise_config = []
        self.wo_noise_config = []
        self.wffn1_noise_config = []
        self.wffn2_noise_config = []

        self.prev_action_indices = np.array(self._sos_tokens, dtype=np.int64)
        self.prev_scalings = {
            "x": max(self._input_noise_allowed),
            "wq": max(self._weight_noise_allowed),
            "wk": max(self._weight_noise_allowed),
            "wv": max(self._weight_noise_allowed),
            "wo": max(self._weight_noise_allowed),
            "wffn1": max(self._weight_noise_allowed),
            "wffn2": max(self._weight_noise_allowed),
        }

        self.accumulated_dense_reward = 0.0
        self.input_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wq_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wk_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wv_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wo_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn1_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        self.wffn2_history = np.full(self.total_layers, self._history_mask, dtype=np.float32)
        return self._get_state()

    def _get_budget_features(self):
        prev_loss = self.prev_episode_metrics["loss"]
        prev_m1 = self.prev_episode_metrics["metric1"]
        prev_m2 = self.prev_episode_metrics["metric2"]
        current_limits = self._get_current_constraint_limits()

        loss_budget = 1.0 - prev_loss / (current_limits["loss"] + 1e-8)
        m1_budget = prev_m1 / (current_limits["metric1"] + 1e-8) - 1.0
        if self.num_metrics == 1:
            m2_budget = 0.0
        else:
            m2_budget = prev_m2 / (current_limits["metric2"] + 1e-8) - 1.0

        return (
            np.clip(loss_budget, -1.0, 1.0),
            np.clip(m1_budget, -1.0, 1.0),
            np.clip(m2_budget, -1.0, 1.0),
        )

    def get_continuous_features(self):
        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0
        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)

        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0
        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)

        progress = self.current_layer / self.total_layers
        loss_budget, m1_budget, m2_budget = self._get_budget_features()
        return np.array(
            [cost_deviation, complexity_debt, progress, loss_budget, m1_budget, m2_budget],
            dtype=np.float32,
        )

    def update_prev_metrics(self):
        if self.current_episode_metrics is not None:
            self.prev_episode_metrics = self.current_episode_metrics.copy()

    def get_reward_estimator_features(self):
        cost_ratio = self.accumulated_cost / (self.baseline_cost + 1e-8)
        return np.concatenate([
            self.input_history,
            self.wq_history,
            self.wk_history,
            self.wv_history,
            self.wo_history,
            self.wffn1_history,
            self.wffn2_history,
            np.array([cost_ratio], dtype=np.float32),
        ]).astype(np.float32)

    def _get_current_constraint_limits(self):
        base_limits = {
            "loss": float(self.constraint_limits["loss"]),
            "metric1": float(self.constraint_limits["metric1"]),
            "metric2": float(self.constraint_limits["metric2"]),
        }
        base_evaluator = getattr(self.evaluator, "evaluator", None)
        if base_evaluator is not None and hasattr(base_evaluator, "get_curriculum_constraints"):
            return base_evaluator.get_curriculum_constraints(base_limits)
        return base_limits

    def _evaluate_noise_config_mc(self):
        losses = []
        metric1s = []
        metric2s = []
        times = []
        noise_kwargs = {
            "input_noise_scaling_factors": np.array(self.input_noise_config, dtype=int),
            "wq_noise_scaling_factors": np.array(self.wq_noise_config, dtype=int),
            "wk_noise_scaling_factors": np.array(self.wk_noise_config, dtype=int),
            "wv_noise_scaling_factors": np.array(self.wv_noise_config, dtype=int),
            "wo_noise_scaling_factors": np.array(self.wo_noise_config, dtype=int),
            "wffn1_noise_scaling_factors": np.array(self.wffn1_noise_config, dtype=int),
            "wffn2_noise_scaling_factors": np.array(self.wffn2_noise_config, dtype=int),
        }
        for _ in range(self._mc_samples):
            loss, metric1, metric2, eval_time = self.evaluator.evaluate_noise_model(**noise_kwargs)
            losses.append(float(loss))
            metric1s.append(float(metric1))
            metric2s.append(float(metric2))
            times.append(float(eval_time))
        return {
            "num_samples": self._mc_samples,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "metric1_mean": float(np.mean(metric1s)),
            "metric1_std": float(np.std(metric1s)),
            "metric2_mean": float(np.mean(metric2s)),
            "metric2_std": float(np.std(metric2s)),
            "time_mean_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
        }

    def _assemble_final_reward(self, loss, m1, m2):
        limits = self._get_current_constraint_limits()

        delta_loss = self.prev_episode_metrics["loss"] - loss
        delta_m1 = m1 - self.prev_episode_metrics["metric1"]
        delta_m2 = m2 - self.prev_episode_metrics["metric2"]

        def amplify_signal(delta):
            sign = 1.0 if delta >= 0 else -1.0
            return sign * (abs(delta) ** self._diff_reward_power) * self._diff_reward_scale_acc

        r_loss_diff = amplify_signal(delta_loss)
        r_m1_diff = amplify_signal(delta_m1)
        r_m2_diff = amplify_signal(delta_m2)
        if self.num_metrics == 1:
            r_diff = (r_loss_diff + r_m1_diff) / 2.0
        else:
            r_diff = (r_loss_diff + r_m1_diff + r_m2_diff) / 3.0

        def violation_penalty(curr_value, limit_value, is_upper_bound=True):
            margin = (limit_value - curr_value) if is_upper_bound else (curr_value - limit_value)
            if margin < 0:
                penalty = -self._log_barrier_viol_scale * np.exp(-margin * self._log_barrier_viol_steep)
            else:
                penalty = 0.0
            return penalty, margin

        r_loss_barrier, margin_loss = violation_penalty(loss, limits["loss"], is_upper_bound=True)
        r_m1_barrier, margin_m1 = violation_penalty(m1, limits["metric1"], is_upper_bound=False)
        r_m2_barrier, margin_m2 = violation_penalty(m2, limits["metric2"], is_upper_bound=False)
        if self.num_metrics == 1:
            r_barrier = (r_loss_barrier + r_m1_barrier) / 2.0
            positive_margins = [max(0.0, margin_loss), max(0.0, margin_m1)]
            constraints_ok = (margin_loss >= 0) and (margin_m1 >= 0)
        else:
            r_barrier = (r_loss_barrier + r_m1_barrier + r_m2_barrier) / 3.0
            positive_margins = [max(0.0, margin_loss), max(0.0, margin_m1), max(0.0, margin_m2)]
            constraints_ok = (margin_loss >= 0) and (margin_m1 >= 0) and (margin_m2 >= 0)

        cost_saving = (self.baseline_cost - self.accumulated_cost) / (self.baseline_cost + 1e-8)
        r_cost = cost_saving * self._reward_cost_weight
        r_safe = 0.0
        if constraints_ok:
            r_safe = self._reward_safety_bonus + self._log_barrier_sat_scale * float(np.mean(positive_margins))

        raw_reward = np.clip(
            (r_cost + r_diff + r_barrier + r_safe) / self._reward_norm_scale,
            self._reward_clip_min,
            self._reward_clip_max,
        )
        reward_components = {
            "cost": float(r_cost),
            "diff": float(r_diff),
            "barrier": float(r_barrier),
            "safety": float(r_safe),
            "constraints_ok": bool(constraints_ok),
            "loss_limit": float(limits["loss"]),
            "metric1_limit": float(limits["metric1"]),
            "metric2_limit": float(limits["metric2"]),
            "margin_loss": float(margin_loss),
            "margin_metric1": float(margin_m1),
            "margin_metric2": float(margin_m2),
            "raw_final_reward": float(raw_reward),
        }
        self.last_cost_reward = float(r_cost / self._reward_norm_scale)
        self.last_acc_reward = float((r_diff + r_barrier + r_safe) / self._reward_norm_scale)
        self.last_reward_components = reward_components
        return float(raw_reward), reward_components

    def _get_state(self):
        position = np.zeros(self.total_layers, dtype=np.float32)
        if self.current_layer < self.total_layers:
            position[self.current_layer] = 1.0

        prev_norms = np.array([
            self._input_noise_to_norm[self.prev_scalings["x"]],
            self._weight_noise_to_norm[self.prev_scalings["wq"]],
            self._weight_noise_to_norm[self.prev_scalings["wk"]],
            self._weight_noise_to_norm[self.prev_scalings["wv"]],
            self._weight_noise_to_norm[self.prev_scalings["wo"]],
            self._weight_noise_to_norm[self.prev_scalings["wffn1"]],
            self._weight_noise_to_norm[self.prev_scalings["wffn2"]],
        ], dtype=np.float32)

        cont = self.get_continuous_features()
        state = np.concatenate([
            position,
            [cont[0]],
            prev_norms,
            [cont[1]],
            [cont[2]],
            self.input_history,
            self.wq_history,
            self.wk_history,
            self.wv_history,
            self.wo_history,
            self.wffn1_history,
            self.wffn2_history,
            cont[3:],
        ])
        return state.astype(np.float32)

    def _compute_dense_step_reward(self, step_cost):
        cost_saving = (self.max_cost_per_layer - step_cost) / self.max_cost_per_layer
        cost_reward = self._reward_dense_scale * cost_saving

        layers_completed = self.current_layer + 1
        expected_cost_so_far = layers_completed * self.expected_cost_per_layer
        actual_cost_so_far = self.accumulated_cost + step_cost
        if expected_cost_so_far > 0:
            budget_deviation = (actual_cost_so_far - expected_cost_so_far) / expected_cost_so_far
        else:
            budget_deviation = 0.0

        if budget_deviation <= 0:
            budget_reward = self._budget_dev_scale * (1.0 - abs(budget_deviation) * 0.5)
        else:
            budget_reward = -self._budget_dev_scale * budget_deviation
        return cost_reward + budget_reward

    def step(self, input_action_idx, wq_action_idx, wk_action_idx, wv_action_idx,
             wo_action_idx, wffn1_action_idx, wffn2_action_idx):
        input_sf = self._input_noise_scaling_map[int(input_action_idx)]
        wq_sf = self._wq_map[int(wq_action_idx)]
        wk_sf = self._wk_map[int(wk_action_idx)]
        wv_sf = self._wv_map[int(wv_action_idx)]
        wo_sf = self._wo_map[int(wo_action_idx)]
        wffn1_sf = self._wffn1_map[int(wffn1_action_idx)]
        wffn2_sf = self._wffn2_map[int(wffn2_action_idx)]

        self.input_noise_config.append(input_sf)
        self.wq_noise_config.append(wq_sf)
        self.wk_noise_config.append(wk_sf)
        self.wv_noise_config.append(wv_sf)
        self.wo_noise_config.append(wo_sf)
        self.wffn1_noise_config.append(wffn1_sf)
        self.wffn2_noise_config.append(wffn2_sf)

        step_cost = (
            self._input_noise_cost_map[input_sf] +
            self._weight_noise_cost_map[wq_sf] +
            self._weight_noise_cost_map[wk_sf] +
            self._weight_noise_cost_map[wv_sf] +
            self._weight_noise_cost_map[wo_sf] +
            self._weight_noise_cost_map[wffn1_sf] +
            self._weight_noise_cost_map[wffn2_sf]
        )
        self.accumulated_cost += step_cost

        self.prev_action_indices = np.array([
            int(input_action_idx), int(wq_action_idx), int(wk_action_idx), int(wv_action_idx),
            int(wo_action_idx), int(wffn1_action_idx), int(wffn2_action_idx)
        ], dtype=np.int64)
        self.prev_scalings = {
            "x": input_sf,
            "wq": wq_sf,
            "wk": wk_sf,
            "wv": wv_sf,
            "wo": wo_sf,
            "wffn1": wffn1_sf,
            "wffn2": wffn2_sf,
        }

        self.input_history[self.current_layer] = self._input_noise_to_norm[input_sf]
        self.wq_history[self.current_layer] = self._weight_noise_to_norm[wq_sf]
        self.wk_history[self.current_layer] = self._weight_noise_to_norm[wk_sf]
        self.wv_history[self.current_layer] = self._weight_noise_to_norm[wv_sf]
        self.wo_history[self.current_layer] = self._weight_noise_to_norm[wo_sf]
        self.wffn1_history[self.current_layer] = self._weight_noise_to_norm[wffn1_sf]
        self.wffn2_history[self.current_layer] = self._weight_noise_to_norm[wffn2_sf]

        dense_reward = self._compute_dense_step_reward(step_cost)
        self.accumulated_dense_reward += dense_reward

        info = {
            "layer_index": self.current_layer,
            "curr_input_noise_scaling_factor": input_sf,
            "curr_wq_noise_scaling_factor": wq_sf,
            "curr_wk_noise_scaling_factor": wk_sf,
            "curr_wv_noise_scaling_factor": wv_sf,
            "curr_wo_noise_scaling_factor": wo_sf,
            "curr_wffn1_noise_scaling_factor": wffn1_sf,
            "curr_wffn2_noise_scaling_factor": wffn2_sf,
            "accumulated_cost": self.accumulated_cost,
            "input_noise_config": self.input_noise_config.copy(),
            "wq_noise_config": self.wq_noise_config.copy(),
            "wk_noise_config": self.wk_noise_config.copy(),
            "wv_noise_config": self.wv_noise_config.copy(),
            "wo_noise_config": self.wo_noise_config.copy(),
            "wffn1_noise_config": self.wffn1_noise_config.copy(),
            "wffn2_noise_config": self.wffn2_noise_config.copy(),
            "dense_reward": dense_reward,
        }

        self.current_layer += 1
        if self.current_layer < self.total_layers:
            return self._get_state(), dense_reward, False, info

        final_reward = self._compute_final_reward()
        info["final_reward"] = final_reward["raw_final_reward"]
        info["raw_final_reward"] = final_reward["raw_final_reward"]
        info["mc_eval"] = final_reward["mc_eval"]
        info["reward_components"] = final_reward["reward_components"]
        info["accumulated_dense_reward"] = self.accumulated_dense_reward
        return self._get_state(), final_reward["raw_final_reward"] + dense_reward, True, info

    def _compute_final_reward(self):
        mc_eval = self._evaluate_noise_config_mc()
        loss = mc_eval["loss_mean"]
        m1 = mc_eval["metric1_mean"]
        m2 = mc_eval["metric2_mean"]

        self.current_episode_metrics = {
            "loss": loss,
            "metric1": m1,
            "metric2": m2,
            "cost": self.accumulated_cost,
        }
        raw_final_reward, reward_components = self._assemble_final_reward(loss, m1, m2)
        self.last_mc_eval = mc_eval
        return {
            "raw_final_reward": raw_final_reward,
            "mc_eval": mc_eval,
            "reward_components": reward_components,
        }


# ---------------------------------------------------------------------------
# Module-private helper functions
# ---------------------------------------------------------------------------

def _write_noise_step_info(step_info, f):
    f.write(f"  step_global: {step_info['step_global']}\n")
    f.write(f"  episode_id: {step_info['episode_id']}\n")
    f.write(f"  layer_index: {step_info['layer_index']}\n")
    f.write(f"  state_vector: {step_info['state_vector']}\n")
    f.write(f"  curr_input_noise_scaling_factor: {step_info['curr_input_noise_scaling_factor']}\n")
    f.write(f"  curr_wq_noise_scaling_factor: {step_info['curr_wq_noise_scaling_factor']}\n")
    f.write(f"  curr_wk_noise_scaling_factor: {step_info['curr_wk_noise_scaling_factor']}\n")
    f.write(f"  curr_wv_noise_scaling_factor: {step_info['curr_wv_noise_scaling_factor']}\n")
    f.write(f"  curr_wo_noise_scaling_factor: {step_info['curr_wo_noise_scaling_factor']}\n")
    f.write(f"  curr_wffn1_noise_scaling_factor: {step_info['curr_wffn1_noise_scaling_factor']}\n")
    f.write(f"  curr_wffn2_noise_scaling_factor: {step_info['curr_wffn2_noise_scaling_factor']}\n")
    f.write(f"  x_prob_dist: {step_info['x_prob_dist']}\n")
    f.write(f"  wq_prob_dist: {step_info['wq_prob_dist']}\n")
    f.write(f"  wk_prob_dist: {step_info['wk_prob_dist']}\n")
    f.write(f"  wv_prob_dist: {step_info['wv_prob_dist']}\n")
    f.write(f"  wo_prob_dist: {step_info['wo_prob_dist']}\n")
    f.write(f"  wffn1_prob_dist: {step_info['wffn1_prob_dist']}\n")
    f.write(f"  wffn2_prob_dist: {step_info['wffn2_prob_dist']}\n")
    f.write(f"  critic_value: {step_info['critic_value']}\n")
    f.write(f"  accumulated_cost: {step_info['accumulated_cost']}\n")
    f.write(f"  input_noise_config: {step_info['input_noise_config']}\n")
    f.write(f"  wq_noise_config: {step_info['wq_noise_config']}\n")
    f.write(f"  wk_noise_config: {step_info['wk_noise_config']}\n")
    f.write(f"  wv_noise_config: {step_info['wv_noise_config']}\n")
    f.write(f"  wo_noise_config: {step_info['wo_noise_config']}\n")
    f.write(f"  wffn1_noise_config: {step_info['wffn1_noise_config']}\n")
    f.write(f"  wffn2_noise_config: {step_info['wffn2_noise_config']}\n")
    if "current_lr" in step_info:
        f.write(f"  current_lr: {step_info['current_lr']:.6f}\n")
    if "current_entropy_coef" in step_info:
        f.write(f"  current_entropy_coef: {step_info['current_entropy_coef']:.6f}\n")
    if step_info.get("mc_samples") is not None:
        f.write(f"  mc_samples: {step_info['mc_samples']}\n")
        f.write(f"  mc_loss_mean: {step_info['mc_loss_mean']}\n")
        f.write(f"  mc_loss_std: {step_info['mc_loss_std']}\n")
        f.write(f"  mc_metric1_mean: {step_info['mc_metric1_mean']}\n")
        f.write(f"  mc_metric1_std: {step_info['mc_metric1_std']}\n")
        f.write(f"  mc_metric2_mean: {step_info['mc_metric2_mean']}\n")
        f.write(f"  mc_metric2_std: {step_info['mc_metric2_std']}\n")
    if step_info.get("raw_reward") is not None:
        f.write(f"  raw_reward: {step_info['raw_reward']}\n")
    if step_info.get("blended_reward") is not None:
        f.write(f"  blended_reward: {step_info['blended_reward']}\n")
    if step_info.get("reward_estimator_pred") is not None:
        f.write(f"  reward_estimator_pred: {step_info['reward_estimator_pred']}\n")
    if step_info.get("reward_estimator_loss") is not None:
        f.write(f"  reward_estimator_loss: {step_info['reward_estimator_loss']}\n")
    if step_info.get("reward_blend_alpha") is not None:
        f.write(f"  reward_blend_alpha: {step_info['reward_blend_alpha']}\n")
    if step_info.get("buffer_step_reward") is not None:
        f.write(f"  buffer_step_reward: {step_info['buffer_step_reward']}\n")


def _predict_noise_reward_estimator(model, feature, device):
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.tensor(feature, dtype=torch.float32, device=device).view(1, -1)
        prediction = model(feature_tensor).squeeze(0)
    return float(prediction.item())


def _train_noise_reward_estimator(
        model, optimizer, replay, device,
        warmup_episodes=32,
        batch_size=64,
        epochs=4):
    if len(replay) < max(1, int(warmup_episodes)):
        return None

    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(int(max(1, epochs))):
        features, targets = replay.sample(batch_size)
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        prediction = model(feature_tensor)
        loss = loss_fn(prediction, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return float(np.mean(losses)) if losses else None


def _compute_noise_reward_blend_alpha(
        episode_idx, total_episodes,
        start_alpha=0.2, end_alpha=0.8):
    progress = float(episode_idx) / max(1, int(total_episodes) - 1)
    progress = np.clip(progress, 0.0, 1.0)
    return float(start_alpha + (end_alpha - start_alpha) * progress)


def _ppo_update_noise_gtrxl(evaluator, noise_net, optimizer, buffer, device,
                            mini_batch_episodes=8, entropy_coef=None,
                            ppo_update_step=0,
                            ppo_eps_clip=0.2, ppo_k_epochs=4,
                            ppo_value_coef=0.5,
                            gtrxl_warmup_steps=500,
                            gtrxl_entropy_lower_bound=0.005,
                            gtrxl_mini_batch_episodes=8,
                            value_clip_range=0.2):
    if entropy_coef is None:
        entropy_coef = evaluator.get_current_entropy_coef()

    if ppo_update_step < gtrxl_warmup_steps:
        warmup_factor = (ppo_update_step + 1) / gtrxl_warmup_steps
        current_lr = evaluator.ppo_lr_initial * warmup_factor
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    (cont_features, layer_indices, prev_actions, actions,
     old_logprobs, rewards, values, dones) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    all_advantages = []
    all_returns = []
    for i in range(n_eps):
        adv, ret = evaluator.compute_gae(
            rewards[i].cpu().numpy(),
            values[i].cpu().numpy(),
            dones[i].cpu().numpy(),
        )
        all_advantages.append(adv)
        all_returns.append(ret)

    advantages = torch.stack(all_advantages).to(device)
    returns = torch.stack(all_returns).to(device)
    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    evaluator.return_normalizer.update(returns)
    returns_normalized = torch.tensor(
        evaluator.return_normalizer.normalize(returns.cpu().numpy()),
        dtype=torch.float32
    ).to(device)
    values_normalized = torch.tensor(
        evaluator.return_normalizer.normalize(values.cpu().numpy()),
        dtype=torch.float32
    ).to(device)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(ppo_k_epochs):
        ep_indices = torch.randperm(n_eps)
        for start in range(0, n_eps, gtrxl_mini_batch_episodes):
            end = min(start + gtrxl_mini_batch_episodes, n_eps)
            mb_idx = ep_indices[start:end]

            mb_cont = cont_features[mb_idx]
            mb_layer = layer_indices[mb_idx]
            mb_prev_actions = prev_actions[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_lp = old_logprobs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns_normalized[mb_idx]
            mb_old_val = values_normalized[mb_idx]

            new_logprobs, entropy, new_values_raw = noise_net.evaluate_actions(
                mb_cont, mb_layer, mb_prev_actions, mb_actions
            )

            new_logprobs_flat = new_logprobs.reshape(-1)
            entropy_flat = entropy.reshape(-1)
            new_values_flat = new_values_raw.reshape(-1)
            mb_old_lp_flat = mb_old_lp.reshape(-1)
            mb_adv_flat = mb_adv.reshape(-1)
            mb_ret_flat = mb_ret.reshape(-1)
            mb_old_val_flat = mb_old_val.reshape(-1)

            ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
            surr1 = ratios * mb_adv_flat
            surr2 = torch.clamp(ratios, 1 - ppo_eps_clip, 1 + ppo_eps_clip) * mb_adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values_norm = (new_values_flat - evaluator.return_normalizer.mean) / evaluator.return_normalizer.std
            value_clipped = mb_old_val_flat + torch.clamp(
                new_values_norm - mb_old_val_flat,
                -value_clip_range, value_clip_range
            )
            huber_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
            vl_unclipped = huber_loss_fn(new_values_norm, mb_ret_flat)
            vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
            value_loss = torch.max(vl_unclipped, vl_clipped).mean()

            mean_entropy = entropy_flat.mean()
            effective_entropy_coef = entropy_coef
            if mean_entropy.item() < gtrxl_entropy_lower_bound:
                entropy_deficit = gtrxl_entropy_lower_bound - mean_entropy.item()
                effective_entropy_coef = entropy_coef + 10.0 * entropy_deficit

            entropy_loss = -mean_entropy
            loss = policy_loss + ppo_value_coef * value_loss + effective_entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(noise_net.parameters(), 0.5)
            optimizer.step()

            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = mean_entropy.item()

    return last_policy_loss, last_value_loss, last_entropy


def _plot_noise_training_curves(
        evaluator,
        episode_rewards, episode_losses, episode_metric1s, episode_metric2s,
        episode_entropies,
        base_loss, base_p, base_s,
        training_curve_path="noise_ppo_training_curve.png",
        entropy_curve_path="noise_ppo_entropy_curve.png",
        ppo_update_interval=170,
        use_validation=True):
    if len(episode_rewards) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        episodes = np.arange(1, len(episode_rewards) + 1)
        rewards = np.array(episode_rewards, dtype=np.float32)
        losses = np.array(episode_losses, dtype=np.float32)
        metric1s = np.array(episode_metric1s, dtype=np.float32)
        metric2s = np.array(episode_metric2s, dtype=np.float32)
        metric_names_tuple = evaluator.get_metric_names()
        _num_m = evaluator.get_num_metrics()
        _m1_name = metric_names_tuple[0]
        _m2_name = metric_names_tuple[1] if _num_m > 1 else metric_names_tuple[0]
        window = min(50, max(1, len(rewards) // 10))

        def compute_ma(data):
            if len(data) < window:
                return data
            kernel = np.ones(window, dtype=np.float32) / window
            return np.convolve(data, kernel, mode="valid")

        rewards_ma = compute_ma(rewards)
        losses_ma = compute_ma(losses)
        metric1s_ma = compute_ma(metric1s)
        metric2s_ma = compute_ma(metric2s) if _num_m > 1 else None
        episodes_ma = episodes[window - 1:] if len(rewards) >= window else episodes

        dataset_info = f" ({evaluator.data_path})"
        val_guided_info = " [Validation Guided]" if use_validation else ""

        if _num_m == 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Noise PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight="bold")
            ax1, ax2, ax3 = axes
            ax4 = None
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Noise PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight="bold")
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color="blue")
        ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkblue")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward"); ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(episodes, losses, label="Loss", alpha=0.6, color="red")
        ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkred")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss"); ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
        ax2.axhline(y=base_loss, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax2.legend()

        ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color="green")
        ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkgreen")
        ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name); ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
        ax3.axhline(y=base_p, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
        ax3.legend()

        if ax4 is not None:
            ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color="purple")
            ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color="darkviolet")
            ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name); ax4.set_title(f"{_m2_name} (higher is better)"); ax4.grid(True, alpha=0.3)
            ax4.axhline(y=base_s, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")
            ax4.legend()

        plt.tight_layout()
        plt.savefig(training_curve_path, dpi=150)
        plt.close()
        evaluator.log(f"Noise PPO training curves saved to: {training_curve_path}")

        if episode_entropies:
            update_episodes = np.arange(ppo_update_interval, len(episode_rewards) + 1, ppo_update_interval)
            entropies = np.array(episode_entropies, dtype=np.float32)
            if len(update_episodes) == len(entropies):
                fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color="teal", marker="o", markersize=3)
                window_ent = min(5, max(1, len(entropies) // 5))
                if len(entropies) >= window_ent:
                    kernel_ent = np.ones(window_ent, dtype=np.float32) / window_ent
                    ent_ma = np.convolve(entropies, kernel_ent, mode="valid")
                    ax_ent.plot(update_episodes[window_ent - 1:], ent_ma, label=f"Moving Avg ({window_ent})", linewidth=2, color="darkgreen")
                ax_ent.set_xlabel("Episode (at PPO update)")
                ax_ent.set_ylabel("Entropy")
                ax_ent.set_title("Noise PPO Training: Policy Entropy over Episodes")
                ax_ent.grid(True, alpha=0.3)
                ax_ent.legend()
                plt.tight_layout()
                plt.savefig(entropy_curve_path, dpi=150)
                plt.close()
                evaluator.log(f"Noise PPO entropy curve saved to: {entropy_curve_path}")
    except Exception as e:
        evaluator.log(f"[Warning] Failed to plot Noise PPO training curves: {e}")
