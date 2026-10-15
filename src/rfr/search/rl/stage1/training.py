"""Stage-1 GAE, PPO updates and rollout collection over the shared workflow state."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from rfr.search.rl.stage1 import config as stage1_config
from rfr.search.rl.stage1.rollout import (
    _get_stage1_gelu_mask_templates,
    _stage1_prob_tensors_to_nested_lists,
    _stage1_scalar_tensors_to_float_list,
)
from rfr.search.rl.stage1.env import TransformerOptEnv

def compute_gae(workflow, rewards, values, dones, gamma=stage1_config.PPO_GAMMA, lam=stage1_config.PPO_LAMBDA):
    """Compute generalized advantage estimates for one rollout."""
    advantages = []
    gae = 0


    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values

    return advantages, returns

def compute_gae_batch(workflow, rewards, values, dones, gamma=stage1_config.PPO_GAMMA, lam=stage1_config.PPO_LAMBDA):
    """Batch GAE over all episodes while staying on the tensor device."""
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    gae = torch.zeros(rewards.size(0), dtype=torch.float32, device=rewards.device)

    for t in range(rewards.size(1) - 1, -1, -1):
        if t == rewards.size(1) - 1:
            next_value = torch.zeros_like(gae)
        else:
            next_value = values[:, t + 1]
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_value * not_done - values[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae

    returns = advantages + values
    return advantages, returns

def ppo_update_gtrxl(workflow, gtrxl_net, optimizer, buffer, device,
                      mini_batch_episodes=stage1_config.GTRXL_MINI_BATCH_EPISODES, entropy_coef=None,
                      ppo_update_step=0):
    """Run one warm-started, entropy-regularized GTrXL PPO update."""
    if entropy_coef is None:
        entropy_coef = workflow.get_current_entropy_coef()


    if ppo_update_step < stage1_config.GTRXL_WARMUP_STEPS:
        warmup_factor = (ppo_update_step + 1) / stage1_config.GTRXL_WARMUP_STEPS
        current_lr = workflow.ppo_lr_initial * warmup_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr


    _u_seed = (
        int(getattr(workflow, "random_seed", 42))
        ^ (int(ppo_update_step) * 2654435761)
    ) & 0x7FFFFFFFFFFFFFFF
    torch.manual_seed(_u_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_u_seed)

    (cont_features, layer_indices, prev_g_actions,
     actions_g, old_logprobs, rewards, values, dones, gelu_masks) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    advantages, returns = compute_gae_batch(workflow, rewards, values, dones)

    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    workflow.return_normalizer.update(returns)
    returns_normalized = workflow.return_normalizer.normalize(returns).to(
        device=device, dtype=torch.float32
    )
    values_normalized = workflow.return_normalizer.normalize(values).to(
        device=device, dtype=torch.float32
    )

    last_policy_loss_t = None
    last_value_loss_t = None
    last_entropy_t = None

    kl_early_stop = False
    for epoch in range(stage1_config.PPO_K_EPOCHS):
        if kl_early_stop:
            break
        ep_indices = torch.randperm(n_eps, device=device)
        epoch_kl_acc_t = torch.zeros((), device=device)
        epoch_kl_count = 0

        for start in range(0, n_eps, mini_batch_episodes):
            end = min(start + mini_batch_episodes, n_eps)
            mb_idx = ep_indices[start:end]

            mb_cont = cont_features[mb_idx]
            mb_layer = layer_indices[mb_idx]
            mb_prev_g = prev_g_actions[mb_idx]
            mb_act_g = actions_g[mb_idx]
            mb_old_lp = old_logprobs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns_normalized[mb_idx]
            mb_old_val = values_normalized[mb_idx]
            mb_gelu_mask = gelu_masks[mb_idx] if gelu_masks is not None else None

            new_logprobs, entropy, new_values_raw = gtrxl_net.evaluate_actions(
                mb_cont, mb_layer, mb_prev_g, mb_act_g,
                gelu_mask=mb_gelu_mask
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
            surr2 = torch.clamp(ratios, 1 - stage1_config.PPO_EPS_CLIP, 1 + stage1_config.PPO_EPS_CLIP) * mb_adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            new_values_norm = (new_values_flat - workflow.return_normalizer.mean) / workflow.return_normalizer.std
            value_clipped = mb_old_val_flat + torch.clamp(
                new_values_norm - mb_old_val_flat,
                -stage1_config.VALUE_CLIP_RANGE, stage1_config.VALUE_CLIP_RANGE
            )
            huber_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
            vl_unclipped = huber_loss_fn(new_values_norm, mb_ret_flat)
            vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
            value_loss = torch.max(vl_unclipped, vl_clipped).mean()


            mean_entropy = entropy_flat.mean()
            mean_entropy_detached = mean_entropy.detach()
            entropy_deficit = torch.relu(
                mean_entropy_detached.new_tensor(stage1_config.GTRXL_ENTROPY_LOWER_BOUND)
                - mean_entropy_detached
            )
            effective_entropy_coef = (
                entropy_coef
                + stage1_config.GTRXL_ENTROPY_RECOVERY_MULTIPLIER * entropy_deficit
            )

            entropy_loss = -mean_entropy

            loss = policy_loss + stage1_config.PPO_VALUE_COEF * value_loss + effective_entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gtrxl_net.parameters(), 0.5)
            optimizer.step()


            with torch.no_grad():
                approx_kl_t = (mb_old_lp_flat - new_logprobs_flat).mean()
            epoch_kl_acc_t += approx_kl_t.detach()
            epoch_kl_count += 1

            last_policy_loss_t = policy_loss.detach()
            last_value_loss_t = value_loss.detach()
            last_entropy_t = mean_entropy.detach()

        if epoch_kl_count > 0:
            avg_kl_t = epoch_kl_acc_t / float(epoch_kl_count)
            avg_kl = float(avg_kl_t.item())
            if avg_kl > 1.5 * stage1_config.GTRXL_KL_TARGET:
                kl_early_stop = True

    last_policy_loss = (
        float(last_policy_loss_t.item()) if last_policy_loss_t is not None else 0.0
    )
    last_value_loss = (
        float(last_value_loss_t.item()) if last_value_loss_t is not None else 0.0
    )
    last_entropy = float(last_entropy_t.item()) if last_entropy_t is not None else 0.0
    return last_policy_loss, last_value_loss, last_entropy


def _stage1_collect_episode_in_worker(workflow, *, worker, episode_seed):
    """Run one Stage-1 episode entirely on ``worker``'s own GPU.

    Both the GTrXL policy rollout AND the terminal BERT forward run on
    ``worker.device`` using ``worker.gtrxl_replica`` (a per-worker eval-mode
    copy of the central policy, weight-synced each window). There is NO
    shared policy + lock, so workers no longer serialize on policy access —
    that is the speedup. Determinism is preserved: the action sample is
    seeded from the worker's own device RNG with the GLOBAL-episode-index
    ``episode_seed`` from the runner, and CUDA Philox is device-independent,
    so episode ``g`` samples the same action on any GPU / any GPU count.

    Returns an ``EpisodeRollout`` carrying per-step transitions plus
    per-episode summary metrics for the central bookkeeping.
    """
    from rfr.search.rl.stage1.parallel_runner import EpisodeRollout

    device = worker.device
    policy = worker.gtrxl_replica
    if device.type == "cuda":


        torch.cuda.set_device(device)

    env = worker.env
    state = env.reset()


    prev_g_idx = stage1_config.SOS_TOKEN_GELU

    seq_cont_feats = torch.empty(
        (1, workflow.total_layers, stage1_config.POLICY_CONTINUOUS_DIM),
        dtype=torch.float32,
        device=device,
    )
    seq_layer_indices = torch.empty(
        (1, workflow.total_layers), dtype=torch.long, device=device,
    )
    seq_prev_g = torch.empty(
        (1, workflow.total_layers), dtype=torch.long, device=device,
    )
    seq_gelu_masks = torch.empty(
        (1, workflow.total_layers, int(stage1_config.STAGE1_GELU_ACTION_MASK.size)),
        dtype=torch.bool,
        device=device,
    )
    gelu_mask_np, gelu_mask_t = _get_stage1_gelu_mask_templates(device)
    seq_gelu_masks.copy_(gelu_mask_t.view(1, 1, -1).expand_as(seq_gelu_masks))

    rollout = EpisodeRollout(
        cont_features=[], layer_indices=[], prev_g_actions=[],
        actions_g=[], logprobs=[], rewards=[], values=[],
        dones=[], gelu_masks=[],
        episode_reward=0.0, episode_loss=0.0, episode_metric1=0.0,
        episode_metric2=0.0, episode_cost=0.0,
        gelu_config=[], softmax_config=[], final_config_metrics=None,
        step_infos=[],
    )

    episode_reward = 0.0
    logprob_tensors = []
    value_tensors = []
    gelu_prob_tensors = []
    for step in range(workflow.total_layers):
        N = workflow.total_layers
        layer_idx = int(np.argmax(state[0:N]))


        cont_feat_np = env.get_policy_cont_features()
        cont_feat_record = np.asarray(cont_feat_np, dtype=np.float32)

        cont_feat_t = torch.as_tensor(
            cont_feat_record, dtype=torch.float32, device=device
        )

        seq_cont_feats[0, step].copy_(cont_feat_t)
        seq_layer_indices[0, step] = int(layer_idx)
        seq_prev_g[0, step] = int(prev_g_idx)

        full_cont = seq_cont_feats[:, : step + 1, :]
        full_layer = seq_layer_indices[:, : step + 1]
        full_prev_g = seq_prev_g[:, : step + 1]
        full_gelu_mask = seq_gelu_masks[:, : step + 1, :]


        if device.type == "cuda":
            torch.cuda.manual_seed(int(episode_seed) + step)
        else:
            torch.manual_seed(int(episode_seed) + step)
        with torch.no_grad():
            gelu_action, logprob, value, gelu_probs =\
                policy.get_action_and_logprob(
                    full_cont, full_layer, full_prev_g,
                    return_probs=True, gelu_mask=full_gelu_mask,
                )


        gelu_action_idx = int(gelu_action.item())
        next_state, reward, done, info = env.step(gelu_action_idx)
        logprob_tensors.append(logprob.detach().reshape(()))
        value_tensors.append(value.detach().reshape(()))
        gelu_prob_tensors.append(gelu_probs.detach())

        rollout.cont_features.append(cont_feat_record)
        rollout.layer_indices.append(layer_idx)
        rollout.prev_g_actions.append(prev_g_idx)
        rollout.actions_g.append(gelu_action_idx)
        rollout.rewards.append(float(reward))
        rollout.dones.append(float(done))
        rollout.gelu_masks.append(gelu_mask_np)
        rollout.step_infos.append({
            "episode_id": -1,
            "layer_index": info["layer_index"],
            "state_vector": state.tolist(),
            "gelu_action_index": gelu_action_idx,
            "gelu_action_degree": int(stage1_config.GELU_MAP[gelu_action_idx]),
            "step_reward": float(reward),
            "done": bool(done),
            "logprob": None,
            "curr_gelu_degree": info["curr_gelu_degree"],
            "curr_softmax_degree": info["curr_softmax_degree"],
            "gelu_prob_dist": None,
            "critic_value": None,
            "accumulated_cost": info["accumulated_cost"],
            "gelu_config": info["gelu_config"],
            "softmax_config": info["softmax_config"],
            "current_lr": None,
            "current_entropy_coef": None,
        })

        prev_g_idx = gelu_action_idx

        episode_reward += reward
        state = next_state

    logprob_values = _stage1_scalar_tensors_to_float_list(logprob_tensors)
    critic_values = _stage1_scalar_tensors_to_float_list(value_tensors)
    gelu_prob_dists = _stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)
    rollout.logprobs.extend(logprob_values)
    rollout.values.extend(critic_values)
    for idx, step_info in enumerate(rollout.step_infos):
        step_info["logprob"] = logprob_values[idx]
        step_info["critic_value"] = critic_values[idx]
        step_info["gelu_prob_dist"] = gelu_prob_dists[idx]

    rollout.episode_reward = float(episode_reward)
    rollout.gelu_config = list(env.gelu_config)
    rollout.softmax_config = list(env.softmax_config)
    rollout.episode_cost = float(env.accumulated_cost)
    if env.current_episode_metrics is not None:
        rollout.episode_loss = float(env.current_episode_metrics["loss"])
        rollout.episode_metric1 = float(env.current_episode_metrics["metric1"])
        rollout.episode_metric2 = float(env.current_episode_metrics["metric2"])
        rollout.final_config_metrics = {
            "loss": float(env.current_episode_metrics["loss"]),
            "metric1": float(env.current_episode_metrics["metric1"]),
            "metric2": float(env.current_episode_metrics["metric2"]),
            "cost": float(env.current_episode_metrics["cost"]),
        }
    return rollout

def _build_stage1_parallel_runner(workflow, *, baseline_metrics, base_tot_c,
                                  constraint_limits, eval_split_name,
                                  proxy_prev_metrics):
    """Construct the Stage-1 multi-GPU rollout runner once at training start.

    Returns ``None`` only when ``workflow.stage1_rl_devices`` is empty (then the
    the single-GPU central loop runs). Pass an explicit device list — even
    a single id like ``0`` — to take the global-seeded rollout path, which
    produces identical results for any GPU count (1, 4, 5, ...).

    Arguments mirror the per-worker env setup the single-GPU path does
    inline in ``on_evaluate`` (baseline_metrics, base_tot_c, constraint
    limits, proxy prev metrics for the differential reward chain).
    """
    from rfr.search.rl.stage1.parallel_runner import build_stage1_parallel_runner
    from rfr.search.runtime.device_utils import parse_device_ids

    device_ids = parse_device_ids(workflow.stage1_rl_devices)
    if len(device_ids) < 1:
        return None

    num_metrics = workflow.get_num_metrics()
    total_layers = workflow.total_layers
    evaluator_self = workflow

    def env_factory(model, handler, device, eval_wrapper):
        env = TransformerOptEnv(
            total_layers,
            base_tot_c,
            baseline_metrics,
            eval_wrapper,
            constraint_limits=dict(constraint_limits),
            num_metrics=num_metrics,
        )
        env.prev_episode_metrics = dict(proxy_prev_metrics)
        return env

    def collect_episode(*, worker, episode_seed):
        return _stage1_collect_episode_in_worker(evaluator_self, worker=worker,
            episode_seed=episode_seed,
        )

    return build_stage1_parallel_runner(
        primary_model=workflow.model,
        primary_handler=workflow.reversible_handler,
        evaluator=workflow,
        env_factory=env_factory,
        collect_episode_fn=collect_episode,
        device_ids=device_ids,
        eval_split_name=eval_split_name,
        log_fn=workflow.log,
    )
