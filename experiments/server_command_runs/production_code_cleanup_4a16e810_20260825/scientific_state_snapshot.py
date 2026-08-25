from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import random
import tempfile

import numpy as np
import torch

from blb_stage2_rl.candidate_store import (
    action_hash,
    build_candidate_identity_context,
    candidate_key,
    candidate_rank_key,
    effective_action_hash,
    rescale_cost_rank_key,
)
from blb_stage2_rl.reward import (
    BaselineCostStats,
    EpisodeMetrics,
    RewardWeights,
    compute_reward,
)
from blb_stage2_rl.statistical_constraints import ConstraintAssessment
from final_evaluation_module import UnifiedFinalEvaluationModule

try:
    from stage1_rl.checkpoint import save_stage1_rl_checkpoint
except ImportError:
    from noise_rl_module_v2 import save_stage1_rl_checkpoint


def normalize(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "torch.Tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy.ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


baseline = BaselineCostStats(
    total_bits_sum=820,
    total_fusion_count=31,
    avg_k=10.25,
    loss_mean=0.35,
    loss_std=0.012,
    metric1_mean=0.87,
    metric2_mean=0.84,
    metric1_std=0.009,
    metric2_std=0.011,
    typical_bits_drop=35.0,
    typical_fusion_count=4.0,
    typical_k_drop=0.75,
)
metrics = EpisodeMetrics(
    loss_mean=0.349,
    loss_std=0.008,
    metric1_mean=0.872,
    metric2_mean=0.845,
    metric1_std=0.007,
    metric2_std=0.009,
    loss_max=0.36,
    metric1_min=0.865,
    metric2_min=0.838,
    loss_trials=(0.347, 0.351),
    metric1_trials=(0.87, 0.874),
    metric2_trials=(0.843, 0.847),
    trial_seeds=(101, 102),
)
probability_cases = {
    "p1_precision": (0.40, 0.79),
    "p2_stability": (0.78, 0.41),
    "p3_feasible": (0.82, 0.76),
}
rewards = {}
for name, (precision, stability) in probability_cases.items():
    assessment = ConstraintAssessment(
        loss_precision_probability=precision + 0.04,
        metric1_precision_probability=precision + 0.02,
        metric2_precision_probability=precision,
        loss_stability_probability=stability + 0.04,
        metric1_stability_probability=stability + 0.02,
        metric2_stability_probability=stability,
        precision_probability=precision,
        stability_probability=stability,
        gate_probability=min(precision, stability),
        online_precision_pass=precision >= 0.5,
        online_stability_pass=stability >= 0.5,
    )
    breakdown = compute_reward(
        metrics,
        SimpleNamespace(any_invalid=False),
        8.75,
        baseline,
        weights=RewardWeights(reward_design="robust_constrained"),
        any_invalid=False,
        external_cost_score=0.625,
        external_cost_rank=17.25,
        external_resource_objective={
            "compute_saving": 0.60,
            "communication_saving": 0.65,
            "robust_floor": 0.60,
            "secondary_progress": 0.25,
            "ppo_resource_score": 0.625,
            "compute_shapley_credit": 0.30,
            "communication_shapley_credit": 0.325,
            "layer_resource_rewards": [0.1, 0.2, 0.3],
            "slot_resource_rewards": [[0.1, 0.2], [0.3]],
        },
        constraint_assessment=assessment,
    )
    rewards[name] = normalize(asdict(breakdown))
rewards["invalid"] = normalize(asdict(compute_reward(
    metrics,
    SimpleNamespace(any_invalid=True),
    8.75,
    baseline,
    weights=RewardWeights(reward_design="robust_constrained"),
    any_invalid=True,
    external_cost_score=0.0,
    constraint_assessment=None,
)))

action = [0, 2, 1, 2, 0, 1, 1, 0, 2, 2, 1, 0]
baseline_action = [2] * len(action)
registry = [
    {"global_index": index, "effective": index % 4 != 1}
    for index in range(len(action))
]
identity = build_candidate_identity_context(
    action_space_version="blb_stage2_layerwise_v1",
    registry_hash="r" * 64,
    max_sfs_hash="m" * 64,
    stage1_hash="s" * 64,
    stage1_degrees={"gelu": [4] * 12, "softmax": [6] * 12},
    profile="bert-base",
    rescale_optimizer_mode="in_process",
    rescale_optimizer_root="Rescale_optimizer",
    rescale_optimizer_hash="o" * 64,
    decode_version="layerwise-v1",
    dataset="mrpc",
    model="bert-base",
    metric_policy_version="bootstrap_5x3_v1",
    threshold_policy_hash="t" * 64,
    fidelity="F1",
    mask_schedule_hash="k" * 64,
)
rank_records = {
    "p3": {
        "valid": True, "terminal_priority": 3, "invalid_steps": 0,
        "terminal_reward": 1.625, "total_reward": 1.75,
        "terminal_cost_rank_score": 17.25, "terminal_fusion_gain": 5.0,
        "terminal_k_gain": 2.5, "terminal_bits_gain": 41.0,
    },
    "p2": {
        "valid": True, "terminal_priority": 2, "invalid_steps": 0,
        "stability_violation": 0.03, "terminal_metric1_mean": 0.872,
        "terminal_metric2_mean": 0.845, "terminal_reward": -1.4,
        "total_reward": -1.3,
    },
    "p1": {
        "valid": True, "terminal_priority": 1, "invalid_steps": 0,
        "acc_violation": 0.02, "terminal_metric1_mean": 0.84,
        "terminal_metric2_mean": 0.81, "terminal_reward": -3.2,
        "total_reward": -3.1,
    },
    "fallback": {
        "valid": True, "acc_violation": 0.0, "stability_violation": 0.0,
        "rescale_cost": {"rank_key": [731, 18]},
    },
}
candidate = {
    "raw_hash": action_hash(action),
    "effective_hash": effective_action_hash(action, registry, baseline_action),
    "candidate_key": candidate_key(action, identity),
    "candidate_key_effective": candidate_key(
        action,
        identity,
        effective_action_hash_value=effective_action_hash(
            action, registry, baseline_action,
        ),
    ),
    "identity": identity,
    "rank_keys": {
        name: list(candidate_rank_key(row)) for name, row in rank_records.items()
    },
    "rescale_rank_keys": {
        name: list(rescale_cost_rank_key(row)) for name, row in rank_records.items()
    },
}

evaluator = SimpleNamespace(
    total_layers=12,
    model_type="bert-base",
    dataset_key="mrpc",
    final_eval_only=False,
    final_eval_dir="/tmp/final-eval-unused",
)
final_module = UnifiedFinalEvaluationModule(
    evaluator,
    config_source="json",
    config_path="glue_final_configs_best_ppo.json",
    permutation_trials=0,
    cost_equivalent_trials=0,
    budget_equivalent_trials=0,
    stage1_budget_trials=0,
    stage2_budget_trials=0,
)
final_gelu, final_softmax, final_source = final_module.resolve_stage1_only(None, 12)
final_noise = final_module._resolve_stage2_from_json(12)
final_eval = {
    "stage1_source": final_source,
    "gelu": normalize(final_gelu),
    "softmax": normalize(final_softmax),
    "stage2": normalize(final_noise),
}

random.seed(123456)
np.random.seed(123456)
torch.manual_seed(123456)
model = torch.nn.Linear(3, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
loss = model(torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)).square().sum()
loss.backward()
optimizer.step()
with tempfile.TemporaryDirectory() as td:
    checkpoint_path = str(Path(td) / "stage1.pt")
    fixed_config = {
        "gelu": np.array([4] * 12),
        "softmax": np.array([6] * 12),
    }
    save_stage1_rl_checkpoint(
        path=checkpoint_path,
        gtrxl_net=model,
        optimizer=optimizer,
        episode=119,
        gtrxl_ppo_update_count=1,
        episode_rewards=[1.0, 1.25],
        episode_losses=[0.5, 0.45],
        episode_metric1s=[0.8, 0.82],
        episode_metric2s=[0.7, 0.73],
        episode_entropies=[0.6, 0.55],
        best_reward=1.25,
        best_cost=2.0,
        best_config=fixed_config,
        search_best_config=fixed_config,
        global_best_config=fixed_config,
        window_best_reward=1.25,
        window_best_cost=2.0,
        window_best_config=fixed_config,
        ev_runtime_state={"probe": 256, "split": "train"},
        stage1_prev_avg_reward=1.125,
        stage1_warnings=[],
        dataset_protocol_hash="p" * 64,
        structured_run_id="fixed-run",
        structured_jsonl_sizes={"episodes.jsonl": 101, "ppo_updates.jsonl": 202},
        detail_file_sizes={"ppo_step_info_1-120.txt": 303},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

output = {
    "schema": "production_cleanup_scientific_state_parity_v1",
    "reward": rewards,
    "candidate": normalize(candidate),
    "checkpoint": normalize(checkpoint),
    "final_evaluation": final_eval,
}
print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
