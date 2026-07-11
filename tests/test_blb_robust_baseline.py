import json
from types import SimpleNamespace

import numpy as np
import pytest

from blb_stage2_rl.reward import EpisodeMetrics
from blb_stage2_rl.seed_utils import derive_baseline_group_probe_seed
from blb_stage2_rl.statistical_constraints import DegenerateBaselineVariance


class FakeBaseEnv:
    def __init__(self, group_metrics):
        self.env_cfg = SimpleNamespace(num_trials_per_step=2)
        self.probe_noise_seed = 777
        self._group_metrics = group_metrics
        self.group_seeds = []
        self.actions = []
        self.reset_calls = 0
        self.clear_calls = 0

    def clear_installed_blb(self):
        self.clear_calls += 1

    def reset(self, *, seed):
        self.reset_calls += 1
        return None

    def step(self, action):
        group_idx = len(self.group_seeds)
        self.group_seeds.append(self.probe_noise_seed)
        self.actions.append(tuple(np.asarray(action, dtype=int).tolist()))
        return None, 0.0, True, {"metrics": self._group_metrics(group_idx)}


def _metrics(group_idx, *, degenerate_until=0):
    if group_idx < degenerate_until:
        values = np.ones(5)
        return EpisodeMetrics(
            loss_trials=tuple(values),
            metric1_trials=tuple(values),
            metric2_trials=tuple(values),
            trial_seeds=tuple(range(group_idx * 5, group_idx * 5 + 5)),
        )
    trial = np.arange(5, dtype=float)
    return EpisodeMetrics(
        loss_trials=tuple(0.2 + group_idx * 0.01 + trial * 0.001),
        metric1_trials=tuple(0.8 + group_idx * 0.01 + trial * 0.001),
        metric2_trials=tuple(0.7 + group_idx * 0.01 + trial * 0.001),
        trial_seeds=tuple(range(group_idx * 5, group_idx * 5 + 5)),
    )


def _collect(env):
    from blb_stage2_rl.sequential_runner import _collect_robust_baseline_reference

    return _collect_robust_baseline_reference(
        base_env=env,
        baseline_action_vec=np.asarray([0, 13, 0, 13], dtype=np.int64),
        base_seed=42,
        precision_tolerance=0.001,
        stability_multiplier=2.0,
        bootstrap_samples=64,
    )


def test_healthy_baseline_collects_five_disjoint_groups_and_restores_env_state():
    env = FakeBaseEnv(_metrics)

    reference, summary = _collect(env)

    assert reference.trial_count == 25
    assert len(summary["groups"]) == 5
    assert env.group_seeds == [derive_baseline_group_probe_seed(42, i) for i in range(5)]
    assert len(set(env.group_seeds)) == 5
    assert env.actions == [(0, 13, 0, 13)] * 5
    assert env.env_cfg.num_trials_per_step == 2
    assert env.probe_noise_seed == 777
    assert env.reset_calls == 5
    assert env.clear_calls == 6


def test_degenerate_channels_extend_one_group_at_a_time_until_recovered():
    env = FakeBaseEnv(lambda group_idx: _metrics(group_idx, degenerate_until=5))

    reference, summary = _collect(env)

    assert reference.trial_count == 30
    assert len(summary["groups"]) == 6
    assert env.group_seeds == [derive_baseline_group_probe_seed(42, i) for i in range(6)]


def test_persistent_degenerate_baseline_uses_ten_groups_then_reraises():
    env = FakeBaseEnv(lambda group_idx: _metrics(group_idx, degenerate_until=10))

    with pytest.raises(DegenerateBaselineVariance):
        _collect(env)

    assert len(env.group_seeds) == 10
    assert env.env_cfg.num_trials_per_step == 2
    assert env.probe_noise_seed == 777


def test_non_degeneracy_errors_propagate_without_retrying():
    env = FakeBaseEnv(_metrics)

    def explode(_action):
        raise RuntimeError("bridge broken")

    env.step = explode
    with pytest.raises(RuntimeError, match="bridge broken"):
        _collect(env)

    assert env.env_cfg.num_trials_per_step == 2
    assert env.probe_noise_seed == 777
    assert env.clear_calls == 2


def test_reset_failure_restores_env_state_and_clears_final_install():
    env = FakeBaseEnv(_metrics)

    def explode(*, seed):
        raise RuntimeError("reset broken")

    env.reset = explode
    with pytest.raises(RuntimeError, match="reset broken"):
        _collect(env)

    assert env.env_cfg.num_trials_per_step == 2
    assert env.probe_noise_seed == 777
    assert env.clear_calls == 2


def test_summary_includes_raw_groups_and_direct_two_x_std_limits():
    env = FakeBaseEnv(_metrics)

    reference, summary = _collect(env)

    first_group = summary["groups"][0]
    expected_metrics = _metrics(0)
    assert first_group["group_index"] == 0
    assert first_group["group_probe_seed"] == derive_baseline_group_probe_seed(42, 0)
    assert first_group["trial_seeds"] == list(expected_metrics.trial_seeds)
    assert first_group["loss_trials"] == list(expected_metrics.loss_trials)
    assert first_group["metric1_trials"] == list(expected_metrics.metric1_trials)
    assert first_group["metric2_trials"] == list(expected_metrics.metric2_trials)
    assert summary["pooled"]["trial_count"] == 25
    assert summary["bootstrap"] == {"samples": 64, "seed": 42}
    assert summary["limits"]["loss_std"] == pytest.approx(reference.loss_std * 2.0)
    assert summary["limits"]["metric1_std"] == pytest.approx(reference.metric1_std * 2.0)
    assert summary["limits"]["metric2_std"] == pytest.approx(reference.metric2_std * 2.0)
    json.dumps(summary)


def test_baseline_group_probe_seed_is_reproducible_and_unique_for_first_ten_groups():
    seeds = [derive_baseline_group_probe_seed(314, group_idx) for group_idx in range(10)]

    assert seeds == [derive_baseline_group_probe_seed(314, group_idx) for group_idx in range(10)]
    assert len(set(seeds)) == 10


@pytest.mark.parametrize("base_seed,group_idx", [
    (True, 0),
    (np.bool_(False), 0),
    (1.5, 0),
    ("1", 0),
    (-1, 0),
    (1, True),
    (1, np.bool_(False)),
    (1, 1.5),
    (1, "1"),
    (1, -1),
])
def test_baseline_group_probe_seed_rejects_invalid_integer_inputs(base_seed, group_idx):
    with pytest.raises((TypeError, ValueError)):
        derive_baseline_group_probe_seed(base_seed, group_idx)


def test_baseline_preflight_dispatch_skips_legacy_callback_for_robust_mode():
    from blb_stage2_rl.sequential_runner import _run_legacy_preflight_if_needed

    calls = []
    _run_legacy_preflight_if_needed(
        robust_mode=True,
        run_legacy_preflight=lambda: calls.append("legacy"),
    )
    assert calls == []

    _run_legacy_preflight_if_needed(
        robust_mode=False,
        run_legacy_preflight=lambda: calls.append("legacy"),
    )
    assert calls == ["legacy"]


def test_robust_baseline_config_reads_public_bootstrap_samples_name():
    from blb_stage2_rl.sequential_runner import _resolve_robust_baseline_config

    evaluator = SimpleNamespace(
        stage2_limit_tolerance=0.001,
        stage2_stability_tolerance=9.0,
    )
    configured = SimpleNamespace(
        stage2_stability_multiplier=2.0,
        constraint_bootstrap_samples=123,
    )
    defaulted = SimpleNamespace(stage2_stability_multiplier=2.0)

    assert _resolve_robust_baseline_config(configured, evaluator) == (0.001, 2.0, 123)
    assert _resolve_robust_baseline_config(defaulted, evaluator) == (0.001, 2.0, 4096)


def test_install_robust_reference_replaces_legacy_stability_and_margin_state():
    from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
    from blb_stage2_rl.sequential_runner import _install_robust_baseline_reference

    reference, summary = _collect(FakeBaseEnv(_metrics))
    baseline = BaselineCostStats(
        loss_std=9.0,
        metric1_std=8.0,
        metric2_std=7.0,
        metric1_mean=0.1,
        metric2_mean=0.2,
    )
    weights = RewardWeights(
        baseline_metric1=0.1,
        baseline_metric2=0.2,
        stab_tolerance=9.0,
        stab_floor=0.01,
    )
    base_env = SimpleNamespace(
        loss_threshold=None,
        acc_threshold=None,
        acc_threshold_m2=None,
        stab_threshold=None,
    )

    _install_robust_baseline_reference(base_env, baseline, weights, reference)

    pooled = summary["pooled"]
    assert weights.baseline_metric1 == pooled["metric1_mean"]
    assert weights.baseline_metric2 == pooled["metric2_mean"]
    assert weights.stab_tolerance == reference.stability_multiplier == 2.0
    assert weights.stab_floor == 0.0
    assert baseline.loss_std * weights.stab_tolerance == reference.loss_std_limit
    assert baseline.metric1_std * weights.stab_tolerance == reference.metric1_std_limit
    assert baseline.metric2_std * weights.stab_tolerance == reference.metric2_std_limit
    assert base_env.stab_threshold == reference.loss_std_limit
    assert base_env.acc_threshold == reference.metric1_limit
    assert base_env.acc_threshold_m2 == reference.metric2_limit
    assert base_env.loss_threshold == reference.loss_limit
    assert base_env.statistical_reference is reference


def test_collection_bypasses_robust_dispatch_then_restores_loud_candidate_gate(monkeypatch):
    from blb_stage2_rl import env as env_module
    from blb_stage2_rl.action_space import make_all_max_action_vector
    from blb_stage2_rl.env import BLBStage2Env
    from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
    from blb_stage2_rl.sequential_runner import _collect_robust_baseline_reference

    env = BLBStage2Env.__new__(BLBStage2Env)
    env.baseline = BaselineCostStats(total_bits_sum=100.0, avg_k=13.0)
    env.reward_weights = RewardWeights(reward_design="robust_constrained")
    env.statistical_reference = None
    env.acc_threshold = 0.0
    env.acc_threshold_m2 = 0.0
    env.stab_threshold = 1.0
    env.loss_threshold = None
    env.pareto_cost_archive = None
    env.num_layers = 1
    env.total_action_dim = len(make_all_max_action_vector(1))
    env.env_cfg = SimpleNamespace(
        profile="mrpc", num_trials_per_step=2, persistent_probe_install=False,
    )
    env.probe_noise_seed = 777
    env.max_sfs = object()
    env.rescale_bridge = SimpleNamespace(invoker=SimpleNamespace(baselines={}))
    env.gelu_degree = 4
    env.attn_degree = 6
    env.sync_degree_vectors_from_model = lambda: {}
    env.probe_runner = None
    env.bridge = SimpleNamespace(apply=lambda **_kwargs: None)
    env.clear_installed_blb = lambda: None
    env.reset = lambda *, seed: np.asarray([0.0], dtype=np.float32)
    env._installed_action_hash = None
    env._last_probe_diagnostics = {}
    env._step_idx = 0
    env._last_invalid_rate = 0.0
    env._last_total_bits_norm = 0.0
    env._last_fusion_count = 0.0
    env._build_state = lambda: np.asarray([0.0], dtype=np.float32)
    group_index = {"value": 0}

    def evaluate_probe(_trial_count):
        metrics = _metrics(group_index["value"])
        group_index["value"] += 1
        return metrics

    env._eval_on_probe = evaluate_probe
    env._maybe_borderline_retest = lambda metrics, _info: metrics
    decoded = SimpleNamespace(
        block1_cfgs=[], block2_cfgs=[], block3_cfgs=[], block4_cfgs=[], block5_cfgs=[],
    )
    signals = SimpleNamespace(
        any_invalid=False,
        total_bits_sum=100.0,
        total_fusion_count=0.0,
        valid_block_count=1,
        total_block_count=1,
    )
    cost_eval = SimpleNamespace(
        decoded=decoded, cfgs_dict={}, outputs={}, signals=signals, optimizer_eval_mode="fake",
    )
    monkeypatch.setattr(env_module, "evaluate_action_for_cost", lambda *_args, **_kwargs: cost_eval)

    reference, _summary = _collect_robust_baseline_reference(
        base_env=env,
        baseline_action_vec=make_all_max_action_vector(1),
        base_seed=42,
        precision_tolerance=0.001,
        stability_multiplier=2.0,
        bootstrap_samples=64,
    )

    assert reference.trial_count == 25
    assert env.reward_weights.reward_design == "robust_constrained"
    assert env.statistical_reference is None
    with pytest.raises(RuntimeError, match="statistical_reference"):
        env._compute_terminal_reward(
            _metrics(6), signals,
            action_vec=make_all_max_action_vector(1),
            action_vec_hash="candidate",
            any_invalid=False,
            external_cost_score=0.5,
            external_cost_rank=0.5,
            info={},
        )
