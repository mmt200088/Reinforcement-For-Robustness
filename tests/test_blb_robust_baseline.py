import json
import inspect
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


def test_robust_mode_guards_legacy_preflight_and_uses_public_bootstrap_config_name():
    import blb_stage2_rl.sequential_runner as runner_mod

    source = inspect.getsource(runner_mod.run_sequential_via_runner)
    legacy_preflight = source.index("base_env.step(baseline_action_vec)")
    assert "if not robust_mode:" in source[:legacy_preflight]
    resolver_source = inspect.getsource(runner_mod._resolve_robust_baseline_config)
    assert "constraint_bootstrap_samples" in resolver_source
    assert "stage2_bootstrap_samples" not in resolver_source


def test_robust_baseline_config_reads_public_bootstrap_samples_name():
    from blb_stage2_rl.sequential_runner import _resolve_robust_baseline_config

    evaluator = SimpleNamespace(stage2_limit_tolerance=0.001)
    configured = SimpleNamespace(
        stage2_stability_multiplier=2.0,
        constraint_bootstrap_samples=123,
    )
    defaulted = SimpleNamespace(stage2_stability_multiplier=2.0)

    assert _resolve_robust_baseline_config(configured, evaluator) == (0.001, 2.0, 123)
    assert _resolve_robust_baseline_config(defaulted, evaluator) == (0.001, 2.0, 4096)
