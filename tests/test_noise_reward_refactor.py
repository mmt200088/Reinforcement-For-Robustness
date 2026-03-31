"""尾部安全（tail-safe）重构后的 reward 和 tail 指标单元测试。"""
import math

from noise_rl_module import (
    _compute_stage2_reward_components,
    _compute_tail_metrics_from_trials,
    _compute_wilson_lower_bound,
    _bootstrap_cvar_ucb,
    _resolve_stage2_metric_weights,
)


BASELINE_METRICS = (0.30, 0.90, 0.88)
CONSTRAINT_LIMITS = {
    "loss": 0.33,
    "metric1": 0.87,
    "metric2": 0.85,
}
PERF_WEIGHTS = {
    "loss": 0.15,
    "metric1": 0.425,
    "metric2": 0.425,
}
BARRIER_WEIGHTS = {
    "loss": 0.10,
    "metric1": 0.45,
    "metric2": 0.45,
}


def _make_trials(loss_list, m1_list, m2_list):
    return [
        {"loss": l, "metric1": m1, "metric2": m2}
        for l, m1, m2 in zip(loss_list, m1_list, m2_list)
    ]


def _compute_reward(loss, metric1, metric2, *, num_metrics=2, cost_value=40.0, mc_eval=None):
    return _compute_stage2_reward_components(
        loss=loss,
        metric1=metric1,
        metric2=metric2,
        baseline_metrics=BASELINE_METRICS,
        constraint_limits=CONSTRAINT_LIMITS,
        cost_value=cost_value,
        cost_lower_bound=30.0,
        cost_upper_bound=40.0,
        final_reward_alpha_perf=0.75,
        final_reward_alpha_cost=0.25,
        perf_weights=_resolve_stage2_metric_weights(num_metrics, PERF_WEIGHTS, "perf"),
        barrier_weights=_resolve_stage2_metric_weights(num_metrics, BARRIER_WEIGHTS, "barrier"),
        stability_proxy_std_ref=0.008,
        stability_weight=0.10,
        num_metrics=num_metrics,
        mc_eval=mc_eval,
    )


def test_tail_metrics_all_safe_trials():
    """所有试验都安全通过时，violation=0, safe_rate=1。"""
    trials = _make_trials(
        [0.30, 0.31, 0.30, 0.30, 0.30],
        [0.90, 0.89, 0.90, 0.90, 0.90],
        [0.88, 0.87, 0.88, 0.88, 0.88],
    )
    perf_w = _resolve_stage2_metric_weights(2, PERF_WEIGHTS, "perf")
    bar_w = _resolve_stage2_metric_weights(2, BARRIER_WEIGHTS, "barrier")
    tail = _compute_tail_metrics_from_trials(
        trials, CONSTRAINT_LIMITS, BASELINE_METRICS, perf_w, bar_w, 2,
    )
    assert tail["safe_rate"] == 1.0
    assert tail["unsafe_count"] == 0
    assert tail["tail_violation_cvar"] == 0.0
    assert tail["tail_margin_score"] >= 0.0


def test_tail_metrics_mixed_safe_unsafe():
    """部分试验违约时，safe_rate < 1，violation > 0。"""
    trials = _make_trials(
        [0.30, 0.35, 0.30, 0.36, 0.30],
        [0.90, 0.85, 0.90, 0.84, 0.90],
        [0.88, 0.83, 0.88, 0.82, 0.88],
    )
    perf_w = _resolve_stage2_metric_weights(2, PERF_WEIGHTS, "perf")
    bar_w = _resolve_stage2_metric_weights(2, BARRIER_WEIGHTS, "barrier")
    tail = _compute_tail_metrics_from_trials(
        trials, CONSTRAINT_LIMITS, BASELINE_METRICS, perf_w, bar_w, 2,
    )
    assert tail["safe_rate"] < 1.0
    assert tail["unsafe_count"] > 0
    assert tail["tail_violation_cvar"] > 0.0


def test_reward_with_trials_returns_tail_fields():
    """带 trials 的 mc_eval 应返回 tail 风险字段。"""
    trials = _make_trials(
        [0.30, 0.31, 0.32, 0.30, 0.30],
        [0.90, 0.89, 0.88, 0.90, 0.90],
        [0.88, 0.87, 0.86, 0.88, 0.88],
    )
    mc_eval = {
        "trials": trials,
        "loss_mean": 0.306,
        "loss_std": 0.008,
        "metric1_mean": 0.894,
        "metric1_std": 0.008,
        "metric2_mean": 0.874,
        "metric2_std": 0.008,
    }
    raw_reward, components = _compute_reward(0.306, 0.894, 0.874, mc_eval=mc_eval)
    assert "safe_rate" in components
    assert "tail_violation_cvar" in components
    assert "tail_margin_score" in components
    assert "raw_tail_reward" in components
    assert math.isclose(
        components["final_selection_score"],
        components["raw_final_reward"],
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def test_reward_without_trials_falls_back():
    """无 trials 时回退到均值方式（兼容旧调用）。"""
    raw_reward, components = _compute_reward(0.30, 0.90, 0.88, cost_value=40.0)
    assert "safe_rate" in components
    assert components["safe_rate"] == 1.0
    assert components["tail_violation_cvar"] == 0.0


def test_single_metric_mode_renormalizes_active_weights():
    perf_weights = _resolve_stage2_metric_weights(1, PERF_WEIGHTS, "perf")
    barrier_weights = _resolve_stage2_metric_weights(1, BARRIER_WEIGHTS, "barrier")
    assert set(perf_weights.keys()) == {"loss", "metric1"}
    assert set(barrier_weights.keys()) == {"loss", "metric1"}
    assert math.isclose(sum(perf_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(sum(barrier_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)


def test_wilson_lower_bound_reasonable():
    """Wilson 下界应在合理范围内。"""
    lb = _compute_wilson_lower_bound(61, 64)
    assert 0.85 <= lb <= 0.98
    lb_perfect = _compute_wilson_lower_bound(64, 64)
    assert lb_perfect > 0.93


def test_bootstrap_cvar_ucb_reasonable():
    """Bootstrap CVaR UCB 应 >= 均值。"""
    costs = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
    cvar_mean, cvar_ucb = _bootstrap_cvar_ucb(costs, tail_k=2, n_bootstrap=100)
    assert cvar_ucb >= cvar_mean
    assert cvar_mean > 0.0
