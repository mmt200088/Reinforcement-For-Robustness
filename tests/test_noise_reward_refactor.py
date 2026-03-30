import math

from noise_rl_module import (
    _build_candidate_score_sort_key,
    _compute_stage2_reward_components,
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


def test_baseline_quality_at_max_cost_hits_perf_not_cost():
    raw_reward, components = _compute_reward(0.30, 0.90, 0.88, cost_value=40.0)
    assert math.isclose(components["perf_score"], 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(components["barrier_penalty"], 0.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(components["cost_score"], 0.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(raw_reward, 0.75, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(
        components["final_selection_score"],
        components["raw_final_reward"],
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def test_threshold_quality_at_min_cost_hits_cost_not_perf():
    raw_reward, components = _compute_reward(0.33, 0.87, 0.85, cost_value=30.0)
    assert math.isclose(components["perf_score"], 0.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(components["barrier_penalty"], 0.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(components["cost_score"], 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(raw_reward, 0.25, rel_tol=0.0, abs_tol=1e-8)


def test_metric_violations_are_penalized_more_than_loss_violations():
    _, loss_components = _compute_reward(0.345, 0.90, 0.88, cost_value=35.0)
    _, metric1_components = _compute_reward(0.30, 0.855, 0.88, cost_value=35.0)
    _, metric2_components = _compute_reward(0.30, 0.90, 0.835, cost_value=35.0)

    assert math.isclose(loss_components["loss_violation"], 0.5, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(metric1_components["metric1_violation"], 0.5, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(metric2_components["metric2_violation"], 0.5, rel_tol=0.0, abs_tol=1e-8)
    assert metric1_components["barrier_penalty"] > loss_components["barrier_penalty"]
    assert metric2_components["barrier_penalty"] > loss_components["barrier_penalty"]


def test_single_metric_mode_renormalizes_active_weights():
    perf_weights = _resolve_stage2_metric_weights(1, PERF_WEIGHTS, "perf")
    barrier_weights = _resolve_stage2_metric_weights(1, BARRIER_WEIGHTS, "barrier")

    assert set(perf_weights.keys()) == {"loss", "metric1"}
    assert set(barrier_weights.keys()) == {"loss", "metric1"}
    assert math.isclose(sum(perf_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(sum(barrier_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8)

    raw_reward, components = _compute_reward(
        0.30,
        0.90,
        0.88,
        num_metrics=1,
        cost_value=40.0,
    )
    assert math.isclose(components["perf_score"], 1.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(components["barrier_penalty"], 0.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(raw_reward, 0.75, rel_tol=0.0, abs_tol=1e-8)


def test_candidate_sort_key_is_score_first_then_tiebreakers():
    better_score = _build_candidate_score_sort_key(0.80, 9.0, 40.0, 0.40, 1.0)
    worse_score = _build_candidate_score_sort_key(0.70, 0.1, 30.0, 0.20, 2.0)
    assert better_score < worse_score

    equal_score_better_stability = _build_candidate_score_sort_key(0.80, 0.2, 40.0, 0.40, 1.0)
    equal_score_worse_stability = _build_candidate_score_sort_key(0.80, 0.3, 30.0, 0.20, 2.0)
    assert equal_score_better_stability < equal_score_worse_stability
