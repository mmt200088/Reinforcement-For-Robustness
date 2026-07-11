import numpy as np
import pytest

from blb_stage2_rl.statistical_constraints import (
    DegenerateBaselineVariance,
    InsufficientBaselineTrials,
    TrialSeries,
    assess_candidate,
    build_baseline_reference,
)


def _baseline_groups():
    offsets = np.linspace(-0.0012, 0.0012, 25)
    loss = 1.0 + offsets
    metric1 = 0.8 + offsets
    metric2 = 0.7 + offsets
    groups = []
    for start in range(0, 25, 5):
        stop = start + 5
        groups.append(
            TrialSeries(
                loss=loss[start:stop],
                metric1=metric1[start:stop],
                metric2=metric2[start:stop],
                seeds=range(100 + start, 100 + stop),
            )
        )
    return groups, loss, metric1, metric2


def _reference():
    groups, *_ = _baseline_groups()
    return build_baseline_reference(
        groups,
        precision_tolerance=0.001,
        stability_multiplier=2.0,
        bootstrap_samples=1024,
        seed=17,
    )


def _candidate(*, loss=None, metric1=None, metric2=None):
    count = 25
    return TrialSeries(
        loss=np.full(count, 0.95) if loss is None else loss,
        metric1=np.full(count, 0.85) if metric1 is None else metric1,
        metric2=np.full(count, 0.75) if metric2 is None else metric2,
    )


def test_trial_series_normalizes_values_and_protects_invariants():
    loss = [1, 2]
    metric1 = [0.8, 0.9]
    metric2 = [0.7, 0.75]
    seeds = [10, 11]

    trials = TrialSeries(loss=loss, metric1=metric1, metric2=metric2, seeds=seeds)
    loss[0] = 99
    seeds[0] = 99

    assert trials.loss == (1.0, 2.0)
    assert trials.metric1 == (0.8, 0.9)
    assert trials.metric2 == (0.7, 0.75)
    assert trials.seeds == (10, 11)


def test_build_baseline_reference_pools_groups_and_uses_unbiased_std():
    groups, loss, metric1, metric2 = _baseline_groups()

    reference = build_baseline_reference(
        groups,
        precision_tolerance=0.001,
        stability_multiplier=2.0,
        bootstrap_samples=512,
        seed=17,
    )

    assert reference.trial_count == 25
    assert reference.trials.loss == tuple(loss)
    assert reference.trials.metric1 == tuple(metric1)
    assert reference.trials.metric2 == tuple(metric2)
    assert reference.trials.seeds == tuple(range(100, 125))
    assert reference.loss_mean == pytest.approx(np.mean(loss))
    assert reference.metric1_mean == pytest.approx(np.mean(metric1))
    assert reference.metric2_mean == pytest.approx(np.mean(metric2))
    assert reference.loss_std == pytest.approx(np.std(loss, ddof=1))
    assert reference.metric1_std == pytest.approx(np.std(metric1, ddof=1))
    assert reference.metric2_std == pytest.approx(np.std(metric2, ddof=1))
    assert reference.loss_limit == pytest.approx(np.mean(loss) * 1.001)
    assert reference.metric1_limit == pytest.approx(np.mean(metric1) * 0.999)
    assert reference.metric2_limit == pytest.approx(np.mean(metric2) * 0.999)
    assert reference.loss_std_limit == pytest.approx(np.std(loss, ddof=1) * 2.0)
    assert reference.metric1_std_limit == pytest.approx(np.std(metric1, ddof=1) * 2.0)
    assert reference.metric2_std_limit == pytest.approx(np.std(metric2, ddof=1) * 2.0)
    assert reference.loss_std_limit < 0.01
    assert reference.bootstrap_seed == 17
    assert reference.bootstrap_samples == 512
    assert set(reference.bootstrap_means) == {"loss", "metric1", "metric2"}
    assert set(reference.bootstrap_stds) == {"loss", "metric1", "metric2"}
    assert all(values.shape == (512,) for values in reference.bootstrap_means.values())
    assert all(values.shape == (512,) for values in reference.bootstrap_stds.values())


@pytest.mark.parametrize(
    ("candidate", "failed_field"),
    [
        (_candidate(loss=np.full(25, 1.05)), "loss_precision_probability"),
        (_candidate(metric1=np.full(25, 0.75)), "metric1_precision_probability"),
        (_candidate(metric2=np.full(25, 0.65)), "metric2_precision_probability"),
        (
            _candidate(loss=np.resize(np.array([0.2, 0.8]), 25)),
            "loss_stability_probability",
        ),
        (
            _candidate(metric1=np.resize(np.array([0.82, 0.98]), 25)),
            "metric1_stability_probability",
        ),
        (
            _candidate(metric2=np.resize(np.array([0.72, 0.88]), 25)),
            "metric2_stability_probability",
        ),
    ],
)
def test_each_constraint_probability_gates_independently(candidate, failed_field):
    assessment = assess_candidate(
        candidate,
        _reference(),
        gate_probability=0.8,
        bootstrap_seed=29,
    )
    probability_fields = (
        "loss_precision_probability",
        "metric1_precision_probability",
        "metric2_precision_probability",
        "loss_stability_probability",
        "metric1_stability_probability",
        "metric2_stability_probability",
    )

    assert getattr(assessment, failed_field) < assessment.gate_probability
    for field in probability_fields:
        if field != failed_field:
            assert getattr(assessment, field) >= assessment.gate_probability

    precision_fields = probability_fields[:3]
    stability_fields = probability_fields[3:]
    assert assessment.precision_probability == min(
        getattr(assessment, field) for field in precision_fields
    )
    assert assessment.stability_probability == min(
        getattr(assessment, field) for field in stability_fields
    )
    assert assessment.online_precision_pass is (
        assessment.precision_probability >= assessment.gate_probability
    )
    assert assessment.online_stability_pass is (
        assessment.stability_probability >= assessment.gate_probability
    )


def test_candidate_assessment_is_deterministic_for_the_same_seed():
    reference = _reference()
    candidate = _candidate(
        loss=np.linspace(0.93, 0.97, 25),
        metric1=np.linspace(0.82, 0.88, 25),
        metric2=np.linspace(0.72, 0.78, 25),
    )

    first = assess_candidate(
        candidate,
        reference,
        gate_probability=0.5,
        bootstrap_seed=101,
    )
    second = assess_candidate(
        candidate,
        reference,
        gate_probability=0.5,
        bootstrap_seed=101,
    )

    assert first == second
    for probability in (
        first.loss_precision_probability,
        first.metric1_precision_probability,
        first.metric2_precision_probability,
        first.loss_stability_probability,
        first.metric1_stability_probability,
        first.metric2_stability_probability,
        first.precision_probability,
        first.stability_probability,
    ):
        assert 0.0 <= probability <= 1.0


def test_build_baseline_reference_requires_at_least_25_pooled_trials():
    groups, *_ = _baseline_groups()

    with pytest.raises(InsufficientBaselineTrials):
        build_baseline_reference(
            groups[:4],
            precision_tolerance=0.001,
            stability_multiplier=2.0,
            bootstrap_samples=64,
            seed=17,
        )


def test_degenerate_baseline_reports_every_zero_variance_channel():
    varying = np.linspace(0.7, 0.9, 25)
    groups = [
        TrialSeries(
            loss=np.ones(5),
            metric1=varying[start : start + 5],
            metric2=np.full(5, 0.7),
        )
        for start in range(0, 25, 5)
    ]

    with pytest.raises(DegenerateBaselineVariance) as error:
        build_baseline_reference(
            groups,
            precision_tolerance=0.001,
            stability_multiplier=2.0,
            bootstrap_samples=64,
            seed=17,
        )

    assert error.value.channels == ("loss", "metric2")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss": [], "metric1": [], "metric2": []},
        {"loss": [1.0], "metric1": [0.8, 0.9], "metric2": [0.7]},
        {"loss": [1.0], "metric1": [0.8], "metric2": [0.7], "seeds": [1, 2]},
        {"loss": [np.nan], "metric1": [0.8], "metric2": [0.7]},
        {"loss": [1.0], "metric1": [np.inf], "metric2": [0.7]},
    ],
)
def test_trial_series_rejects_invalid_channels(kwargs):
    with pytest.raises(ValueError):
        TrialSeries(**kwargs)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("precision_tolerance", -0.001),
        ("precision_tolerance", 1.0),
        ("precision_tolerance", np.nan),
        ("stability_multiplier", 0.0),
        ("stability_multiplier", np.inf),
        ("bootstrap_samples", 0),
        ("bootstrap_samples", 1.5),
        ("seed", -1),
        ("seed", 1.5),
    ],
)
def test_build_baseline_reference_rejects_invalid_arguments(override, value):
    groups, *_ = _baseline_groups()
    kwargs = {
        "precision_tolerance": 0.001,
        "stability_multiplier": 2.0,
        "bootstrap_samples": 64,
        "seed": 17,
    }
    kwargs[override] = value

    with pytest.raises((TypeError, ValueError)):
        build_baseline_reference(groups, **kwargs)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("gate_probability", 0.0),
        ("gate_probability", 1.01),
        ("gate_probability", np.nan),
        ("bootstrap_seed", -1),
        ("bootstrap_seed", 1.5),
    ],
)
def test_assess_candidate_rejects_invalid_arguments(override, value):
    kwargs = {"gate_probability": 0.5, "bootstrap_seed": 29}
    kwargs[override] = value

    with pytest.raises((TypeError, ValueError)):
        assess_candidate(_candidate(), _reference(), **kwargs)


def test_assess_candidate_requires_two_finite_trials():
    reference = _reference()

    with pytest.raises(ValueError, match="at least two"):
        assess_candidate(
            TrialSeries(loss=[0.95], metric1=[0.85], metric2=[0.75]),
            reference,
            gate_probability=0.5,
            bootstrap_seed=29,
        )
    with pytest.raises(ValueError, match="finite"):
        TrialSeries(loss=[0.95, np.inf], metric1=[0.85, 0.85], metric2=[0.75, 0.75])
