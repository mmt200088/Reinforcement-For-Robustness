import math

import numpy as np
import pytest

pytest.importorskip("torch")

try:
    from blb_stage2_rl.eval_metrics import finalize_probe_trial_metrics
    from blb_stage2_rl import inference_eval
except Exception as exc:  # pragma: no cover - torch-free local environments
    pytest.skip(f"Stage-2 probe modules are unavailable: {exc}", allow_module_level=True)


def test_probe_metrics_weight_tail_batches_by_sample_count():
    metrics = finalize_probe_trial_metrics(
        losses=[0.0, 10.0],
        m1s=[1.0, 0.0],
        m2s=[1.0, 0.0],
        counts=[4, 1],
        metric_profile="sst2",
        is_regression=False,
        preds=[np.array([1, 1, 1, 1]), np.array([0])],
        labels=[np.array([1, 1, 1, 1]), np.array([1])],
    )

    assert metrics is not None
    loss, metric1, metric2 = metrics
    assert math.isclose(loss, 2.0)
    assert math.isclose(metric1, 0.8)
    assert math.isclose(metric2, 0.8)


def test_mrpc_metric2_is_weighted_f1_not_accuracy():
    preds = np.array([0, 1, 0, 0, 1])
    labels = np.array([0, 1, 1, 1, 1])

    metrics = finalize_probe_trial_metrics(
        losses=[0.0],
        m1s=[0.6],
        m2s=[0.6],
        counts=[5],
        metric_profile="mrpc",
        is_regression=False,
        preds=[preds],
        labels=[labels],
    )

    assert metrics is not None
    _loss, metric1, metric2 = metrics
    assert math.isclose(metric1, 0.6)
    assert math.isclose(metric2, 0.6333333333333333)


def test_probe_metric_aggregation_is_invariant_to_batch_partitioning():
    preds = [np.array([0, 1, 0]), np.array([1, 1])]
    labels = [np.array([0, 1, 1]), np.array([0, 1])]

    partitioned = finalize_probe_trial_metrics(
        losses=[0.3, 0.6],
        m1s=[2.0 / 3.0, 0.5],
        m2s=[2.0 / 3.0, 0.5],
        counts=[3, 2],
        metric_profile="mrpc",
        is_regression=False,
        preds=preds,
        labels=labels,
    )
    combined = finalize_probe_trial_metrics(
        losses=[0.42],
        m1s=[0.6],
        m2s=[0.6],
        counts=[5],
        metric_profile="mrpc",
        is_regression=False,
        preds=[np.concatenate(preds)],
        labels=[np.concatenate(labels)],
    )

    assert partitioned == pytest.approx(combined, rel=0.0, abs=1e-15)


def test_ordered_batch_contributions_preserve_mrpc_weighted_f1():
    contribution_type = inference_eval.ProbeBatchContribution
    finalize_contributions = inference_eval.finalize_probe_batch_contributions
    contributions = [
        contribution_type(
            trial_index=7,
            batch_index=0,
            loss=0.3,
            metric1=2.0 / 3.0,
            metric2=2.0 / 3.0,
            sample_count=3,
            predictions=np.array([0, 1, 0]),
            labels=np.array([0, 1, 1]),
        ),
        contribution_type(
            trial_index=7,
            batch_index=1,
            loss=0.6,
            metric1=0.5,
            metric2=0.5,
            sample_count=2,
            predictions=np.array([1, 1]),
            labels=np.array([0, 1]),
        ),
    ]

    actual = finalize_contributions(
        contributions,
        expected_trial_index=7,
        expected_batch_count=2,
        metric_profile="mrpc",
        is_regression=False,
    )
    expected = finalize_probe_trial_metrics(
        losses=[0.3, 0.6],
        m1s=[2.0 / 3.0, 0.5],
        m2s=[2.0 / 3.0, 0.5],
        counts=[3, 2],
        metric_profile="mrpc",
        is_regression=False,
        preds=[
            np.array([0, 1, 0]),
            np.array([1, 1]),
        ],
        labels=[
            np.array([0, 1, 1]),
            np.array([0, 1]),
        ],
    )

    assert actual == expected


@pytest.mark.parametrize(
    "batch_indices",
    ([0, 0], [0, 2], [1, 0]),
)
def test_batch_contribution_finalizer_rejects_noncanonical_identities(
        batch_indices,
):
    contribution_type = inference_eval.ProbeBatchContribution
    finalize_contributions = inference_eval.finalize_probe_batch_contributions
    contributions = [
        contribution_type(
            trial_index=3,
            batch_index=batch_index,
            loss=0.1,
            metric1=1.0,
            metric2=1.0,
            sample_count=1,
            predictions=np.array([1]),
            labels=np.array([1]),
        )
        for batch_index in batch_indices
    ]

    with pytest.raises(ValueError, match="canonical"):
        finalize_contributions(
            contributions,
            expected_trial_index=3,
            expected_batch_count=2,
            metric_profile="mrpc",
            is_regression=False,
        )
