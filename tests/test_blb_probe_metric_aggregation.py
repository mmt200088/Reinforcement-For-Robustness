import math

import numpy as np
import pytest

pytest.importorskip("torch")

try:
    from rfr.search.common.eval_metrics import finalize_probe_trial_metrics
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
    assert math.isclose(metric2, 0.8888888888888888)


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
