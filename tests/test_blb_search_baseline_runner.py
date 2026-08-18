from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from blb_stage2_rl.candidate_store import action_hash, candidate_key
from blb_stage2_rl.fusion_count_map import FusionCountMap
from blb_stage2_rl.layerwise_action import (
    layerwise_schedule,
    materialize_layerwise_counterfactuals,
)
from blb_stage2_rl.precision_presets import allocated_precision_tolerances
from blb_stage2_rl.reward import EpisodeMetrics
import blb_stage2_rl.search_baseline_runner as search_runner_module
from blb_stage2_rl.search_baseline_runner import (
    LayerwiseRuntimeEvaluator,
    _atomic_json,
    _strict_fallback_rank,
    _strict_selected_rank,
    canonical_strict_validation,
    limits_from_reference,
    load_search_preload,
    persist_search_result,
    run_layerwise_search_baseline,
)
from blb_stage2_rl.search_baselines import (
    ConstraintLimits,
    LayerwiseSearchSpace,
    SearchConfig,
    SearchEvaluation,
    SearchMetrics,
    SearchResult,
    candidate_rank_key,
    run_search,
)
from blb_stage2_rl.seed_utils import (
    derive_layerwise_episode_probe_seed,
    derive_probe_trial_seed,
)


class _Reference:
    loss_limit = 1.01
    metric1_limit = 0.89
    metric2_limit = 0.84
    loss_std_limit = 0.02
    metric1_std_limit = 0.015
    metric2_std_limit = 0.018


def _reference(*, metric1_limit=0.89):
    return SimpleNamespace(
        loss_limit=1.01,
        metric1_limit=float(metric1_limit),
        metric2_limit=0.84,
        loss_std_limit=0.02,
        metric1_std_limit=0.015,
        metric2_std_limit=0.018,
    )


def _strict_identity_context(marker="a"):
    return {"profile": "mrpc", "fixture_marker": str(marker)}


def _search_manifest(**overrides):
    payload = {"profile": "mrpc"}
    payload.update(overrides)
    return payload


def _canonical_test_materializations(action_matrix):
    rows = tuple(tuple(int(value) for value in row) for row in action_matrix)
    fusion_map = FusionCountMap.load("mrpc")
    schedule = layerwise_schedule(
        len(rows),
        fusion_map,
        profile="mrpc",
        gelu_degrees=[4] * len(rows),
    )
    baseline = np.zeros(len(rows) * 73 + 1, dtype=np.int64)
    return materialize_layerwise_counterfactuals(
        baseline,
        rows,
        schedule,
        fusion_map,
    )


def _serialized_test_boosted_overrides(value):
    return [
        {
            "block_idx": int(block_idx),
            "layer_idx": int(layer_idx),
            "field_values": {
                str(name): int(field_value)
                for name, field_value in fields.items()
            },
        }
        for (block_idx, layer_idx), fields in sorted(
            value.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]






def _candidate_store_stub(path="/tmp/candidates.jsonl"):
    return SimpleNamespace(path=path)


class _BaseEnv:
    def __init__(self):
        self.probe_noise_seed = None
        self.clear_count = 0
        self.env_cfg = SimpleNamespace(num_trials_per_step=3)
        self.prepare_calls = 0

    def prepare_action_for_terminal_probe(
            self, action_indices, **_kwargs,
            ):
        self.prepare_calls += 1
        return {
            "requires_forward": True,
            "any_invalid": False,
            "decoded": {
                "action_indices": tuple(
                    int(value) for value in action_indices
                ),
            },
            "final_config_fingerprint": "f" * 64,
        }

    def clear_installed_blb(self):
        self.clear_count += 1


class _LayerwiseEnv:
    horizon = 2
    communication_importance_ratio = 1.0

    def __init__(
            self,
            *,
            forward_ran=True,
            model_uses_replan=True,
            final_config_fingerprint="f" * 64,
            ):
        self.base = _BaseEnv()
        self.forward_ran = forward_ran
        self.model_uses_replan = model_uses_replan
        self.final_config_fingerprint = final_config_fingerprint
        self.rows = []
        self.runtime_terminal_info = None
        self.boosted_overrides = {(4, 0): {"slot": 47}}
        self.pending_full_vector = [1, 2, 3]

    def reset(self, *, seed=None):
        self.rows = []
        self.runtime_terminal_info = None
        self.reset_seed = seed
        return [0.0]

    def step(self, row):
        self.rows.append([int(value) for value in row])
        done = len(self.rows) == self.horizon
        if not done:
            return [0.0], 0.0, False, {"layer_idx": len(self.rows) - 1}
        joint = _canonical_test_materializations(self.rows)["joint"]
        self.pending_full_vector = [
            int(value) for value in joint.full_vector
        ]
        self.boosted_overrides = dict(joint.boosted_overrides)
        self.runtime_terminal_info = {
            "metrics": EpisodeMetrics(
                loss_mean=1.0,
                metric1_mean=0.90,
                metric2_mean=0.85,
                loss_std=0.01,
                metric1_std=0.01,
                metric2_std=0.01,
                loss_trials=(0.99, 1.01, 1.0),
                metric1_trials=(0.89, 0.90, 0.91),
                metric2_trials=(0.84, 0.85, 0.86),
                trial_seeds=(11, 12, 13),
            ),
            "invalid": False,
            "forward_ran": self.forward_ran,
            "replan_application": {
                "model_uses_replan_config": self.model_uses_replan,
            },
            "final_config_fingerprint": self.final_config_fingerprint,
            "statistical_assessment": {
                "loss_precision_probability": 0.91,
                "metric1_precision_probability": 0.92,
                "metric2_precision_probability": 0.93,
                "loss_stability_probability": 0.81,
                "metric1_stability_probability": 0.82,
                "metric2_stability_probability": 0.83,
                "precision_probability": 0.9,
                "stability_probability": 0.8,
                "gate_probability": 0.5,
                "online_precision_pass": True,
                "online_stability_pass": True,
                "bootstrap_seed": 1234,
            },
            "reward_breakdown": {"priority": 3},
        }
        return [0.0], 1.25, True, {
            "pending_full_vector": list(self.pending_full_vector),
        }


class _SeededLayerwiseEnv(_LayerwiseEnv):
    def step(self, row):
        state, reward, done, info = super().step(row)
        if done:
            probe_seed = int(self.base.probe_noise_seed)
            metrics = self.runtime_terminal_info["metrics"]
            self.runtime_terminal_info["metrics"] = EpisodeMetrics(
                loss_mean=metrics.loss_mean,
                metric1_mean=metrics.metric1_mean,
                metric2_mean=metrics.metric2_mean,
                loss_std=metrics.loss_std,
                metric1_std=metrics.metric1_std,
                metric2_std=metrics.metric2_std,
                loss_trials=metrics.loss_trials,
                metric1_trials=metrics.metric1_trials,
                metric2_trials=metrics.metric2_trials,
                trial_seeds=tuple(probe_seed + index for index in range(3)),
            )
        return state, reward, done, info


class _InvalidCandidateLayerwiseEnv(_LayerwiseEnv):
    def step(self, row):
        state, reward, done, info = super().step(row)
        if done and self.rows[0][0] == 1:
            self.runtime_terminal_info.update({
                "invalid": True,
                "forward_ran": False,
                "materialization_failure_reason": "optimizer_invalid_chain",
                "forward_skipped_reason": "optimizer_invalid_chain",
                "optimizer_invalid_summary": "block4 invalid_chain",
            })
        return state, reward, done, info


class _AlwaysInvalidLayerwiseEnv(_InvalidCandidateLayerwiseEnv):
    def step(self, row):
        state, reward, done, info = super().step(row)
        if done:
            self.runtime_terminal_info.update({
                "invalid": True,
                "forward_ran": False,
                "materialization_failure_reason": "optimizer_invalid_chain",
                "forward_skipped_reason": "optimizer_invalid_chain",
                "optimizer_invalid_summary": "all candidates invalid",
            })
        return state, reward, done, info




def _search_evaluation(
        action_matrix,
        *,
        metric1_mean=0.90,
        valid=True,
        materializable=True,
        canonical_materializations=None,
        ):
    materializations = canonical_materializations
    if materializations is None and materializable and len(action_matrix) == 2:
        materializations = _canonical_test_materializations(action_matrix)
    joint = (
        materializations["joint"]
        if materializations is not None else None
    )
    return SearchEvaluation(
        action_matrix=action_matrix,
        metrics=SearchMetrics(
            loss_mean=1.0,
            metric1_mean=metric1_mean,
            metric2_mean=0.85,
            loss_std=0.01,
            metric1_std=0.01,
            metric2_std=0.01,
        ),
        limits=ConstraintLimits(
            loss_max=1.01,
            metric1_min=0.89,
            metric2_min=0.84,
            loss_std_max=0.02,
            metric1_std_max=0.015,
            metric2_std_max=0.018,
        ),
        valid=valid,
        reward=1.0,
        metadata={
            "pending_full_vector": (
                [int(value) for value in joint.full_vector]
                if joint is not None
                else (
                    [value for row in action_matrix for value in row]
                    if materializable else []
                )
            ),
            "boosted_overrides": (
                _serialized_test_boosted_overrides(
                    joint.boosted_overrides
                )
                if joint is not None else []
            ),
            "statistical_assessment": {"bootstrap_seed": 1234},
            "materializable": bool(materializable),
            "final_config_fingerprint": "f" * 64,
        },
    )


def _search_result(*evaluations):
    return SearchResult(
        algorithm="greedy",
        best=max(evaluations, key=candidate_rank_key),
        observations=tuple(evaluations),
        history=(),
        termination_reason="test",
    )


def _five_eligible_evaluations(*evaluations):
    owned = list(evaluations)
    actions = {evaluation.action_matrix for evaluation in owned}
    for action in (
            ((0, 0), (0, 0)),
            ((1, 0), (0, 0)),
            ((0, 1), (0, 0)),
            ((0, 0), (1, 0)),
            ((1, 1), (0, 0)),
            ((0, 2), (1, 2)),
            ((1, 2), (1, 2)),
            ):
        if action in actions:
            continue
        owned.append(_search_evaluation(action))
        actions.add(action)
        if len(owned) == 5:
            break
    if len(owned) != 5:
        raise AssertionError("test fixture requires five distinct evaluations")
    return tuple(owned)


_STRICT_BANK_PROBE_SEEDS = {
    "A": (101, 102, 103, 104, 105),
    "B": (201, 202, 203, 204, 205),
    "C": (301, 302, 303, 304, 305),
}
_STRICT_TRIALS_PER_PROBE = 3
_STRICT_BATCH_SET_KEY = "validation_full"


def _strict_metrics_from_trial_results(results):
    rows = tuple(dict(result) for result in results)
    values = {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in ("loss", "metric1", "metric2")
    }
    return {
        "loss_mean": float(np.mean(values["loss"])),
        "metric1_mean": float(np.mean(values["metric1"])),
        "metric2_mean": float(np.mean(values["metric2"])),
        "loss_std": float(np.std(values["loss"], ddof=1)),
        "metric1_std": float(np.std(values["metric1"], ddof=1)),
        "metric2_std": float(np.std(values["metric2"], ddof=1)),
    }


def _strict_fixture_metrics(trial_count=45):
    result = {"loss": 1.0, "metric1": 0.90, "metric2": 0.85}
    return _strict_metrics_from_trial_results(
        result for _ in range(int(trial_count))
    )


def _strict_validation_bank_contract():
    banks = {}
    for label, probe_seeds in _STRICT_BANK_PROBE_SEEDS.items():
        trial_seeds = [
            derive_probe_trial_seed(base_seed, trial_index)
            for base_seed in probe_seeds
            for trial_index in range(_STRICT_TRIALS_PER_PROBE)
        ]
        banks[label] = {
            "probe_seeds": list(probe_seeds),
            "trial_seeds": trial_seeds,
            "trials_per_probe": _STRICT_TRIALS_PER_PROBE,
            "trial_count": len(trial_seeds),
        }
    return {
        "schema_version": "layerwise_validation_banks_v1",
        "banks": banks,
        "promotion_trial_count": 30,
        "final_trial_count": 45,
        "hard_gate": "canonical",
        "bootstrap_probability_role": "diagnostic_tiebreak_only",
    }


def _strict_banks_for_trial_count(strict_trial_count):
    trial_count = int(strict_trial_count)
    if trial_count not in (15, 30, 45):
        raise ValueError("strict trial count must be a complete bank prefix")
    return ("A", "B", "C")[:trial_count // 15]


def _strict_axis_action(full_vector, axis_name):
    offset = 100 if axis_name == "compute" else 200
    return tuple(offset + int(value) for value in full_vector)


def _strict_axis_counterfactuals(materializations, banks_run):
    bank_contract = _strict_validation_bank_contract()["banks"]
    out = {}
    for axis_name, materialization_name in (
            ("compute", "compute_only"),
            ("communication", "communication_only"),
    ):
        cumulative = 0
        axis_banks = {}
        for label in banks_run:
            cumulative += int(bank_contract[label]["trial_count"])
            axis_banks[label] = {
                "trial_count": cumulative,
                "metrics": _strict_fixture_metrics(cumulative),
            }
        materialization = materializations[materialization_name]
        full_vector = [
            int(value) for value in materialization.full_vector
        ]
        out[axis_name] = {
            "mode": materialization.mode,
            "full_vector": full_vector,
            "action_hash": action_hash(full_vector),
            "boosted_overrides": _serialized_test_boosted_overrides(
                materialization.boosted_overrides
            ),
            "final_config_fingerprint": "f" * 64,
            "precision_tolerance": 0.001,
            "banks": axis_banks,
            "metrics": _strict_fixture_metrics(cumulative),
        }
    return out


def _strict_artifact(
        result,
        *,
        strict_feasible=True,
        strict_trial_count=45,
        selected_updates=None,
        strict_metrics=None,
        ):
    ranked = sorted(
        (
            candidate for candidate in result.observations
            if (
                candidate.valid
                and candidate.inference_performed
                and bool(candidate.metadata.get("materializable", False))
                and bool(candidate.metadata.get("pending_full_vector"))
            )
        ),
        key=candidate_rank_key,
        reverse=True,
    )[:5]
    if len(ranked) != 5:
        raise ValueError("strict test artifact requires five eligible candidates")
    banks_run = _strict_banks_for_trial_count(strict_trial_count)
    not_run_banks = [
        label for label in ("A", "B", "C") if label not in banks_run
    ]
    default_metrics = _strict_fixture_metrics(strict_trial_count)
    if not strict_feasible:
        default_metrics = {**default_metrics, "metric1_mean": 0.88}
    joint_metrics = SearchMetrics.from_dict(
        default_metrics if strict_metrics is None else strict_metrics
    )
    axis_metrics = SearchMetrics.from_dict(default_metrics)
    records = []
    strict_evaluations = []
    for online in ranked:
        materializations = _canonical_test_materializations(
            online.action_matrix
        )
        violations = {
            "families": {
                name: {
                    "available": True,
                    "point_pass": bool(strict_feasible),
                    "trial_count": int(strict_trial_count),
                    "banks_run": list(banks_run),
                    "not_run_banks": list(not_run_banks),
                    "metrics": (
                        joint_metrics if name == "joint" else axis_metrics
                    ).as_dict(),
                }
                for name in (
                    "joint", "compute_only", "communication_only",
                )
            },
            "aggregate": {
                "failed_constraint_count": 0 if strict_feasible else 1,
                "total_normalized_violation": (
                    0.0 if strict_feasible else 0.1
                ),
                "worst_normalized_violation": (
                    0.0 if strict_feasible else 0.1
                ),
                "unavailable_family_count": 0,
            },
        }
        strict_payload = SearchEvaluation(
            action_matrix=online.action_matrix,
            metrics=joint_metrics,
            limits=online.limits,
            valid=True,
            reward=online.reward,
            communication_importance_ratio=1.0,
            constraint_probabilities=online.constraint_probabilities,
            gate_probability=online.gate_probability,
            metadata={
                **online.metadata,
                "strict_trial_count": int(strict_trial_count),
                "strict_final_assessment": {
                    name: 0.99
                    for name in (
                        "loss_precision_probability",
                        "metric1_precision_probability",
                        "metric2_precision_probability",
                        "loss_stability_probability",
                        "metric1_stability_probability",
                        "metric2_stability_probability",
                    )
                },
                "strict_axis_counterfactuals": (
                    _strict_axis_counterfactuals(
                        materializations, banks_run,
                    )
                ),
                "strict_materialization_fingerprints": {
                    family: "f" * 64
                    for family in (
                        "joint", "compute_only", "communication_only",
                    )
                },
                "strict_candidate_key": action_hash(
                    materializations["joint"].full_vector
                ),
                "strict_violations": violations,
            },
        ).as_dict()
        strict_payload.update(dict(selected_updates or {}))
        strict_evaluation = SearchEvaluation.from_dict(strict_payload)
        strict_evaluations.append(strict_evaluation)
        records.append({
            "online_candidate": online.as_dict(),
            "strict_evaluated": True,
            "selection_eligible": True,
            "strict_point_pass": bool(strict_feasible),
            "strict_feasible": bool(strict_feasible),
            "strict_trial_count": int(strict_trial_count),
            "strict_evaluation": strict_evaluation.as_dict(),
            "violations": violations,
        })
    selection_status = (
        "strict_feasible" if strict_feasible else "strict_least_violating"
    )
    rank_key = _strict_selected_rank if strict_feasible else _strict_fallback_rank
    selected_evaluation = (
        min(strict_evaluations, key=rank_key)
        if strict_feasible
        else max(strict_evaluations, key=rank_key)
    )
    selected_violations = selected_evaluation.metadata["strict_violations"]
    selected = {
        **selected_evaluation.as_dict(),
        "selection_status": selection_status,
        "strict_feasible": bool(strict_feasible),
        "violations": selected_violations,
    }
    return {
        "schema_version": "stage2_search_strict_validation_v3",
        "requested_top_n": 5,
        "strict_evaluated_candidate_count": 5,
        "online_best": result.best.as_dict(),
        "selection_status": selection_status,
        "strict_feasible": bool(strict_feasible),
        "selected_violations": selected_violations,
        "selected": selected,
        "validation_banks": _strict_validation_bank_contract(),
        "records": records,
    }








def _promotion_result(
        *, status, trial_count, metrics, fresh_trial_count=0,
        assessment=None, axis_counterfactuals=None, evidence=None,
        ):
    if evidence is None and metrics is not None:
        evidence = SimpleNamespace(groups=[{
            "final_config_fingerprint": "f" * 64,
        }])
    return SimpleNamespace(
        status=status,
        trial_count=trial_count,
        fresh_trial_count=fresh_trial_count,
        metrics=metrics,
        assessment=assessment,
        axis_counterfactuals=axis_counterfactuals,
        evidence=evidence,
    )


def _bind_strict_result(result, call_kwargs):
    full_vector = call_kwargs.get("action_indices")
    if full_vector is None:
        full_vector = call_kwargs["candidate"]["full_vector"]
    full_vector = tuple(int(value) for value in full_vector)
    identity_context = {
        **dict(call_kwargs["identity_context"]),
        "fidelity": "F4",
    }
    evidence = result.evidence
    evidence_payload = dict(vars(evidence))
    evidence_payload.update({
        "candidate_key": candidate_key(full_vector, identity_context),
        "action_indices": full_vector,
    })
    result_payload = dict(vars(result))
    result_payload["evidence"] = SimpleNamespace(**evidence_payload)
    return SimpleNamespace(**result_payload)


def _strict_result_side_effect(result):
    return lambda **kwargs: _bind_strict_result(result, kwargs)


def _strict_materialization_fingerprints():
    return {
        family: "f" * 64
        for family in ("joint", "compute_only", "communication_only")
    }


def _patch_strict_materialization_preparation():
    return patch(
        "blb_stage2_rl.search_baseline_runner."
        "_prepare_strict_materialization_fingerprints",
        return_value=_strict_materialization_fingerprints(),
    )


def _passing_axis_counterfactuals(metrics):
    payload = {
        "final_config_fingerprint": "f" * 64,
        "loss_limit": 1.01,
        "metric1_limit": 0.89,
        "metric2_limit": 0.84,
        "loss_std_limit": 0.02,
        "metric1_std_limit": 0.015,
        "metric2_std_limit": 0.018,
        "banks": {
            label: {
                "trial_count": trial_count,
                "fresh_trial_count": 15,
                "metrics": metrics,
                "point_pass": True,
            }
            for label, trial_count in (("A", 15), ("B", 30), ("C", 45))
        },
        "point_pass": True,
        "metrics": metrics,
    }
    return {"compute": payload, "communication": payload}


def _passing_strict_violations():
    return {
        "families": {
            name: {
                "available": True,
                "point_pass": True,
                "not_run_banks": [],
            }
            for name in ("joint", "compute_only", "communication_only")
        },
        "aggregate": {
            "failed_constraint_count": 0,
            "total_normalized_violation": 0.0,
            "worst_normalized_violation": 0.0,
            "unavailable_family_count": 0,
        },
    }


def _validation_banks(
        *, bank_a_reference=None, promotion_reference=None,
        final_reference=None,
        ):
    bank_a_reference = bank_a_reference or _Reference()
    promotion_reference = promotion_reference or _Reference()
    final_reference = final_reference or _Reference()
    return SimpleNamespace(
        bank_a=SimpleNamespace(
            trial_count=15,
            reference=bank_a_reference,
        ),
        promotion_trial_count=30,
        final_trial_count=45,
        promotion_reference=promotion_reference,
        final_reference=final_reference,
        contract_payload=_strict_validation_bank_contract,
    )






























class RuntimeEvaluatorTests(unittest.TestCase):
    def test_strict_rank_uses_ppo_full_vector_tiebreak_not_compact_action(self):
        def strict_evaluation(action_matrix, full_vector):
            payload = _search_evaluation(action_matrix).as_dict()
            metadata = dict(payload["metadata"])
            metadata.update({
                "pending_full_vector": list(full_vector),
                "strict_candidate_key": action_hash(full_vector),
                "strict_final_assessment": {
                    name: 0.99 for name in (
                        "loss_precision_probability",
                        "metric1_precision_probability",
                        "metric2_precision_probability",
                        "loss_stability_probability",
                        "metric1_stability_probability",
                        "metric2_stability_probability",
                    )
                },
            })
            payload["metadata"] = metadata
            return SearchEvaluation.from_dict(payload)

        compact_first = strict_evaluation(
            ((0, 1), (1, 0)), (9, 0, 0, 0),
        )
        full_vector_first = strict_evaluation(
            ((1, 0), (0, 1)), (1, 9, 9, 9),
        )

        selected = min(
            (compact_first, full_vector_first), key=_strict_selected_rank,
        )
        self.assertEqual(selected.action_matrix, full_vector_first.action_matrix)

    def test_duplicate_persisted_observation_row_fails_closed(self):
        evaluation = _search_evaluation(((0, 0), (0, 0)))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "observations.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                row = json.dumps(evaluation.as_dict(), sort_keys=True)
                handle.write(row + "\n" + row + "\n")

            with self.assertRaisesRegex(
                    ValueError, "duplicate Stage-2 observation",
            ):
                load_search_preload(path)

    def test_atomic_json_durably_commits_before_return(self):
        events = []
        real_replace = os.replace

        def tracked_replace(source, target):
            events.append("replace")
            return real_replace(source, target)

        with tempfile.TemporaryDirectory() as tmpdir, patch(
                "blb_stage2_rl.search_baseline_runner.os.fsync",
                side_effect=lambda _fd: events.append("fsync"),
        ), patch(
                "blb_stage2_rl.search_baseline_runner.os.replace",
                side_effect=tracked_replace,
        ):
            _atomic_json(
                os.path.join(tmpdir, "artifact.json"),
                {"status": "complete"},
            )

        self.assertEqual(events, ["fsync", "replace", "fsync"])

    def test_real_layerwise_path_yields_all_six_metrics_and_audit_fields(self):
        env = _LayerwiseEnv()
        callback_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=callback_rows.append,
        )

        result = evaluator(((1, 2), (0, 1)))

        self.assertTrue(result.feasible)
        self.assertEqual(env.rows, [[1, 2], [0, 1]])
        self.assertEqual(result.metrics.metric2_mean, 0.85)
        self.assertEqual(result.metrics.metric2_std, 0.01)
        self.assertTrue(result.metadata["forward_ran"])
        self.assertTrue(result.metadata["model_uses_replan_config"])
        self.assertEqual(result.metadata["trial_seeds"], [11, 12, 13])
        self.assertEqual(
            result.constraint_probabilities,
            (0.91, 0.92, 0.93, 0.81, 0.82, 0.83),
        )
        self.assertEqual(result.gate_probability, 0.5)
        self.assertEqual(result.metadata["bootstrap_seed"], 1234)
        self.assertEqual(
            result.metadata["boosted_overrides"],
            _serialized_test_boosted_overrides(
                _canonical_test_materializations(
                    ((1, 2), (0, 1))
                )["joint"].boosted_overrides
            ),
        )
        self.assertEqual(len(callback_rows), 1)
        self.assertGreaterEqual(env.base.clear_count, 1)

    def test_real_layerwise_path_preserves_final_config_fingerprint(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(final_config_fingerprint="e" * 64),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        result = evaluator(((1, 2), (0, 1)))

        self.assertEqual(
            result.metadata["final_config_fingerprint"], "e" * 64,
        )



















    def test_completed_resume_does_not_publish_pending_strict_context(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 36,
            "seed": 42,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = run_layerwise_search_baseline(
                layerwise_env=_LayerwiseEnv(),
                output_dir=tmpdir,
                strict_validator=lambda result: _strict_artifact(
                    result, strict_feasible=True,
                ),
                **kwargs,
            )

            reopened = run_layerwise_search_baseline(
                layerwise_env=_LayerwiseEnv(),
                output_dir=tmpdir,
                strict_validator=lambda _result: self.fail(
                    "completed strict validation must not rerun"
                ),
                pending_strict_context_writer=lambda _contract: self.fail(
                    "completed reopen must not publish pending strict context"
                ),
                **kwargs,
            )

        self.assertEqual(
            reopened["result"].best.as_dict(),
            completed["result"].best.as_dict(),
        )





    def test_missing_real_forward_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(forward_ran=False),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "forward"):
            evaluator(((0, 0), (0, 0)))

    def test_missing_replan_install_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(model_uses_replan=False),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "replan"):
            evaluator(((0, 0), (0, 0)))

    def test_truthy_nonboolean_forward_evidence_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(forward_ran="false"),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "forward"):
            evaluator(((0, 0), (0, 0)))

    def test_truthy_nonboolean_replan_evidence_fails_closed(self):
        evaluator = LayerwiseRuntimeEvaluator(
            env=_LayerwiseEnv(model_uses_replan=1),
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "replan"):
            evaluator(((0, 0), (0, 0)))

    def test_missing_or_malformed_final_config_fingerprint_fails_closed(self):
        for fingerprint in (None, "", "A" * 64, "f" * 63, 7):
            with self.subTest(fingerprint=fingerprint):
                evaluator = LayerwiseRuntimeEvaluator(
                    env=_LayerwiseEnv(
                        final_config_fingerprint=fingerprint,
                    ),
                    reference=_Reference(),
                    base_seed=17,
                    expected_trials=3,
                )

                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    evaluator(((0, 0), (0, 0)))

    def test_probe_seed_stream_matches_ppo_global_episode_indices(self):
        env = _SeededLayerwiseEnv()
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )
        action_a = ((0, 0), (0, 0))
        action_b = ((1, 2), (1, 2))

        first_a = evaluator(action_a)
        observed_b = evaluator(action_b)
        second_a = evaluator(action_a)

        observed = (first_a, observed_b, second_a)
        for index, result in enumerate(observed):
            self.assertEqual(result.metadata["online_stream_index"], index)
            self.assertEqual(
                result.metadata["probe_seed"],
                derive_layerwise_episode_probe_seed(
                    17, index, trial_count=3,
                ),
            )
        self.assertEqual(env.reset_seed, 19)
        self.assertEqual(len({
            result.metadata["probe_seed"] for result in observed
        }), 3)

    def test_same_action_matches_ppo_episode_metrics_at_same_stream_index(self):
        from blb_stage2_rl.layerwise_runner import _collect_layerwise_episode
        from blb_stage2_rl.seed_utils import (
            derive_layerwise_online_evaluation_seeds,
        )

        class SeedSensitiveEnv(_LayerwiseEnv):
            max_step_dim = 2

            @staticmethod
            def current_spec():
                return None

            def step(self, row):
                state, reward, done, info = super().step(row)
                if done:
                    probe_seed = int(self.base.probe_noise_seed)
                    offset = float(probe_seed % 997) * 1.0e-7
                    loss_trials = tuple(
                        0.99 + offset + delta
                        for delta in (0.0, 0.01, 0.005)
                    )
                    metric1_trials = tuple(
                        0.89 + offset + delta
                        for delta in (0.0, 0.01, 0.005)
                    )
                    metric2_trials = tuple(
                        0.84 + offset + delta
                        for delta in (0.0, 0.01, 0.005)
                    )
                    self.runtime_terminal_info["metrics"] = EpisodeMetrics(
                        loss_mean=float(np.mean(loss_trials)),
                        metric1_mean=float(np.mean(metric1_trials)),
                        metric2_mean=float(np.mean(metric2_trials)),
                        loss_std=float(np.std(loss_trials, ddof=1)),
                        metric1_std=float(np.std(metric1_trials, ddof=1)),
                        metric2_std=float(np.std(metric2_trials, ddof=1)),
                        loss_trials=loss_trials,
                        metric1_trials=metric1_trials,
                        metric2_trials=metric2_trials,
                        trial_seeds=tuple(
                            derive_probe_trial_seed(probe_seed, trial_index)
                            for trial_index in range(3)
                        ),
                    )
                return state, reward, done, info

        class FixedPolicy:
            def __init__(self, rows):
                self.rows = iter(rows)

            def sample_action(self, *_args, **_kwargs):
                row = np.asarray(next(self.rows), dtype=np.int64)
                return (
                    row[None, :],
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([0.0], dtype=np.float32),
                    np.zeros((1, row.size), dtype=np.float32),
                )

        class Buffer:
            def __init__(self):
                self.rows = []

            def add(self, **payload):
                self.rows.append(payload)
                return len(self.rows) - 1

        action = ((1, 2), (0, 1))
        base_seed = 17
        stream_index = 7
        expected_reset_seed, expected_probe_seed = (
            derive_layerwise_online_evaluation_seeds(
                base_seed,
                stream_index,
                trial_count=3,
            )
        )

        ppo_env = SeedSensitiveEnv()
        draft = _collect_layerwise_episode(
            env=ppo_env,
            policy=FixedPolicy(action),
            rollout_buffer=Buffer(),
            entropy_samples=[],
            absolute_episode=stream_index,
            base_seed=base_seed,
            expected_online_trials=3,
            horizon=len(action),
            device="cpu",
            step_adapter_fn=lambda *_args: (
                np.ones(2, dtype=bool),
                np.full(2, 3, dtype=np.int64),
            ),
        )

        comparator_env = SeedSensitiveEnv()
        evaluator = LayerwiseRuntimeEvaluator(
            env=comparator_env,
            reference=_Reference(),
            base_seed=base_seed,
            expected_trials=3,
        )
        evaluator.evaluation_count = stream_index
        comparator = evaluator(action)

        ppo_metrics = draft.runtime_info["metrics"]
        self.assertEqual(ppo_env.reset_seed, expected_reset_seed)
        self.assertEqual(comparator_env.reset_seed, expected_reset_seed)
        self.assertEqual(ppo_env.base.probe_noise_seed, expected_probe_seed)
        self.assertEqual(
            comparator_env.base.probe_noise_seed,
            expected_probe_seed,
        )
        self.assertEqual(ppo_env.rows, comparator_env.rows)
        self.assertEqual(
            comparator.metadata["pending_full_vector"],
            list(ppo_env.pending_full_vector),
        )
        self.assertEqual(
            comparator.metadata["final_config_fingerprint"],
            draft.runtime_info["final_config_fingerprint"],
        )
        self.assertEqual(
            comparator.metadata["trial_seeds"],
            list(ppo_metrics.trial_seeds),
        )
        self.assertEqual(
            comparator.metadata["trial_results"],
            {
                "loss": list(ppo_metrics.loss_trials),
                "metric1": list(ppo_metrics.metric1_trials),
                "metric2": list(ppo_metrics.metric2_trials),
            },
        )
        self.assertEqual(
            comparator.metrics.as_dict(),
            {
                "loss_mean": ppo_metrics.loss_mean,
                "metric1_mean": ppo_metrics.metric1_mean,
                "metric2_mean": ppo_metrics.metric2_mean,
                "loss_std": ppo_metrics.loss_std,
                "metric1_std": ppo_metrics.metric1_std,
                "metric2_std": ppo_metrics.metric2_std,
            },
        )

    def test_fixed_action_parity_gate_compares_complete_runtime_surface(self):
        from blb_stage2_rl.same_action_parity import (
            run_same_action_parity_gate,
        )
        from blb_stage2_rl.seed_utils import (
            derive_layerwise_online_evaluation_seeds,
        )

        class ParityEnv(_LayerwiseEnv):
            max_step_dim = 2

            @property
            def schedule(self):
                return [
                    SimpleNamespace(slot_dims=(2, 3))
                    for _ in range(self.horizon)
                ]

            @staticmethod
            def current_spec():
                return SimpleNamespace(
                    fusion_num_options=2,
                    k_num_levels=3,
                )

        action = ((1, 2), (0, 1))
        base_seed = 17
        stream_index = 7
        expected_reset_seed, expected_probe_seed = (
            derive_layerwise_online_evaluation_seeds(
                base_seed,
                stream_index,
                trial_count=3,
            )
        )
        evidence = run_same_action_parity_gate(
            layerwise_env=ParityEnv(),
            robust_reference=_Reference(),
            action_matrix=action,
            base_seed=base_seed,
            stream_index=stream_index,
            expected_trials=3,
            device="cpu",
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(
            evidence["schema_version"],
            "stage2_same_action_parity_v1",
        )
        self.assertEqual(
            evidence["ppo_projection"],
            evidence["comparator_projection"],
        )
        self.assertEqual(
            evidence["ppo_projection"]["action_matrix"],
            [[1, 2], [0, 1]],
        )
        self.assertEqual(
            evidence["ppo_projection"]["reset_seed"],
            expected_reset_seed,
        )
        self.assertEqual(
            evidence["ppo_projection"]["probe_seed"],
            expected_probe_seed,
        )
        self.assertEqual(
            evidence["ppo_projection"]["constraint_probabilities"],
            {
                "loss_precision_probability": 0.91,
                "metric1_precision_probability": 0.92,
                "metric2_precision_probability": 0.93,
                "loss_stability_probability": 0.81,
                "metric1_stability_probability": 0.82,
                "metric2_stability_probability": 0.83,
            },
        )
        self.assertEqual(len(evidence["projection_sha256"]), 64)

    def test_fixed_action_parity_gate_rejects_any_semantic_drift(self):
        from blb_stage2_rl.same_action_parity import (
            assert_same_action_projection_equal,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                "same-action parity failed",
                ):
            assert_same_action_projection_equal(
                {"metrics": {"loss_mean": 0.1}},
                {"metrics": {"loss_mean": 0.2}},
            )

    def test_fixed_action_strict_gate_uses_five_real_unique_candidates(self):
        from blb_stage2_rl.same_action_parity import (
            run_same_action_parity_gate,
        )

        class ParityEnv(_LayerwiseEnv):
            max_step_dim = 2

            @property
            def schedule(self):
                return [
                    SimpleNamespace(slot_dims=(2, 3))
                    for _ in range(self.horizon)
                ]

            @staticmethod
            def current_spec():
                return SimpleNamespace(
                    fusion_num_options=2,
                    k_num_levels=3,
                )

        target = ((0, 0), (0, 0))

        def strict_validator(result):
            self.assertEqual(len(result.observations), 5)
            self.assertEqual(
                len({item.action_matrix for item in result.observations}),
                5,
            )
            target_evaluation = next(
                item for item in result.observations
                if item.action_matrix == target
            )
            target_payload = target_evaluation.as_dict()
            target_payload["metadata"] = {
                **target_payload["metadata"],
                "strict_trial_count": 45,
                "strict_violations": {
                    "families": {
                        family: {"banks_run": ["A", "B", "C"]}
                        for family in (
                            "joint",
                            "compute_only",
                            "communication_only",
                        )
                    },
                },
            }
            return {
                "strict_evaluated_candidate_count": 5,
                "selected": target_payload,
                "records": [{
                    "online_candidate": target_evaluation.as_dict(),
                    "strict_evaluated": True,
                    "strict_evaluation": target_payload,
                }],
            }

        evidence = run_same_action_parity_gate(
            layerwise_env=ParityEnv(),
            robust_reference=_Reference(),
            action_matrix=target,
            base_seed=17,
            expected_trials=3,
            strict_validator=strict_validator,
        )

        self.assertEqual(evidence["strict_candidate_count"], 5)
        self.assertEqual(evidence["strict_target_trial_count"], 45)
        self.assertEqual(
            evidence["strict_target_banks_run"],
            ["A", "B", "C"],
        )

    def test_optimizer_invalid_candidate_returns_invalid_and_search_continues(self):
        env = _InvalidCandidateLayerwiseEnv()
        callback_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=callback_rows.append,
        )

        invalid = evaluator(((1, 0), (0, 0)))
        valid = evaluator(((0, 0), (0, 0)))

        self.assertFalse(invalid.valid)
        self.assertFalse(invalid.feasible)
        self.assertFalse(invalid.metadata["inference_performed"])
        self.assertFalse(invalid.metadata["materializable"])
        self.assertTrue(valid.valid)
        self.assertEqual(evaluator.evaluation_count, 2)
        self.assertEqual(len(callback_rows), 2)

    def test_invalid_candidate_with_eval_failure_is_infrastructure_error(self):
        env = _InvalidCandidateLayerwiseEnv()
        original_step = env.step

        def failed_step(row):
            state, reward, done, info = original_step(row)
            if done:
                env.runtime_terminal_info["eval_failed"] = True
                env.runtime_terminal_info["error"] = "cuda failure"
            return state, reward, done, info

        env.step = failed_step
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
        )

        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            evaluator(((1, 0), (0, 0)))

    def test_persistence_keeps_manifest_observations_history_and_summary(self):
        env = _LayerwiseEnv()
        evaluation_rows = []
        evaluator = LayerwiseRuntimeEvaluator(
            env=env,
            reference=_Reference(),
            base_seed=17,
            expected_trials=3,
            on_evaluation=evaluation_rows.append,
        )
        result = run_search(
            "greedy",
            LayerwiseSearchSpace(2),
            evaluator,
            SearchConfig(evaluation_budget=3, seed=17),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = persist_search_result(
                output_dir=tmpdir,
                result=result,
                manifest=_search_manifest(scientific_status="smoke"),
                observation_rows=evaluation_rows,
            )

            self.assertEqual(
                set(paths),
                {"manifest", "observations", "history", "summary"},
            )
            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)
            with open(paths["observations"], encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), result.evaluation_count)
            with open(paths["summary"], encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["best"]["metrics"]["metric2_mean"], 0.85)
            self.assertEqual(
                summary["best"]["metadata"]["installed_action"]["layers"][0][
                    "precision_preset_name"
                ],
                result.best.metadata["installed_action"]["layers"][0][
                    "precision_preset_name"
                ],
            )

    def test_reference_limits_include_all_six_channels(self):
        limits = limits_from_reference(_Reference())

        self.assertEqual(limits.loss_max, 1.01)
        self.assertEqual(limits.metric2_min, 0.84)
        self.assertEqual(limits.metric2_std_max, 0.018)

    def test_complete_greedy_run_persists_crash_recoverable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=2,
                seed=17,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest=_search_manifest(scientific_status="smoke"),
            )

            self.assertEqual(run["result"].evaluation_count, 2)
            self.assertEqual(
                run["manifest"]["status"], "smoke_only_complete",
            )
            self.assertNotIn("scientific_export_allowed", run)
            self.assertEqual(
                run["manifest"]["scientific_status"],
                "smoke_only_no_validation_full_gate",
            )
            self.assertEqual(
                run["manifest"]["search_config"]["population_size"], 4,
            )
            self.assertEqual(
                run["manifest"]["inference_reaching_candidate_count"], 2,
            )
            self.assertEqual(
                run["manifest"]["online_candidate_trial_count"], 6,
            )
            self.assertGreaterEqual(run["manifest"]["total_wall_seconds"], 0.0)
            with open(
                    run["artifact_paths"]["observations"],
                    encoding="utf-8",
            ) as handle:
                self.assertEqual(
                    sum(1 for line in handle if line.strip()),
                    2,
                )

    def test_completed_run_resumes_without_reinference(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 2,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_env = _LayerwiseEnv()
            first = run_layerwise_search_baseline(
                layerwise_env=first_env,
                output_dir=tmpdir,
                **kwargs,
            )
            self.assertGreater(first_env.base.clear_count, 0)

            second_env = _LayerwiseEnv()
            second = run_layerwise_search_baseline(
                layerwise_env=second_env,
                output_dir=tmpdir,
                **kwargs,
            )
            self.assertEqual(
                second["manifest"]["status"], "smoke_only_complete",
            )
            self.assertEqual(second_env.base.clear_count, 0)
            self.assertEqual(
                second["selected"].action_matrix,
                first["selected"].action_matrix,
            )
            self.assertEqual(
                second["result"].evaluation_count,
                first["result"].evaluation_count,
            )



    def test_pending_strict_context_writer_receives_resume_contract_before_strict(self):
        context_calls = []
        requested_manifest = _search_manifest()

        def interrupted_strict(_result):
            self.assertEqual(len(context_calls), 1)
            raise RuntimeError("strict infrastructure interruption")

        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
                RuntimeError, "strict infrastructure interruption"):
            run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=36,
                seed=42,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest=requested_manifest,
                strict_validator=interrupted_strict,
                pending_strict_context_writer=context_calls.append,
            )

        self.assertEqual(len(context_calls), 1)
        resume_contract = context_calls[0]
        self.assertEqual(resume_contract["search_backend"], "greedy")
        self.assertTrue(resume_contract["strict_validation_requested"])
        self.assertEqual(
            resume_contract["requested_manifest"], requested_manifest,
        )




















    def test_completed_resume_rejects_truncated_observation_journal(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 2,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_layerwise_search_baseline(
                layerwise_env=_LayerwiseEnv(),
                output_dir=tmpdir,
                **kwargs,
            )
            observation_path = first["artifact_paths"]["observations"]
            with open(observation_path, encoding="utf-8") as handle:
                first_row = next(handle)
            with open(observation_path, "w", encoding="utf-8") as handle:
                handle.write(first_row)

            with self.assertRaisesRegex(
                    RuntimeError,
                    "completed observation count does not match manifest",
            ):
                run_layerwise_search_baseline(
                    layerwise_env=_LayerwiseEnv(),
                    output_dir=tmpdir,
                    **kwargs,
                )

    def test_completed_resume_rejects_mismatched_termination_metadata(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 2,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_layerwise_search_baseline(
                layerwise_env=_LayerwiseEnv(),
                output_dir=tmpdir,
                **kwargs,
            )
            summary_path = first["artifact_paths"]["summary"]
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            summary["termination_reason"] = "tampered"
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle)

            with self.assertRaisesRegex(
                    RuntimeError,
                    "completed termination reason does not match manifest",
            ):
                run_layerwise_search_baseline(
                    layerwise_env=_LayerwiseEnv(),
                    output_dir=tmpdir,
                    **kwargs,
                )

    def test_strict_phase_resume_does_not_repeat_online_search(self):
        def completed_strict(result):
            return _strict_artifact(
                result, strict_feasible=True, strict_trial_count=45,
            )

        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 36,
            "seed": 42,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_env = _LayerwiseEnv()
            with self.assertRaisesRegex(RuntimeError, "strict interrupted"):
                run_layerwise_search_baseline(
                    layerwise_env=first_env,
                    output_dir=tmpdir,
                    strict_validator=lambda _result: (_ for _ in ()).throw(
                        RuntimeError("strict interrupted")
                    ),
                    **kwargs,
                )
            self.assertGreater(first_env.base.clear_count, 0)
            with open(
                    os.path.join(tmpdir, "manifest.json"),
                    encoding="utf-8",
            ) as handle:
                self.assertEqual(
                    json.load(handle)["status"],
                    "search_complete_pending_strict",
                )

            second_env = _LayerwiseEnv()
            resumed = run_layerwise_search_baseline(
                layerwise_env=second_env,
                output_dir=tmpdir,
                strict_validator=completed_strict,
                **kwargs,
            )
            self.assertEqual(second_env.base.clear_count, 0)
            self.assertEqual(
                resumed["strict_validation"]["selection_status"],
                "strict_feasible",
            )
            self.assertTrue(resumed["manifest"]["strict_feasible"])
            self.assertTrue(resumed["manifest"]["strict_validation_passed"])
            self.assertNotIn("scientific_export_allowed", resumed)
            self.assertEqual(
                resumed["manifest"]["strict_trial_count"], 225,
            )
            self.assertEqual(resumed["manifest"]["strict_attempt_count"], 2)
            self.assertGreaterEqual(
                resumed["manifest"]["strict_attempt_wall_seconds_total"],
                resumed["manifest"]["last_strict_attempt_wall_seconds"],
            )



















    def test_strict_runner_requires_seed_42_before_search(self):
        scientific_manifest = _search_manifest()

        with tempfile.TemporaryDirectory() as tmpdir, patch(
                "blb_stage2_rl.search_baseline_runner.run_search",
                side_effect=AssertionError("search must not run"),
        ):
            with self.assertRaisesRegex(ValueError, "seed 42"):
                run_layerwise_search_baseline(
                    backend="bo_rf",
                    layerwise_env=_LayerwiseEnv(),
                    robust_reference=_Reference(),
                    output_dir=tmpdir,
                    evaluation_budget=50_000,
                    seed=17,
                    initial_design_size=64,
                    candidate_pool_size=2_048,
                    population_size=64,
                    patience_generations=2_000,
                    mutation_max_coordinates=4,
                    rf_n_estimators=128,
                    rf_min_samples_leaf=2,
                    communication_importance_ratio=1.0,
                    manifest=scientific_manifest,
                    strict_validator=lambda _result: {},
                )

    def test_strict_bo_requires_canonical_search_configuration(self):
        scientific_manifest = _search_manifest()
        canonical = {
            "initial_design_size": 64,
            "candidate_pool_size": 2_048,
            "patience_generations": 2_000,
            "rf_n_estimators": 128,
            "rf_min_samples_leaf": 2,
        }
        for field, value in (
            ("initial_design_size", 63),
            ("candidate_pool_size", 2_047),
            ("patience_generations", 1_999),
            ("rf_n_estimators", 127),
            ("rf_min_samples_leaf", 3),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir, patch(
                    "blb_stage2_rl.search_baseline_runner.run_search",
                    side_effect=AssertionError("search must not run"),
            ):
                with self.assertRaisesRegex(ValueError, "Bayesian"):
                    run_layerwise_search_baseline(
                        backend="bo_rf",
                        layerwise_env=_LayerwiseEnv(),
                        robust_reference=_Reference(),
                        output_dir=tmpdir,
                        evaluation_budget=50_000,
                        seed=42,
                        population_size=64,
                        mutation_max_coordinates=4,
                        communication_importance_ratio=1.0,
                        manifest=scientific_manifest,
                        strict_validator=lambda _result: {},
                        **{**canonical, field: value},
                    )

    def test_strict_bo_accepts_canonical_search_configuration(self):
        scientific_manifest = _search_manifest()

        with tempfile.TemporaryDirectory() as tmpdir, patch(
                "blb_stage2_rl.search_baseline_runner.run_search",
                side_effect=RuntimeError("canonical search reached"),
        ):
            with self.assertRaisesRegex(RuntimeError, "canonical search reached"):
                run_layerwise_search_baseline(
                    backend="bo_rf",
                    layerwise_env=_LayerwiseEnv(),
                    robust_reference=_Reference(),
                    output_dir=tmpdir,
                    evaluation_budget=50_000,
                    seed=42,
                    initial_design_size=64,
                    candidate_pool_size=2_048,
                    population_size=64,
                    patience_generations=2_000,
                    mutation_max_coordinates=4,
                    rf_n_estimators=128,
                    rf_min_samples_leaf=2,
                    communication_importance_ratio=1.0,
                    manifest=scientific_manifest,
                    strict_validator=lambda _result: {},
                )

    def test_ga_accepts_five_generation_stagnation(self):
        validator = getattr(
            search_runner_module,
            "_validate_ga_completion_proof",
            None,
        )
        self.assertIsNotNone(validator)
        space = LayerwiseSearchSpace(4)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _search_evaluation(action),
            SearchConfig(
                evaluation_budget=100,
                seed=17,
                ga_population_size=12,
                ga_elite_count=2,
                ga_generations=10,
                patience_generations=5,
            ),
        )

        validator(
            result,
            patience_generations=5,
            generation_cap=10,
            maximum_evaluations=100,
        )

    def test_ga_full_run_completion_requires_exact_generation_cap(self):
        validator = getattr(
            search_runner_module,
            "_validate_ga_completion_proof",
            None,
        )
        self.assertIsNotNone(validator)
        result = run_search(
            "coinn_ga",
            LayerwiseSearchSpace(4),
            lambda action: _search_evaluation(action),
            SearchConfig(
                evaluation_budget=12 + 3 * 10,
                seed=17,
                ga_population_size=12,
                ga_elite_count=2,
                ga_generations=3,
                patience_generations=1,
                ga_stop_on_no_improvement=False,
                ga_require_full_generations=True,
            ),
        )

        validator(
            result,
            patience_generations=1,
            generation_cap=3,
            maximum_evaluations=42,
            stop_on_no_improvement=False,
            require_full_generations=True,
        )

        forged = SearchResult(
            algorithm=result.algorithm,
            best=result.best,
            observations=result.observations,
            history=result.history,
            termination_reason="ga_no_incumbent_improvement",
        )
        with self.assertRaisesRegex(RuntimeError, "full-generation"):
            validator(
                forged,
                patience_generations=1,
                generation_cap=3,
                maximum_evaluations=42,
                stop_on_no_improvement=False,
                require_full_generations=True,
            )

    def test_ga_rejects_stagnation_before_five_generations(self):
        validator = getattr(
            search_runner_module,
            "_validate_ga_completion_proof",
            None,
        )
        self.assertIsNotNone(validator)
        space = LayerwiseSearchSpace(4)
        result = run_search(
            "coinn_ga",
            space,
            lambda action: _search_evaluation(action),
            SearchConfig(
                evaluation_budget=100,
                seed=17,
                ga_population_size=12,
                ga_elite_count=2,
                ga_generations=10,
                patience_generations=5,
            ),
        )
        updates = [
            row for row in result.history
            if row.get("phase") == "ga_update_generation"
        ]
        fourth_observation_count = int(updates[3]["observations"])
        forged_observations = result.observations[:fourth_observation_count]
        forged = SearchResult(
            algorithm=result.algorithm,
            best=max(forged_observations, key=candidate_rank_key),
            observations=forged_observations,
            history=tuple(
                row for row in result.history
                if int(row.get("generation", 0)) <= 4
            ),
            termination_reason="ga_no_incumbent_improvement",
        )

        with self.assertRaisesRegex(RuntimeError, "five-generation"):
            validator(
                forged,
                patience_generations=5,
                generation_cap=10,
                maximum_evaluations=100,
            )


    def test_ga_cannot_complete_after_observation_guard(self):
        evaluation = _search_evaluation(((0, 0), (0, 0)))
        scientific_manifest = _search_manifest()
        incomplete = SearchResult(
            algorithm="coinn_ga",
            best=evaluation,
            observations=(evaluation,),
            history=(),
            termination_reason="observation_attempt_guard",
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch(
                "blb_stage2_rl.search_baseline_runner.run_search",
                return_value=incomplete,
        ), patch(
                "blb_stage2_rl.search_baseline_runner.persist_search_result",
                return_value={
                    "manifest": os.path.join(tmpdir, "manifest.json"),
                    "observations": os.path.join(tmpdir, "observations.jsonl"),
                    "history": os.path.join(tmpdir, "history.json"),
                    "summary": os.path.join(tmpdir, "summary.json"),
                },
        ):
            with self.assertRaisesRegex(
                    RuntimeError, "full-generation",
            ):
                run_layerwise_search_baseline(
                    backend="coinn_ga",
                    layerwise_env=_LayerwiseEnv(),
                    robust_reference=_Reference(),
                    output_dir=tmpdir,
                    evaluation_budget=11_464,
                    seed=42,
                    initial_design_size=64,
                    candidate_pool_size=2_048,
                    population_size=64,
                    patience_generations=5,
                    mutation_max_coordinates=4,
                    rf_n_estimators=128,
                    rf_min_samples_leaf=2,
                    communication_importance_ratio=1.0,
                    manifest=scientific_manifest,
                    strict_validator=lambda _result: {},
                )

    def test_greedy_resume_accumulates_pre_interruption_online_time(self):
        kwargs = {
            "backend": "greedy",
            "robust_reference": _Reference(),
            "evaluation_budget": 2,
            "seed": 17,
            "initial_design_size": 2,
            "candidate_pool_size": 8,
            "population_size": 4,
            "patience_generations": 5,
            "mutation_max_coordinates": 1,
            "rf_n_estimators": 8,
            "rf_min_samples_leaf": 1,
            "communication_importance_ratio": 1.0,
            "manifest": _search_manifest(),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            def interrupted_search(_backend, space, evaluator, _config, **_kwargs):
                evaluator(space.safe_action)
                raise RuntimeError("online interrupted")

            with patch(
                    "blb_stage2_rl.search_baseline_runner.run_search",
                    side_effect=interrupted_search,
            ):
                with self.assertRaisesRegex(RuntimeError, "online interrupted"):
                    run_layerwise_search_baseline(
                        layerwise_env=_LayerwiseEnv(),
                        output_dir=tmpdir,
                        **kwargs,
                    )
            preload = load_search_preload(os.path.join(
                tmpdir, "observations.jsonl",
            ))
            interrupted_wall = float(
                preload[-1].metadata["search_cumulative_wall_seconds"]
            )

            resumed = run_layerwise_search_baseline(
                layerwise_env=_LayerwiseEnv(),
                output_dir=tmpdir,
                **kwargs,
            )
            self.assertGreaterEqual(
                resumed["manifest"]["online_search_wall_seconds"],
                interrupted_wall,
            )

    def test_partial_bo_and_ga_resume_replays_prefix_without_forward(self):
        class TrackingLayerwiseEnv(_LayerwiseEnv):
            def __init__(self):
                super().__init__()
                self.action_history = []

            def step(self, row):
                state, reward, done, info = super().step(row)
                if done:
                    self.action_history.append(tuple(
                        tuple(value for value in layer) for layer in self.rows
                    ))
                return state, reward, done, info

        for backend, budget in (("bo_rf", 2), ("coinn_ga", 804)):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmpdir:
                kwargs = {
                    "backend": backend,
                    "layerwise_env": _LayerwiseEnv(),
                    "robust_reference": _Reference(),
                    "output_dir": tmpdir,
                    "evaluation_budget": budget,
                    "seed": 17,
                    "initial_design_size": 2,
                    "candidate_pool_size": 8,
                    "population_size": 4,
                    "patience_generations": 5,
                    "mutation_max_coordinates": 4,
                    "rf_n_estimators": 8,
                    "rf_min_samples_leaf": 1,
                    "communication_importance_ratio": 1.0,
                    "manifest": _search_manifest(),
                }

                def interrupted_search(_backend, space, evaluator, _config, **_kwargs):
                    evaluator(space.safe_action)
                    raise RuntimeError("online interrupted")

                with patch(
                        "blb_stage2_rl.search_baseline_runner.run_search",
                        side_effect=interrupted_search,
                ):
                    with self.assertRaisesRegex(RuntimeError, "online interrupted"):
                        run_layerwise_search_baseline(**kwargs)

                resumed_env = TrackingLayerwiseEnv()
                resumed = run_layerwise_search_baseline(
                    **{**kwargs, "layerwise_env": resumed_env}
                )

                self.assertEqual(
                    resumed["manifest"]["preloaded_observation_count"], 1,
                )
                self.assertNotIn(
                    LayerwiseSearchSpace(resumed_env.horizon).safe_action,
                    resumed_env.action_history,
                )
                self.assertEqual(
                    len(resumed_env.action_history),
                    resumed["result"].evaluation_count - 1,
                )

    def test_resume_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            common = {
                "backend": "greedy",
                "layerwise_env": _LayerwiseEnv(),
                "robust_reference": _Reference(),
                "output_dir": tmpdir,
                "evaluation_budget": 2,
                "seed": 17,
                "initial_design_size": 2,
                "candidate_pool_size": 8,
                "population_size": 4,
                "patience_generations": 5,
                "mutation_max_coordinates": 1,
                "rf_n_estimators": 8,
                "rf_min_samples_leaf": 1,
                "communication_importance_ratio": 1.0,
                "manifest": _search_manifest(),
            }
            run_layerwise_search_baseline(**common)
            with self.assertRaisesRegex(RuntimeError, "resume contract"):
                run_layerwise_search_baseline(
                    **{
                        **common,
                        "layerwise_env": _LayerwiseEnv(),
                        "seed": 18,
                    }
                )


    def test_no_strict_feasible_returns_materializable_fallback_and_artifacts(self):
        def strict_validator(result):
            return _strict_artifact(
                result, strict_feasible=False, strict_trial_count=15,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=36,
                seed=42,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest=_search_manifest(),
                strict_validator=strict_validator,
            )

            self.assertIsNotNone(run["selected"])
            self.assertFalse(run["strict_feasible"])
            self.assertNotIn("scientific_export_allowed", run)
            self.assertEqual(
                run["manifest"]["status"],
                "complete_least_violating",
            )
            self.assertEqual(
                run["manifest"]["selection_status"],
                "strict_least_violating",
            )
            self.assertTrue(os.path.isfile(
                run["artifact_paths"]["final_selected_configuration"]
            ))
            self.assertTrue(os.path.isfile(
                run["artifact_paths"]["online_best"]
            ))
            self.assertEqual(
                run["online_best"].action_matrix,
                run["result"].best.action_matrix,
            )





    def test_canonical_strict_validation_reuses_shared_bank_gates(self):
        result = _search_result(*_five_eligible_evaluations())
        strict_metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }
        axis_counterfactuals = _passing_axis_counterfactuals(strict_metrics)
        strict_evidence = SimpleNamespace(groups=[{
            "final_config_fingerprint": "f" * 64,
        }])
        promotion = SimpleNamespace(
            status="promoted",
            trial_count=30,
            fresh_trial_count=30,
            metrics=None,
            assessment=None,
            axis_counterfactuals=axis_counterfactuals,
            evidence=strict_evidence,
        )
        certification = SimpleNamespace(
            status="final_revalidation_passed",
            trial_count=45,
            fresh_trial_count=15,
            metrics=strict_metrics,
            assessment={
                "loss_precision_probability": 0.99,
                "metric1_precision_probability": 0.99,
                "metric2_precision_probability": 0.99,
                "loss_stability_probability": 0.99,
                "metric1_stability_probability": 0.99,
                "metric2_stability_probability": 0.99,
            },
            axis_counterfactuals=axis_counterfactuals,
            evidence=strict_evidence,
        )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=_strict_result_side_effect(promotion),
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
                side_effect=_strict_result_side_effect(certification),
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(
                    "/tmp/search_strict_candidates.jsonl"
                ),
                identity_context=_strict_identity_context(),
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 5)
        self.assertEqual(certify_mock.call_count, 5)
        self.assertTrue(all(
            "physical_trial_invocation_hash" not in call.kwargs
            for call in promote_mock.call_args_list + certify_mock.call_args_list
        ))
        self.assertEqual(strict["requested_top_n"], 5)
        self.assertEqual(strict["strict_evaluated_candidate_count"], 5)
        self.assertTrue(strict["strict_feasible"])
        self.assertTrue(all(
            record["strict_point_pass"] for record in strict["records"]
        ))
        self.assertIsNotNone(strict["selected"])
        self.assertEqual(
            strict["selected"]["metrics"]["metric2_mean"], 0.85,
        )
        self.assertNotIn("physical_trial_invocation_hash", strict)
        self.assertNotIn("physical_trial_accounting", strict)
        self.assertNotIn("formal_run_identity", strict)








    def test_already_certified_candidate_requires_complete_axis_evidence(self):
        result = _search_result(*_five_eligible_evaluations())
        metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }
        promotion = _promotion_result(
            status="already_promoted", trial_count=30, metrics=metrics,
        )
        certification = _promotion_result(
            status="already_final_certified",
            trial_count=45,
            metrics=metrics,
            axis_counterfactuals=None,
        )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
                side_effect=_strict_result_side_effect(promotion),
        ), patch(
                "blb_stage2_rl.layerwise_runner.certify_candidate_with_bank_c",
                side_effect=_strict_result_side_effect(certification),
        ), self.assertRaisesRegex(RuntimeError, "axis"):
            canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub("strict.jsonl"),
                identity_context=_strict_identity_context(),
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

    def test_canonical_validation_processes_all_top_five_candidates(self):
        result = _search_result(*_five_eligible_evaluations())
        metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }
        promotion = _promotion_result(
            status="promoted", trial_count=30, metrics=metrics,
        )
        certification = _promotion_result(
            status="final_revalidation_passed",
            trial_count=45,
            fresh_trial_count=15,
            metrics=metrics,
            assessment={name: 0.99 for name in (
                "loss_precision_probability",
                "metric1_precision_probability",
                "metric2_precision_probability",
                "loss_stability_probability",
                "metric1_stability_probability",
                "metric2_stability_probability",
            )},
            axis_counterfactuals=_passing_axis_counterfactuals(metrics),
        )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=_strict_result_side_effect(promotion),
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
                side_effect=_strict_result_side_effect(certification),
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 5)
        self.assertEqual(certify_mock.call_count, 5)
        self.assertEqual(len(strict["records"]), 5)
        self.assertTrue(all(
            record["strict_evaluated"] for record in strict["records"]
        ))
        self.assertEqual(strict["selection_status"], "strict_feasible")
        self.assertTrue(strict["strict_feasible"])
        self.assertNotIn("formal_feasible", strict)

    def test_top_five_includes_online_infeasible_and_strict_fallback(self):
        online_feasible = _search_evaluation(((0, 0), (0, 0)))
        online_infeasible = _search_evaluation(
            ((1, 2), (1, 2)), metric1_mean=0.70,
        )
        result = _search_result(*_five_eligible_evaluations(
            online_infeasible, online_feasible,
        ))

        def promotion_side_effect(**kwargs):
            action = tuple(tuple(row) for row in kwargs["action_matrix"])
            metric1 = (
                0.88 if action == online_infeasible.action_matrix else 0.80
            )
            return _bind_strict_result(
                _promotion_result(
                    status="bank_a_point_failed",
                    trial_count=15,
                    fresh_trial_count=15,
                    metrics={
                        "loss_mean": 1.0,
                        "metric1_mean": metric1,
                        "metric2_mean": 0.85,
                        "loss_std": 0.01,
                        "metric1_std": 0.01,
                        "metric2_std": 0.01,
                    },
                ),
                kwargs,
            )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=promotion_side_effect,
        ) as promote_mock, patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(
                    bank_a_reference=_reference(metric1_limit=0.89),
                    promotion_reference=_reference(metric1_limit=0.87),
                    final_reference=_reference(metric1_limit=0.85),
                ),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 5)
        certify_mock.assert_not_called()
        self.assertTrue(any(
            not record["online_candidate"]["feasible"]
            for record in strict["records"]
        ))
        self.assertEqual(strict["selection_status"], "strict_least_violating")
        self.assertFalse(strict["strict_feasible"])
        self.assertIsNotNone(strict["selected"])
        self.assertEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in online_infeasible.action_matrix],
        )
        selected_record = next(
            record for record in strict["records"]
            if record.get("strict_evaluation", {}).get("action_matrix")
            == strict["selected"]["action_matrix"]
        )
        self.assertEqual(selected_record["strict_trial_count"], 15)
        self.assertEqual(
            strict["selected"]["metadata"]["strict_trial_count"], 15,
        )
        self.assertEqual(
            strict["selected"]["metadata"]["strict_limits_source"],
            "bank_a",
        )
        self.assertEqual(strict["selected"]["limits"]["metric1_min"], 0.89)
        self.assertEqual(
            strict["selected_violations"]["families"]["joint"][
                "failed_constraint_count"
            ],
            1,
        )
        self.assertIn(
            "metric1_mean",
            strict["selected_violations"]["families"]["joint"][
                "constraints"
            ],
        )
        self.assertEqual(
            strict["selected_violations"]["families"]["compute_only"][
                "status"
            ],
            "not_run",
        )

    def test_strict_fallback_prioritizes_violation_before_unavailable_families(self):
        def with_violations(action, *, failed, total, worst, unavailable):
            payload = _search_evaluation(action).as_dict()
            payload["metadata"]["strict_violations"] = {
                "aggregate": {
                    "failed_constraint_count": failed,
                    "total_normalized_violation": total,
                    "worst_normalized_violation": worst,
                    "unavailable_family_count": unavailable,
                }
            }
            return SearchEvaluation.from_dict(payload)

        mild_incomplete = with_violations(
            ((0, 0), (0, 0)),
            failed=1,
            total=0.01,
            worst=0.01,
            unavailable=1,
        )
        severe_complete = with_violations(
            ((1, 2), (1, 2)),
            failed=2,
            total=2.0,
            worst=1.0,
            unavailable=0,
        )

        self.assertGreater(
            _strict_fallback_rank(mild_incomplete),
            _strict_fallback_rank(severe_complete),
        )

    def test_strict_fallback_preserves_sub_femtoscale_violation_order(self):
        def with_total(action, total):
            payload = _search_evaluation(action).as_dict()
            payload["metadata"]["strict_violations"] = {
                "aggregate": {
                    "failed_constraint_count": 1,
                    "total_normalized_violation": total,
                    "worst_normalized_violation": total,
                    "unavailable_family_count": 0,
                }
            }
            return SearchEvaluation.from_dict(payload)

        lower_violation = with_total(((0, 0), (0, 0)), 0.1)
        higher_violation = with_total(
            ((1, 2), (1, 2)), math.nextafter(0.1, math.inf),
        )
        self.assertGreater(
            higher_violation.resource.ppo_resource_score,
            lower_violation.resource.ppo_resource_score,
        )
        self.assertGreater(
            _strict_fallback_rank(lower_violation),
            _strict_fallback_rank(higher_violation),
        )

    def test_strict_fallback_ranks_joint_and_axis_violation_families(self):
        mild_axis = _search_evaluation(((0, 0), (0, 0)))
        severe_axis = _search_evaluation(((1, 0), (0, 0)))
        two_axis_failures = _search_evaluation(((1, 2), (1, 2)))
        result = _search_result(*_five_eligible_evaluations(
            mild_axis, severe_axis, two_axis_failures,
        ))
        joint_metrics = {
            "loss_mean": 1.0,
            "metric1_mean": 0.90,
            "metric2_mean": 0.85,
            "loss_std": 0.01,
            "metric1_std": 0.01,
            "metric2_std": 0.01,
        }

        def axis_payload(*, metric1_mean, point_pass):
            metrics = {**joint_metrics, "metric1_mean": metric1_mean}
            return {
                "final_config_fingerprint": "f" * 64,
                "loss_limit": 1.01,
                "metric1_limit": 0.89,
                "metric2_limit": 0.84,
                "loss_std_limit": 0.02,
                "metric1_std_limit": 0.015,
                "metric2_std_limit": 0.018,
                "banks": {
                    "A": {
                        "trial_count": 15,
                        "fresh_trial_count": 15,
                        "metrics": metrics,
                        "point_pass": point_pass,
                    },
                    "B": {
                        "trial_count": 30,
                        "fresh_trial_count": 15,
                        "metrics": metrics,
                        "point_pass": point_pass,
                    },
                },
                "point_pass": point_pass,
                "metrics": metrics,
            }

        def promotion_side_effect(**kwargs):
            action = tuple(tuple(row) for row in kwargs["action_matrix"])
            if action == mild_axis.action_matrix:
                compute = axis_payload(metric1_mean=0.88, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.90, point_pass=True,
                )
            elif action == severe_axis.action_matrix:
                compute = axis_payload(metric1_mean=0.70, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.90, point_pass=True,
                )
            else:
                compute = axis_payload(metric1_mean=0.80, point_pass=False)
                communication = axis_payload(
                    metric1_mean=0.80, point_pass=False,
                )
            return _bind_strict_result(
                _promotion_result(
                    status="axis_counterfactual_point_failed",
                    trial_count=30,
                    fresh_trial_count=30,
                    metrics=joint_metrics,
                    axis_counterfactuals={
                        "compute": compute,
                        "communication": communication,
                    },
                ),
                kwargs,
            )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=promotion_side_effect,
        ), patch(
                "blb_stage2_rl.layerwise_runner."
                "certify_candidate_with_bank_c",
        ) as certify_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        certify_mock.assert_not_called()
        self.assertEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in mild_axis.action_matrix],
        )
        aggregate = strict["selected_violations"]["aggregate"]
        self.assertEqual(aggregate["failed_constraint_count"], 1)
        selected_compute = strict["selected_violations"]["families"][
            "compute_only"
        ]
        self.assertEqual(selected_compute["trial_count"], 30)
        self.assertEqual(selected_compute["banks_run"], ["A", "B"])
        self.assertEqual(selected_compute["not_run_banks"], ["C"])
        self.assertEqual(selected_compute["status"], "partial_early_stopped")
        self.assertEqual(
            strict["selected"]["metadata"]["strict_trial_count"], 30,
        )

    def test_strict_validation_requires_five_eligible_candidates(self):
        valid = _search_evaluation(((0, 0), (0, 0)))
        invalid = _search_evaluation(
            ((1, 2), (1, 2)), valid=False, materializable=False,
        )
        four_valid = _five_eligible_evaluations(valid)[:4]
        with self.assertRaisesRegex(RuntimeError, "eligible=4 requested=5"):
            canonical_strict_validation(
                result=_search_result(invalid, *four_valid),
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

    def test_strict_infrastructure_failure_is_not_least_violating(self):
        promotion = _promotion_result(
            status="failed_evaluation",
            trial_count=15,
            metrics={
                "loss_mean": 1.0,
                "metric1_mean": 0.90,
                "metric2_mean": 0.85,
                "loss_std": 0.01,
                "metric1_std": 0.01,
                "metric2_std": 0.01,
            },
        )
        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=_strict_result_side_effect(promotion),
        ), self.assertRaisesRegex(RuntimeError, "infrastructure evaluation failed"):
            canonical_strict_validation(
                result=_search_result(*_five_eligible_evaluations()),
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

    def test_strict_candidate_identity_mismatch_fails_closed(self):
        promotion = _promotion_result(
            status="bank_a_point_failed",
            trial_count=15,
            metrics={
                "loss_mean": 1.0,
                "metric1_mean": 0.88,
                "metric2_mean": 0.85,
                "loss_std": 0.01,
                "metric1_std": 0.01,
                "metric2_std": 0.01,
            },
            evidence=SimpleNamespace(
                candidate_key="0" * 64,
                action_indices=(999,),
                groups=[{"final_config_fingerprint": "f" * 64}],
            ),
        )
        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                return_value=promotion,
        ), self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            canonical_strict_validation(
                result=_search_result(*_five_eligible_evaluations()),
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

    def test_invalid_and_nonmaterializable_candidates_cannot_be_selected(self):
        invalid = _search_evaluation(
            ((1, 2), (1, 2)), valid=False, materializable=False,
        )
        materializable = _search_evaluation(((0, 0), (0, 0)))
        eligible = _five_eligible_evaluations(materializable)
        result = _search_result(invalid, *eligible)
        promotion = _promotion_result(
            status="bank_a_point_failed",
            trial_count=15,
            metrics={
                "loss_mean": 1.0,
                "metric1_mean": 0.88,
                "metric2_mean": 0.85,
                "loss_std": 0.01,
                "metric1_std": 0.01,
                "metric2_std": 0.01,
            },
        )

        with _patch_strict_materialization_preparation(), patch(
                "blb_stage2_rl.layerwise_runner."
                "promote_candidate_if_eligible",
                side_effect=_strict_result_side_effect(promotion),
        ) as promote_mock:
            strict = canonical_strict_validation(
                result=result,
                layerwise_env=object(),
                promotion_base_env=object(),
                candidate_store=_candidate_store_stub(),
                identity_context={"profile": "mrpc"},
                validation_banks=_validation_banks(),
                top_n=5,
                communication_importance_ratio=1.0,
                promotion_probability=0.8,
                final_probability=0.95,
            )

        self.assertEqual(promote_mock.call_count, 5)
        self.assertNotEqual(
            strict["selected"]["action_matrix"],
            [list(row) for row in invalid.action_matrix],
        )
        self.assertEqual(strict["requested_top_n"], 5)
        self.assertEqual(strict["eligible_online_candidate_count"], 5)
        self.assertEqual(len(strict["records"]), 5)


    def test_point_gated_strict_selection_round_trips_without_probabilities(self):
        def strict_validator(result):
            artifact = _strict_artifact(
                result,
                strict_feasible=True,
                selected_updates={
                    "constraint_probabilities": {},
                    "gate_probability": None,
                },
            )
            return artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            run = run_layerwise_search_baseline(
                backend="greedy",
                layerwise_env=_LayerwiseEnv(),
                robust_reference=_Reference(),
                output_dir=tmpdir,
                evaluation_budget=36,
                seed=42,
                initial_design_size=2,
                candidate_pool_size=8,
                population_size=4,
                patience_generations=5,
                mutation_max_coordinates=1,
                rf_n_estimators=8,
                rf_min_samples_leaf=1,
                communication_importance_ratio=1.0,
                manifest=_search_manifest(),
                strict_validator=strict_validator,
            )

        self.assertTrue(run["manifest"]["strict_feasible"])
        self.assertTrue(run["manifest"]["strict_validation_passed"])
        self.assertNotIn("scientific_export_allowed", run)
        self.assertEqual(run["selected"].constraint_probabilities, ())
        self.assertIsNone(run["selected"].gate_probability)


if __name__ == "__main__":
    unittest.main()
