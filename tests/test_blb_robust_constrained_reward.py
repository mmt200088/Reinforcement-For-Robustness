import pathlib
import ast
import importlib.machinery
import importlib.util
import json
import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BLB_DIR = _REPO_ROOT / "blb_stage2_rl"
for _path in (str(_REPO_ROOT), str(_BLB_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from reward import (
    BaselineCostStats,
    EpisodeMetrics,
    RewardBreakdown,
    RewardWeights,
    boundary_signal,
    compute_reward,
    robust_constrained_reward,
)
from statistical_constraints import ConstraintAssessment
import layerwise_action


def _assessment(**overrides):
    values = {
        "loss_precision_probability": 0.9,
        "metric1_precision_probability": 0.9,
        "metric2_precision_probability": 0.9,
        "loss_stability_probability": 0.9,
        "metric1_stability_probability": 0.9,
        "metric2_stability_probability": 0.9,
        "precision_probability": 0.9,
        "stability_probability": 0.9,
        "gate_probability": 0.5,
        "online_precision_pass": True,
        "online_stability_pass": True,
    }
    values.update(overrides)
    values["precision_probability"] = min(
        values["loss_precision_probability"],
        values["metric1_precision_probability"],
        values["metric2_precision_probability"],
    )
    values["stability_probability"] = min(
        values["loss_stability_probability"],
        values["metric1_stability_probability"],
        values["metric2_stability_probability"],
    )
    values["online_precision_pass"] = values["precision_probability"] >= 0.5
    values["online_stability_pass"] = values["stability_probability"] >= 0.5
    return ConstraintAssessment(**values)


def test_boundary_signal_matches_clipped_log_ratio():
    assert boundary_signal(0.5) == 0.0
    assert boundary_signal(0.0) == -1.0
    assert np.isclose(boundary_signal(1.0), np.log((1.0 + 1e-8) / (0.5 + 1e-8)))


def test_pure_helper_returns_formula_tuple_and_boundary_components():
    assessment = _assessment(
        loss_precision_probability=0.8,
        metric1_stability_probability=0.7,
    )
    precision_signal = boundary_signal(0.8)
    stability_signal = boundary_signal(0.7)

    result = robust_constrained_reward(
        assessment, invalid=False, variable_cost=0.4,
    )

    assert result == (
        1.0 + 0.4 + 0.0005 * (precision_signal + stability_signal),
        3,
        precision_signal,
        stability_signal,
    )


def test_strict_tier_ordering_holds_over_probability_and_cost_boundaries():
    probabilities = np.linspace(0.0, 1.0, 101)
    costs = np.linspace(0.0, 1.0, 101)
    p1_rewards = [
        robust_constrained_reward(
            _assessment(loss_precision_probability=float(p)),
            invalid=False,
            variable_cost=float(cost),
        )[0]
        for p in probabilities if p < 0.5
        for cost in costs
    ]
    p2_rewards = [
        robust_constrained_reward(
            _assessment(loss_stability_probability=float(p)),
            invalid=False,
            variable_cost=float(cost),
        )[0]
        for p in probabilities if p < 0.5
        for cost in costs
    ]
    p3_rewards = [
        robust_constrained_reward(
            _assessment(
                loss_precision_probability=float(p),
                loss_stability_probability=float(s),
            ),
            invalid=False,
            variable_cost=float(cost),
        )[0]
        for p in probabilities if p >= 0.5
        for s in probabilities if s >= 0.5
        for cost in costs
    ]

    assert min(p3_rewards) > max(p2_rewards)
    assert min(p2_rewards) > max(p1_rewards)


def test_cost_never_affects_failed_constraint_tiers():
    for assessment in (
        _assessment(metric1_precision_probability=0.25),
        _assessment(metric1_stability_probability=0.25),
    ):
        rewards = {
            robust_constrained_reward(assessment, invalid=False, variable_cost=cost)[0]
            for cost in np.linspace(0.0, 1.0, 21)
        }
        assert len(rewards) == 1


def test_each_precision_channel_independently_yields_p1():
    for field in (
        "loss_precision_probability",
        "metric1_precision_probability",
        "metric2_precision_probability",
    ):
        reward, priority, precision_signal, stability_signal = robust_constrained_reward(
            _assessment(**{field: 0.49}), invalid=False, variable_cost=1.0,
        )
        assert priority == 1
        assert reward == -3.0 + 0.5 * precision_signal
        assert precision_signal == boundary_signal(0.49)
        assert stability_signal == boundary_signal(0.9)


def test_each_stability_channel_independently_yields_p2_after_precision_passes():
    for field in (
        "loss_stability_probability",
        "metric1_stability_probability",
        "metric2_stability_probability",
    ):
        reward, priority, precision_signal, stability_signal = robust_constrained_reward(
            _assessment(**{field: 0.49}), invalid=False, variable_cost=1.0,
        )
        assert priority == 2
        assert reward == -1.5 + 0.5 * stability_signal
        assert precision_signal == boundary_signal(0.9)
        assert stability_signal == boundary_signal(0.49)


def test_invalid_is_exactly_minus_five_and_preserves_probability_diagnostics():
    assessment = _assessment(
        loss_precision_probability=0.2,
        metric2_stability_probability=0.3,
    )
    result = robust_constrained_reward(assessment, invalid=True, variable_cost=1.0)

    assert result == (-5.0, 1, boundary_signal(0.2), boundary_signal(0.3))


def test_invalid_without_assessment_has_floor_signals():
    assert robust_constrained_reward(None, invalid=True, variable_cost=0.0) == (
        -5.0, 1, -1.0, -1.0,
    )


def _captured_value_error(call):
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("call did not reject invalid unit-interval input")


def test_helper_and_dispatcher_reject_bad_variable_cost_identically():
    for value in (-0.01, 1.01, float("nan"), float("inf"), float("-inf")):
        helper_message = _captured_value_error(lambda: robust_constrained_reward(
            _assessment(), invalid=False, variable_cost=value,
        ))
        dispatcher_message = _captured_value_error(lambda: compute_reward(
            EpisodeMetrics(), SimpleNamespace(any_invalid=False),
            action_avg_k=13.0,
            baseline=BaselineCostStats(),
            weights=RewardWeights(reward_design="robust_constrained"),
            external_cost_score=value,
            constraint_assessment=_assessment(),
        ))
        assert helper_message == dispatcher_message
        assert "variable_cost" in helper_message


def test_helper_and_dispatcher_reject_all_bad_probabilities_identically():
    fields = (
        "loss_precision_probability",
        "metric1_precision_probability",
        "metric2_precision_probability",
        "loss_stability_probability",
        "metric1_stability_probability",
        "metric2_stability_probability",
    )
    for field in fields:
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            assessment = _assessment(**{field: value})
            helper_message = _captured_value_error(lambda: robust_constrained_reward(
                assessment, invalid=False, variable_cost=0.5,
            ))
            dispatcher_message = _captured_value_error(lambda: compute_reward(
                EpisodeMetrics(), SimpleNamespace(any_invalid=False),
                action_avg_k=13.0,
                baseline=BaselineCostStats(),
                weights=RewardWeights(reward_design="robust_constrained"),
                external_cost_score=0.5,
                constraint_assessment=assessment,
            ))
            assert helper_message == dispatcher_message
            assert field in helper_message


def test_robust_breakdown_exposes_probability_q_signal_and_cost_fields():
    result = compute_reward(
        EpisodeMetrics(), SimpleNamespace(any_invalid=False),
        action_avg_k=13.0,
        baseline=BaselineCostStats(),
        weights=RewardWeights(reward_design="robust_constrained"),
        external_cost_score=0.4,
        constraint_assessment=_assessment(),
    )
    assert isinstance(result, RewardBreakdown)
    assert result.constraint_policy == "bootstrap_5x5_v1"
    assert result.loss_precision_probability == 0.9
    assert result.metric1_precision_probability == 0.9
    assert result.metric2_precision_probability == 0.9
    assert result.loss_stability_probability == 0.9
    assert result.metric1_stability_probability == 0.9
    assert result.metric2_stability_probability == 0.9
    assert result.q_precision == 0.9
    assert result.q_stability == 0.9
    assert result.precision_signal == boundary_signal(0.9)
    assert result.stability_signal == boundary_signal(0.9)
    assert result.variable_cost == 0.4
    assert result.cost_score == 0.4


def test_compute_reward_dispatches_only_when_robust_design_is_selected():
    result = compute_reward(
        EpisodeMetrics(),
        SimpleNamespace(any_invalid=False),
        action_avg_k=13.0,
        baseline=BaselineCostStats(),
        weights=RewardWeights(reward_design="robust_constrained"),
        external_cost_score=0.25,
        constraint_assessment=_assessment(),
    )
    assert result.priority == 3
    assert result.variable_cost == 0.25
    assert result.constraint_policy == "bootstrap_5x5_v1"


def test_compute_reward_canonicalizes_robust_design_without_routing_unknown_names():
    for reward_design in (" ROBUST_CONSTRAINED ", "Robust_Constrained"):
        result = compute_reward(
            EpisodeMetrics(), SimpleNamespace(any_invalid=False),
            action_avg_k=13.0,
            baseline=BaselineCostStats(),
            weights=RewardWeights(reward_design=reward_design),
            external_cost_score=0.25,
            constraint_assessment=_assessment(),
        )
        assert result.constraint_policy == "bootstrap_5x5_v1"

    result = compute_reward(
        EpisodeMetrics(), SimpleNamespace(any_invalid=False),
        action_avg_k=13.0,
        baseline=BaselineCostStats(),
        weights=RewardWeights(reward_design="robust-constrained"),
    )
    assert result.constraint_policy == ""


def test_valid_robust_compute_reward_requires_explicit_normalized_cost():
    try:
        compute_reward(
            EpisodeMetrics(), SimpleNamespace(any_invalid=False),
            action_avg_k=13.0,
            baseline=BaselineCostStats(),
            weights=RewardWeights(reward_design="robust_constrained"),
            external_cost_score=None,
            constraint_assessment=_assessment(),
        )
    except ValueError as exc:
        assert "external_cost_score" in str(exc)
    else:
        raise AssertionError("robust candidate without cost did not fail")


def test_invalid_robust_compute_reward_needs_neither_assessment_nor_cost():
    result = compute_reward(
        EpisodeMetrics(), SimpleNamespace(any_invalid=True),
        action_avg_k=13.0,
        baseline=BaselineCostStats(),
        weights=RewardWeights(reward_design="robust_constrained"),
    )
    assert result.reward == -5.0
    assert result.q_precision == 0.0
    assert result.q_stability == 0.0
    assert result.precision_signal == -1.0
    assert result.stability_signal == -1.0


def test_legacy_breakdown_defaults_remain_compatible():
    result = RewardBreakdown(reward=0.0, priority=3, invalid=False)
    assert result.constraint_policy == ""
    assert result.variable_cost == 0.0
    assert result.loss_precision_probability == 0.0
    assert result.q_precision == 0.0
    assert result.q_stability == 0.0
    assert result.precision_signal == 0.0
    assert result.stability_signal == 0.0


def _method_calls(method_name):
    tree = ast.parse((_BLB_DIR / "env.py").read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }


def test_prepared_terminal_route_calls_statistical_reward_integration():
    assert "_compute_terminal_reward" in _method_calls("_finish_prepared_terminal_probe")


def test_normal_terminal_route_calls_statistical_reward_integration():
    assert "_compute_terminal_reward" in _method_calls("step")


def _runtime_reference(statistical_constraints):
    groups = []
    for group_idx in range(5):
        offsets = np.linspace(-0.002, 0.002, 5) + group_idx * 0.0001
        groups.append(statistical_constraints.TrialSeries(
            loss=1.0 + offsets,
            metric1=0.8 + offsets,
            metric2=0.7 + offsets,
            seeds=range(group_idx * 5, group_idx * 5 + 5),
        ))
    return statistical_constraints.build_baseline_reference(
        groups,
        precision_tolerance=0.01,
        stability_multiplier=2.0,
        bootstrap_samples=64,
        seed=17,
    )


_RUNTIME_MODULES = None


def _runtime_modules():
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is not None:
        return _RUNTIME_MODULES

    package_name = "_blb_robust_reward_test_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_BLB_DIR)]
    sys.modules[package_name] = package

    def load(module_name, path):
        loader = importlib.machinery.SourceFileLoader(module_name, str(path))
        spec = importlib.util.spec_from_loader(module_name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        return module

    reward_module = load(f"{package_name}.reward", _BLB_DIR / "reward.py")
    statistical_constraints = load(
        f"{package_name}.statistical_constraints",
        _BLB_DIR / "statistical_constraints.py",
    )

    action_space = types.ModuleType(f"{package_name}.action_space")
    for name in ("ActionDecodeResult", "MaxSFsTable"):
        setattr(action_space, name, type(name, (), {}))
    action_space.BLB_FIRST_INPUT_N = 0
    action_space.K_LEVELS = (8, 9, 10, 11, 12, 13)
    action_space.action_dims_for_config = lambda layers: [1] * int(layers)
    action_space.action_vector_to_cfgs = lambda *_args, **_kwargs: None
    action_space.avg_truncation_k_in_action = lambda *_args, **_kwargs: 13.0
    action_space.build_optimizer_requests = lambda *_args, **_kwargs: {}
    action_space.layer_dims = lambda *_args, **_kwargs: []
    action_space.make_all_max_action_vector = lambda layers: np.full(int(layers), 9)
    action_space.parse_config_name = lambda _name: (0, "", -1)

    candidate_store = types.ModuleType(f"{package_name}.candidate_store")
    candidate_store.action_hash = lambda action: "".join(
        f"{int(value):016x}" for value in np.asarray(action).reshape(-1)
    )
    optimizer_cost = types.ModuleType(f"{package_name}.optimizer_cost")
    optimizer_cost.apply_optimizer_outputs_to_cfgs = lambda **_kwargs: {
        "optimizer_cfg_overrides": {},
    }
    optimizer_cost.evaluate_action_for_cost = lambda *_args, **_kwargs: None
    optimizer_cost.materialize_action_for_model = lambda *_args, **_kwargs: None
    probe_runner = types.ModuleType(f"{package_name}.probe_runner")
    probe_runner.ProbeRunner = type("ProbeRunner", (), {})
    probe_runner.diagnostics_payload = lambda _diag: {}
    probe_runner.format_diagnostics_line = lambda _diag: ""

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.device = object
    torch_stub.long = object()
    blb_bridge = types.ModuleType("blb_rl_bridge")
    blb_bridge.BLBNoiseRLBridge = type("BLBNoiseRLBridge", (), {})
    rescale_bridge = types.ModuleType("rescale_optimizer_bridge")
    rescale_bridge.RescaleOptimizerBridge = type("RescaleOptimizerBridge", (), {})
    for name in (
        "aggregate_optimizer_signals",
        "apply_optimizer_output_to_cfg",
        "apply_rotation_flags_to_cfg",
        "sync_block2_aux_fresh_binding",
        "sync_block2_qk_binding",
        "sync_block4_v_mask_binding",
        "sync_block5_aux_fresh_binding",
    ):
        setattr(rescale_bridge, name, lambda *_args, **_kwargs: [])
    rescale_bridge._strip_layer_suffix = lambda name: (name, None)

    relative_modules = {
        action_space.__name__: action_space,
        candidate_store.__name__: candidate_store,
        optimizer_cost.__name__: optimizer_cost,
        probe_runner.__name__: probe_runner,
    }
    with mock.patch.dict(sys.modules, {
        **relative_modules,
        "torch": torch_stub,
        "blb_rl_bridge": blb_bridge,
        "rescale_optimizer_bridge": rescale_bridge,
    }):
        env_module = load(f"{package_name}.env", _BLB_DIR / "env.py")

    _RUNTIME_MODULES = env_module, reward_module, statistical_constraints
    return _RUNTIME_MODULES


def _runtime_metrics(reward_module):
    offsets = np.linspace(-0.001, 0.001, 5)
    return reward_module.EpisodeMetrics(
        loss_mean=0.95,
        loss_std=float(np.std(0.95 + offsets, ddof=1)),
        metric1_mean=0.85,
        metric2_mean=0.75,
        metric1_std=float(np.std(0.85 + offsets, ddof=1)),
        metric2_std=float(np.std(0.75 + offsets, ddof=1)),
        loss_trials=0.95 + offsets,
        metric1_trials=0.85 + offsets,
        metric2_trials=0.75 + offsets,
        trial_seeds=range(100, 105),
    )


def _runtime_env(env_module, reward_module, statistical_constraints):
    env = env_module.BLBStage2Env.__new__(env_module.BLBStage2Env)
    env.baseline = reward_module.BaselineCostStats(total_bits_sum=100.0, avg_k=13.0)
    env.reward_weights = reward_module.RewardWeights(reward_design="robust_constrained")
    env.acc_threshold = 0.0
    env.acc_threshold_m2 = 0.0
    env.stab_threshold = 1.0
    env.loss_threshold = None
    env.pareto_cost_archive = None
    env.statistical_reference = _runtime_reference(statistical_constraints)
    env.num_layers = 1
    env._step_idx = 0
    env._last_invalid_rate = 0.0
    env._last_total_bits_norm = 0.0
    env._last_fusion_count = 0.0
    env._build_state = lambda: np.asarray([0.0], dtype=np.float32)
    return env


def _runtime_materialized(decoded, signals):
    return SimpleNamespace(
        decoded=decoded,
        cfgs_dict={},
        outputs={},
        signals=signals,
        optimizer_eval_mode="fake",
        optimizer_invalid=False,
        model_ready=True,
        failure_reason=None,
        final_config_fingerprint="test-materialized-action",
        replan_application={
            "model_uses_replan_config": True,
            "optimizer_cfg_overrides": {},
        },
    )


def test_prepared_terminal_runtime_assesses_trials_deterministically_and_threads_cost():
    env_module, reward_module, statistical_constraints = _runtime_modules()

    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    decoded_actions = []
    for layer_idx in range(12):
        k_by_block = {2: 13, 3: 13, 4: 13, 5: 13}
        if layer_idx:
            k_by_block[1] = 13
        decoded_actions.append(layerwise_action.LayerwiseDecodedAction(1, k_by_block))
    resource = layerwise_action.compute_variable_cost(decoded_actions)
    variable_cost = resource.ppo_resource_score
    assert resource.compute_saving == 1.0
    assert resource.communication_saving == 0.0
    assert resource.robust_floor == 0.0
    assert resource.secondary_progress == 0.5
    assert np.isclose(
        variable_cost,
        layerwise_action.dual_resource_score(1.0, 0.0)[2],
    )
    external_resource_objective = {
        field_name: getattr(resource, field_name)
        for field_name in (
            "compute_saving",
            "communication_saving",
            "robust_floor",
            "secondary_progress",
            "ppo_resource_score",
            "compute_shapley_credit",
            "communication_shapley_credit",
            "layer_resource_rewards",
            "slot_resource_rewards",
        )
    }

    env = _runtime_env(env_module, reward_module, statistical_constraints)
    env.total_action_dim = 3
    env.env_cfg = SimpleNamespace(profile="mrpc")
    env.max_sfs = object()
    env.rescale_bridge = SimpleNamespace(invoker=SimpleNamespace(baselines={}))
    env.gelu_degree = 4
    env.attn_degree = 6
    env.sync_degree_vectors_from_model = lambda: {}
    decoded = SimpleNamespace(
        block1_cfgs=[], block2_cfgs=[], block3_cfgs=[], block4_cfgs=[], block5_cfgs=[],
    )
    materialized = _runtime_materialized(decoded, signals)
    with mock.patch.object(env_module, "make_all_max_action_vector", return_value=np.asarray([9, 9, 9])), \
            mock.patch.object(env_module, "materialize_action_for_model", return_value=materialized):
        prepared = env.prepare_action_for_terminal_probe(
            np.asarray([3, 1, 4]),
            external_cost_score=variable_cost,
            external_cost_rank=variable_cost,
            external_resource_objective=external_resource_objective,
        )
    metrics = _runtime_metrics(reward_module)
    assessments = []
    for _ in range(2):
        env = _runtime_env(env_module, reward_module, statistical_constraints)
        with mock.patch.object(env_module, "avg_truncation_k_in_action", return_value=13.0):
            _state, terminal_reward, done, info = env._finish_prepared_terminal_probe(
                prepared, metrics,
            )
        assert done is True
        assert info["reward_breakdown"].variable_cost == variable_cost
        assert info["reward_breakdown"].compute_saving == 1.0
        assert info["reward_breakdown"].communication_saving == 0.0
        assert info["reward_breakdown"].robust_floor == 0.0
        assert info["reward_breakdown"].secondary_progress == 0.5
        assert info["statistical_trials"]["seeds"] == [100, 101, 102, 103, 104]
        json.dumps(info["statistical_assessment"])
        expected_reward = reward_module.robust_constrained_reward(
            SimpleNamespace(**info["statistical_assessment"]),
            invalid=False,
            variable_cost=variable_cost,
        )[0]
        assert np.isclose(terminal_reward, expected_reward)
        assessments.append(info["statistical_assessment"])
    assert assessments[0] == assessments[1]


def test_valid_robust_prepared_terminal_rejects_missing_or_out_of_range_cost():
    env_module, reward_module, statistical_constraints = _runtime_modules()
    env = _runtime_env(env_module, reward_module, statistical_constraints)
    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    base = {
        "action_vec": np.asarray([3, 1, 4], dtype=int),
        "action_hash": "1234567890abcdef" * 4,
        "opt_signals": signals,
        "any_invalid": False,
        "info": {},
    }
    for cost in (None, -0.01, 1.01):
        prepared = dict(base)
        if cost is not None:
            prepared["external_cost_score"] = cost
        try:
            env._finish_prepared_terminal_probe(prepared, _runtime_metrics(reward_module))
        except ValueError as exc:
            assert "cost" in str(exc)
        else:
            raise AssertionError(f"prepared robust cost {cost!r} did not fail")


def test_normal_terminal_runtime_assesses_trials_and_threads_external_cost():
    env_module, reward_module, statistical_constraints = _runtime_modules()

    env = _runtime_env(env_module, reward_module, statistical_constraints)
    env.total_action_dim = 3
    env.env_cfg = SimpleNamespace(
        profile="mrpc", num_trials_per_step=5, persistent_probe_install=False,
    )
    env.max_sfs = object()
    env.rescale_bridge = SimpleNamespace(invoker=SimpleNamespace(baselines={}))
    env.gelu_degree = 4
    env.attn_degree = 6
    env.sync_degree_vectors_from_model = lambda: {}
    env.probe_runner = None
    env.bridge = SimpleNamespace(apply=lambda **_kwargs: None)
    env.clear_installed_blb = lambda: None
    env._installed_config_fingerprint = None
    env._eval_on_probe = lambda _k: _runtime_metrics(reward_module)
    env._maybe_borderline_retest = lambda metrics, _info: metrics
    env._last_probe_diagnostics = {}
    decoded = SimpleNamespace(
        block1_cfgs=[], block2_cfgs=[], block3_cfgs=[], block4_cfgs=[], block5_cfgs=[],
    )
    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    materialized = _runtime_materialized(decoded, signals)

    with mock.patch.object(env_module, "make_all_max_action_vector", return_value=np.asarray([9, 9, 9])), \
            mock.patch.object(env_module, "materialize_action_for_model", return_value=materialized), \
            mock.patch.object(env_module, "avg_truncation_k_in_action", return_value=13.0):
        _state, terminal_reward, done, info = env.step(
            np.asarray([3, 1, 4]), external_cost_score=0.6, external_cost_rank=0.6,
        )

    assert done is True
    assert terminal_reward > 1.6
    assert info["reward_breakdown"].variable_cost == 0.6
    assert info["statistical_assessment"]["precision_probability"] >= 0.5
    assert info["statistical_assessment"]["stability_probability"] >= 0.5


def test_env_terminal_dispatch_canonicalizes_only_known_robust_spelling():
    env_module, reward_module, statistical_constraints = _runtime_modules()
    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    for reward_design in (" ROBUST_CONSTRAINED ", "Robust_Constrained"):
        env = _runtime_env(env_module, reward_module, statistical_constraints)
        env.reward_weights.reward_design = reward_design
        breakdown = env._compute_terminal_reward(
            _runtime_metrics(reward_module), signals,
            action_vec=np.asarray([1, 2, 3]),
            action_vec_hash="canonical",
            any_invalid=False,
            external_cost_score=0.5,
            external_cost_rank=0.5,
            info={},
        )
        assert breakdown.constraint_policy == "bootstrap_5x5_v1"

    env = _runtime_env(env_module, reward_module, statistical_constraints)
    env.reward_weights.reward_design = "robust-constrained"
    breakdown = env._compute_terminal_reward(
        reward_module.EpisodeMetrics(), signals,
        action_vec=np.asarray([1, 2, 3]),
        action_vec_hash="unknown",
        any_invalid=False,
        external_cost_score=None,
        external_cost_rank=None,
        info={},
    )
    assert breakdown.constraint_policy == ""


def test_robust_invalid_terminal_needs_no_reference_or_fake_trials():
    env_module, reward_module, _statistical_constraints = _runtime_modules()

    env = _runtime_env(env_module, reward_module, _statistical_constraints)
    env.statistical_reference = None
    info = {}
    breakdown = env._compute_terminal_reward(
        reward_module.EpisodeMetrics(loss_mean=float("inf"), loss_std=float("inf")),
        SimpleNamespace(any_invalid=True),
        action_vec=np.asarray([1, 2, 3]),
        action_vec_hash="abcdef" * 10,
        any_invalid=True,
        external_cost_score=1.0,
        external_cost_rank=1.0,
        info=info,
    )
    assert breakdown.reward == -5.0
    assert breakdown.invalid is True
    assert "statistical_assessment" not in info


def test_valid_robust_terminal_fails_loudly_without_reference_or_two_trials():
    env_module, reward_module, statistical_constraints = _runtime_modules()
    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    env = _runtime_env(env_module, reward_module, statistical_constraints)
    env.statistical_reference = None
    try:
        env._compute_terminal_reward(
            _runtime_metrics(reward_module), signals,
            action_vec=np.asarray([1, 2, 3]),
            action_vec_hash="abc",
            any_invalid=False,
            external_cost_score=0.0,
            external_cost_rank=0.0,
            info={},
        )
    except RuntimeError as exc:
        assert "statistical_reference" in str(exc)
    else:
        raise AssertionError("missing robust reference did not fail")

    env = _runtime_env(env_module, reward_module, statistical_constraints)
    one_trial = reward_module.EpisodeMetrics(
        loss_trials=[0.95], metric1_trials=[0.85], metric2_trials=[0.75], trial_seeds=[1],
    )
    try:
        env._compute_terminal_reward(
            one_trial, signals,
            action_vec=np.asarray([1, 2, 3]),
            action_vec_hash="abc",
            any_invalid=False,
            external_cost_score=0.0,
            external_cost_rank=0.0,
            info={},
        )
    except ValueError as exc:
        assert "at least two trials" in str(exc)
    else:
        raise AssertionError("insufficient robust trial evidence did not fail")


def test_normal_terminal_eval_failure_is_minus_five_without_reference():
    env_module, reward_module, statistical_constraints = _runtime_modules()
    env = _runtime_env(env_module, reward_module, statistical_constraints)
    env.statistical_reference = None
    env.total_action_dim = 3
    env.env_cfg = SimpleNamespace(
        profile="mrpc", num_trials_per_step=5, persistent_probe_install=False,
    )
    env.max_sfs = object()
    env.rescale_bridge = SimpleNamespace(invoker=SimpleNamespace(baselines={}))
    env.gelu_degree = 4
    env.attn_degree = 6
    env.sync_degree_vectors_from_model = lambda: {}
    env.probe_runner = None
    env.bridge = SimpleNamespace(apply=lambda **_kwargs: None)
    env.clear_installed_blb = lambda: None
    env._installed_config_fingerprint = None
    env._eval_on_probe = lambda _k: (_ for _ in ()).throw(RuntimeError("probe failed"))
    decoded = SimpleNamespace(
        block1_cfgs=[], block2_cfgs=[], block3_cfgs=[], block4_cfgs=[], block5_cfgs=[],
    )
    signals = SimpleNamespace(any_invalid=False, total_bits_sum=100.0, total_fusion_count=0.0)
    materialized = _runtime_materialized(decoded, signals)
    with mock.patch.object(env_module, "make_all_max_action_vector", return_value=np.asarray([9, 9, 9])), \
            mock.patch.object(env_module, "materialize_action_for_model", return_value=materialized), \
            mock.patch.object(env_module, "avg_truncation_k_in_action", return_value=13.0):
        _state, terminal_reward, done, info = env.step(
            np.asarray([3, 1, 4]), external_cost_score=1.0,
        )
    assert done is True
    assert terminal_reward == -5.0
    assert info["eval_failed"] is True
    assert info["reward_breakdown"].invalid is True


if __name__ == "__main__":
    for _name, _test in sorted(globals().items()):
        if _name.startswith("test_") and callable(_test):
            _test()
