"""Real-evaluator adapter and persistence runner for Stage-1 search baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from json_utils import (
    json_default,
    read_json_file,
    to_jsonable,
    write_json_file,
)
from jsonl_utils import read_jsonl, recover_jsonl_file, write_jsonl_rows

from .search_baselines import (
    FIXED_SOFTMAX_DEGREE,
    GELU_DEGREES,
    SearchConfig,
    SearchEvaluation,
    SearchResult,
    Stage1Action,
    Stage1Constraints,
    Stage1SearchSpace,
    SurrogateFactory,
    _select_hamming_diverse_elites,
    candidate_rank_key,
    normalize_search_backend,
    run_search,
    structured_maximin_initial_design,
)


ResultCheckpointCallback = Callable[[Mapping[str, Any]], None]
_REPLAY_SEMANTICS = (
    "deterministic ordered observation replay reconstructs surrogate, "
    "population, RNG, and local-search state; no serialized optimizer state "
    "is restored"
)
_MANIFEST_SCHEMA = "stage1_gelu_search_manifest_v1"
_REQUIRED_COMPLETED_ARTIFACTS = (
    "observations.jsonl",
    "history.json",
    "result.json",
    "summary.json",
    "checkpoint.json",
    "manifest.json",
)
def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _runtime_metrics(
        runtime_result: Any,
        constraints: Stage1Constraints,
        ) -> tuple[float, tuple[float, ...], float | None, Mapping[str, Any]]:
    metric_count = len(constraints.metric_mins)
    if isinstance(runtime_result, Mapping):
        loss = float(runtime_result["loss"])
        metrics_value = runtime_result.get("metrics")
        if metrics_value is None:
            metrics = tuple(float(runtime_result[name]) for name in constraints.metric_names)
        else:
            metrics = tuple(float(value) for value in metrics_value)
        elapsed = runtime_result.get("time_ms", runtime_result.get("elapsed_ms"))
        metadata = dict(runtime_result.get("metadata") or {})
    else:
        values = tuple(runtime_result)
        if len(values) < 1 + metric_count:
            raise RuntimeError(
                "stage1_evaluate returned fewer values than the configured loss "
                "and metric channels"
            )
        loss = float(values[0])
        metrics = tuple(float(value) for value in values[1:1 + metric_count])
        elapsed = values[-1] if len(values) >= metric_count + 2 else None
        metadata = {}
    if len(metrics) != metric_count:
        raise RuntimeError(
            f"stage1_evaluate returned {len(metrics)} metrics; expected {metric_count}"
        )
    return loss, metrics, None if elapsed is None else float(elapsed), metadata


class Stage1EvaluatorAdapter:
    """Adapt the real evaluator while enforcing canonical Stage-1 semantics."""

    def __init__(
            self,
            *,
            evaluator: Any,
            num_layers: int,
            constraints: Stage1Constraints,
            on_evaluation: Callable[[Mapping[str, Any]], None] | None = None,
            ):
        self.evaluator = evaluator
        self.space = Stage1SearchSpace(int(num_layers))
        self.constraints = constraints
        self.on_evaluation = on_evaluation
        self.evaluation_count = 0
        self._cost_cache: dict[Stage1Action, tuple[float, list[float]]] = {}
        if not callable(getattr(evaluator, "stage1_evaluate", None)):
            raise TypeError("evaluator must provide stage1_evaluate")
        if not callable(getattr(evaluator, "get_simulated_cost", None)):
            raise TypeError("evaluator must provide get_simulated_cost")

    def _cost_details(self, action: Stage1Action) -> tuple[float, list[float]]:
        owned = self.space.validate(action)
        cached = self._cost_cache.get(owned)
        if cached is not None:
            return cached[0], list(cached[1])
        gelu_degrees = self.space.decode_gelu(owned)
        softmax_degrees = self.space.fixed_softmax()
        cost_result = self.evaluator.get_simulated_cost(gelu_degrees, softmax_degrees)
        if isinstance(cost_result, Sequence) and not isinstance(cost_result, (str, bytes)):
            if not cost_result:
                raise RuntimeError("get_simulated_cost returned an empty sequence")
            total_cost = float(cost_result[0])
            components = [float(value) for value in cost_result[1:]]
        else:
            total_cost = float(cost_result)
            components = []
        self._cost_cache[owned] = (total_cost, list(components))
        return total_cost, components

    def cost_for_action(self, action: Stage1Action) -> float:
        """Return exact simulated cost without running model inference."""

        return float(self._cost_details(action)[0])

    def __call__(self, action: Stage1Action) -> SearchEvaluation:
        owned = self.space.validate(action)
        gelu_degrees = self.space.decode_gelu(owned)
        softmax_degrees = self.space.fixed_softmax()
        evaluation_index = int(self.evaluation_count)
        started = time.perf_counter()
        runtime_result = self.evaluator.stage1_evaluate(
            gelu_degrees,
            softmax_degrees,
            use_train=False,
            split="validation_full",
        )
        loss, metrics, inference_time_ms, runtime_metadata = _runtime_metrics(
            runtime_result,
            self.constraints,
        )
        total_cost, cost_components = self._cost_details(owned)
        evaluation = SearchEvaluation(
            action=owned,
            loss=loss,
            metrics=metrics,
            cost=total_cost,
            constraints=self.constraints,
            valid=bool(_field(runtime_result, "valid", True)),
            metadata={
                **dict(runtime_metadata),
                "evaluation_index": evaluation_index,
                "split": "validation_full",
                "use_train": False,
                "wall_seconds": float(time.perf_counter() - started),
                "inference_time_ms": inference_time_ms,
                "cost_components": cost_components,
                "baseline_loss": float(self.constraints.baseline_loss),
                "baseline_metrics": list(self.constraints.baseline_metrics),
                "loss_limit": float(self.constraints.loss_max),
                "metric_limits": list(self.constraints.metric_mins),
                "fixed_softmax_degree": int(FIXED_SOFTMAX_DEGREE),
            },
        )
        self.evaluation_count += 1
        if self.on_evaluation is not None:
            self.on_evaluation(evaluation.as_dict())
        return evaluation


def _atomic_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    write_json_file(temporary, payload, sort_keys=True)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(
        os.fspath(target.parent),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _append_jsonl_row(path: str | Path, row: Mapping[str, Any]) -> int:
    """Append one normalized row and fsync its complete JSONL boundary."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    creates_directory_entry = not target.exists()
    encoded = json.dumps(
        to_jsonable(dict(row), preserve_native=True),
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
    ).encode("utf-8") + b"\n"
    with target.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        committed_size = int(handle.tell())
    if creates_directory_entry:
        directory_fd = os.open(
            os.fspath(target.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return committed_size


def _read_strict_object_jsonl(
        path: str | Path,
        *,
        gzip_fallback: bool = False,
        ) -> list[dict[str, Any]]:
    rows = read_jsonl(
        path,
        errors="raise",
        dict_only=False,
        missing_ok=False,
        gzip_fallback=gzip_fallback,
    )
    for line_no, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
    return [dict(row) for row in rows]


def _relative_path(path: Path, owner: Path) -> str:
    try:
        return os.fspath(path.relative_to(owner.parent))
    except ValueError:
        return os.fspath(path)


def _observation_store(
        *,
        observation_path: Path | None,
        metadata_path: Path | None,
        observation_count: int,
        committed_size: int | None,
        ) -> dict[str, Any] | None:
    if observation_path is None:
        return None
    return {
        "format": "jsonl",
        "path": (
            os.fspath(observation_path)
            if metadata_path is None
            else _relative_path(observation_path, metadata_path)
        ),
        "observation_count": int(observation_count),
        "committed_size": None if committed_size is None else int(committed_size),
        "append_fsync": True,
    }


def _compact_result(result: SearchResult) -> dict[str, Any]:
    payload = result.as_dict(include_observations=False, include_history=False)
    payload["schema_version"] = "stage1_gelu_search_compact_result_v3"
    return payload


def _checkpoint_payload(
        *,
        backend: str,
        config: SearchConfig,
        observations: Sequence[SearchEvaluation],
        status: str,
        observation_store: Mapping[str, Any] | None,
        result: SearchResult | None = None,
        best_evaluation: SearchEvaluation | None = None,
        latest_evaluation: SearchEvaluation | None = None,
        observation_count: int | None = None,
        search_wall_seconds: float | None = None,
        contract: Mapping[str, Any] | None = None,
        error: str | None = None,
        ) -> dict[str, Any]:
    if best_evaluation is None:
        if result is not None:
            best_evaluation = result.best
        elif observations:
            best_evaluation = max(observations, key=candidate_rank_key)
    if latest_evaluation is None and observations:
        latest_evaluation = observations[-1]
    count = len(observations) if observation_count is None else int(observation_count)
    payload = {
        "schema_version": "stage1_gelu_search_checkpoint_v2",
        "status": str(status),
        "backend": normalize_search_backend(backend),
        "config": config.as_dict(),
        "contract": dict(contract or {}),
        "observation_count": count,
        "observation_store": (
            None if observation_store is None else dict(observation_store)
        ),
        "best": (
            None if best_evaluation is None else best_evaluation.as_dict()
        ),
        "latest": (
            None if latest_evaluation is None else latest_evaluation.as_dict()
        ),
        "result": None if result is None else _compact_result(result),
        "resume_semantics": _REPLAY_SEMANTICS,
        "optimizer_state_restored": False,
        "search_wall_seconds": (
            None
            if search_wall_seconds is None
            else float(search_wall_seconds)
        ),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if error is not None:
        payload["error"] = str(error)
    return payload


def save_search_checkpoint(
        path: str | Path,
        *,
        backend: str,
        config: SearchConfig,
        observations: Sequence[SearchEvaluation],
        status: str = "running",
        result: SearchResult | None = None,
        observation_path: str | Path | None = None,
        write_observations: bool = True,
        contract: Mapping[str, Any] | None = None,
        search_wall_seconds: float | None = None,
        error: str | None = None,
        ) -> Path:
    """Persist ordinary progress metadata referencing the observation JSONL."""

    checkpoint_path = Path(path)
    store_path = (
        Path(observation_path)
        if observation_path is not None
        else checkpoint_path.with_name("observations.jsonl")
    )
    if write_observations:
        write_jsonl_rows(
            store_path,
            (item.as_dict() for item in observations),
            sort_keys=True,
        )
    committed_size = recover_jsonl_file(store_path)
    checkpoint_contract = (
        dict(contract)
        if contract is not None
        else (
            {}
            if not observations
            else {
                "num_layers": len(observations[0].action),
                "gelu_degree_categories": list(GELU_DEGREES),
                "fixed_softmax_degree": int(FIXED_SOFTMAX_DEGREE),
                "constraints": observations[0].constraints.as_dict(),
                "split": "validation_full",
                "use_train": False,
            }
        )
    )
    return _atomic_json(
        checkpoint_path,
        _checkpoint_payload(
            backend=backend,
            config=config,
            observations=observations,
            status=status,
            observation_store=_observation_store(
                observation_path=store_path,
                metadata_path=checkpoint_path,
                observation_count=len(observations),
                committed_size=committed_size,
            ),
            result=result,
            search_wall_seconds=search_wall_seconds,
            contract=checkpoint_contract,
            error=error,
        ),
    )


def _resolve_store(
        metadata_path: Path,
        payload: Mapping[str, Any],
        ) -> Optional[tuple[Path, Mapping[str, Any]]]:
    store = payload.get("observation_store")
    if not isinstance(store, Mapping) or not store.get("path"):
        return None
    observation_path = Path(os.fspath(store["path"]))
    if not observation_path.is_absolute():
        observation_path = metadata_path.parent / observation_path
    return observation_path, store


def load_search_preload(path: str | Path) -> tuple[SearchEvaluation, ...]:
    """Load cached observations for replay, not exact optimizer-state resume."""

    source = Path(path)
    if source.suffix in (".jsonl", ".gz"):
        return tuple(
            SearchEvaluation.from_dict(item)
            for item in _read_strict_object_jsonl(
                source,
                gzip_fallback=True,
            )
        )
    payload = read_json_file(source)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 search preload must be a JSON object")
    resolved = _resolve_store(source, payload)
    if resolved is not None:
        observation_path, store = resolved
        # The JSONL row is fsynced before checkpoint metadata is published.
        # A crash may therefore leave a valid append-only suffix beyond the last
        # checkpoint interval; recover the complete suffix instead of truncating
        # it to stale checkpoint metadata.
        recover_jsonl_file(observation_path)
        rows = _read_strict_object_jsonl(observation_path)
        expected = int(store.get("observation_count", payload.get("observation_count", len(rows))))
        if len(rows) < expected:
            raise RuntimeError(
                "Stage-1 checkpoint observation JSONL is shorter than metadata: "
                f"{len(rows)} < {expected}"
            )
        return tuple(SearchEvaluation.from_dict(item) for item in rows)

    # Backward compatibility with the initial all-in-memory schema.
    if payload.get("result") and payload["result"].get("observations") is not None:
        return SearchResult.from_dict(payload["result"]).observations
    if payload.get("schema_version") == "stage1_gelu_search_result_v1":
        return SearchResult.from_dict(payload).observations
    return tuple(
        SearchEvaluation.from_dict(item)
        for item in payload.get("observations", ())
    )


def _without_search_runtime_marker(
        evaluation: SearchEvaluation,
        ) -> dict[str, Any]:
    payload = evaluation.as_dict()
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("search_cumulative_wall_seconds", None)
    payload["metadata"] = metadata
    return payload


def _validate_greedy_neighborhood_proof(
        result: SearchResult,
        space: Stage1SearchSpace,
        expected_starts: int,
        ) -> None:
    observations = {item.action: item for item in result.observations}

    def observed_neighbors(
            actions: Iterable[Stage1Action],
            *,
            label: str,
            ) -> list[SearchEvaluation]:
        expected = tuple(actions)
        missing = [action for action in expected if action not in observations]
        if missing:
            raise RuntimeError(
                "Stage-1 Greedy completion contract neighborhood proof is "
                f"missing {label} observations"
            )
        return [observations[action] for action in expected]

    for start_index in range(expected_starts):
        rows = [
            row for row in result.history
            if (
                isinstance(row, Mapping)
                and row.get("start_index") is not None
                and int(row["start_index"]) == start_index
            )
        ]
        if not rows or rows[0].get("phase") != "start":
            raise RuntimeError(
                "Stage-1 Greedy completion contract neighborhood proof has "
                "no canonical start row"
            )
        expected_start = space.anchors[start_index]
        try:
            current = space.validate(rows[0].get("current_action") or ())
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Stage-1 Greedy completion contract neighborhood proof has "
                "an invalid start action"
            ) from exc
        if current != expected_start:
            raise RuntimeError(
                "Stage-1 Greedy completion contract neighborhood proof starts "
                "from the wrong anchor"
            )
        verified = False
        for row in rows[1:]:
            phase = str(row.get("phase"))
            try:
                recorded_current = space.validate(
                    row.get("current_action") or current
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Stage-1 Greedy completion contract neighborhood proof has "
                    "an invalid accepted action"
                ) from exc
            one_neighbors = observed_neighbors(
                space.one_opt_neighbors(current),
                label="1-opt",
            )
            one_best = max(
                one_neighbors,
                key=candidate_rank_key,
                default=observations[current],
            )
            if phase == "one_opt" and row.get("accepted") is True:
                if (
                        candidate_rank_key(one_best)
                        <= candidate_rank_key(observations[current])
                        or recorded_current != one_best.action
                ):
                    raise RuntimeError(
                        "Stage-1 Greedy completion contract neighborhood proof "
                        "does not justify its accepted 1-opt move"
                    )
                current = recorded_current
                continue
            if candidate_rank_key(one_best) > candidate_rank_key(
                    observations[current]
            ):
                raise RuntimeError(
                    "Stage-1 Greedy completion contract neighborhood proof "
                    "ignored an improving 1-opt move"
                )
            two_neighbors = observed_neighbors(
                space.two_opt_neighbors(current),
                label="2-opt",
            )
            two_best = max(
                two_neighbors,
                key=candidate_rank_key,
                default=observations[current],
            )
            if phase == "two_opt" and row.get("accepted") is True:
                if (
                        candidate_rank_key(two_best)
                        <= candidate_rank_key(observations[current])
                        or recorded_current != two_best.action
                ):
                    raise RuntimeError(
                        "Stage-1 Greedy completion contract neighborhood proof "
                        "does not justify its accepted 2-opt move"
                    )
                current = recorded_current
                continue
            if (
                    phase != "verified_local_optimum"
                    or row.get("one_opt_verified") is not True
                    or row.get("two_opt_verified") is not True
                    or recorded_current != current
                    or candidate_rank_key(two_best)
                    > candidate_rank_key(observations[current])
            ):
                raise RuntimeError(
                    "Stage-1 Greedy completion contract neighborhood proof "
                    "does not establish a local optimum"
                )
            if row is not rows[-1] or verified:
                raise RuntimeError(
                    "Stage-1 Greedy completion contract neighborhood proof has "
                    "rows after the verified local optimum"
                )
            verified = True
        if not verified:
            raise RuntimeError(
                "Stage-1 Greedy completion contract neighborhood proof is "
                "incomplete"
            )


def _validate_bo_history_proof(
        result: SearchResult,
        space: Stage1SearchSpace,
        ) -> None:
    if result.termination_reason not in {
            "no_improvement_convergence", "evaluation_cap",
            "candidate_space_exhausted",
    }:
        raise RuntimeError(
            "Stage-1 BO-RF completion contract has an invalid termination reason"
        )
    if (
            not result.history
            or result.history[0].get("phase")
            != "structured_maximin_initial_design"
            or int(result.history[0].get("iteration", -1)) != 0
            or any(
                row.get("phase") != "feasibility_aware_acquisition"
                for row in result.history[1:]
            )
    ):
        raise RuntimeError(
            "Stage-1 BO-RF completion contract has invalid history phases"
        )
    initial_count = min(
        int(result.config.bo_initial_design_size),
        int(result.config.evaluation_cap),
        int(space.cardinality),
    )
    initial_row = result.history[0]
    expected_initial = tuple(structured_maximin_initial_design(
        space,
        count=initial_count,
        seed=int(result.config.seed),
        maximin_candidate_pool_size=int(
            result.config.maximin_candidate_pool_size
        ),
    ))
    actual_initial = tuple(
        item.action for item in result.observations[:initial_count]
    )
    if (
            int(initial_row.get("evaluations", -1)) != initial_count
            or actual_initial != expected_initial
    ):
        raise RuntimeError(
            "Stage-1 BO-RF completion contract initial design does not match "
            "the observation journal"
        )

    incumbent_key = max(
        (candidate_rank_key(item) for item in result.observations[:initial_count]),
    )
    no_improvement = 0
    cumulative = initial_count
    for expected_iteration, row in enumerate(result.history[1:], start=1):
        if (
                int(row.get("iteration", -1)) != expected_iteration
                or int(row.get("evaluations", -1)) != cumulative + 1
                or cumulative >= result.evaluation_count
        ):
            raise RuntimeError(
                "Stage-1 BO-RF completion contract acquisitions are not "
                "contiguous"
            )
        observation = result.observations[cumulative]
        try:
            selected_action = space.validate(
                row.get("selected_action") or ()
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Stage-1 BO-RF completion contract has an invalid selected action"
            ) from exc
        if selected_action != observation.action:
            raise RuntimeError(
                "Stage-1 BO-RF completion contract selected action does not "
                "match the observation journal"
            )
        new_key = max(
            incumbent_key,
            candidate_rank_key(observation),
        )
        improved = new_key > incumbent_key
        no_improvement = 0 if improved else no_improvement + 1
        if (
                row.get("improved") is not improved
                or int(row.get("no_improvement", -1)) != no_improvement
        ):
            raise RuntimeError(
                "Stage-1 BO-RF completion contract has inconsistent incumbent "
                "progress"
            )
        incumbent_key = new_key
        cumulative += 1
    if cumulative != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 BO-RF completion contract leaves acquisitions unassigned"
        )
    if result.termination_reason == "evaluation_cap":
        if result.evaluation_count != min(
                int(result.config.evaluation_cap), int(space.cardinality)
        ):
            raise RuntimeError(
                "Stage-1 BO-RF completion contract did not reach its "
                "evaluation cap"
            )
    elif result.termination_reason == "candidate_space_exhausted":
        if result.evaluation_count != int(space.cardinality):
            raise RuntimeError(
                "Stage-1 BO-RF completion contract did not exhaust the "
                "candidate space"
            )
    elif (
            not result.history[1:]
            or no_improvement
            < int(result.config.bo_no_improvement_patience)
    ):
        raise RuntimeError(
            "Stage-1 BO-RF completion contract does not prove convergence"
        )


def _validate_ga_generation_proof(
        result: SearchResult,
        space: Stage1SearchSpace,
        ) -> None:
    if result.termination_reason not in {
            "completed_generations", "evaluation_cap",
            "candidate_space_exhausted", "ga_no_incumbent_improvement",
    }:
        raise RuntimeError(
            "Stage-1 GA completion contract has an invalid termination reason"
        )
    initial_rows = [
        row for row in result.history
        if (
            isinstance(row, Mapping)
            and row.get("phase") == "initial_population"
            and int(row.get("generation", -1)) == 0
        )
    ]
    update_rows = [
        row for row in result.history
        if isinstance(row, Mapping) and row.get("phase") == "elitist_update"
    ]
    if (
            len(initial_rows) != 1
            or not result.history
            or result.history[0] is not initial_rows[0]
            or len(result.history) != 1 + len(update_rows)
    ):
        raise RuntimeError(
            "Stage-1 GA completion contract has an invalid initial history"
        )
    population_size = min(
        int(result.config.ga_population_size),
        int(space.cardinality),
        int(result.config.evaluation_cap),
    )
    elite_count = min(
        int(result.config.ga_elite_count),
        max(0, population_size - 1),
    )
    initial = initial_rows[0]
    if (
            int(initial.get("evaluations", -1)) != population_size
            or int(initial.get("population_size", -1)) != population_size
    ):
        raise RuntimeError(
            "Stage-1 GA completion contract has the wrong initial population"
        )
    expected_initial_actions = tuple(structured_maximin_initial_design(
        space,
        count=population_size,
        seed=int(result.config.seed),
        maximin_candidate_pool_size=int(
            result.config.maximin_candidate_pool_size
        ),
    ))
    initial_actions = tuple(
        item.action for item in result.observations[:population_size]
    )
    if expected_initial_actions != initial_actions:
        raise RuntimeError(
            "Stage-1 GA completion contract initial design does not match "
            "the observation journal"
        )

    population = list(result.observations[:population_size])
    incumbent_key = candidate_rank_key(max(
        population, key=candidate_rank_key,
    ))
    no_improvement_generations = 0
    cumulative = population_size
    expected_generation = 1
    full_offspring_count = population_size - elite_count
    for row in update_rows:
        if (
                int(row.get("generation", -1)) != expected_generation
                or int(row.get("iteration", -1)) != expected_generation
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract generations are not contiguous"
            )
        expected_elites = _select_hamming_diverse_elites(
            space, population, elite_count,
        )
        try:
            recorded_elites = tuple(
                space.validate(action)
                for action in row.get("elite_actions", ())
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Stage-1 GA completion contract has invalid elite actions"
            ) from exc
        if (
                recorded_elites
                != tuple(item.action for item in expected_elites)
                or int(row.get("elite_count", -1)) != elite_count
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract does not prove exact elite "
                "retention"
            )
        next_cumulative = int(row.get("evaluations", -1))
        new_count = int(row.get("new_unique_evaluations", -1))
        if (
                next_cumulative <= cumulative
                or next_cumulative - cumulative != new_count
                or new_count != full_offspring_count
                or next_cumulative > result.evaluation_count
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract has invalid generation "
                "evaluation accounting"
            )
        offspring = list(result.observations[cumulative:next_cumulative])
        if len(offspring) != new_count:
            raise RuntimeError(
                "Stage-1 GA completion contract generation does not match "
                "the observation journal"
            )
        population = list(expected_elites) + offspring
        next_incumbent_key = candidate_rank_key(max(
            population, key=candidate_rank_key,
        ))
        improved = next_incumbent_key > incumbent_key
        no_improvement_generations = (
            0 if improved else no_improvement_generations + 1
        )
        recorded_no_improvement = row.get("no_improvement_generations")
        if (
                row.get("improved") is not improved
                or type(recorded_no_improvement) is not int
                or recorded_no_improvement != no_improvement_generations
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract has inconsistent incumbent "
                "stagnation evidence"
            )
        incumbent_key = next_incumbent_key
        cumulative = next_cumulative
        expected_generation += 1
    if cumulative != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 GA completion contract leaves unassigned observations"
        )

    completed_generations = expected_generation - 1
    configured_generations = int(result.config.ga_update_generations)
    if completed_generations > configured_generations:
        raise RuntimeError(
            "Stage-1 GA completion contract records more update rows than "
            "the configured generation count"
        )
    if result.termination_reason == "ga_no_incumbent_improvement":
        patience = int(result.config.ga_no_improvement_patience)
        if (
                completed_generations >= int(
                    result.config.ga_update_generations
                )
                or no_improvement_generations != patience
                or not update_rows
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract lacks a configured-patience "
                "incumbent stagnation proof"
            )
    elif result.termination_reason == "completed_generations":
        expected_count = (
            population_size
            + int(result.config.ga_update_generations)
            * full_offspring_count
        )
        if (
                completed_generations
                != int(result.config.ga_update_generations)
                or result.evaluation_count != expected_count
                or any(
                    int(row.get("new_unique_evaluations", -1))
                    != full_offspring_count
                    for row in update_rows
                )
        ):
            raise RuntimeError(
                "Stage-1 GA completion contract does not contain every "
                "configured generation"
            )
    elif result.termination_reason == "evaluation_cap":
        if completed_generations == int(result.config.ga_update_generations):
            raise RuntimeError(
                "Stage-1 GA completion contract reached every configured "
                "generation before claiming the evaluation cap"
            )
        effective_cap = min(
            int(result.config.evaluation_cap), int(space.cardinality),
        )
        unused_budget = effective_cap - result.evaluation_count
        if not 0 <= unused_budget < full_offspring_count:
            raise RuntimeError(
                "Stage-1 GA completion contract did not reach its evaluation cap "
                "boundary"
            )
    elif result.evaluation_count != int(space.cardinality):
        raise RuntimeError(
            "Stage-1 GA completion contract did not exhaust the candidate space"
        )


def _constraints_from_contract(
        contract: Mapping[str, Any] | None,
        ) -> Stage1Constraints:
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "Stage-1 completion contract must be a JSON object"
        )
    payload = contract.get("constraints")
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "Stage-1 completion contract constraints must be a JSON object"
        )
    try:
        return Stage1Constraints.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Stage-1 completion contract constraints are structurally invalid"
        ) from exc


def _validate_completed_search_contract(
        result: SearchResult,
        *,
        constraints: Stage1Constraints | None = None,
        comparator_smoke: bool = False,
        ) -> None:
    if not result.observations:
        raise RuntimeError(
            "Stage-1 completion contract has no observations"
        )
    expected_constraints = constraints or result.best.constraints
    if (
            result.best.constraints != expected_constraints
            or any(
                item.constraints != expected_constraints
                for item in result.observations
            )
    ):
        raise RuntimeError(
            "Stage-1 completion contract observation constraints do not match"
        )
    if result.unique_evaluation_count != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 completion contract contains duplicate observations"
        )
    valid = [item for item in result.observations if item.valid]
    if not valid:
        raise RuntimeError(
            "Stage-1 completion contract has no valid model-forward evaluation"
        )
    ranked_best = max(valid, key=candidate_rank_key)
    matching = [
        item for item in result.observations
        if item.action == result.best.action
    ]
    if (
            len(matching) != 1
            or _without_search_runtime_marker(matching[0])
            != _without_search_runtime_marker(result.best)
            or _without_search_runtime_marker(ranked_best)
            != _without_search_runtime_marker(result.best)
    ):
        raise RuntimeError(
            "Stage-1 completion contract best evaluation is absent, stale, "
            "or not best under the configured rank"
        )
    if comparator_smoke:
        if (
                int(result.config.evaluation_cap) != 1
                or result.evaluation_count != 1
                or result.termination_reason != "evaluation_cap"
        ):
            raise RuntimeError(
                "Stage-1 comparator smoke must terminate at its single real "
                "evaluation budget"
            )
        return
    if result.algorithm == "greedy":
        if result.termination_reason != "verified_local_optimum":
            raise RuntimeError(
                "Stage-1 Greedy completion contract did not terminate at a "
                "verified local optimum"
            )
        space = Stage1SearchSpace(len(result.best.action))
        expected_starts = min(
            int(result.config.greedy_max_starts),
            len(space.anchors),
        )
        verified_starts = {
            int(row["start_index"])
            for row in result.history
            if (
                isinstance(row, Mapping)
                and row.get("phase") == "verified_local_optimum"
                and row.get("one_opt_verified") is True
                and row.get("two_opt_verified") is True
                and row.get("start_index") is not None
            )
        }
        if verified_starts != set(range(expected_starts)):
            raise RuntimeError(
                "Stage-1 Greedy completion contract does not prove every "
                "configured start is a verified 1-opt/2-opt local optimum"
            )
        _validate_greedy_neighborhood_proof(
            result, space, expected_starts,
        )
    elif result.algorithm == "bo_rf":
        _validate_bo_history_proof(
            result, Stage1SearchSpace(len(result.best.action)),
        )
    elif result.algorithm == "coinn_ga":
        _validate_ga_generation_proof(
            result, Stage1SearchSpace(len(result.best.action)),
        )


def _search_result_from_artifacts(
        output: Path,
        payload: Mapping[str, Any],
        ) -> SearchResult:
    observations = load_search_preload(output / "observations.jsonl")
    history = read_json_file(output / "history.json", default=[])
    return SearchResult.from_dict({
        **dict(payload),
        "observations": [item.as_dict() for item in observations],
        "history": list(history or []),
    })


def _load_finalizing_search_result(
        output_dir: str | Path,
        *,
        backend: str,
        config: SearchConfig,
        checkpoint: Mapping[str, Any],
        comparator_smoke: bool = False,
        ) -> SearchResult:
    """Rebuild derived completion artifacts without running the evaluator."""

    output = Path(output_dir)
    for name in ("observations.jsonl", "history.json"):
        if not (output / name).is_file():
            raise RuntimeError(f"Stage-1 finalizing artifact missing: {name}")
    checkpoint_result = checkpoint.get("result")
    if not isinstance(checkpoint_result, Mapping):
        raise RuntimeError("Stage-1 finalizing checkpoint has no result")

    result_path = output / "result.json"
    if result_path.is_file():
        payload = read_json_file(result_path)
        if not isinstance(payload, Mapping):
            raise RuntimeError("Stage-1 finalizing result must be a JSON object")
    else:
        payload = dict(checkpoint_result)

    summary_path = output / "summary.json"
    if result_path.is_file() and summary_path.is_file():
        summary = read_json_file(summary_path)
        if not isinstance(summary, Mapping) or dict(summary) != dict(payload):
            raise RuntimeError(
                "Stage-1 finalizing summary does not match result"
            )

    result = _search_result_from_artifacts(output, payload)
    if dict(checkpoint_result) != _compact_result(result):
        raise RuntimeError(
            "Stage-1 finalizing checkpoint result does not match artifacts"
        )
    if result.algorithm != normalize_search_backend(backend):
        raise RuntimeError("Stage-1 finalizing backend does not match")
    if result.config.as_dict() != config.as_dict():
        raise RuntimeError(
            "Stage-1 finalizing search configuration does not match"
        )
    if int(checkpoint.get("observation_count", -1)) != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 finalizing observation count does not match checkpoint"
        )
    expected_best = checkpoint.get("best")
    if (
            not isinstance(expected_best, Mapping)
            or dict(expected_best) != result.best.as_dict()
    ):
        raise RuntimeError("Stage-1 finalizing best does not match checkpoint")
    expected_latest = checkpoint.get("latest")
    if (
            not isinstance(expected_latest, Mapping)
            or _without_search_runtime_marker(
                SearchEvaluation.from_dict(expected_latest)
            )
            != _without_search_runtime_marker(result.observations[-1])
    ):
        raise RuntimeError(
            "Stage-1 finalizing latest observation does not match checkpoint"
        )
    _validate_completed_search_contract(
        result,
        constraints=_constraints_from_contract(checkpoint.get("contract")),
        comparator_smoke=comparator_smoke,
    )
    return result


def load_completed_search_result(output_dir: str | Path) -> SearchResult:
    """Load and validate an ordinarily completed Stage-1 search."""

    output = Path(output_dir)
    missing = [
        name for name in _REQUIRED_COMPLETED_ARTIFACTS
        if not (output / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Stage-1 completed artifacts are missing: " + ", ".join(missing)
        )
    if not (output / "COMPLETED").is_file():
        raise RuntimeError("Stage-1 completed marker is missing")

    manifest = read_json_file(output / "manifest.json")
    checkpoint = read_json_file(output / "checkpoint.json")
    payload = read_json_file(output / "result.json")
    summary = read_json_file(output / "summary.json")
    for label, value in (
            ("manifest", manifest),
            ("checkpoint", checkpoint),
            ("result", payload),
            ("summary", summary),
    ):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Stage-1 completed {label} must be a JSON object")
    try:
        _ordinary_manifest_fields(manifest)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if str(manifest.get("schema_version")) != _MANIFEST_SCHEMA:
        raise RuntimeError("Stage-1 manifest schema version is unsupported")
    if str(manifest.get("status")) != "complete":
        raise RuntimeError("Stage-1 manifest status is not complete")
    if str(checkpoint.get("schema_version")) != (
            "stage1_gelu_search_checkpoint_v2"
    ):
        raise RuntimeError("Stage-1 checkpoint schema version is unsupported")
    if str(checkpoint.get("status")) != "complete":
        raise RuntimeError("Stage-1 checkpoint status is not complete")
    if dict(summary) != dict(payload):
        raise RuntimeError("Stage-1 completed summary does not match result")

    result = _search_result_from_artifacts(output, payload)
    normalized_backend = normalize_search_backend(manifest.get("backend"))
    if result.algorithm != normalized_backend:
        raise RuntimeError("Stage-1 completed backend does not match manifest")
    if normalize_search_backend(checkpoint.get("backend")) != result.algorithm:
        raise RuntimeError("Stage-1 completed backend does not match checkpoint")
    manifest_config = SearchConfig.from_dict(manifest.get("config") or {})
    checkpoint_config = SearchConfig.from_dict(checkpoint.get("config") or {})
    if (
            result.config.as_dict() != manifest_config.as_dict()
            or result.config.as_dict() != checkpoint_config.as_dict()
    ):
        raise RuntimeError(
            "Stage-1 completed search config does not match persisted metadata"
        )
    if int(manifest.get("evaluation_count", -1)) != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 completed observation count does not match manifest"
        )
    if int(checkpoint.get("observation_count", -1)) != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 completed observation count does not match checkpoint"
        )
    if int(manifest.get("unique_evaluation_count", -1)) != (
            result.unique_evaluation_count
    ):
        raise RuntimeError(
            "Stage-1 completed unique count does not match manifest"
        )
    if str(manifest.get("termination_reason")) != result.termination_reason:
        raise RuntimeError(
            "Stage-1 completed termination reason does not match manifest"
        )
    if checkpoint.get("result") != _compact_result(result):
        raise RuntimeError(
            "Stage-1 completed checkpoint result does not match artifacts"
        )
    if checkpoint.get("best") != result.best.as_dict():
        raise RuntimeError(
            "Stage-1 completed checkpoint best does not match artifacts"
        )
    checkpoint_latest = checkpoint.get("latest")
    if (
            not isinstance(checkpoint_latest, Mapping)
            or _without_search_runtime_marker(
                SearchEvaluation.from_dict(checkpoint_latest)
            )
            != _without_search_runtime_marker(result.observations[-1])
    ):
        raise RuntimeError(
            "Stage-1 completed checkpoint latest does not match observations"
        )
    _validate_completed_search_contract(
        result,
        constraints=_constraints_from_contract(checkpoint.get("contract")),
        comparator_smoke=manifest.get("comparator_smoke") is True,
    )
    return result


def build_stage1_search_accounting(
        *,
        result: SearchResult,
        manifest: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Return ordinary logical accounting for a completed Stage-1 search."""

    if int(manifest.get("evaluation_count", -1)) != result.evaluation_count:
        raise RuntimeError(
            "Stage-1 completed observation count is inconsistent"
        )
    if str(manifest.get("termination_reason")) != result.termination_reason:
        raise RuntimeError(
            "Stage-1 completed termination reason is inconsistent"
        )
    return {
        "observation_count": int(result.evaluation_count),
        "termination_reason": str(result.termination_reason),
        "search_wall_seconds": float(
            manifest.get("search_wall_seconds", 0.0)
        ),
    }


def _validate_preload_contract(
        path: str | Path,
        *,
        backend: str,
        config: SearchConfig,
        contract: Mapping[str, Any],
        ) -> None:
    source = Path(path)
    if source.suffix in (".jsonl", ".gz"):
        raise ValueError(
            "raw Stage-1 observation preload has no contract metadata; use "
            "checkpoint.json for production resume"
        )
    payload = read_json_file(source)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 resume metadata must be a JSON object")
    if normalize_search_backend(payload.get("backend")) != backend:
        raise RuntimeError("Stage-1 resume backend does not match")
    saved_config = SearchConfig.from_dict(payload.get("config") or {})
    if saved_config.as_dict() != config.as_dict():
        raise RuntimeError("Stage-1 resume search configuration does not match")
    saved_contract = dict(payload.get("contract") or {})
    for name, value in contract.items():
        if saved_contract.get(name) != value:
            raise RuntimeError(
                f"Stage-1 resume contract field {name!r} does not match"
            )


def _ordinary_manifest_fields(
        manifest: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
    return {
        str(name): value
        for name, value in dict(manifest or {}).items()
    }


def persist_search_result(
        *,
        output_dir: str | Path,
        result: SearchResult,
        manifest: Mapping[str, Any] | None = None,
        write_observations: bool = True,
        contract: Mapping[str, Any] | None = None,
        ) -> dict[str, str]:
    """Publish ordinary, recoverable Stage-1 search artifacts."""

    _validate_completed_search_contract(
        result,
        constraints=_constraints_from_contract(contract),
        comparator_smoke=(manifest or {}).get("comparator_smoke") is True,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed_marker = output / "COMPLETED"
    completed_marker.unlink(missing_ok=True)
    paths = {
        "manifest": output / "manifest.json",
        "observations": output / "observations.jsonl",
        "history": output / "history.json",
        "result": output / "result.json",
        "summary": output / "summary.json",
        "checkpoint": output / "checkpoint.json",
    }
    if write_observations:
        write_jsonl_rows(
            paths["observations"],
            (item.as_dict() for item in result.observations),
            sort_keys=True,
        )
    committed_size = recover_jsonl_file(paths["observations"])
    persisted_observations = load_search_preload(paths["observations"])
    if tuple(persisted_observations) != tuple(result.observations):
        raise RuntimeError(
            "Stage-1 completed observations do not match the search result"
        )
    store = _observation_store(
        observation_path=paths["observations"],
        metadata_path=paths["result"],
        observation_count=result.evaluation_count,
        committed_size=committed_size,
    )
    compact = {
        **_compact_result(result),
        "observation_store": store,
        "history_path": _relative_path(paths["history"], paths["result"]),
        "resume_semantics": _REPLAY_SEMANTICS,
        "optimizer_state_restored": False,
    }
    manifest_payload = _ordinary_manifest_fields(manifest)
    search_wall_seconds = float(
        manifest_payload.get("search_wall_seconds", 0.0) or 0.0
    )

    # The complete checkpoint records algorithm completion.  The manifest and
    # COMPLETED marker are published only after all derived artifacts exist.
    _atomic_json(paths["history"], list(result.history))
    _atomic_json(paths["result"], compact)
    _atomic_json(paths["summary"], compact)
    save_search_checkpoint(
        paths["checkpoint"],
        backend=result.algorithm,
        config=result.config,
        observations=result.observations,
        status="complete",
        result=result,
        observation_path=paths["observations"],
        write_observations=False,
        contract=dict(contract or {}),
        search_wall_seconds=search_wall_seconds,
    )
    _atomic_json(paths["manifest"], {
        **manifest_payload,
        "schema_version": _MANIFEST_SCHEMA,
        "status": "complete",
        "backend": result.algorithm,
        "feasible": bool(result.best.feasible),
        "selection_status": (
            "feasible" if result.best.feasible else "least_violating"
        ),
        "evaluation_count": result.evaluation_count,
        "unique_evaluation_count": result.unique_evaluation_count,
        "termination_reason": result.termination_reason,
        "config": result.config.as_dict(),
        "observation_store": store,
        "resume_semantics": _REPLAY_SEMANTICS,
        "optimizer_state_restored": False,
        "search_wall_seconds": search_wall_seconds,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _atomic_json(completed_marker, {"status": "complete"})
    return {name: str(path) for name, path in paths.items()}


def _deduplicate_preload(
        observations: Iterable[SearchEvaluation],
        ) -> list[SearchEvaluation]:
    ordered: list[SearchEvaluation] = []
    by_action: dict[Stage1Action, SearchEvaluation] = {}
    for observation in observations:
        previous = by_action.get(observation.action)
        if previous is not None:
            if previous.as_dict() != observation.as_dict():
                raise ValueError("conflicting preload evaluations for one action")
            raise ValueError("duplicate preload evaluation for one action")
        by_action[observation.action] = observation
        ordered.append(observation)
    return ordered


class Stage1SearchRunner:
    """Run and persist search with deterministic observation replay."""

    def __init__(
            self,
            *,
            adapter: Stage1EvaluatorAdapter,
            config: SearchConfig | None = None,
            output_dir: str | Path | None = None,
            manifest: Mapping[str, Any] | None = None,
            checkpoint_callback: ResultCheckpointCallback | None = None,
            checkpoint_interval: int = 50,
            ):
        self.adapter = adapter
        self.config = config or SearchConfig()
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.manifest = _ordinary_manifest_fields(manifest)
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

    def run(
            self,
            backend: str,
            *,
            surrogate_factory: SurrogateFactory | None = None,
            preload: Iterable[SearchEvaluation] = (),
            preload_path: str | Path | None = None,
            ) -> SearchResult:
        normalized = normalize_search_backend(backend)
        run_started_monotonic = time.perf_counter()
        prior_search_wall_seconds = 0.0
        preload_metadata: Mapping[str, Any] | None = None
        contract = {
            "num_layers": self.adapter.space.num_layers,
            "gelu_degree_categories": list(GELU_DEGREES),
            "fixed_softmax_degree": int(FIXED_SOFTMAX_DEGREE),
            "constraints": self.adapter.constraints.as_dict(),
            "split": "validation_full",
            "use_train": False,
        }
        if (
                preload_path is None
                and self.output_dir is not None
                and (self.output_dir / "checkpoint.json").is_file()
        ):
            preload_path = self.output_dir / "checkpoint.json"

        preload_rows = list(preload)
        if preload_path is not None:
            _validate_preload_contract(
                preload_path,
                backend=normalized,
                config=self.config,
                contract=contract,
            )
            loaded_metadata = read_json_file(preload_path)
            if isinstance(loaded_metadata, Mapping):
                preload_metadata = loaded_metadata
                prior_search_wall_seconds = float(
                    preload_metadata.get("search_wall_seconds", 0.0) or 0.0
                )
            if (
                    preload_metadata is not None
                    and str(preload_metadata.get("status")) == "complete"
            ):
                if self.output_dir is None:
                    raise RuntimeError(
                        "completed Stage-1 resume requires its output directory"
                    )
                if (
                        (self.output_dir / "manifest.json").is_file()
                        and (self.output_dir / "COMPLETED").is_file()
                ):
                    return load_completed_search_result(self.output_dir)
                recovered_result = _load_finalizing_search_result(
                    self.output_dir,
                    backend=normalized,
                    config=self.config,
                    checkpoint=preload_metadata,
                    comparator_smoke=(
                        self.manifest.get("comparator_smoke") is True
                    ),
                )
                persist_search_result(
                    output_dir=self.output_dir,
                    result=recovered_result,
                    manifest={
                        **self.manifest,
                        "search_wall_seconds": prior_search_wall_seconds,
                    },
                    write_observations=False,
                    contract=contract,
                )
                return load_completed_search_result(self.output_dir)
            preload_rows.extend(load_search_preload(preload_path))

        preload_rows = _deduplicate_preload(preload_rows)
        if any(
                item.constraints != self.adapter.constraints
                for item in preload_rows
        ):
            raise ValueError(
                "Stage-1 preload constraints do not match current constraints"
            )
        if preload_rows:
            prior_search_wall_seconds = max(
                prior_search_wall_seconds,
                max(
                    float(item.metadata.get(
                        "search_cumulative_wall_seconds", 0.0,
                    ) or 0.0)
                    for item in preload_rows
                ),
            )
        self.adapter.evaluation_count = len(preload_rows)

        checkpoint_path: Path | None = None
        observation_path: Path | None = None
        committed_size: int | None = None
        persisted_count = len(preload_rows)
        best_so_far = (
            None
            if not preload_rows
            else max(preload_rows, key=candidate_rank_key)
        )
        latest_observation = (
            None if not preload_rows else preload_rows[-1]
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.output_dir / "checkpoint.json"
            observation_path = self.output_dir / "observations.jsonl"
            if observation_path.exists() and observation_path.stat().st_size > 0:
                if not preload_rows:
                    raise RuntimeError(
                        "output already contains observations; provide its "
                        "checkpoint or JSONL as replay preload, or use a fresh "
                        "directory"
                    )
                committed_size = recover_jsonl_file(observation_path)
                existing_rows = load_search_preload(observation_path)
                if tuple(existing_rows) != tuple(preload_rows):
                    raise RuntimeError(
                        "Stage-1 resume observations do not match the output JSONL"
                    )
            else:
                write_jsonl_rows(
                    observation_path,
                    (item.as_dict() for item in preload_rows),
                    sort_keys=True,
                )
                committed_size = recover_jsonl_file(observation_path)

        def cumulative_wall_seconds() -> float:
            return float(
                prior_search_wall_seconds
                + time.perf_counter() - run_started_monotonic
            )

        def progress_payload(
                *,
                observation_count: int,
                status: str,
                latest_evaluation: SearchEvaluation | None = None,
                result: SearchResult | None = None,
                error: str | None = None,
                ) -> dict[str, Any]:
            return _checkpoint_payload(
                backend=normalized,
                config=self.config,
                observations=(),
                observation_count=int(observation_count),
                latest_evaluation=latest_evaluation,
                status=status,
                observation_store=_observation_store(
                    observation_path=observation_path,
                    metadata_path=checkpoint_path,
                    observation_count=int(observation_count),
                    committed_size=committed_size,
                ),
                result=result,
                best_evaluation=best_so_far,
                search_wall_seconds=cumulative_wall_seconds(),
                contract=contract,
                error=error,
            )

        def publish(payload: Mapping[str, Any], *, notify: bool = True) -> None:
            if checkpoint_path is not None:
                _atomic_json(checkpoint_path, payload)
            if notify and self.checkpoint_callback is not None:
                self.checkpoint_callback(
                    to_jsonable(payload, preserve_native=True)
                )

        if preload_rows or checkpoint_path is not None:
            publish(progress_payload(
                observation_count=len(preload_rows),
                latest_evaluation=latest_observation,
                status="running",
            ))

        def on_observation(
                observation: SearchEvaluation,
                observation_count: int,
                ) -> None:
            nonlocal best_so_far, committed_size, persisted_count
            nonlocal latest_observation
            if int(observation_count) != persisted_count + 1:
                raise RuntimeError(
                    "Stage-1 observation callback count is not append-only"
                )
            if observation_path is not None:
                observation.metadata[
                    "search_cumulative_wall_seconds"
                ] = cumulative_wall_seconds()
            latest_observation = observation
            if (
                    best_so_far is None
                    or candidate_rank_key(observation)
                    > candidate_rank_key(best_so_far)
            ):
                best_so_far = observation
            if observation_path is not None:
                committed_size = _append_jsonl_row(
                    observation_path,
                    observation.as_dict(),
                )
            persisted_count = int(observation_count)
            if (
                    observation_count == 1
                    or observation_count % self.checkpoint_interval == 0
            ):
                publish(progress_payload(
                    observation_count=observation_count,
                    latest_evaluation=observation,
                    status="running",
                ))

        try:
            result = run_search(
                normalized,
                self.adapter.space,
                self.adapter,
                self.config,
                surrogate_factory=surrogate_factory,
                preload=preload_rows,
                incremental_checkpoint_callback=on_observation,
                replay_greedy_preload_in_order=True,
            )
            if (
                    observation_path is not None
                    and persisted_count != result.evaluation_count
            ):
                raise RuntimeError(
                    "observation JSONL did not receive every unique evaluation"
                )
            _validate_completed_search_contract(
                result,
                constraints=self.adapter.constraints,
                comparator_smoke=(
                    self.manifest.get("comparator_smoke") is True
                ),
            )
        except Exception as exc:
            publish(progress_payload(
                observation_count=int(persisted_count),
                latest_evaluation=latest_observation,
                status="failed",
                error=repr(exc),
            ))
            raise

        if self.output_dir is not None:
            persist_search_result(
                output_dir=self.output_dir,
                result=result,
                manifest={
                    **self.manifest,
                    "split": "validation_full",
                    "gelu_degree_categories": list(GELU_DEGREES),
                    "softmax_degrees": (
                        [FIXED_SOFTMAX_DEGREE]
                        * self.adapter.space.num_layers
                    ),
                    "constraints": self.adapter.constraints.as_dict(),
                    "preloaded_observation_count": len(preload_rows),
                    "search_wall_seconds": cumulative_wall_seconds(),
                },
                write_observations=False,
                contract=contract,
            )
        elif self.checkpoint_callback is not None:
            self.checkpoint_callback(to_jsonable(
                progress_payload(
                    observation_count=result.evaluation_count,
                    latest_evaluation=result.observations[-1],
                    status="complete",
                    result=result,
                ),
                preserve_native=True,
            ))
        return result


def run_stage1_search(
        *,
        backend: str,
        evaluator: Any,
        num_layers: int,
        constraints: Stage1Constraints,
        config: Optional[SearchConfig] = None,
        output_dir: Optional[str | Path] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        surrogate_factory: Optional[SurrogateFactory] = None,
        preload: Iterable[SearchEvaluation] = (),
        preload_path: Optional[str | Path] = None,
        checkpoint_callback: Optional[ResultCheckpointCallback] = None,
        checkpoint_interval: int = 50,
        ) -> SearchResult:
    runner = Stage1SearchRunner(
        adapter=Stage1EvaluatorAdapter(
            evaluator=evaluator,
            num_layers=int(num_layers),
            constraints=constraints,
        ),
        config=config,
        output_dir=output_dir,
        manifest=manifest,
        checkpoint_callback=checkpoint_callback,
        checkpoint_interval=int(checkpoint_interval),
    )
    return runner.run(
        backend,
        surrogate_factory=surrogate_factory,
        preload=preload,
        preload_path=preload_path,
    )


__all__ = [
    "ResultCheckpointCallback",
    "Stage1EvaluatorAdapter",
    "Stage1SearchRunner",
    "build_stage1_search_accounting",
    "load_completed_search_result",
    "load_search_preload",
    "persist_search_result",
    "run_stage1_search",
    "save_search_checkpoint",
]
