"""Real-evaluator adapter and persistence runner for Stage-1 search baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from json_utils import json_default, read_json_file, to_jsonable, write_json_file
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
    candidate_rank_key,
    normalize_search_backend,
    run_search,
)


ResultCheckpointCallback = Callable[[Mapping[str, Any]], None]
_REPLAY_SEMANTICS = (
    "observation_preload_replay_only; optimizer population, surrogate, RNG, "
    "and local-search position are not restored"
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _runtime_metrics(
        runtime_result: Any,
        constraints: Stage1Constraints,
        ) -> tuple[float, tuple[float, ...], Optional[float], Mapping[str, Any]]:
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
            on_evaluation: Optional[Callable[[Mapping[str, Any]], None]] = None,
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
    os.replace(temporary, target)
    return target


def _append_jsonl_row(path: str | Path, row: Mapping[str, Any]) -> int:
    """Append one normalized row and fsync its complete JSONL boundary."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
        return int(handle.tell())


def _relative_path(path: Path, owner: Path) -> str:
    try:
        return os.fspath(path.relative_to(owner.parent))
    except ValueError:
        return os.fspath(path)


def _observation_store(
        *,
        observation_path: Optional[Path],
        metadata_path: Optional[Path],
        observation_count: int,
        committed_size: Optional[int],
        ) -> Optional[dict[str, Any]]:
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
    payload["schema_version"] = "stage1_gelu_search_compact_result_v2"
    return payload


def _checkpoint_payload(
        *,
        backend: str,
        config: SearchConfig,
        observations: Sequence[SearchEvaluation],
        status: str,
        observation_store: Optional[Mapping[str, Any]],
        result: Optional[SearchResult] = None,
        best_evaluation: Optional[SearchEvaluation] = None,
        latest_evaluation: Optional[SearchEvaluation] = None,
        observation_count: Optional[int] = None,
        search_wall_seconds: Optional[float] = None,
        contract: Optional[Mapping[str, Any]] = None,
        ) -> dict[str, Any]:
    if best_evaluation is None and observations:
        best_evaluation = max(observations, key=candidate_rank_key)
    if latest_evaluation is None and observations:
        latest_evaluation = observations[-1]
    count = len(observations) if observation_count is None else int(observation_count)
    best = None if best_evaluation is None else best_evaluation.as_dict()
    latest = None if latest_evaluation is None else latest_evaluation.as_dict()
    return {
        "schema_version": "stage1_gelu_search_checkpoint_v2",
        "status": str(status),
        "backend": normalize_search_backend(backend),
        "config": config.as_dict(),
        "contract": dict(contract or {}),
        "observation_count": count,
        "observation_store": None if observation_store is None else dict(observation_store),
        "best": best,
        "latest": latest,
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


def save_search_checkpoint(
        path: str | Path,
        *,
        backend: str,
        config: SearchConfig,
        observations: Sequence[SearchEvaluation],
        status: str = "running",
        result: Optional[SearchResult] = None,
        observation_path: Optional[str | Path] = None,
        write_observations: bool = True,
        contract: Optional[Mapping[str, Any]] = None,
        ) -> Path:
    """Persist compact progress metadata referencing an external JSONL store."""

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
            contract=(
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
                    }
                )
            ),
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
            for item in read_jsonl(
                source,
                errors="raise",
                dict_only=True,
                missing_ok=False,
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
        rows = read_jsonl(
            observation_path,
            errors="raise",
            dict_only=True,
            missing_ok=False,
        )
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


def load_completed_search_result(output_dir: str | Path) -> SearchResult:
    output = Path(output_dir)
    payload = read_json_file(output / "result.json")
    manifest = read_json_file(output / "manifest.json")
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 completed result must be a JSON object")
    if not isinstance(manifest, Mapping):
        raise ValueError("Stage-1 completed manifest must be a JSON object")
    observations = load_search_preload(output / "observations.jsonl")
    history = read_json_file(output / "history.json", default=[])
    result = SearchResult.from_dict({
        **dict(payload),
        "observations": [item.as_dict() for item in observations],
        "history": list(history or []),
    })
    if result.algorithm != normalize_search_backend(manifest.get("backend")):
        raise RuntimeError(
            "Stage-1 completed backend does not match manifest"
        )
    if result.config.as_dict() != SearchConfig.from_dict(
            manifest.get("config") or {}
    ).as_dict():
        raise RuntimeError(
            "Stage-1 completed search config does not match manifest"
        )
    expected_count = int(manifest.get("evaluation_count", -1))
    expected_unique = int(manifest.get("unique_evaluation_count", -1))
    if result.evaluation_count != expected_count:
        raise RuntimeError(
            "Stage-1 completed observation count does not match manifest: "
            f"{result.evaluation_count} != {expected_count}"
        )
    if result.unique_evaluation_count != expected_unique:
        raise RuntimeError(
            "Stage-1 completed unique count does not match manifest"
        )
    if str(result.termination_reason) != str(manifest.get("termination_reason")):
        raise RuntimeError(
            "Stage-1 completed termination reason does not match manifest"
        )
    matching = [
        item for item in observations if item.action == result.best.action
    ]
    if (
            len(matching) != 1
            or _without_search_runtime_marker(matching[0])
            != _without_search_runtime_marker(result.best)
    ):
        raise RuntimeError(
            "Stage-1 completed best evaluation is absent or stale in JSONL"
        )
    return result


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


def persist_search_result(
        *,
        output_dir: str | Path,
        result: SearchResult,
        manifest: Optional[Mapping[str, Any]] = None,
        write_observations: bool = True,
        contract: Optional[Mapping[str, Any]] = None,
        ) -> dict[str, str]:
    """Persist compact summaries while observations remain in JSONL."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    _atomic_json(paths["manifest"], {
        **dict(manifest or {}),
        "schema_version": "stage1_gelu_search_manifest_v1",
        "status": "complete",
        "backend": result.algorithm,
        "formal_feasible": bool(result.best.feasible),
        "selection_status": (
            "feasible" if result.best.feasible else "least_violating"
        ),
        "scientific_export_allowed": bool(result.best.feasible),
        "evaluation_count": result.evaluation_count,
        "unique_evaluation_count": result.unique_evaluation_count,
        "termination_reason": result.termination_reason,
        "config": result.config.as_dict(),
        "observation_store": store,
        "resume_semantics": _REPLAY_SEMANTICS,
        "optimizer_state_restored": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
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
        contract=contract,
    )
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
            continue
        by_action[observation.action] = observation
        ordered.append(observation)
    return ordered


class Stage1SearchRunner:
    """Run and persist search with observation-replay-only preload semantics."""

    def __init__(
            self,
            *,
            adapter: Stage1EvaluatorAdapter,
            config: Optional[SearchConfig] = None,
            output_dir: Optional[str | Path] = None,
            manifest: Optional[Mapping[str, Any]] = None,
            checkpoint_callback: Optional[ResultCheckpointCallback] = None,
            checkpoint_interval: int = 50,
            ):
        self.adapter = adapter
        self.config = config or SearchConfig()
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.manifest = dict(manifest or {})
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

    def run(
            self,
            backend: str,
            *,
            surrogate_factory: Optional[SurrogateFactory] = None,
            preload: Iterable[SearchEvaluation] = (),
            preload_path: Optional[str | Path] = None,
            ) -> SearchResult:
        normalized = normalize_search_backend(backend)
        run_started_monotonic = time.perf_counter()
        prior_search_wall_seconds = 0.0
        contract = {
            "num_layers": self.adapter.space.num_layers,
            "gelu_degree_categories": list(GELU_DEGREES),
            "fixed_softmax_degree": int(FIXED_SOFTMAX_DEGREE),
            "constraints": self.adapter.constraints.as_dict(),
            "split": "validation_full",
            "manifest_identity": to_jsonable(
                self.manifest, preserve_native=True,
            ),
        }
        if (
                preload_path is None
                and self.output_dir is not None
                and (self.output_dir / "checkpoint.json").is_file()
        ):
            preload_path = self.output_dir / "checkpoint.json"
        preload_rows = list(preload)
        if preload_rows and normalized in ("bo_rf", "coinn_ga"):
            raise RuntimeError(
                f"partial {normalized} resume is disabled because exact "
                "surrogate/population state is not available; use a completed "
                "artifact or restart fresh"
            )
        if preload_path is not None:
            _validate_preload_contract(
                preload_path,
                backend=normalized,
                config=self.config,
                contract=contract,
            )
            preload_metadata = read_json_file(preload_path)
            if isinstance(preload_metadata, Mapping):
                prior_search_wall_seconds = float(
                    preload_metadata.get("search_wall_seconds", 0.0) or 0.0
                )
            if (
                    isinstance(preload_metadata, Mapping)
                    and str(preload_metadata.get("status")) == "complete"
            ):
                if self.output_dir is None:
                    raise RuntimeError(
                        "completed Stage-1 resume requires its output directory"
                    )
                return load_completed_search_result(self.output_dir)
            if normalized in ("bo_rf", "coinn_ga"):
                raise RuntimeError(
                    f"partial {normalized} resume is disabled because exact "
                    "surrogate/population state is not available; restart fresh"
                )
            preload_rows.extend(load_search_preload(preload_path))
        preload_rows = _deduplicate_preload(preload_rows)
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

        checkpoint_path: Optional[Path] = None
        observation_path: Optional[Path] = None
        committed_size: Optional[int] = None
        persisted_count = len(preload_rows)
        best_so_far = (
            None
            if not preload_rows
            else max(preload_rows, key=candidate_rank_key)
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.output_dir / "checkpoint.json"
            observation_path = self.output_dir / "observations.jsonl"
            if observation_path.exists() and observation_path.stat().st_size > 0:
                if not preload_rows:
                    raise RuntimeError(
                        "output already contains observations; provide its checkpoint "
                        "or JSONL as replay preload, or use a fresh directory"
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

        def progress_payload(
                *,
                observation_count: int,
                status: str,
                latest_evaluation: Optional[SearchEvaluation] = None,
                result: Optional[SearchResult] = None,
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
                search_wall_seconds=float(
                    prior_search_wall_seconds
                    + time.perf_counter() - run_started_monotonic
                ),
                contract=contract,
            )

        def publish(payload: Mapping[str, Any], *, notify: bool = True) -> None:
            if checkpoint_path is not None:
                _atomic_json(checkpoint_path, payload)
            if notify and self.checkpoint_callback is not None:
                self.checkpoint_callback(to_jsonable(payload, preserve_native=True))

        if preload_rows or checkpoint_path is not None:
            publish(progress_payload(
                observation_count=len(preload_rows),
                latest_evaluation=(None if not preload_rows else preload_rows[-1]),
                status="running",
            ))

        def on_observation(
                observation: SearchEvaluation,
                observation_count: int,
                ) -> None:
            nonlocal best_so_far, committed_size, persisted_count
            if int(observation_count) != persisted_count + 1:
                raise RuntimeError(
                    "Stage-1 observation callback count is not append-only"
                )
            if (
                    best_so_far is None
                    or candidate_rank_key(observation)
                    > candidate_rank_key(best_so_far)
            ):
                best_so_far = observation
            if observation_path is not None:
                observation_payload = observation.as_dict()
                observation_metadata = dict(
                    observation_payload.get("metadata") or {}
                )
                observation_metadata["search_cumulative_wall_seconds"] = float(
                    prior_search_wall_seconds
                    + time.perf_counter() - run_started_monotonic
                )
                observation_payload["metadata"] = observation_metadata
                committed_size = _append_jsonl_row(
                    observation_path,
                    observation_payload,
                )
            persisted_count = int(observation_count)
            publish_checkpoint = bool(
                observation_count == 1
                or observation_count % self.checkpoint_interval == 0
            )
            if publish_checkpoint:
                publish(progress_payload(
                    observation_count=observation_count,
                    latest_evaluation=observation,
                    status="running",
                ))

        result = run_search(
            normalized,
            self.adapter.space,
            self.adapter,
            self.config,
            surrogate_factory=surrogate_factory,
            preload=preload_rows,
            incremental_checkpoint_callback=on_observation,
        )
        if observation_path is not None and persisted_count != result.evaluation_count:
            raise RuntimeError("observation JSONL did not receive every unique evaluation")
        if self.output_dir is not None:
            persist_search_result(
                output_dir=self.output_dir,
                result=result,
                manifest={
                    **self.manifest,
                    "split": "validation_full",
                    "gelu_degree_categories": list(GELU_DEGREES),
                    "softmax_degrees": [FIXED_SOFTMAX_DEGREE] * self.adapter.space.num_layers,
                    "constraints": self.adapter.constraints.as_dict(),
                    "preloaded_observation_count": len(preload_rows),
                    "search_wall_seconds": float(
                        prior_search_wall_seconds
                        + time.perf_counter() - run_started_monotonic
                    ),
                    "model_inference_count": int(result.evaluation_count),
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
    "load_completed_search_result",
    "load_search_preload",
    "persist_search_result",
    "run_stage1_search",
    "save_search_checkpoint",
]
