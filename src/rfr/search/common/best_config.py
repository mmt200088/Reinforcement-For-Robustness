"""Validated JSON handoffs between production search stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rfr.common.json_utils import read_json_file, to_jsonable
from rfr.search.common.data_points import write_strict_json_file


STAGE1_BEST_CONFIG_SCHEMA = "stage1_best_config_v1"
SEARCH_BEST_CONFIG_SCHEMA = "search_best_config_v1"
STAGE1_BEST_CONFIG_FILENAME = "stage1_best_config.json"
SEARCH_BEST_CONFIG_FILENAME = "search_best_config.json"

SUPPORTED_ALGORITHMS = ("rl", "bo_rf", "greedy", "coinn_ga")
SUPPORTED_MODEL_TYPES = ("bert-base", "bert-large")
SUPPORTED_DATASETS = ("mrpc", "rte", "sst2")
ALLOWED_GELU_DEGREES = (0, 1, 2, 4)
ALLOWED_SOFTMAX_DEGREES = (2, 3, 4, 5, 6)
ALLOWED_FUSION_ACTIONS = (0, 1)
ALLOWED_PRECISION_ACTIONS = (0, 1, 2)

_COMMON_FIELDS = {
    "schema_version",
    "algorithm",
    "model_type",
    "dataset",
    "num_layers",
    "stage1",
    "selection",
    "provenance",
}


def normalize_algorithm(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {"ppo": "rl", "bo": "bo_rf", "coinn": "coinn_ga", "ga": "coinn_ga"}
    normalized = aliases.get(raw, raw)
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported search algorithm: {value!r}")
    return normalized


def expected_num_layers(model_type: Any) -> int:
    normalized = str(model_type or "").strip().lower().replace("_", "-")
    if normalized not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model type: {model_type!r}")
    return 12 if normalized == "bert-base" else 24


def profile_for(model_type: Any, dataset: Any) -> str:
    model = str(model_type or "").strip().lower().replace("_", "-")
    task = str(dataset or "").strip().lower()
    expected_num_layers(model)
    if task not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    return task if model == "bert-base" else f"{task}_large"


def stage1_best_config_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / STAGE1_BEST_CONFIG_FILENAME


def search_best_config_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / SEARCH_BEST_CONFIG_FILENAME


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _require_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} field set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer_vector(
        value: Any,
        *,
        name: str,
        length: int,
        allowed: Sequence[int],
        ) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be an integer")
        try:
            normalized = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be an integer") from exc
        if normalized != item or normalized not in allowed:
            raise ValueError(
                f"{name}[{index}]={item!r} is outside {tuple(allowed)}"
            )
        result.append(normalized)
    if len(result) != int(length):
        raise ValueError(f"{name} has {len(result)} values, expected {length}")
    return result


def _normalize_common(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    algorithm = normalize_algorithm(payload.get("algorithm"))
    model_type = str(payload.get("model_type") or "").strip().lower().replace("_", "-")
    dataset = str(payload.get("dataset") or "").strip().lower()
    layers = expected_num_layers(model_type)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    raw_layers = payload.get("num_layers")
    if isinstance(raw_layers, bool) or int(raw_layers) != layers:
        raise ValueError(
            f"num_layers={raw_layers!r} does not match {model_type} ({layers})"
        )
    stage1 = _require_mapping(payload.get("stage1"), "stage1")
    _require_fields(stage1, {"gelu", "softmax"}, "stage1")
    stage1 = {
        "gelu": _integer_vector(
            stage1["gelu"], name="stage1.gelu", length=layers,
            allowed=ALLOWED_GELU_DEGREES,
        ),
        "softmax": _integer_vector(
            stage1["softmax"], name="stage1.softmax", length=layers,
            allowed=ALLOWED_SOFTMAX_DEGREES,
        ),
    }
    selection = _require_mapping(payload.get("selection"), "selection")
    provenance = _require_mapping(payload.get("provenance"), "provenance")
    return {
        "schema_version": schema,
        "algorithm": algorithm,
        "model_type": model_type,
        "dataset": dataset,
        "num_layers": layers,
        "stage1": stage1,
        "selection": to_jsonable(selection, preserve_native=True),
        "provenance": to_jsonable(provenance, preserve_native=True),
    }


def validate_stage1_best_config(payload: Any) -> dict[str, Any]:
    data = _require_mapping(payload, "Stage-1 best config")
    _require_fields(data, _COMMON_FIELDS, "Stage-1 best config")
    if data.get("schema_version") != STAGE1_BEST_CONFIG_SCHEMA:
        raise ValueError("unsupported Stage-1 best config schema")
    return _normalize_common(data, schema=STAGE1_BEST_CONFIG_SCHEMA)


def load_stage1_best_config(path: str | Path) -> dict[str, Any]:
    return validate_stage1_best_config(read_json_file(path))


def write_stage1_best_config(
        run_dir: str | Path,
        *,
        algorithm: Any,
        model_type: Any,
        dataset: Any,
        gelu: Sequence[int],
        softmax: Sequence[int],
        selection: Mapping[str, Any],
        provenance: Mapping[str, Any],
        ) -> Path:
    layers = expected_num_layers(model_type)
    payload = validate_stage1_best_config({
        "schema_version": STAGE1_BEST_CONFIG_SCHEMA,
        "algorithm": normalize_algorithm(algorithm),
        "model_type": str(model_type),
        "dataset": str(dataset),
        "num_layers": layers,
        "stage1": {"gelu": list(gelu), "softmax": list(softmax)},
        "selection": dict(selection),
        "provenance": dict(provenance),
    })
    path = stage1_best_config_path(run_dir)
    write_strict_json_file(path, payload)
    return path


def _action_matrix(value: Any, *, layers: int) -> list[list[int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("stage2.action_matrix must be an array")
    rows = []
    for layer_idx, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"stage2.action_matrix[{layer_idx}] must be an array")
        if len(row) != 2:
            raise ValueError(
                f"stage2.action_matrix[{layer_idx}] must have two values"
            )
        fusion = _integer_vector(
            [row[0]], name=f"stage2.action_matrix[{layer_idx}].fusion",
            length=1, allowed=ALLOWED_FUSION_ACTIONS,
        )[0]
        precision = _integer_vector(
            [row[1]], name=f"stage2.action_matrix[{layer_idx}].precision",
            length=1, allowed=ALLOWED_PRECISION_ACTIONS,
        )[0]
        rows.append([fusion, precision])
    if len(rows) != int(layers):
        raise ValueError(
            f"stage2.action_matrix has {len(rows)} rows, expected {layers}"
        )
    return rows


def validate_search_best_config(payload: Any) -> dict[str, Any]:
    data = _require_mapping(payload, "search best config")
    expected = set(_COMMON_FIELDS) | {"stage2"}
    _require_fields(data, expected, "search best config")
    if data.get("schema_version") != SEARCH_BEST_CONFIG_SCHEMA:
        raise ValueError("unsupported search best config schema")
    normalized = _normalize_common(data, schema=SEARCH_BEST_CONFIG_SCHEMA)
    stage2 = _require_mapping(data.get("stage2"), "stage2")
    _require_fields(stage2, {"action_matrix"}, "stage2")
    normalized["stage2"] = {
        "action_matrix": _action_matrix(
            stage2["action_matrix"], layers=normalized["num_layers"],
        )
    }
    eligible = normalized["selection"].get("final_eval_eligible")
    if type(eligible) is not bool:
        raise ValueError("selection.final_eval_eligible must be boolean")
    strict_feasible = normalized["selection"].get("strict_feasible")
    if type(strict_feasible) is not bool:
        raise ValueError("selection.strict_feasible must be boolean")
    if eligible and not strict_feasible:
        raise ValueError(
            "final-eval eligibility requires a strict-feasible selection"
        )
    return normalized


def load_search_best_config(
        path: str | Path,
        *,
        require_final_eval_eligible: bool = False,
        ) -> dict[str, Any]:
    payload = validate_search_best_config(read_json_file(path))
    if require_final_eval_eligible and not payload["selection"]["final_eval_eligible"]:
        raise ValueError("selected search configuration is not final-eval eligible")
    return payload


def write_search_best_config(
        run_dir: str | Path,
        *,
        algorithm: Any,
        model_type: Any,
        dataset: Any,
        gelu: Sequence[int],
        softmax: Sequence[int],
        action_matrix: Sequence[Sequence[int]],
        selection: Mapping[str, Any],
        provenance: Mapping[str, Any],
        ) -> Path:
    layers = expected_num_layers(model_type)
    payload = validate_search_best_config({
        "schema_version": SEARCH_BEST_CONFIG_SCHEMA,
        "algorithm": normalize_algorithm(algorithm),
        "model_type": str(model_type),
        "dataset": str(dataset),
        "num_layers": layers,
        "stage1": {"gelu": list(gelu), "softmax": list(softmax)},
        "stage2": {"action_matrix": [list(row) for row in action_matrix]},
        "selection": dict(selection),
        "provenance": dict(provenance),
    })
    path = search_best_config_path(run_dir)
    write_strict_json_file(path, payload)
    return path


__all__ = [
    "SEARCH_BEST_CONFIG_FILENAME",
    "SEARCH_BEST_CONFIG_SCHEMA",
    "STAGE1_BEST_CONFIG_FILENAME",
    "STAGE1_BEST_CONFIG_SCHEMA",
    "load_search_best_config",
    "load_stage1_best_config",
    "normalize_algorithm",
    "profile_for",
    "search_best_config_path",
    "stage1_best_config_path",
    "validate_search_best_config",
    "validate_stage1_best_config",
    "write_search_best_config",
    "write_stage1_best_config",
]
