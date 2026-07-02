"""Shared helpers for fusion-count fixed-action evaluation scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, ValuesView

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli_parse_utils import parse_json_int_list  # noqa: E402
from json_utils import stable_json_hash, stable_json_key  # noqa: E402


def iter_action_config_paths(action_dir: Path) -> Iterable[Path]:
    try:
        with os.scandir(action_dir) as entries:
            names = sorted(
                entry.name
                for entry in entries
                if entry.is_file()
                and entry.name.endswith(".json")
                and not entry.name.startswith(("._", "_"))
            )
    except OSError:
        names = []
    for name in names:
        yield action_dir / name


def load_paean_action_configs(action_dir: Path) -> List[dict]:
    configs = []
    for path in iter_action_config_paths(action_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        action = payload.get("action_vec")
        slots = payload.get("slots")
        base = payload.get("base")
        legacy = payload.get("legacy_action_vec")
        has_executable_slots = isinstance(slots, (list, dict))
        has_executable_vec = isinstance(action, list)
        if not (has_executable_slots or has_executable_vec):
            continue
        group = payload.get("group") or {}
        name = str(group.get("name") or path.stem)
        hash_payload = (
            {
                "slots": slots,
                "base": base,
                "legacy_action_vec": legacy,
            }
            if has_executable_slots
            else [int(v) for v in action]
        )
        configs.append({
            "name": name,
            "path": path,
            "action_hash": stable_json_hash(hash_payload),
            "group": group,
        })
    if not configs:
        raise RuntimeError(f"no action config JSON files found under {action_dir}")
    return configs


def rlpath_group_key(cfg: Mapping[str, Any]) -> str:
    group = cfg.get("group") or {}
    key_payload = {
        "option_by_graph": group.get("option_by_graph") or {},
        "option_by_step": group.get("option_by_step") or {},
        "baseline_k_index": cfg.get("baseline_k_index", 3),
    }
    return stable_json_key(key_payload)


def load_rlpath_action_configs(action_dir: Path) -> List[dict]:
    configs: List[dict] = []
    for path in iter_action_config_paths(action_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        group = payload.get("group") or {}
        name = str(group.get("name") or path.stem)
        cfg = {
            "name": name,
            "path": path,
            "group": group,
            "baseline_k_index": int(payload.get("baseline_k_index", 3)),
        }
        cfg["group_key"] = rlpath_group_key(cfg)
        configs.append(cfg)
    if not configs:
        raise RuntimeError(f"no action configs found under {action_dir}")
    return configs


def unique_configs_by_key(
        configs: Sequence[Mapping[str, Any]],
        *,
        key_name: str,
        fallback_key_fn,
        ) -> ValuesView[Mapping[str, Any]]:
    seen: Dict[str, Mapping[str, Any]] = {}
    for cfg in configs:
        key = cfg.get(key_name)
        if key is None:
            key = fallback_key_fn(cfg)
        seen.setdefault(str(key), cfg)
    return seen.values()
