"""Build a BLB Stage-2 Trust-0 run manifest."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.action_space import action_dims_for_config, per_layer_field_offsets  # noqa: E402


def _run_git(args: Sequence[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_hash_from_file(h: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)


def _dir_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    h = hashlib.sha256()
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for file_path in _iter_sorted_tree_paths([path]):
        if not file_path.is_file() or any(part in skip_dirs for part in file_path.parts):
            continue
        rel = file_path.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        file_hash = _file_sha256(file_path)
        if file_hash:
            h.update(file_hash.encode("ascii"))
    return h.hexdigest()


def _path_hash(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = (REPO_ROOT / path_text).resolve() if not os.path.isabs(path_text) else Path(path_text)
    if path.is_file():
        return _file_sha256(path)
    if path.is_dir():
        return _dir_sha256(path)
    return None


def _resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _iter_sorted_tree_paths(paths: Iterable[Path]) -> Iterable[Path]:
    heap: list[tuple[str, Path]] = []
    for path in paths:
        heapq.heappush(heap, (path.as_posix(), path))
    while heap:
        _key, path = heapq.heappop(heap)
        if path.is_dir():
            children = list(path.iterdir())
            children.sort(key=lambda child: child.as_posix())
            for child in children:
                heapq.heappush(heap, (child.as_posix(), child))
        else:
            yield path


def _canonical_rescale_optimizer_hash(root_text: str | None, profile: str) -> str | None:
    root = _resolve_path(root_text)
    if root is None or not root.exists():
        return None
    relevant_roots = [
        root / "rescale_optimizer",
        root / "configs" / str(profile),
        root / "replan_configs" / str(profile),
    ]
    saw_files = False
    h = hashlib.sha256()
    for file_path in _iter_sorted_tree_paths(relevant_roots):
        if not file_path.is_file() or file_path.suffix not in {".py", ".json"}:
            continue
        saw_files = True
        try:
            rel = file_path.relative_to(root).as_posix()
        except ValueError:
            rel = file_path.as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        _update_hash_from_file(h, file_path)
        h.update(b"\0")
    if not saw_files:
        return _path_hash(root_text)
    return h.hexdigest()


def _load_stage1_config(
        path_text: str | None,
        source_text: str | None,
        *,
        model: str,
        profile: str,
        ) -> Dict[str, Any]:
    path = _resolve_path(path_text) or _resolve_path(source_text)
    if path is None or not path.is_file():
        return {
            "source": source_text or "",
            "config_path": "" if path is None else str(path),
            "config_content_hash": "",
            "gelu_degrees": None,
            "softmax_degrees": None,
        }
    payload: Any = None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    candidates = payload
    if isinstance(payload, dict):
        model_node = payload.get(str(model))
        if isinstance(model_node, dict) and isinstance(model_node.get(str(profile)), dict):
            candidates = model_node[str(profile)].get("stage1", model_node[str(profile)])
        elif isinstance(payload.get(str(profile)), dict):
            candidates = payload[str(profile)].get("stage1", payload[str(profile)])
        for key in ("stage1", "stage1_search_best", "best_stage1"):
            if isinstance(payload.get(key), dict):
                candidates = payload[key]
                break
    gelu = None
    softmax = None
    if isinstance(candidates, dict):
        gelu = candidates.get("gelu", candidates.get("gelu_degrees"))
        softmax = candidates.get("softmax", candidates.get("softmax_degrees"))
    return {
        "source": source_text or str(path),
        "config_path": str(path),
        "config_content_hash": _file_sha256(path) or "",
        "gelu_degrees": gelu,
        "softmax_degrees": softmax,
    }


def _load_registry_hash(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _file_sha256(path)
    if isinstance(payload, dict) and payload.get("registry_hash"):
        return str(payload["registry_hash"])
    return _sha256_bytes(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))


def _git_diff_hash() -> str:
    diff = _run_git(["diff", "--no-ext-diff", "--binary"]) or ""
    return _sha256_bytes(diff.encode("utf-8", errors="replace"))


def _block_slot_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for block_idx, _field, _kind in per_layer_field_offsets():
        key = f"block{int(block_idx)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    status = _run_git(["status", "--short", "--branch"]) or ""
    registry_hash = _load_registry_hash(args.registry_path)
    max_sfs_hash = _path_hash(args.max_sfs_path)
    rescale_full_tree_hash = _path_hash(args.rescale_optimizer_root)
    rescale_hash = _canonical_rescale_optimizer_hash(args.rescale_optimizer_root, args.profile)
    stage1 = _load_stage1_config(
        args.stage1_config_path,
        args.stage1_source,
        model=args.model,
        profile=args.profile,
    )
    missing = []
    for name, value in (
            ("registry_hash", registry_hash),
            ("max_sfs_hash", max_sfs_hash),
            ("rescale_optimizer_hash", rescale_hash),
            ("stage1_config_content_hash", stage1.get("config_content_hash")),
            ("threshold_source", args.threshold_source),
    ):
        if value in (None, "") or str(value).strip().lower() in ("unknown", "none", "null"):
            missing.append(name)
    return {
        "schema": "blb_trust0_run_manifest_v1",
        "git": {
            "head": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "upstream": _run_git(["rev-parse", "@{u}"]),
            "diff_hash": _git_diff_hash(),
            "dirty": bool(status.strip()),
            "status_short": status,
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "data": {
            "dataset": args.dataset,
            "model": args.model,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
            "glue_local_dataset_dir": os.environ.get("GLUE_LOCAL_DATASET_DIR"),
        },
        "rescale_optimizer": {
            "mode": args.rescale_optimizer_mode,
            "root": args.rescale_optimizer_root,
            "profile": args.profile,
            "hash": rescale_hash,
            "canonical_hash": rescale_hash,
            "full_tree_hash": rescale_full_tree_hash,
        },
        "action_space": {
            "version": args.action_space_version,
            "registry_path": args.registry_path,
            "registry_hash": registry_hash,
            "slot_counts": _block_slot_counts(),
            "per_layer_slot_count": len(per_layer_field_offsets()),
            "full_action_length": len(action_dims_for_config(int(args.num_layers))),
            "decode_version": args.decode_version,
        },
        "max_sfs": {
            "path": args.max_sfs_path,
            "hash": max_sfs_hash,
        },
        "stage1": {
            "source": stage1.get("source", ""),
            "config_path": stage1.get("config_path", ""),
            "hash": stage1.get("config_content_hash", ""),
            "config_content_hash": stage1.get("config_content_hash", ""),
            "gelu_degrees": stage1.get("gelu_degrees"),
            "softmax_degrees": stage1.get("softmax_degrees"),
        },
        "thresholds": {
            "source": args.threshold_source,
            "acc_limit": args.acc_limit,
            "f1_limit": args.f1_limit,
            "acc_std_limit": args.acc_std_limit,
            "f1_std_limit": args.f1_std_limit,
            "strict_z": args.strict_z,
            "strict_z_source": "default" if float(args.strict_z) == 1.0 else "explicit",
        },
        "cost_policy": {
            "optimizer_cost_terms": ["total_bits_sum", "fusion_count"],
            "optimizer_validity_terms": ["invalid_chain", "optimizer_valid", "any_invalid"],
            "optimizer_diagnostic_terms": ["q_bits", "q_head_bits", "q_tail_bits"],
            "mpc_truncation_cost_enabled": bool(args.mpc_truncation_cost_enabled),
            "mpc_truncation_term": "avg_k" if args.mpc_truncation_cost_enabled else None,
        },
        "fidelity_policy": {
            "F0": "optimizer-only Rescale_optimizer (no model forward)",
            "F1": "small probe + few MC trials during training",
            "F4": "formal final eval with real BLB install on validation_full and repeated evaluation",
        },
        "missing_or_todo": missing,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "run_manifest.json"
    md_path = out / "run_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# BLB Trust-0 Run Manifest",
        "",
        f"- git HEAD: `{manifest['git']['head']}`",
        f"- diff hash: `{manifest['git']['diff_hash']}`",
        f"- dirty: `{manifest['git']['dirty']}`",
        f"- registry hash: `{manifest['action_space']['registry_hash']}`",
        f"- max_sfs hash: `{manifest['max_sfs']['hash']}`",
        f"- Rescale_optimizer mode/root/hash: `{manifest['rescale_optimizer']['mode']}` / `{manifest['rescale_optimizer']['root']}` / `{manifest['rescale_optimizer']['hash']}`",
        f"- full action length: `{manifest['action_space']['full_action_length']}`",
        f"- strict_z: `{manifest['thresholds']['strict_z']}`",
        "",
        "## Cost Policy",
        "",
        "- Rescale optimizer final cost terms: `total_bits_sum`, `fusion_count`",
        "- `invalid_chain` is a validity gate, not numeric cost.",
        "- `q_bits`, `q_head_bits`, `q_tail_bits` are debug-only.",
        "",
        "## Missing / TODO",
        "",
    ]
    if manifest["missing_or_todo"]:
        lines.extend(f"- {item}" for item in manifest["missing_or_todo"])
    else:
        lines.append("- none")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/blb_opt/trust0_manifest")
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--dataset", default="mrpc")
    parser.add_argument("--model", default="bert-base")
    parser.add_argument("--action-space-version", default="current-code-v1")
    parser.add_argument("--decode-version", default="action_space_v1")
    parser.add_argument("--registry-path", default="reports/blb_opt/trust0_registry/current_code_action_registry.json")
    parser.add_argument("--max-sfs-path", default="blb_stage2_rl/max_sfs/mrpc.json")
    parser.add_argument("--stage1-source", default="")
    parser.add_argument("--stage1-config-path", default="")
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument("--rescale-optimizer-mode", default="in_process_real")
    parser.add_argument("--threshold-source", default="unknown")
    parser.add_argument("--acc-limit", type=float, default=None)
    parser.add_argument("--f1-limit", type=float, default=None)
    parser.add_argument("--acc-std-limit", type=float, default=None)
    parser.add_argument("--f1-std-limit", type=float, default=None)
    parser.add_argument("--strict-z", type=float, default=1.0)
    parser.add_argument("--mpc-truncation-cost-enabled", action="store_true")
    args = parser.parse_args(argv)
    paths = write_manifest(build_manifest(args), args.output_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
