#!/usr/bin/env python3
"""Audit required Stage-2 fusion maps and enforce a fusion-count ceiling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence


def required_graph_keys(profile: str) -> List[str]:
    return [
        f"block2_{profile}",
        "block4",
        "block5_n1",
        "block5_n2",
        "block5_n4",
    ]


def _map_paths(profile_dir: Path) -> List[Path]:
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(profile_dir)
            if entry.is_file()
            and entry.name.startswith("block")
            and entry.name.endswith(".json")
        )
    except OSError:
        return []
    return [profile_dir / name for name in names]


def audit_profile_dir(profile_dir: Path | str, *, max_allowed: int = 1) -> Dict[str, Any]:
    root = Path(profile_dir)
    profile = root.name
    required = required_graph_keys(profile)
    present: set[str] = set()
    violations: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    graphs: List[Dict[str, Any]] = []

    for path in _map_paths(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            graph_key = str(payload["graph_key"])
            options = list(payload["options"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        present.add(graph_key)
        fusion_counts: List[int] = []
        for option in options:
            try:
                option_id = int(option["option_id"])
                fusion_count = int(option["fusion_count"])
                slots = dict(option.get("slots") or {})
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            fusion_counts.append(fusion_count)
            if fusion_count > int(max_allowed):
                violations.append({
                    "graph_key": graph_key,
                    "option_id": option_id,
                    "fusion_count": fusion_count,
                    "slots": slots,
                })

        graphs.append({
            "graph_key": graph_key,
            "path": str(path),
            "num_options": len(options),
            "fusion_counts": fusion_counts,
        })

    missing = [graph_key for graph_key in required if graph_key not in present]
    status = "pass" if not missing and not violations and not errors else "fail"
    return {
        "profile": profile,
        "profile_dir": str(root),
        "max_allowed": int(max_allowed),
        "status": status,
        "required_graph_keys": required,
        "present_graph_keys": sorted(present),
        "missing_graph_keys": missing,
        "violations": violations,
        "errors": errors,
        "graphs": graphs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", action="append", required=True)
    parser.add_argument("--max-allowed", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = [
        audit_profile_dir(path, max_allowed=int(args.max_allowed))
        for path in args.profile_dir
    ]
    status = "pass" if all(item["status"] == "pass" for item in profiles) else "fail"
    payload = {
        "status": status,
        "max_allowed": int(args.max_allowed),
        "profiles": profiles,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fusion map audit: {status} ({len(profiles)} profile(s)); report={output}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
