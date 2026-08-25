"""
gen_replan_actions.py
=====================

Generate per-config ``replan_actions_<name>.json`` templates from a
static-skeletons archive (``configs/static_skeletons.json``).

Each generated file contains:

  * ``t_new``           — initialized to the baseline t (sf_post / sf at source)
                          so that running replan with no edits is an identity.
  * ``delta_overrides`` — initialized to current per-mul propagation deltas.

You can then edit these files and feed them via:

    python scripts/replan_what_if.py \
        --config configs/<name>.json \
        --baseline-from configs/static_skeletons.json \
        --actions-file configs/replan_actions_<name>.json \
        --out configs/replan_<name>.json

Usage:
    python scripts/gen_replan_actions.py
    python scripts/gen_replan_actions.py --filter "block*"
    python scripts/gen_replan_actions.py --archive configs/static_skeletons.json --out-dir configs
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPO_ROOT / "configs" / "preparation" / "rescale"


def _baseline_t_from_entry(entry: Dict[str, Any]) -> List[int]:
    skel = list(entry.get("skeleton", []))
    sf_for_idx: Dict[int, int] = {}
    for row in entry.get("cut_point_sf", []):
        i = int(row["i"])
        if "sf_post" in row:
            sf_for_idx[i] = int(row["sf_post"])
        elif "sf" in row:
            sf_for_idx[i] = int(row["sf"])
    return [sf_for_idx[i] for i in skel if i in sf_for_idx]


def _baseline_delta_overrides(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in entry.get("propagation_deltas", []):
        out[str(row["name"])] = row["delta"]
    return out


def _format_actions_file(name: str, t_new: List[int],
                         delta_overrides: Dict[str, Any]) -> str:
    """Compact-but-readable JSON, matching the rest of the project style."""
    lines: List[str] = []
    lines.append("{")
    lines.append(f'  "config_name": {json.dumps(name)},')
    lines.append('  "notes": "Auto-generated from configs/static_skeletons.json. '
                 'Edit t_new / delta_overrides as needed.",')
    lines.append(f'  "t_new": {json.dumps(t_new, separators=(", ", ": "))},')
    lines.append('  "delta_overrides": {')
    last_idx = len(delta_overrides) - 1
    for k, (name_, val) in enumerate(delta_overrides.items()):
        v_str = ('"x2"' if val == "x2" else json.dumps(val))
        suf = "," if k < last_idx else ""
        lines.append(f'    {json.dumps(name_)}: {v_str}{suf}')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive",
                   default=str(CONFIG_ROOT / "static_skeletons.json"),
                   help="Path to static_skeletons.json")
    p.add_argument("--out-dir",
                   default=str(CONFIG_ROOT),
                   help="Where to write replan_actions_*.json")
    p.add_argument("--filter", default="block*",
                   help="Glob over config_name (default: block*)")
    p.add_argument("--include-failed", action="store_true",
                   help="Also emit templates for failed configs (skipped by default)")
    args = p.parse_args()

    archive_path = Path(args.archive).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    n_written = 0
    for entry in doc.get("results", []):
        name = entry.get("config_name", "")
        if not fnmatch.fnmatch(name, args.filter):
            continue
        if not entry.get("success", False):
            if not args.include_failed:
                print(f"[skip ] {name:<22} <not successful>")
                continue
            else:
                print(f"[warn ] {name:<22} <not successful — emitting empty template>")
                t_new: List[int] = []
                deltas: Dict[str, Any] = {}
        else:
            t_new = _baseline_t_from_entry(entry)
            deltas = _baseline_delta_overrides(entry)

        out_path = out_dir / f"replan_actions_{name}.json"
        out_path.write_text(_format_actions_file(name, t_new, deltas),
                            encoding="utf-8")
        try:
            rel = out_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = out_path
        print(f"[write] {name:<22} -> {rel}")
        n_written += 1

    try:
        rel_out = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        rel_out = out_dir
    print(f"\n[gen] wrote {n_written} actions file(s) under {rel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
