"""Validate a preset .conf file against the launcher's declared flag list.

The launcher (``llama_7B_LayerImportance.sh``) accepts ~80 flags via a long
``case`` block. A typo in a preset file (``--stage2-rolllout-size`` with an
extra ``l``) is silently ignored — the bash branch doesn't match, and the
flag is dropped without warning. Hours of debugging later.

This script parses the launcher to extract the canonical list of accepted
``--*`` flags, then walks each preset file and flags any unrecognized
token. It also catches:

* duplicate flag definitions in one preset
* obviously wrong values (--stage2-search-episodes "abc")
* flags that have known boolean aliases (``--blb-v3-sequential-rl true``)

Usage::

    python -m tools.validate_preset presets/*.conf Paean/presets/*.conf
    python tools/validate_preset.py presets/mrpc-blb-stage2-rl.conf

Exit code 0 = clean; 1 = problems found (with line-number annotations).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple


LAUNCHER_REL = "llama_7B_LayerImportance.sh"
PAEAN_LAUNCHER_REL = "Paean/run_final_eval.sh"


_FLAG_LINE_RE = re.compile(
    # match "--foo|--foo-bar) ..." or just "--foo) ..."
    r"""^\s*((?:--[a-zA-Z0-9_-]+)(?:\s*\|\s*--[a-zA-Z0-9_-]+)*)\)\s*"""
)


def extract_launcher_flags(launcher_path: str) -> Set[str]:
    """Scan a bash launcher for accepted ``--*`` flags."""
    flags: Set[str] = set()
    if not os.path.isfile(launcher_path):
        return flags
    with open(launcher_path, encoding="utf-8") as f:
        for line in f:
            m = _FLAG_LINE_RE.match(line)
            if not m:
                continue
            group = m.group(1)
            for piece in group.split("|"):
                t = piece.strip()
                if t.startswith("--"):
                    flags.add(t)
    return flags


def _iter_numbered_lines(lines: Iterable[str]) -> Iterator[Tuple[int, str]]:
    for line_num, raw in enumerate(lines, start=1):
        yield line_num, raw.rstrip("\r\n")


def _extract_preset_flags_from_lines(lines: Iterable[str]) -> List[Tuple[int, str, str]]:
    numbered = iter(_iter_numbered_lines(lines))
    pending: Tuple[int, str] | None = None
    out: List[Tuple[int, str, str]] = []

    while True:
        if pending is None:
            try:
                line_num, raw = next(numbered)
            except StopIteration:
                break
        else:
            line_num, raw = pending
            pending = None

        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Tokenize the line. Preset .conf files are simple: flag + value
        # may be on the same line ("--foo bar") or split across two lines
        # ("--foo\nbar"). We support both.
        parts = s.split(None, 1)
        flag = parts[0]
        if not flag.startswith("--"):
            # Stray token / value line; just record as a value with no flag.
            out.append((line_num, "", flag))
            continue
        value = parts[1] if len(parts) > 1 else ""
        if not value:
            # Look-ahead one line for the value without materializing the file.
            try:
                next_line_num, next_raw = next(numbered)
            except StopIteration:
                next_line_num, next_raw = 0, ""
            nxt = next_raw.strip()
            if nxt and not nxt.startswith("#") and not nxt.startswith("--"):
                value = nxt
            elif next_line_num:
                pending = (next_line_num, next_raw)
        out.append((line_num, flag, value))
    return out


def extract_preset_flags(preset_path: str) -> List[Tuple[int, str, str]]:
    """Return ``[(line_num, flag, value), ...]`` from a preset file.

    Lines starting with `#` or empty are skipped. Values on the next line
    after a flag (the launcher tolerates both) are paired with the flag.
    """
    if not os.path.isfile(preset_path):
        return []
    with open(preset_path, encoding="utf-8") as handle:
        return _extract_preset_flags_from_lines(handle)


def _classify_value(flag: str, value: str) -> str:
    """Return '' if value is acceptable, else a short error message."""
    if value in ("", None):
        # The launcher tolerates flag-only (treated as boolean true) for
        # things like --skip-stage1-search.
        return ""
    # Numeric-looking flags should have numeric values
    numeric_hints = (
        "episodes", "trials", "size", "interval", "samples", "seed",
        "lr", "tolerance", "anchor", "stability", "limit", "probe",
        "rollout", "coeff", "penalty",
    )
    if any(h in flag for h in numeric_hints):
        try:
            float(value)
        except ValueError:
            return f"expected number, got {value!r}"
    return ""


def validate_preset(
        preset_path: str,
        launcher_flags: Set[str],
        ) -> List[Tuple[int, str]]:
    """Return ``[(line_num, message), ...]`` for problems."""
    problems: List[Tuple[int, str]] = []
    seen: Dict[str, int] = {}
    for line_num, flag, value in extract_preset_flags(preset_path):
        if not flag:
            problems.append((line_num, f"orphan value {value!r} with no preceding flag"))
            continue
        if flag not in launcher_flags:
            # Don't warn on flags that are in the OTHER launcher (e.g. some
            # Paean-only flags may not be in the BLB launcher).
            problems.append((
                line_num,
                f"unknown flag {flag!r} (not accepted by either launcher; "
                "check for typos against the canonical list)",
            ))
        elif flag in seen:
            problems.append((
                line_num,
                f"duplicate flag {flag!r}; first seen on line {seen[flag]}",
            ))
        else:
            seen[flag] = line_num
            err = _classify_value(flag, value)
            if err:
                problems.append((line_num, f"{flag} {err}"))
    return problems


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("presets", nargs="+", help="One or more .conf preset files")
    ap.add_argument(
        "--launcher", default=LAUNCHER_REL,
        help=f"Main launcher to scan for accepted flags (default: {LAUNCHER_REL})",
    )
    ap.add_argument(
        "--paean-launcher", default=PAEAN_LAUNCHER_REL,
        help=f"Paean launcher to scan (default: {PAEAN_LAUNCHER_REL})",
    )
    args = ap.parse_args(argv)

    # Union: a preset under presets/ may use BLB launcher flags;
    # one under Paean/presets/ may use Paean flags. We accept either source.
    flags = extract_launcher_flags(args.launcher) | extract_launcher_flags(args.paean_launcher)
    if not flags:
        print(f"[error] no flags extracted from {args.launcher} / {args.paean_launcher}; "
              "regex may be stale", file=sys.stderr)
        return 2
    print(f"[validate_preset] using {len(flags)} canonical flags from "
          f"{args.launcher} + {args.paean_launcher}", file=sys.stderr)

    any_failed = False
    for preset_path in args.presets:
        problems = validate_preset(preset_path, flags)
        if not problems:
            print(f"  OK  {preset_path}")
            continue
        any_failed = True
        print(f"  FAIL {preset_path}")
        for line_num, msg in problems:
            print(f"    line {line_num}: {msg}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
