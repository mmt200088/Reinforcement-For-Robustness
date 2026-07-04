"""
update_noise_tables_from_csv.py
================================

Update the ``noise_table`` block (and its ``_noise_table_doc``) in every
config under ``configs/<profile>/`` from a measured-noise CSV.

CSV format (one row per (N, scale_bits)) — produced by CKKS_noise_seal:

    N,scale_bits,B_enc,B_fresh,B_rs

Mapping into the config:
    "rescale" <- B_rs
    "fresh"   <- B_fresh

Per-config N choice (defaults configurable below):

    EIGHT_K_BASES = {"block1", "block3_exp_n2", "block5_n0", "block5_n1"}
        -> N = 8192
    All others      -> N = 16384

The "base" name is the config stem with any ``_wnli`` / ``_mrpc`` suffix
stripped, so ``block1_wnli`` and ``block1_mrpc`` both map to "block1".

The output range is sf = SF_MIN..SF_MAX with a configurable step
(defaults: 12..60 step 1 — i.e. one entry per scale bit). For sf
greater than the max sf available in the CSV, values are extrapolated
using the empirical rule "B(s+1) = B(s)/2" (verified by the CSV).

Only the noise_table block of each file is rewritten; the rest of the
JSON file is left untouched (we use a brace-matching text replacement so
formatting / comments are preserved).

Usage::

    python scripts/update_noise_tables_from_csv.py
    python scripts/update_noise_tables_from_csv.py --csv path/to/noise.csv
    python scripts/update_noise_tables_from_csv.py --dirs configs/mrpc
    python scripts/update_noise_tables_from_csv.py --step 1
    python scripts/update_noise_tables_from_csv.py --sf-min 12 --sf-max 60
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent

DEFAULT_CSV = Path(
    "/var/tmp/root-home/ckks_noise_measure/CKKS_noise_seal/noise_inf_table.csv"
)
DEFAULT_DIRS = [REPO / "configs/wnli", REPO / "configs/mrpc"]

EIGHT_K_BASES = {"block1", "block3_exp_n2", "block5_n0", "block5_n1"}

SF_MIN_DEFAULT = 12
SF_MAX_DEFAULT = 60
SF_STEP_DEFAULT = 1  # one entry per scale bit (was 2 previously)
ITEMS_PER_LINE = 4


def base_name(stem: str) -> str:
    for suf in ("_wnli", "_mrpc"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def n_for_stem(stem: str) -> int:
    return 8192 if base_name(stem) in EIGHT_K_BASES else 16384


def load_csv(csv_path: Path) -> Dict[Tuple[int, int], Tuple[float, float]]:
    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        columns = {str(name): idx for idx, name in enumerate(header)}
        n_col = columns["N"]
        sf_col = columns["scale_bits"]
        fresh_col = columns["B_fresh"]
        rescale_col = columns["B_rs"]
        for row in rdr:
            if not row or all(not cell or cell.isspace() for cell in row):
                continue
            N = int(row[n_col])
            sf = int(row[sf_col])
            out[(N, sf)] = (float(row[fresh_col]), float(row[rescale_col]))
    return out


def build_tables(
    table: Dict[Tuple[int, int], Tuple[float, float]],
    N: int,
    sfs: List[int],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    sfs_for_n = sorted(sf for (n, sf) in table if n == N)
    if not sfs_for_n:
        raise ValueError(f"No CSV rows for N={N}")
    min_csv_sf, max_csv_sf = sfs_for_n[0], sfs_for_n[-1]
    f_max, r_max = table[(N, max_csv_sf)]

    fresh: Dict[int, float] = {}
    rescale: Dict[int, float] = {}
    for sf in sfs:
        if (N, sf) in table:
            f, r = table[(N, sf)]
        elif sf > max_csv_sf:
            shift = sf - max_csv_sf
            f = f_max / (2 ** shift)
            r = r_max / (2 ** shift)
        else:
            raise ValueError(
                f"sf={sf} below CSV minimum {min_csv_sf} for N={N}; "
                f"can only extrapolate above the table."
            )
        fresh[sf] = f
        rescale[sf] = r
    return fresh, rescale


def fmt_dict_block(d: Dict[int, float], indent: str) -> str:
    lines: List[str] = []
    parts: List[str] = []
    last_idx = len(d) - 1
    for idx, (sf, val) in enumerate(d.items()):
        parts.append(f'"{sf}": {val:.6e}')
        if len(parts) < ITEMS_PER_LINE and idx != last_idx:
            continue
        line = indent + ", ".join(parts)
        if idx != last_idx:
            line += ","
        lines.append(line)
        parts = []
    return "\n".join(lines)


def replace_block(text: str, key: str, new_value_block: str) -> str:
    """
    Replace the JSON block for ``"key"`` (whose value is an object literal
    starting with '{') in ``text``. ``new_value_block`` should be the
    value text only, starting with '{' and ending with '}'.
    """
    pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*')
    m = pat.search(text)
    if not m:
        raise ValueError(f"key '{key}' not found")
    open_idx = text.find("{", m.end() - 1)
    if open_idx == -1:
        raise ValueError(f"value of '{key}' is not an object")
    depth = 0
    end_idx = -1
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        raise ValueError(f"unterminated value for '{key}'")
    return text[: m.end()] + new_value_block + text[end_idx + 1 :]


def replace_string_field(text: str, key: str, new_value: str) -> str:
    pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
    m = pat.search(text)
    if not m:
        return text  # silently skip if missing
    return text[: m.start()] + f'"{key}": "{new_value}"' + text[m.end() :]


def update_one(
    path: Path,
    csv_table: Dict[Tuple[int, int], Tuple[float, float]],
    sf_min: int,
    sf_max: int,
    sf_step: int,
) -> Tuple[bool, str]:
    stem = path.stem
    N = n_for_stem(stem)
    sfs = list(range(sf_min, sf_max + 1, sf_step))
    fresh, rescale = build_tables(csv_table, N, sfs)

    text = path.read_text(encoding="utf-8")

    base_indent = "  "
    inner_indent = base_indent + "  "
    item_indent = inner_indent + "  "

    new_block = (
        "{\n"
        + inner_indent + '"rescale": {\n'
        + fmt_dict_block(rescale, item_indent) + "\n"
        + inner_indent + "},\n"
        + inner_indent + '"fresh": {\n'
        + fmt_dict_block(fresh, item_indent) + "\n"
        + inner_indent + "}\n"
        + base_indent + "}"
    )

    try:
        new_text = replace_block(text, "noise_table", new_block)
    except ValueError as e:
        return False, f"skip (no noise_table?): {e}"

    csv_max_sf = max(
        sf for (n, sf) in csv_table if n == N
    )
    if sf_max > csv_max_sf:
        extrap_msg = (
            f"sf in [{sf_min},{csv_max_sf}] from CSV; "
            f"sf in [{csv_max_sf + 1},{sf_max}] extrapolated by "
            "halving per +1 bit."
        )
    else:
        extrap_msg = f"sf in [{sf_min},{sf_max}] all from CSV (no extrapolation)."
    new_doc = (
        f"Measured noise from noise_inf_table.csv (N={N}). "
        f"Columns: rescale<-B_rs, fresh<-B_fresh. "
        f"sf range {sf_min}..{sf_max} step {sf_step}. {extrap_msg}"
    )
    new_text = replace_string_field(new_text, "_noise_table_doc", new_doc)

    if new_text == text:
        return False, "unchanged"
    path.write_text(new_text, encoding="utf-8")
    return True, f"updated (N={N}, {len(sfs)} entries)"


def _discover_config_paths(config_dir: Path) -> List[Path]:
    names = sorted(
        entry.name
        for entry in os.scandir(config_dir)
        if entry.is_file()
        and entry.name.endswith(".json")
        and not entry.name.startswith("static_skeletons")
    )
    return [config_dir / name for name in names]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument(
        "--dirs", nargs="*", default=[str(d) for d in DEFAULT_DIRS],
        help="Config directories to scan (default: wnli + mrpc).",
    )
    p.add_argument("--sf-min", type=int, default=SF_MIN_DEFAULT)
    p.add_argument("--sf-max", type=int, default=SF_MAX_DEFAULT)
    p.add_argument("--step", type=int, default=SF_STEP_DEFAULT,
                   help="Scale-bit step (default: 1, i.e. one entry per bit).")
    args = p.parse_args()

    csv_table = load_csv(Path(args.csv))

    n_changed = 0
    n_total = 0
    for d in args.dirs:
        dpath = Path(d)
        if not dpath.is_dir():
            print(f"[skip ] {dpath} does not exist")
            continue
        for cfg in _discover_config_paths(dpath):
            n_total += 1
            try:
                changed, msg = update_one(
                    cfg, csv_table, args.sf_min, args.sf_max, args.step,
                )
            except Exception as e:
                print(f"[ERR ] {cfg}: {e}")
                continue
            n_changed += int(changed)
            tag = "WRITE" if changed else "skip "
            try:
                rel = cfg.relative_to(REPO)
            except ValueError:
                rel = cfg
            print(f"[{tag}] {rel}  -- {msg}")

    print(f"\n[noise] updated {n_changed} / {n_total} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
