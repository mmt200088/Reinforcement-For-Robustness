#!/usr/bin/env python3
"""Build the Stage-2 fusion-count map (spec §3, plan Task 4).

For each production fusion block type, enumerate the
effective chain slots, run real replan, group by realized fusion_count, keep the
minimum-installed-noise set (option 0 == baseline by construction), and write a
per-block-type JSON cache under ``blb_stage2_rl/fusion_maps/<profile>/``.

The builder requires Torch and the in-process Rescale optimizer. Cartesian
products are partitioned across worker processes.

Usage:
    python scripts/blb_build_fusion_count_map.py --profile mrpc \
        --out-dir blb_stage2_rl/fusion_maps/mrpc --workers 16
"""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "Rescale_optimizer"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from json_utils import write_json_file  # noqa: E402


def block_types_for_profile(profile: str) -> List[Tuple[str, int, int, int]]:
    """``(graph_key, block_idx, gelu_degree, attn_degree)`` per buildable block-type
    for ``profile``. block1 / block2 graph keys are profile-suffixed
    (``block1_<profile>`` / ``block2_<profile>``); block4 / block5_n* are shared
    names. This generalizes the build to any fine-tuned profile (mrpc / rte / sst2
    and their ``_large`` variants) — the chain STRUCTURE is profile-independent, only
    the per-profile static_skeletons SF values differ."""
    p = str(profile)
    return [
        (f"block1_{p}", 1, 4, 2),
        (f"block2_{p}", 2, 4, 2),
        ("block4", 4, 4, 2),
        ("block5_n1", 5, 1, 2),
        ("block5_n2", 5, 2, 2),
        ("block5_n4", 5, 4, 2),
    ]


BLOCK_TYPES: List[Tuple[str, int, int, int]] = block_types_for_profile("mrpc")


def _utcnow_slug() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _enumerate_shard_worker(payload: Dict[str, Any]) -> Tuple[int, List[Tuple]]:
    """multiprocessing worker: build context + enumerate one shard.

    Returns ``(num_valid_seen, reduced_tuples)``. ``reduced_tuples`` is this shard's
    per-fusion_count minimum-variance set (plain picklable tuples) — NOT all valid
    configs, so a worker returns O(distinct min-var signatures) rows even for
    block4's ~3e8/96 stride (the full list would OOM / blow pickle). Each worker
    rebuilds its own context (ReplanSession is not picklable); the slot
    classification is deterministic, so every worker enumerates the same product
    and the ``i % num_shards`` stride is consistent.
    """
    from blb_stage2_rl import fusion_count_map as fcm
    from blb_stage2_rl import fusion_enum

    ctx = fusion_enum.prepare_block_type_context(
        graph_key=payload["graph_key"],
        block_idx=payload["block_idx"],
        gelu_degree=payload["gelu_degree"],
        attn_degree=payload["attn_degree"],
        profile=payload["profile"],
        rescale_optimizer_root=payload["ro_root"],
        num_layers=payload["num_layers"],
        ref_layer=payload["ref_layer"],
    )
    ecs, num_valid = fusion_enum.enumerate_shard(
        ctx,
        shard_idx=payload["shard_idx"],
        num_shards=payload["num_shards"],
        noise_order=fcm.SummedInstalledVariance(),
    )
    return num_valid, [
        (ec.action_indices, ec.fusion_count, ec.total_bits, ec.total_variance, ec.installed_signature) for ec in ecs
    ]


def _iter_golden_shard_results(payloads: List[Dict[str, Any]], num_shards: int):
    if int(num_shards) == 1:
        yield _enumerate_shard_worker(payloads[0])
        return
    with mp.get_context("spawn").Pool(processes=int(num_shards)) as pool:
        yield from pool.imap_unordered(_enumerate_shard_worker, payloads)


def _merge_golden_shard_results(shard_results) -> Tuple[List[Any], int]:
    from blb_stage2_rl import fusion_enum

    reducer = fusion_enum._MinNoiseReducer()
    nv_g = 0
    for num_valid, shard in shard_results:
        nv_g += int(num_valid)
        for ai, fc, tb, tv, sig in shard:
            reducer.add(
                fusion_enum.EvaluatedConfig(
                    action_indices=tuple(ai), fusion_count=int(fc),
                    total_bits=int(tb), total_variance=float(tv),
                    installed_signature=sig, slots={},
                )
            )
    return reducer.results(), nv_g


def _fast_range_payloads(
    template: Any, total: int, num_shards: int, *, profile: str, ro_root: str,
) -> List[Dict[str, Any]]:
    """Contiguous [start, stop) rank ranges covering [0, total) exactly once."""
    n = max(1, int(num_shards))
    base = int(total) // n
    rem = int(total) % n
    payloads: List[Dict[str, Any]] = []
    cursor = 0
    for s in range(n):
        count = base + (1 if s < rem else 0)
        if count <= 0:
            continue
        payloads.append({
            "template": template,
            "start": cursor,
            "stop": cursor + count,
            "profile": str(profile),
            "ro_root": str(ro_root),
        })
        cursor += count
    assert cursor == int(total)
    return payloads


def build_one_block_type(
    graph_key: str,
    block_idx: int,
    gelu_degree: int,
    attn_degree: int,
    *,
    profile: str,
    ro_root: str,
    num_layers: int,
    ref_layer: int,
    workers: int,
    max_enum_combos: int = 0,
    degeneracy_probe_samples: int = 2000,
    enum_path: str = "fast",
    fast_verify_random: int = 64,
    shards_per_worker: int = 8,
) -> Dict[str, Any]:
    from blb_stage2_rl import fusion_enum


    ctx = fusion_enum.prepare_block_type_context(
        graph_key=graph_key,
        block_idx=block_idx,
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
        profile=profile,
        rescale_optimizer_root=ro_root,
        num_layers=num_layers,
        ref_layer=ref_layer,
    )
    total_combos = ctx.enum_total()
    num_shards = max(1, int(workers))


    if max_enum_combos and total_combos > int(max_enum_combos):
        probe = fusion_enum.degeneracy_probe(ctx, num_random=int(degeneracy_probe_samples))
        if not probe["degenerate"]:
            raise RuntimeError(
                f"{graph_key}: enum_total {total_combos} > budget {max_enum_combos} AND the degeneracy probe "
                f"found fusion {probe['fusion_seen']} (base_fc={probe['base_fc']}) — refusing to emit a shortcut "
                f"map. Raise --max-enum-combos to build it fully, or reduce the action level count."
            )
        baseline_indices = [int(x) for x in ctx.baseline_block_indices]
        options = [
            {
                "option_id": 0,
                "fusion_count": int(probe["base_fc"]),
                "tie_index": 0,
                "total_variance": 0.0,
                "total_bits": 0,
                "slots": fusion_enum.decode_block_slots(ctx, baseline_indices),
                "action_indices": baseline_indices,
            }
        ]
        k_indep = fusion_enum.check_k_independence(ctx, sample_configs=[baseline_indices])
        return {
            "graph_key": graph_key,
            "profile": profile,
            "block_idx": block_idx,
            "gelu_degree": gelu_degree,
            "attn_degree": attn_degree,
            "k_slot_index": ctx.k_slot_index,
            "block_num_slots": ctx.block_num_slots,
            "options": options,
            "over_budget_degenerate": True,
            "build_meta": {
                "active_rescale_fields": ctx.active_rescale_fields,
                "enum_positions": ctx.enum_positions,
                "pinned_positions": ctx.pinned_positions,
                "enum_total_combos": total_combos,
                "valid_configs": 1,
                "num_options": 1,
                "fusion_counts": [int(probe["base_fc"])],
                "wall_seconds": 0.0,
                "workers": 0,
                "budget": int(max_enum_combos),
                "over_budget_degenerate": True,
                "degeneracy_probe": probe,
                "k_independence": k_indep,
            },
        }

    fast_meta: Dict[str, Any] = {}
    evaluated: List[Any] = []
    evaluated_golden: List[Any] = []
    num_valid_total = 0
    num_valid_golden = 0
    t0 = time.time()


    effective_enum_path = enum_path
    fast_fallback_reason = ""
    if enum_path in ("fast", "both"):
        from blb_stage2_rl import fusion_enum_fast
        template = fusion_enum_fast.build_fast_template(ctx)
        try:
            vres = fusion_enum_fast.verify_template(
                template, ctx, num_random=int(fast_verify_random),
            )
        except RuntimeError as exc:
            if enum_path != "fast":
                raise
            fast_fallback_reason = str(exc)
            effective_enum_path = "golden"
            print(
                f"  [fast] template verify FAILED -> falling back to golden for "
                f"{graph_key}: {exc}",
                flush=True,
            )
    if effective_enum_path in ("fast", "both"):

        print(
            f"  [fast] template OK: {len(template.points)} point specs, "
            f"golden-vs-fast verified on {vres['checked']} probes "
            f"(baseline + corner + {vres['num_random']} random)",
            flush=True,
        )
        n_shards = max(1, int(workers) * max(1, int(shards_per_worker)))
        n_shards = min(n_shards, max(1, total_combos))
        payloads = _fast_range_payloads(
            template, total_combos, n_shards, profile=profile, ro_root=ro_root,
        )
        rows_all: List[Tuple] = []
        done = 0
        t_fast = time.time()
        last_log = t_fast
        if len(payloads) == 1:
            nv, rows, _w, cnt = fusion_enum_fast.enumerate_range_worker(payloads[0])
            num_valid_total += int(nv)
            rows_all.extend(rows)
            done += cnt
        else:
            with mp.get_context("spawn").Pool(processes=int(workers)) as pool:
                for nv, rows, _w, cnt in pool.imap_unordered(
                    fusion_enum_fast.enumerate_range_worker, payloads
                ):
                    num_valid_total += int(nv)
                    rows_all.extend(rows)
                    done += int(cnt)
                    now = time.time()
                    if now - last_log >= 30 or done == total_combos:
                        rate = done / max(1e-9, now - t_fast)
                        eta = (total_combos - done) / max(1e-9, rate)
                        print(
                            f"  [fast] {done}/{total_combos} "
                            f"({done / total_combos:.1%}) "
                            f"rate={rate:,.0f} combos/s eta={eta / 60:.1f}min",
                            flush=True,
                        )
                        last_log = now
        for ai, fc, tb, tv, sig in rows_all:
            evaluated.append(
                fusion_enum.EvaluatedConfig(
                    action_indices=tuple(ai), fusion_count=int(fc),
                    total_bits=int(tb), total_variance=float(tv),
                    installed_signature=sig, slots={},
                )
            )
        fast_meta = {
            "enum_path": enum_path,
            "fast_verified_probes": int(vres["checked"]),
            "fast_shards": len(payloads),
            "fast_wall_seconds": round(time.time() - t_fast, 2),
        }

    def _run_golden_enum() -> Tuple[List[Any], int]:
        """Full cfg-path (golden = source of truth) enumeration → (evaluated, num_valid)."""
        payloads_g = [
            {
                "graph_key": graph_key,
                "block_idx": block_idx,
                "gelu_degree": gelu_degree,
                "attn_degree": attn_degree,
                "profile": profile,
                "ro_root": ro_root,
                "num_layers": num_layers,
                "ref_layer": ref_layer,
                "shard_idx": s,
                "num_shards": num_shards,
            }
            for s in range(num_shards)
        ]
        return _merge_golden_shard_results(
            _iter_golden_shard_results(payloads_g, num_shards=num_shards)
        )

    if effective_enum_path in ("golden", "both"):

        evaluated_golden, num_valid_golden = _run_golden_enum()
        if effective_enum_path == "golden":
            evaluated = evaluated_golden
            num_valid_total = num_valid_golden

    def _group(ev: List[Any]) -> List[Dict[str, Any]]:
        return fusion_enum.group_min_noise_options(
            ev, ctx.baseline_block_indices,
            baseline_installed_signature=ctx.baseline_installed_signature,
        )

    options = _group(evaluated)


    if effective_enum_path == "fast":
        kept_problems = fusion_enum.verify_kept_options_golden(ctx, options)
        if kept_problems:
            reason = "; ".join(f"opt{oid}: {msg}" for oid, msg in kept_problems)
            print(
                f"  [fast] KEPT-OPTION golden check FAILED -> falling back to golden "
                f"for {graph_key}: {reason}",
                flush=True,
            )
            fast_fallback_reason = (
                (fast_fallback_reason + " | " if fast_fallback_reason else "")
                + f"kept-option golden mismatch: {reason}"
            )
            effective_enum_path = "golden"
            evaluated_golden, num_valid_golden = _run_golden_enum()
            evaluated = evaluated_golden
            num_valid_total = num_valid_golden
            options = _group(evaluated)

    elapsed = time.time() - t0

    if enum_path == "both":


        if int(num_valid_golden) != int(num_valid_total):
            raise RuntimeError(
                f"{graph_key}: enum-path mismatch: valid fast={num_valid_total} "
                f"golden={num_valid_golden}"
            )
        options_golden = fusion_enum.group_min_noise_options(
            evaluated_golden, ctx.baseline_block_indices,
            baseline_installed_signature=ctx.baseline_installed_signature,
        )
        if len(options) != len(options_golden):
            raise RuntimeError(
                f"{graph_key}: enum-path mismatch: {len(options)} fast options vs "
                f"{len(options_golden)} golden"
            )
        for a, b in zip(options, options_golden, strict=True):
            same = (
                a["action_indices"] == b["action_indices"]
                and a["fusion_count"] == b["fusion_count"]
                and a["total_bits"] == b["total_bits"]
                and a["tie_index"] == b["tie_index"]
                and abs(a["total_variance"] - b["total_variance"])
                <= 1e-12 * max(1.0, abs(b["total_variance"]))
            )
            if not same:
                raise RuntimeError(
                    f"{graph_key}: enum-path option mismatch: fast={a} golden={b}"
                )
        fast_meta["both_paths_identical"] = True
        print(f"  [both] fast == golden on all {len(options)} options ✓", flush=True)

    for opt in options:
        opt["slots"] = fusion_enum.decode_block_slots(ctx, opt["action_indices"])


    options = fusion_enum.boost_options_for_block(ctx, options)
    n_boosted = sum(1 for o in options if o.get("boosted"))
    if n_boosted:
        print(f"  [boost] 加大精度: {n_boosted}/{len(options)} options raised to all-q_max", flush=True)


    sample = [opt["action_indices"] for opt in options[: min(8, len(options))]]
    k_indep = fusion_enum.check_k_independence(ctx, sample_configs=sample)

    fusion_counts = sorted({opt["fusion_count"] for opt in options})
    return {
        "graph_key": graph_key,
        "profile": profile,
        "block_idx": block_idx,
        "gelu_degree": gelu_degree,
        "attn_degree": attn_degree,
        "k_slot_index": ctx.k_slot_index,
        "block_num_slots": ctx.block_num_slots,
        "options": options,
        "build_meta": {
            "active_rescale_fields": ctx.active_rescale_fields,
            "enum_positions": ctx.enum_positions,
            "pinned_positions": ctx.pinned_positions,
            "enum_total_combos": total_combos,
            "valid_configs": num_valid_total,
            "kept_configs": len(evaluated),
            "num_options": len(options),
            "fusion_counts": fusion_counts,
            "wall_seconds": round(elapsed, 2),
            "workers": int(workers),
            "k_independence": k_indep,
            "enum_path_requested": enum_path,
            "enum_path_effective": effective_enum_path,
            "fast_fallback_reason": fast_fallback_reason,
            **fast_meta,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Stage-2 fusion-count map")
    ap.add_argument("--profile", default="mrpc")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "blb_stage2_rl" / "fusion_maps" / "mrpc"))
    ap.add_argument("--rescale-optimizer-root", default=str(REPO_ROOT / "Rescale_optimizer"))
    ap.add_argument("--num-layers", type=int, default=12)
    ap.add_argument("--ref-layer", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument(
        "--enum-path",
        choices=("fast", "golden", "both"),
        default="fast",
        help="fast = direct-replan hot loop (template golden-derived + verified, default); "
        "golden = original cfg-path enumeration; both = run BOTH and require the final "
        "option lists to match exactly (full cross-validation — use on small block-types)",
    )
    ap.add_argument(
        "--fast-verify-random",
        type=int,
        default=64,
        help="random combos (plus baseline + all-min corner) cross-checked golden-vs-fast "
        "before a fast enumeration starts; any mismatch aborts the build",
    )
    ap.add_argument(
        "--shards-per-worker",
        type=int,
        default=8,
        help="fast path splits the combo space into workers*this contiguous rank ranges "
        "(better load balance + progress/ETA granularity)",
    )
    ap.add_argument("--only", default="", help="comma list of graph_keys to build (default: all 7)")
    ap.add_argument(
        "--max-enum-combos",
        type=int,
        default=0,
        help="for any block-type whose enumerated cartesian product exceeds this, run a degeneracy probe "
        "instead of the full build (emits a baseline-only map if degenerate, else raises); 0 = unlimited",
    )
    ap.add_argument(
        "--degeneracy-probe-samples",
        type=int,
        default=2000,
        help="random samples (plus the all-min corner) used by the over-budget degeneracy probe",
    )
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    targets = [bt for bt in block_types_for_profile(args.profile) if (not only or bt[0] in only)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    max_num_options = 6
    for graph_key, block_idx, gelu, attn in targets:
        print(f"[build] {graph_key} (block{block_idx}, gelu={gelu}, attn={attn}) ...", flush=True)
        res = build_one_block_type(
            graph_key,
            block_idx,
            gelu,
            attn,
            profile=args.profile,
            ro_root=args.rescale_optimizer_root,
            num_layers=args.num_layers,
            ref_layer=args.ref_layer,
            workers=args.workers,
            max_enum_combos=args.max_enum_combos,
            enum_path=args.enum_path,
            fast_verify_random=args.fast_verify_random,
            shards_per_worker=args.shards_per_worker,
            degeneracy_probe_samples=args.degeneracy_probe_samples,
        )
        results.append(res)
        m = res["build_meta"]
        max_num_options = max(max_num_options, len(res["options"]))
        write_json_file(out_dir / f"{graph_key}.json", res)
        if res.get("over_budget_degenerate"):
            pr = m["degeneracy_probe"]
            print(
                f"  -> OVER-BUDGET DEGENERATE: enum_total={m['enum_total_combos']} > {m['budget']}; "
                f"probe base_fc={pr['base_fc']} corner_fusion={pr['corner_fusion']} "
                f"fusion_seen={pr['fusion_seen']} samples={pr['samples_checked']} "
                f"-> wrote baseline-only map (new index convention)",
                flush=True,
            )
            continue
        print(
            f"  -> options={m['num_options']} fusion_counts={m['fusion_counts']} "
            f"valid={m['valid_configs']}/{m['enum_total_combos']} "
            f"rescales={len(m['active_rescale_fields'])}{m['active_rescale_fields']} "
            f"pinned={len(m['pinned_positions'])} "
            f"K-indep={m['k_independence']['k_independent']} wall={m['wall_seconds']}s",
            flush=True,
        )

    summary = {
        "profile": args.profile,
        "max_num_options": max_num_options,
        "graph_keys": [r["graph_key"] for r in results],
        "per_type": {r["graph_key"]: r["build_meta"] for r in results},
    }
    write_json_file(out_dir / "_summary.json", summary)
    print("\n=== SUMMARY ===")
    print(f"max_num_options={max_num_options}")
    for r in results:
        m = r["build_meta"]
        tag = " [over-budget degenerate probe]" if r.get("over_budget_degenerate") else ""
        print(
            f"  {r['graph_key']}: #options={m['num_options']} fusion={m['fusion_counts']} "
            f"K-indep={m['k_independence']['k_independent']}{tag}"
        )
    bad_k = [r["graph_key"] for r in results if not r["build_meta"]["k_independence"]["k_independent"]]
    if bad_k:
        print(f"  [WARN] K-dependent fusion in: {bad_k} (reward uses real replan; see spec §3.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
