#!/usr/bin/env python3
"""Build the Stage-2 fusion-count map (spec §3, plan Task 4).

For each of the 7 RL block-types (block3 is frozen → no map), enumerate the
effective chain slots, run real replan, group by realized fusion_count, keep the
minimum-installed-noise set (option 0 == baseline by construction), and write a
per-block-type JSON cache under ``blb_stage2_rl/fusion_maps/<profile>/``.

Server-only (needs torch + Rescale_optimizer). Parallelizes the cartesian
product across processes (each worker owns its own bridge). Run via
SERVER_COMMAND.md per the local/server protocol.

Usage:
    python scripts/blb_build_fusion_count_map.py --profile mrpc \
        --out-dir blb_stage2_rl/fusion_maps/mrpc \
        --report reports/blb_opt/fusion_maps/build_<ts>.html --workers 16
    # local small dry-run (torch needed):
    python scripts/blb_build_fusion_count_map.py --profile mrpc \
        --only block1_mrpc,block5_n1 --out-dir /tmp/fm --workers 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "blb_stage2_rl"), str(REPO_ROOT / "Rescale_optimizer"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# (graph_key, block_idx, gelu_degree, attn_degree). attn=2 keeps the bootstrap's
# block3 baseline valid (block3_exp_n2 exists); it does not affect block1/2/4/5.
# block5_n0 (GELU degree 0 / ReLU) is disabled since 2026-06-06 (Stage-1 stopped
# sampling degree 0); it is no longer built. The committed block5_n0.json map is
# kept as dormant data and FusionCountMap.load still loads it, but a degree-0 layer
# is rejected upstream at baseline_bootstrap, so the fusion schedule never requests it.
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


# Back-compat default (mrpc); main() rebuilds this per --profile.
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
    import fusion_count_map as fcm
    import fusion_enum

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
    import fusion_enum

    # One context in the main process for geometry + baseline + classification
    # diagnostics (the workers rebuild their own bridge for the heavy loop).
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

    # Budget guard: the sound (fusion, total_bits) classification enumerates every
    # fusion-relevant encode JOINTLY, so the deep 10-level grid can be huge for a
    # block with many encodes (block4 ~ 7e8 combos ~ tens of hours), even though
    # such a block is usually fusion-DEGENERATE (the encodes move bits but not
    # fusion). NOTE: we cannot just keep the existing committed map — its
    # action_indices use the OLD per-slot level convention and would mis-decode
    # under the new uniform-10 decode. Instead run a degeneracy probe (all-min
    # corner + random samples); if nothing fuses, emit a correct new-convention
    # baseline-only map cheaply; if any config fuses, REFUSE to shortcut.
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

    if enum_path in ("fast", "both"):
        # ---- direct-replan fast path (template golden-derived + verified) ----
        import fusion_enum_fast
        template = fusion_enum_fast.build_fast_template(ctx)
        vres = fusion_enum_fast.verify_template(
            template, ctx, num_random=int(fast_verify_random),
        )
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

    if enum_path in ("golden", "both"):
        # ---- original cfg-path enumeration (stride shards) ----
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
        if num_shards == 1:
            shard_results = [_enumerate_shard_worker(payloads_g[0])]
        else:
            with mp.get_context("spawn").Pool(processes=num_shards) as pool:
                shard_results = pool.map(_enumerate_shard_worker, payloads_g)
        for num_valid, shard in shard_results:
            num_valid_golden += int(num_valid)
            for ai, fc, tb, tv, sig in shard:
                evaluated_golden.append(
                    fusion_enum.EvaluatedConfig(
                        action_indices=tuple(ai), fusion_count=int(fc),
                        total_bits=int(tb), total_variance=float(tv),
                        installed_signature=sig, slots={},
                    )
                )
        if enum_path == "golden":
            evaluated = evaluated_golden
            num_valid_total = num_valid_golden

    elapsed = time.time() - t0

    options = fusion_enum.group_min_noise_options(evaluated, ctx.baseline_block_indices)

    if enum_path == "both":
        # FULL cross-validation: the two paths' final option lists must agree
        # exactly (variance compared with float-sum-order tolerance only).
        if int(num_valid_golden) != int(num_valid_total):
            raise RuntimeError(
                f"{graph_key}: enum-path mismatch: valid fast={num_valid_total} "
                f"golden={num_valid_golden}"
            )
        options_golden = fusion_enum.group_min_noise_options(
            evaluated_golden, ctx.baseline_block_indices,
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
    # Fill SF/K-first slot view for the kept options only.
    for opt in options:
        opt["slots"] = fusion_enum.decode_block_slots(ctx, opt["action_indices"])

    # 加大精度 (precision boost, 2026-06-19): raise each non-zero-fusion option's
    # short modulus primes to q_max at minimum installed noise (replan-verified),
    # rewriting it as a boosted explicit-SF option. No-op for fc=0 options,
    # block-types without a registered ChainTopology, or options already all-q_max.
    options = fusion_enum.boost_options_for_block(ctx, options)
    n_boosted = sum(1 for o in options if o.get("boosted"))
    if n_boosted:
        print(f"  [boost] 加大精度: {n_boosted}/{len(options)} options raised to all-q_max", flush=True)

    # K-independence self-check on a small sample of options.
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
            **fast_meta,
        },
    }


def write_report(out_html: Path, profile: str, results: List[Dict[str, Any]]) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        m = r["build_meta"]
        rows.append(
            f"<tr><td><code>{html.escape(r['graph_key'])}</code></td>"
            f"<td>{m['enum_total_combos']}</td><td>{m['valid_configs']}</td>"
            f"<td>{m['num_options']}</td><td>{html.escape(str(m['fusion_counts']))}</td>"
            f"<td>{len(m['pinned_positions'])}</td>"
            f"<td>{'yes' if m['k_independence']['k_independent'] else 'NO'}</td>"
            f"<td>{m['wall_seconds']}s</td></tr>"
        )
    out_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>fusion-count map build</title>"
        f"<h1>fusion-count map build — {html.escape(profile)}</h1>"
        "<table border=1 cellpadding=4><thead><tr>"
        "<th>graph_key</th><th>enum combos</th><th>valid</th><th>#options</th>"
        "<th>fusion_counts</th><th>pinned</th><th>K-indep</th><th>wall</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        encoding="utf-8",
    )


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
    ap.add_argument("--report", default="")
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
        (out_dir / f"{graph_key}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
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
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.report:
        write_report(Path(args.report), args.profile, results)

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
