#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from blb_stage2_rl.candidate_store import CandidateStore, candidate_key


def legacy_lookup(store, action_indices, identity_context):
    wanted_key = candidate_key(action_indices, identity_context)
    latest_status = ""
    latest_metadata = {}
    for record in store.iter_active_records():
        if record.get("record_type") != "candidate_promotion_status_v1":
            continue
        if str(record.get("candidate_key", "")) != wanted_key:
            continue
        latest_status = str(record.get("promotion_status", ""))
        latest_metadata = dict(record.get("promotion_metadata") or {})
    return latest_status, latest_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_store", type=Path)
    parser.add_argument("--old-repeats", type=int, default=3)
    parser.add_argument("--hot-repeats", type=int, default=2000)
    args = parser.parse_args()

    path = args.candidate_store.resolve()
    file_size_before = path.stat().st_size
    scan_store = CandidateStore(path)
    active_records = 0
    latest_records = {}
    scan_started = time.perf_counter()
    for record in scan_store.iter_active_records():
        active_records += 1
        if record.get("record_type") != "candidate_promotion_status_v1":
            continue
        key = str(record.get("candidate_key", ""))
        identity_context = record.get("identity_context")
        action_indices = record.get("action_indices")
        if not key or not isinstance(identity_context, dict) or action_indices is None:
            continue
        latest_records[key] = (
            str(record.get("promotion_status", "")),
            dict(record.get("promotion_metadata") or {}),
            list(action_indices),
            dict(identity_context),
        )
    inventory_scan_seconds = time.perf_counter() - scan_started
    if not latest_records:
        raise RuntimeError("candidate store has no indexed promotion records")

    target = next(reversed(latest_records.values()))
    expected_target = target[:2]
    old_seconds = []
    for _ in range(args.old_repeats):
        started = time.perf_counter()
        observed = legacy_lookup(scan_store, target[2], target[3])
        old_seconds.append(time.perf_counter() - started)
        if observed != expected_target:
            raise AssertionError("legacy lookup changed during benchmark")

    indexed_store = CandidateStore(path)
    cold_started = time.perf_counter()
    cold_observed = indexed_store.latest_promotion_status_for_action(
        target[2], target[3],
    )
    cold_index_seconds = time.perf_counter() - cold_started
    if cold_observed != expected_target:
        raise AssertionError("cold indexed lookup differs from legacy lookup")

    parity_started = time.perf_counter()
    mismatches = []
    for key, (status, metadata, action_indices, identity_context) in latest_records.items():
        observed = indexed_store.latest_promotion_status_for_action(
            action_indices, identity_context,
        )
        if observed != (status, metadata):
            mismatches.append(key)
    parity_seconds = time.perf_counter() - parity_started

    hot_batches = []
    for _ in range(5):
        started = time.perf_counter()
        for _ in range(args.hot_repeats):
            observed = indexed_store.latest_promotion_status_for_action(
                target[2], target[3],
            )
        elapsed = time.perf_counter() - started
        if observed != expected_target:
            raise AssertionError("hot indexed lookup changed during benchmark")
        hot_batches.append(elapsed / args.hot_repeats)

    file_size_after = path.stat().st_size
    if file_size_after != file_size_before:
        raise AssertionError("candidate store changed during read-only benchmark")
    hot_median = statistics.median(hot_batches)
    old_median = statistics.median(old_seconds)
    result = {
        "candidate_store": str(path),
        "file_bytes": file_size_before,
        "active_records": active_records,
        "latest_promotion_candidates": len(latest_records),
        "inventory_scan_seconds": inventory_scan_seconds,
        "old_lookup_seconds": old_seconds,
        "old_lookup_median_seconds": old_median,
        "cold_index_and_lookup_seconds": cold_index_seconds,
        "all_candidate_parity_seconds": parity_seconds,
        "all_candidate_parity": not mismatches,
        "mismatch_count": len(mismatches),
        "hot_lookup_seconds_per_call": hot_batches,
        "hot_lookup_median_seconds_per_call": hot_median,
        "hot_lookup_speedup_vs_old_median": old_median / hot_median,
        "indexed_candidate_count": len(
            indexed_store._latest_promotion_by_candidate_key
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
