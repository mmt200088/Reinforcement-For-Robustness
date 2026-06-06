"""
Unit tests for the rescale-fusion algorithm in rescale_optimizer.replan.

These tests target ``_fuse_chain`` directly with synthetic inputs so we
can exercise every branch (fuse-to-prev / fuse-to-next / multi-fusion /
unfusable rejection) without needing a real graph.

Run:
    python scripts/test_replan_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rescale_optimizer.replan import _fuse_chain  # type: ignore


def _check(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def test_identity_no_fusion():
    print("test_identity_no_fusion")
    skel = [0, 1, 3, 5, 6]
    q = [40, 40, 40]
    t = [32, 32, 32, 32]
    valid, s2, q2, t2, evs, bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("skel unchanged", s2 == skel)
        and _check("q unchanged", q2 == q)
        and _check("t unchanged", t2 == t)
        and _check("no events", len(evs) == 0)
    )
    return ok


def test_fuse_to_next():
    """q=[10, 40, 50] -> q[0] small, fuse with next ⇒ q=[50, 50]."""
    print("test_fuse_to_next")
    skel = [0, 1, 3, 5, 6]
    q = [10, 40, 50]
    t = [32, 33, 34, 35]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("q=[50,50]", q2 == [50, 50])
        and _check("skel removed s_1", s2 == [0, 3, 5, 6])
        and _check("t shrank by 1", len(t2) == 3 and t2 == [32, 34, 35])
        and _check("1 fusion event", len(evs) == 1)
        and _check("event side=next", evs[0].fused_into == "next")
    )
    return ok


def test_fuse_to_prev():
    """q=[40, 10, 50] -> q[1] small, prefer next first;
    next would give 10+50=60 ≤ 60 OK ⇒ goes to next: q=[40, 60].
    """
    print("test_fuse_to_prev_or_next_priority")
    skel = [0, 1, 3, 5, 6]
    q = [40, 10, 50]
    t = [32, 33, 34, 35]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        # next is preferred when both sides feasible
        and _check("q=[40,60]", q2 == [40, 60])
        and _check("skel removed s_2", s2 == [0, 1, 5, 6])
        and _check("event side=next", evs[0].fused_into == "next")
    )
    return ok


def test_fuse_to_prev_only():
    """q=[40, 10, 55] -> q[1] small. next: 10+55=65>60 (no);
    prev: 40+10=50 ≤60 (yes). Should fuse to prev.
    """
    print("test_fuse_to_prev_only")
    skel = [0, 1, 3, 5, 6]
    q = [40, 10, 55]
    t = [32, 33, 34, 35]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("q=[50,55]", q2 == [50, 55])
        and _check("skel removed s_2", s2 == [0, 1, 5, 6])
        and _check("event side=prev", evs[0].fused_into == "prev")
    )
    return ok


def test_multi_fusion():
    """q=[10,10,10,30,30] -> sequence of next-fusions consolidates the
    three small primes into one prime ≥ 30.

    Step 1: q[0]=10 < 30. next: 10+10=20 ≤ 60. Fuse-next → q=[20,10,30,30].
    Step 2: q[0]=20 < 30. next: 20+10=30 ≤ 60. Fuse-next → q=[30,30,30].
    Done — all q ≥ 30. 2 fusion events.
    """
    print("test_multi_fusion")
    skel = [0, 1, 2, 3, 4, 5, 6]
    q = [10, 10, 10, 30, 30]
    t = [10, 11, 12, 13, 14, 15]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("2 fusions", len(evs) == 2)
        and _check("q=[30,30,30]", q2 == [30, 30, 30])
        and _check("all q >=30", all(x >= 30 for x in q2))
        and _check("all q <=60", all(x <= 60 for x in q2))
        and _check("R reduced by 2", len(q2) == 3)
    )
    return ok


def test_unfusable():
    """q=[55, 10, 55] -> q[1] small. next: 10+55=65>60 (no);
       prev: 55+10=65>60 (no). UNFUSABLE.
    """
    print("test_unfusable")
    skel = [0, 1, 3, 5, 6]
    q = [55, 10, 55]
    t = [32, 33, 34, 35]
    valid, _s2, _q2, _t2, evs, bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("invalid", not valid)
        and _check("0 events", len(evs) == 0)
        and _check("invalid_chain reported", bad is not None and bad.q_bits == [55, 10, 55])
    )
    return ok


def test_fuse_at_chain_end():
    """q=[40, 10] -> q[1] small. next: doesn't exist; prev: 40+10=50 ≤60. Fuse to prev."""
    print("test_fuse_at_chain_end")
    skel = [0, 1, 3, 5]
    q = [40, 10]
    t = [32, 33, 34]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("q=[50]", q2 == [50])
        and _check("skel removed s_2", s2 == [0, 1, 5])
        and _check("event side=prev", evs[0].fused_into == "prev")
    )
    return ok


def test_fuse_at_chain_start():
    """q=[10, 40] -> q[0] small. next: 10+40=50 ≤60 OK. Fuse to next."""
    print("test_fuse_at_chain_start")
    skel = [0, 1, 3, 5]
    q = [10, 40]
    t = [32, 33, 34]
    valid, s2, q2, t2, evs, _bad = _fuse_chain(skel, q, t, 30, 60)
    ok = (
        _check("valid", valid)
        and _check("q=[50]", q2 == [50])
        and _check("skel removed s_1", s2 == [0, 3, 5])
        and _check("event side=next", evs[0].fused_into == "next")
    )
    return ok


def main() -> int:
    tests = [
        test_identity_no_fusion,
        test_fuse_to_next,
        test_fuse_to_prev,
        test_fuse_to_prev_only,
        test_multi_fusion,
        test_unfusable,
        test_fuse_at_chain_end,
        test_fuse_at_chain_start,
    ]
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"summary: {passed} / {len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
