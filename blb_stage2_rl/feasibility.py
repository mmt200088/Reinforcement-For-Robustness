"""Final-eval feasibility semantics for BLB Trust-0."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), -float("inf")):
        return None
    return out


def _thresholds_known(source: Any, limits: Mapping[str, Any]) -> bool:
    text = str(source or "").strip().lower()
    if text in ("", "unknown", "none", "null"):
        return False
    return all(_finite_or_none(v) is not None for v in limits.values())


def build_final_eval_feasibility(
        *,
        optimizer_valid: bool,
        decode_ok: bool,
        apply_ok: bool,
        eval_ok: bool,
        acc_mean: Any,
        f1_mean: Any,
        acc_std: Any,
        f1_std: Any,
        acc_limit: Any,
        f1_limit: Any,
        acc_std_limit: Any,
        f1_std_limit: Any,
        threshold_source: str,
        loss_mean: Any = None,
        loss_std: Any = None,
        strict_z: float = 1.0,
        ) -> Dict[str, Any]:
    acc = _finite_or_none(acc_mean)
    f1 = _finite_or_none(f1_mean)
    acc_s = _finite_or_none(acc_std)
    f1_s = _finite_or_none(f1_std)
    limits = {
        "acc_limit": acc_limit,
        "f1_limit": f1_limit,
        "acc_std_limit": acc_std_limit,
        "f1_std_limit": f1_std_limit,
    }
    acc_l = _finite_or_none(acc_limit)
    f1_l = _finite_or_none(f1_limit)
    acc_s_l = _finite_or_none(acc_std_limit)
    f1_s_l = _finite_or_none(f1_std_limit)
    gates_ok = bool(optimizer_valid) and bool(decode_ok) and bool(apply_ok) and bool(eval_ok)
    metrics_known = None not in (acc, f1, acc_s, f1_s, acc_l, f1_l, acc_s_l, f1_s_l)
    metric_pass = bool(
        metrics_known
        and acc >= acc_l
        and f1 >= f1_l
        and acc_s <= acc_s_l
        and f1_s <= f1_s_l
    )
    diagnostic_feasible = bool(gates_ok and metric_pass) if metrics_known else None
    formal_available = _thresholds_known(threshold_source, limits)
    feasible = bool(diagnostic_feasible) if formal_available and diagnostic_feasible is not None else None

    strict_z_value = float(strict_z)
    acc_lower = None if acc is None or acc_s is None else float(acc - strict_z_value * acc_s)
    f1_lower = None if f1 is None or f1_s is None else float(f1 - strict_z_value * f1_s)
    strict_pass = bool(
        feasible
        and acc_lower is not None
        and f1_lower is not None
        and acc_l is not None
        and f1_l is not None
        and acc_lower >= acc_l
        and f1_lower >= f1_l
    )
    strict_feasible = strict_pass if feasible is not None else None

    reason = None
    if not formal_available:
        reason = "threshold source unknown or incomplete; formal feasible is unavailable"

    return {
        "schema": "blb_final_eval_feasibility_v1",
        "optimizer_valid": bool(optimizer_valid),
        "decode_ok": bool(decode_ok),
        "apply_ok": bool(apply_ok),
        "eval_ok": bool(eval_ok),
        "threshold_source": str(threshold_source or "unknown"),
        "loss_mean": _finite_or_none(loss_mean),
        "loss_std": _finite_or_none(loss_std),
        "loss_is_hard_constraint": False,
        "acc_mean": acc,
        "f1_mean": f1,
        "acc_std": acc_s,
        "f1_std": f1_s,
        "acc_limit": acc_l,
        "f1_limit": f1_l,
        "acc_std_limit": acc_s_l,
        "f1_std_limit": f1_s_l,
        "feasible": feasible,
        "diagnostic_feasible": diagnostic_feasible,
        "formal_feasible_unavailable_reason": reason,
        "strict_z": strict_z_value,
        "strict_z_source": "default" if strict_z_value == 1.0 else "explicit",
        "acc_lower": acc_lower,
        "f1_lower": f1_lower,
        "strict_feasible": strict_feasible,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# BLB Final Eval Feasibility",
        "",
        f"- feasible: `{report.get('feasible')}`",
        f"- diagnostic_feasible: `{report.get('diagnostic_feasible')}`",
        f"- strict_feasible: `{report.get('strict_feasible')}`",
        f"- strict_z: `{report.get('strict_z')}`",
        f"- threshold_source: `{report.get('threshold_source')}`",
        "- loss 非硬约束: `true`",
        "",
        "| item | value |",
        "|---|---:|",
        f"| acc_mean | {report.get('acc_mean')} |",
        f"| f1_mean | {report.get('f1_mean')} |",
        f"| acc_std | {report.get('acc_std')} |",
        f"| f1_std | {report.get('f1_std')} |",
        f"| acc_limit | {report.get('acc_limit')} |",
        f"| f1_limit | {report.get('f1_limit')} |",
        f"| acc_std_limit | {report.get('acc_std_limit')} |",
        f"| f1_std_limit | {report.get('f1_std_limit')} |",
        "",
    ])


def write_final_eval_feasibility_report(
        report: Mapping[str, Any],
        output_dir: str | Path,
        *,
        stem: str = "final_eval_feasibility",
        ) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
