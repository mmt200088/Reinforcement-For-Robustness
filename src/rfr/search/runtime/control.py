"""Shared progress and graceful-stop controls for production training."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import signal
import time
from typing import Callable, Iterable


STOP_FLAG_FILENAME = "STOP_RL"
PROGRESS_BOX_PPO_INTERVAL = 5

_STOP_STATE = {"requested": False, "installed": False, "previous": None}


def request_graceful_stop() -> None:
    _STOP_STATE["requested"] = True


def reset_graceful_stop_state() -> None:
    _STOP_STATE["requested"] = False


def graceful_stop_requested(stop_flag_path: str | os.PathLike | None = None) -> bool:
    if stop_flag_path and Path(stop_flag_path).is_file():
        _STOP_STATE["requested"] = True
    return bool(_STOP_STATE["requested"])


def install_graceful_stop_handler(
    log_fn: Callable[[str], None] | None = None,
) -> None:
    if _STOP_STATE["installed"]:
        return

    def handler(_signum, _frame):
        if not _STOP_STATE["requested"]:
            _STOP_STATE["requested"] = True
            if log_fn is not None:
                log_fn(
                    "\n  Graceful stop requested; the current checkpoint "
                    "boundary will complete before exit."
                )
            return
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt

    try:
        _STOP_STATE["previous"] = signal.signal(signal.SIGINT, handler)
        _STOP_STATE["installed"] = True
    except (ValueError, OSError):
        pass


def uninstall_graceful_stop_handler() -> None:
    if not _STOP_STATE["installed"]:
        return
    try:
        previous = _STOP_STATE.get("previous")
        signal.signal(
            signal.SIGINT,
            previous if previous is not None else signal.SIG_DFL,
        )
    except (ValueError, OSError):
        pass
    _STOP_STATE["installed"] = False


def consume_stop_flag(stop_flag_path: str | os.PathLike | None) -> None:
    if not stop_flag_path:
        return
    try:
        Path(stop_flag_path).unlink(missing_ok=True)
    except OSError:
        pass


def format_eta_finish(eta_seconds: float) -> str:
    finish = time.time() + max(float(eta_seconds), 0.0)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish))


def log_box(
    log_fn: Callable[[str], None],
    lines: Iterable[object],
    indent: str = "  ",
    min_inner_width: int = 8,
) -> None:
    rendered = [str(line) for line in lines]
    width = max((len(line) for line in rendered), default=0)
    width = max(width, int(min_inner_width))
    border = "─" * (width + 4)
    log_fn(f"{indent}╭{border}╮")
    for line in rendered:
        log_fn(f"{indent}│ {line.ljust(width)} │")
    log_fn(f"{indent}╰{border}╯")


def write_warning_report(
    warning_file: str | os.PathLike,
    warnings: Iterable[dict],
    *,
    stage_label: str = "",
) -> None:
    rows = list(warnings)
    with Path(warning_file).open("w", encoding="utf-8") as handle:
        handle.write("=== RL reward-drop warnings ===\n")
        handle.write(f"stage: {stage_label}\n")
        handle.write(f"updated_at: {datetime.datetime.now().isoformat()}\n")
        handle.write(f"warning_count: {len(rows)}\n\n")
        for index, warning in enumerate(rows, 1):
            start, end = warning["episode_range"]
            handle.write(f"warning {index}\n")
            handle.write(f"  type: {warning['type']}\n")
            handle.write(
                "  drop: "
                f"{warning['drop']:.4f} "
                f"(previous={warning['prev_avg']:.4f}, "
                f"current={warning['curr_avg']:.4f})\n"
            )
            if "threshold" in warning:
                handle.write(f"  threshold: {float(warning['threshold']):.4f}\n")
            handle.write(f"  episodes: {start}-{end}\n")
            for detail_file in warning.get("detail_files", []):
                handle.write(f"  detail: details/{detail_file}\n")
            handle.write("\n")


__all__ = [
    "PROGRESS_BOX_PPO_INTERVAL",
    "STOP_FLAG_FILENAME",
    "consume_stop_flag",
    "format_eta_finish",
    "graceful_stop_requested",
    "install_graceful_stop_handler",
    "log_box",
    "request_graceful_stop",
    "reset_graceful_stop_state",
    "uninstall_graceful_stop_handler",
    "write_warning_report",
]
