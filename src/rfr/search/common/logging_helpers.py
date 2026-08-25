"""Unified logger access for BLB Stage-2 RL modules.

Replaces the mix of ``print()`` / bare ``logging.info`` / ``self.log`` /
``evaluator.log`` patterns scattered throughout the project.

Usage::

    from rfr.search.common.logging_helpers import get_logger
    log = get_logger(__name__)
    log.info("training starts: episodes=%d", n)
    log.warning("skipped resume: %s", reason)

Why a dedicated helper instead of plain ``logging.getLogger``?

1. **First call configures the root** — so a script that just calls
   ``get_logger`` works without anyone setting up handlers. Idempotent;
   later calls don't reset anything.
2. **Environment-driven level** — ``BLB_LOG_LEVEL=DEBUG`` switches
   verbosity without code edits.
3. **Console + file output** — when ``BLB_LOG_FILE=/path/to.log`` is set,
   logs ALSO go to that file (UTF-8, append-only).
4. **Structured / JSON mode** — ``BLB_LOG_JSON=1`` switches the formatter
   to one-line JSON, suitable for shipping to ELK / Datadog / jq.
5. **CJK-safe console** — under Windows GBK, non-GBK chars in messages
   fall back instead of crashing stdout.

The helper does not replace ``evaluator.log(...)``, which owns its own training
file handles. Modules without that evaluator should use ``get_logger``.

Migration tips
--------------
- ``print(f"foo {x}")``                  →  ``log.info("foo %s", x)``
- ``self.log(f"foo {x}")`` (in classes)  →  keep, or switch to module logger
- ``logger.info(f"foo {x}")``            →  use lazy ``%s`` form for cheaper formatting
- For tail-friendly progress lines, use ``log.info`` (INFO level streams to console)
- For internals / RNG state / per-step trace, use ``log.debug`` (silent by default)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Mapping


_ENV_LEVEL = "BLB_LOG_LEVEL"
_ENV_FILE = "BLB_LOG_FILE"
_ENV_JSON = "BLB_LOG_JSON"

_JSON_LOG_ENCODER = json.JSONEncoder(ensure_ascii=False, default=str)
_INITIALIZED = False


class _CJKSafeStreamHandler(logging.StreamHandler):
    """StreamHandler that survives non-encodable characters under GBK consoles.

    Windows console encoding is often GBK; emoji / box drawing chars there
    raise ``UnicodeEncodeError``. We catch and re-emit a ``?``-replaced
    version so a single bad char doesn't kill the run.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                enc = getattr(self.stream, "encoding", None) or "utf-8"
                safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
                self.stream.write(safe + self.terminator)
                self.flush()
            except Exception:

                pass


class _JSONFormatter(logging.Formatter):
    """One-line JSON formatter for shipping to log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _JSON_LOG_ENCODER.encode(payload)


def _resolve_level() -> int:
    raw = os.environ.get(_ENV_LEVEL, "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def _is_json_mode() -> bool:
    return os.environ.get(_ENV_JSON, "").strip().lower() in ("1", "true", "yes", "on")


def _configure_root_once() -> None:
    """First-call init. Subsequent calls are a no-op."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger("blb_stage2_rl")
    root.setLevel(_resolve_level())


    root.propagate = False

    for h in list(root.handlers):
        root.removeHandler(h)

    if _is_json_mode():
        formatter: logging.Formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-5s %(name)s · %(message)s",
            datefmt="%H:%M:%S",
        )


    stream_handler = _CJKSafeStreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


    log_file = os.environ.get(_ENV_FILE, "").strip()
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as exc:
            root.warning("could not attach BLB_LOG_FILE=%s handler: %s", log_file, exc)

    _INITIALIZED = True


def get_logger(name: str = "blb_stage2_rl") -> logging.Logger:
    """Return a configured logger.

    Args:
        name: usually ``__name__`` (so messages tag the source module).
              Defaults to the root BLB logger when called bare.

    The returned logger inherits the root config (level, handlers,
    formatter). Subsequent calls with the same ``name`` return the same
    Logger instance (Python's logging keeps a global cache).
    """
    _configure_root_once()


    if not name.startswith("blb_stage2_rl"):
        if name == "__main__":
            name = "blb_stage2_rl.main"
        else:
            name = f"blb_stage2_rl.{name}"
    return logging.getLogger(name)


def bind_evaluator_log(evaluator, level: int = logging.INFO):
    """Adapt an evaluator-style ``log(message)`` callback to stdlib logging.

    Modules that accept a callable ``log_fn`` can use this adapter while their
    writes flow through the unified logger.
    """
    logger = get_logger(getattr(evaluator, "__class__", type(evaluator)).__name__)

    def _log(message: str, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            try:
                logger.log(level, message, *args, **kwargs)
                return
            except Exception:
                pass
        logger.log(level, str(message))

    return _log


__all__ = ["get_logger", "bind_evaluator_log"]
