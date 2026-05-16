"""Strict-mode helper: replace silent ``except Exception: pass`` with a
single shared guard that respects ``BLB_STRICT`` environment variable.

Background
----------
``runner.py`` alone has ~50 ``except Exception`` blocks. Most are
defensive (writing diagnostic artifacts, optional refresh, etc.) and
shouldn't crash training. But during debugging it's invaluable to
**actually see those errors**.

Usage::

    from blb_stage2_rl.strict import swallow

    @swallow(reason="optional diagnostic write")
    def _flush_some_artifact():
        ...

or inline::

    try:
        do_optional_thing()
    except Exception as exc:
        if is_strict():
            raise
        log.warning("optional_thing failed: %s", exc)

Environment::

    BLB_STRICT=1     re-raise everything (debug mode)
    BLB_STRICT=0     swallow (default, production behavior)

Rationale: most ``except Exception`` blocks in this codebase have a real
swallow-or-die decision behind them. The point of strict mode is to
**reveal** them on demand without forcing every caller to bake the env
check into their own try/except.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Optional, TypeVar


_ENV = "BLB_STRICT"
_LOGGER = logging.getLogger("blb_stage2_rl.strict")

F = TypeVar("F", bound=Callable[..., Any])


def is_strict() -> bool:
    """Return True iff BLB_STRICT env var is set to a truthy value."""
    raw = os.environ.get(_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def swallow(
        *,
        reason: str,
        log_fn: Optional[Callable[[str], None]] = None,
        default: Any = None,
        ) -> Callable[[F], F]:
    """Decorator: wrap a function so it swallows exceptions in normal mode
    and re-raises them in strict mode.

    Args:
        reason: short description used in the swallow log message. Required
                so we never have anonymous swallows.
        log_fn: optional callable; defaults to module logger's warning.
        default: value to return when an exception is swallowed.

    Use this for **optional / best-effort** side-effects (writing a
    diagnostic file, updating a status board, registering to an external
    log). Don't use it on code paths that produce values consumers depend
    on.
    """
    def _decorator(fn: F) -> F:
        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if is_strict():
                    raise
                emit = log_fn or _LOGGER.warning
                emit(f"[strict-swallow] {fn.__qualname__}: {reason} → {type(exc).__name__}: {exc}")
                return default
        return _wrapper  # type: ignore[return-value]
    return _decorator


class strict_guard:
    """Context manager equivalent of :func:`swallow` for inline blocks.

    Usage::

        with strict_guard(reason="optional flush", log_fn=log):
            recorder.flush_periodic()

    Swallows exceptions in non-strict mode; re-raises in strict mode.
    """

    def __init__(
            self,
            *,
            reason: str,
            log_fn: Optional[Callable[[str], None]] = None,
            ) -> None:
        self._reason = str(reason)
        self._log_fn = log_fn or _LOGGER.warning

    def __enter__(self) -> "strict_guard":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            return False
        if is_strict():
            return False  # re-raise
        try:
            self._log_fn(
                f"[strict-swallow] {self._reason} → "
                f"{exc_type.__name__}: {exc_val}"
            )
        except Exception:
            pass
        return True   # swallow


__all__ = ["is_strict", "swallow", "strict_guard"]
