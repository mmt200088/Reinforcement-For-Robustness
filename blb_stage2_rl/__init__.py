"""Layerwise robust Stage-2 search and strict candidate selection."""

__all__ = ["BLBStage2RLRunner"]


def __getattr__(name):
    if name == "BLBStage2RLRunner":
        from .training import BLBStage2RLRunner

        return BLBStage2RLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
