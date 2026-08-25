"""In-process Rescale planning and baseline materialization."""

from pathlib import Path


RESCALE_CONFIG_ROOT = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "preparation"
    / "rescale"
)
