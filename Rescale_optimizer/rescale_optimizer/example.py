#!/usr/bin/env python3
"""
rescale_optimizer/example.py

Demo entry point for the 4-stage rescale optimization pipeline.

All problem data — graph topology, amplitude profiles, SNR
requirements, noise table, per-cut-point amplitude budgets, plaintext-
leaf scaling factors, cost / optimization parameters — lives in a JSON
config.  This script simply:

    1. loads the JSON via :func:`rescale_optimizer.load_graph_from_json`,
    2. calls :func:`optimize_rescale`,
    3. prints the result.

Usage
-----

::

    python -m rescale_optimizer.example [config]
                                        [--log-file PATH]
                                        [--debug | -v]
    
    python3 -m rescale_optimizer.example "configs/block1.json" 2>&1 | tail -25; done

``config`` is a path to a JSON config; if omitted, defaults to
``<repo>/configs/example_graph.json``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rescale_optimizer import (
    load_graph_from_json,
    optimize_rescale,
    print_result,
)
from rescale_optimizer.utils import setup_logging


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "example_graph.json"
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rescale_optimizer.example",
        description="Run the 4-stage rescale optimizer on a JSON config.",
    )
    p.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=f"Path to JSON config (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional path to also write logs to this file.",
    )
    p.add_argument(
        "--debug", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=level, log_file=args.log_file)

    print("=" * 60)
    print(" Rescale Optimizer — Example (4-stage pipeline, JSON config)")
    print(f" config: {cfg_path}")
    if args.log_file:
        print(f" log file: {args.log_file}")
    print("=" * 60)

    graph, opt_config, _amp_budgets = load_graph_from_json(cfg_path)
    result = optimize_rescale(graph, opt_config)
    print()
    print_result(result)


if __name__ == "__main__":
    main()
