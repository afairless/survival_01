#!/usr/bin/env python3
"""Legacy entry point for the Kottas (2002) Section 5.1 example.

Scope
-----
This module is kept for history only.  It used to carry its own copy of the
two-lognormal generator, a ``_demo()``-style script and a ``main()`` that ended
in an undefined ``theta`` behind a ``if __name__ == "__main__()":`` guard, so it
had never run.  Plan decision 10 replaces that body with a thin wrapper:

* the data generator is :class:`src.paper_data.TwoLognormalMix` (re-exported
  here so old imports keep working);
* the run itself is :func:`src.paper_examples.example_5_1`, which does the
  sampling, the prior band and Figures 1-3 under the git-ignored
  ``docs/research/figures/`` directory.

Non-goals
---------
No model code, no plotting and no data generation lives here; nothing in the
package imports this module.  Use the driver directly for anything new:
``python -m src.paper_examples --example 5.1``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.paper_data import TwoLognormalMix
from src.paper_examples import (
    DEFAULT_FIGURE_DIR,
    PRESETS,
    SEED_5_1,
    DemoResult,
    Preset,
    example_5_1,
)

__all__ = ["TwoLognormalMix", "build_parser", "main", "run"]


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the legacy script (a subset of the driver's CLI)."""
    parser = argparse.ArgumentParser(
        prog="kottas_2002_5_1",
        description=(
            "Legacy entry point for Kottas (2002) Section 5.1.  Delegates to "
            "src.paper_examples.example_5_1 and writes Figures 1-3."
        ),
    )
    parser.add_argument(
        "--preset",
        default="quick",
        choices=tuple(PRESETS),
        help="run length preset (default: quick)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_5_1,
        help=f"chain-group base seed (default: {SEED_5_1})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"figure directory (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="chain pool size (default: min(n_chains, cpu count))",
    )
    return parser


def run(
    *,
    preset: str | Preset = "quick",
    seed: int | None = SEED_5_1,
    out_dir: Path | str = DEFAULT_FIGURE_DIR,
    max_workers: int | None = None,
) -> DemoResult:
    """Run the Section 5.1 example through the driver and report its figures."""
    result = example_5_1(
        preset=preset,
        seed=seed,
        out_dir=out_dir,
        max_workers=max_workers,
    )
    print(f"example {result.label}: {len(result.fits)} fit(s)")
    for path in result.figures:
        print(f"wrote {path}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns 0 on success; argparse exits 2 on bad input."""
    args = build_parser().parse_args(argv)
    run(
        preset=args.preset,
        seed=args.seed,
        out_dir=args.out,
        max_workers=args.max_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
