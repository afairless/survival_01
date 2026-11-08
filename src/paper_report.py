"""Report payloads for the artifact-backed Kottas (2002) replication run.

This module is the **recomputation** layer of the replication pipeline
(``docs/research/dp_gibbs_replication_artifacts_plan.md`` §6.3).  It consumes a
loaded :class:`src.paper_examples.DemoResult` and turns it into the numbers the
report tables quote:

* :func:`summarize_5_1` -- the Figure 1-3 quantities (``nu``/``n*`` posteriors,
  ``S(t)``, the density integral and modes, the smoothed hazard's change points
  and band widths);
* :func:`summarize_5_2` -- the Figure 4-5 quantities (prior-curve agreement,
  the A/B survival and hazard contrasts, ``Pr(eta_A > eta_B)`` and the hazard
  ratio at the two reported times);
* :func:`diagnostics` -- split R-hat and bulk ESS for ``nu``, ``n*`` and the
  five-time survival trace, computed on the full draws **and** on the old
  note's first-500-draw slice;
* :func:`delta_vs_baseline` -- signed differences against the committed
  ``docs/research/dp_gibbs_results_baseline.json``, with the rows that cannot
  match by construction marked ``not_comparable``;
* :func:`evaluate_acceptance` -- one pass/fail row per acceptance clause,
  computed from the summary with the thresholds of :data:`ACCEPTANCE_TARGETS`;
* :data:`ACCEPTANCE_TARGETS` -- the target/label table mirrored from
  ``tests/test_paper_5_1.py`` and ``tests/test_paper_5_2.py``;
* :func:`build_payload` -- full-draw and first-500 summaries, diagnostics,
  acceptance and baseline delta, gathered into one JSON-safe payload;
* :func:`render_markdown` -- the generated tables (``report/tables.md``), and
  :func:`write_report_artifacts` -- the writer for ``summary/``, ``report/``
  and the re-rendered ``figures/``;
* :func:`main` -- ``python -m src.paper_report --from-artifacts <section>
  --write``, the runbook's single report entry point.

The first-500 slice is not a second fit.  It is the same stored draws truncated
after 500 per chain: the chains are rebuilt with
``dataclasses.replace(chain.settings, n_save=500)`` (required by
``DPChains.__post_init__``), while the summary arrays are sliced from the
stored :class:`~src.dp_functionals.Functionals`.  Those arrays were computed
with the **full trace's** ``J_common``/``J_prior``, so the slice reuses that
truncation level instead of recomputing a smaller one from 500 draws.  This is
what makes the first-500 column a like-for-like replay of
``docs/research/dp_gibbs_results.md`` (plan §7 Step 2 and §9).

Scope
-----
The payload functions only read containers: they never sample and never
recompute a functional from the chains.  The report writer adds the rendering
half on top: ``summary/stats.json``, ``rhat_ess.csv``, ``quantiles.csv`` and
``acceptance.json``, the generated ``report/tables.md`` and the section's
``figures/``.  Plan R7 makes this stage the *only* figure writer, and it only
ever writes inside the section directory it is given.

``matplotlib`` is imported lazily by ``_pyplot`` and only inside
:func:`write_report_artifacts`'s figure step, so importing this module stays
cheap and headless (the same contract as ``src/paper_examples.py``).  Figure
bytes are deterministic: the recipes pin ``dpi``, ``format`` and the PNG
metadata (no creation time, no matplotlib version string), so re-rendering the
same functionals twice yields byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma as gamma_dist

from src.dp_functionals import Functionals, compare
from src.dp_gibbs import DPChains, ess, r_hat
from src.paper_artifacts import ArtifactError, _jsonable, load_demo_result
from src.paper_examples import (
    REPO_ROOT,
    DemoResult,
    Fit,
    figure_1,
    figure_2,
    figure_3,
    figure_4,
    figure_5,
    smooth_hazard,
)

__all__ = [
    "ACCEPTANCE_TARGETS",
    "BASELINE_PATH",
    "BASELINE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SLICE_DRAWS",
    "build_payload",
    "delta_vs_baseline",
    "diagnostics",
    "evaluate_acceptance",
    "load_baseline",
    "main",
    "manifest_summary",
    "render_markdown",
    "slice_result",
    "summarize_5_1",
    "summarize_5_2",
    "write_report_artifacts",
]

#: The number of stored draws per chain in the old note's probe (plan §4).
SLICE_DRAWS = 500

#: On-disk schema of the committed baseline JSON.
BASELINE_SCHEMA_VERSION = 1

#: On-disk schema of the report payload (``summary/stats.json``) and of the
#: generated ``summary/acceptance.json``.
REPORT_SCHEMA_VERSION = 1

#: Directory names the report owns, relative to the section directory.
_SUMMARY_DIR = "summary"
_REPORT_DIR = "report"
_FIGURES_DIR = "figures"

#: Name of the run-level manifest the report reads when it already exists.
_MANIFEST_NAME = "MANIFEST.json"

#: Cell used for a value the artifacts do not hold (a bandless fit, no
#: manifest yet).  Kept ASCII so the generated tables stay plain text.
_MISSING = "n/a"

#: Sentinel for "the caller did not pass a manifest summary".
_UNSET = object()

#: Default location of the baseline extracted from ``dp_gibbs_results.md``.
BASELINE_PATH = REPO_ROOT / "docs" / "research" / "dp_gibbs_results_baseline.json"

#: Target table mirrored from the committed acceptance tests (plan §6.3).
#:
#: The tests' constants are the authority: the report must compute the same
#: quantities from the same windows and the same amendment thresholds (A1-A4).
#: ``tests/test_paper_report.py`` asserts every entry here against the matching
#: constant of ``tests/test_paper_5_1.py`` / ``tests/test_paper_5_2.py``, so a
#: threshold edited in one file and not the other fails the default gate.
ACCEPTANCE_TARGETS: dict[str, dict[str, Any]] = {
    "5.1": {
        "POPULATION_MEDIAN": 1.1727,
        "SECOND_DENSITY_MODE": 3.26,
        "FIRST_MODE_NEIGHBOUR": 2.4,
        "HAZARD_CHANGE_POINTS": (1.022, 2.366, 4.454),
        "HAZARD_CHANGE_TOLERANCES": (0.25, 0.25, 0.35),
        "NU_PRIOR_MEANS": (2.0 / 0.9, 20.0, 60.0),
        "N_STAR_RANGE": (3, 15),
        "N_STAR_MIN_PROBABILITIES": (0.85, 0.85, 0.70),
        "N_STAR_MEAN_RANGE": (3.0, 15.0),
        "TAIL_DOMINANCE_FACTOR": 2.0,
        "SURVIVAL_AT_MEDIAN_RANGE": (0.42, 0.58),
        "DENSITY_MASS_RANGE": (0.95, 1.05),
        "BULK_UPPER_TIME": 4.0,
        "HAZARD_BAND_RATIO": 1.5,
        "HAZARD_BAND_TIMES": (2.5, 4.5),
        "HAZARD_SMOOTHING_BANDWIDTH": 5.0,
        "HAZARD_WINDOW": (0.0, 5.0),
    },
    "5.2": {
        "REMISSION_WINDOW": (0.0, 20.0),
        "B_HAZARD_PEAK_WINDOW": (10.0, 20.0),
        "HAZARD_BAND_TIMES": (5.0, 25.0),
        "HAZARD_SMOOTHING_BANDWIDTH": 5.0,
        "PRIOR_AGREEMENT_FRACTION": 0.95,
        "ETA_PROBABILITY_RANGE": (0.85, 0.96),
        "DIFFERENCE_MIN_PROBABILITY": 0.50,
        "DIFFERENCE_MEAN_PROBABILITY": 0.75,
        "RATIO_DECLINE_FACTOR": 1.30,
        "RATIO_ORDER_PROBABILITY": 0.75,
    },
}


# ---------------------------------------------------------------------------
# Small numerical helpers shared by both sections
# ---------------------------------------------------------------------------
def _central_interval(
    values: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Central ``level`` interval along axis 0, ignoring non-finite entries.

    ``h`` is ``inf`` where ``S`` underflows and a per-draw hazard ratio is
    ``nan`` where it is undefined, so the interval drops those entries rather
    than propagating them.  This is the same rule the figure recipes use.
    """
    array = np.where(np.isfinite(values), values, np.nan)
    tail = 100.0 * (1.0 - level) / 2.0
    lo = np.nanpercentile(array, tail, axis=0)
    hi = np.nanpercentile(array, 100.0 - tail, axis=0)
    return lo, hi


def _mean_curve(values: np.ndarray) -> np.ndarray:
    """Column means of a ``(n_draws, n_grid)`` array, ignoring non-finite."""
    return np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0)


def _nearest(t_grid: np.ndarray, value: float) -> int:
    """Index of the grid point closest to ``value``."""
    return int(np.argmin(np.abs(np.asarray(t_grid, dtype=np.float64) - value)))


def _window_mask(t_grid: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Strictly interior grid points of ``window``."""
    return (t_grid > window[0]) & (t_grid < window[1])


def _sign_change_times(
    t_grid: np.ndarray, curve: np.ndarray, window: tuple[float, float]
) -> list[float]:
    """Times of the sign changes of ``diff(curve)`` inside ``window``."""
    mask = (t_grid > window[0]) & (t_grid < window[1])
    t = t_grid[mask]
    differences = np.diff(curve[mask])
    signs = np.sign(differences)
    keep = signs != 0.0
    t = t[1:][keep]
    signs = signs[keep]
    changes = np.flatnonzero(np.diff(signs) != 0.0)
    return [float(0.5 * (t[i] + t[i + 1])) for i in changes]


def _pooled(fit: Fit, name: str) -> np.ndarray:
    """Concatenate one stored trace over the chains of a fit."""
    return np.concatenate([np.asarray(getattr(chain, name)) for chain in fit.chains])


def _floats(values: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into JSON-safe Python values."""
    return _jsonable(values)


def _difference(observed: Any, baseline: Any) -> Any:
    """Signed ``observed - baseline`` for a numeric scalar/sequence pair.

    A row whose shape changed has no signed difference.  That happens for
    ``figure_3.change_points`` when a run finds a different *number* of hazard
    change points from the probe; returning ``None`` keeps the row visible in
    the delta table (the report can flag it) instead of crashing the report on
    a finding that is exactly what the table exists to surface.
    """
    if observed is None or baseline is None:
        return None
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape:
        return None
    return _floats(left - right)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into dotted paths; leave leaves untouched."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(item, path))
        return out
    return {prefix: value}


def _not_comparable(path: str, keys: Sequence[str]) -> bool:
    """Whether ``path`` is one of the baseline's ``not_comparable`` rows."""
    return any(path == key or path.startswith(f"{key}.") for key in keys)


# ---------------------------------------------------------------------------
# Slicing: the old note's first 500 draws, with the full trace's J
# ---------------------------------------------------------------------------
def _as_slice_draws(n_draws: object) -> int:
    """Validate a per-chain draw count."""
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise ValueError(f"n_draws must be an integer, got {n_draws!r}")
    value = int(n_draws)
    if value < 1:
        raise ValueError(f"n_draws must be at least 1, got {value}")
    return value


def _slice_chain(chain: DPChains, n_draws: int) -> DPChains:
    """Take the first ``n_draws`` stored draws of one chain.

    ``DPChains.__post_init__`` requires ``theta.shape[0] == settings.n_save``,
    so the settings are replaced together with the arrays.  The rest of the
    chain (data, prior, provenance) is shared as a shallow copy.
    """
    take = min(n_draws, chain.n_draws)
    settings = replace(chain.settings, n_save=take)
    return replace(
        chain,
        settings=settings,
        theta=chain.theta[:take],
        nu=chain.nu[:take],
        gamma=chain.gamma[:take],
        phi=chain.phi[:take],
        n_star=chain.n_star[:take],
        log_post=chain.log_post[:take],
    )


def _slice_functionals(func: Functionals, indices: np.ndarray) -> Functionals:
    """Slice the rows of a stored functional array, keeping its meta verbatim."""
    return replace(
        func,
        S=func.S[indices],
        F=func.F[indices],
        f=func.f[indices],
        h=func.h[indices],
        meta=dict(func.meta),
    )


def _slice_fit(fit: Fit, n_draws: int) -> Fit:
    """Slice one fit's chains and its chain-stacked functionals together.

    ``posterior_functionals`` stacks draws chain by chain, so the first
    ``n_draws`` of every chain are not the first ``n_chains * n_draws`` rows.
    The index list below reconstructs the right rows and therefore keeps the
    ``J_common``/``J_prior`` that produced the full arrays.
    """
    chains = [_slice_chain(chain, n_draws) for chain in fit.chains]
    indices: list[int] = []
    offset = 0
    for chain in fit.chains:
        take = min(n_draws, chain.n_draws)
        indices.extend(range(offset, offset + take))
        offset += chain.n_draws
    if fit.functionals.n_draws != offset:
        raise ValueError(
            "functionals hold "
            f"{fit.functionals.n_draws} draws but the fit's chains hold {offset}; "
            "the functional rows must be stacked chain by chain"
        )
    functionals = _slice_functionals(
        fit.functionals, np.asarray(indices, dtype=np.intp)
    )

    band = fit.prior_band
    if band is not None:
        take = min(n_draws, band.n_draws)
        band = _slice_functionals(band, np.arange(take, dtype=np.intp))

    return replace(fit, chains=chains, functionals=functionals, prior_band=band)


def slice_result(result: DemoResult, n_draws: int = SLICE_DRAWS) -> DemoResult:
    """Return a copy of ``result`` truncated to the first ``n_draws`` per chain.

    The functionals are sliced from the stored arrays, so their ``meta`` still
    records the full trace's ``J_common``/``J_prior``.  That is deliberate:
    recomputing them from the truncated chains would reduce the truncation
    level and make the first-500 column a different estimator from the old
    note's probe (plan §7 Step 2).
    """
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    requested = _as_slice_draws(n_draws)
    if not result.fits:
        raise ValueError("result must hold at least one fit")

    fits = {key: _slice_fit(fit, requested) for key, fit in result.fits.items()}
    meta = dict(result.meta)
    if "n_save" in meta:
        try:
            meta["n_save"] = min(requested, int(meta["n_save"]))
        except TypeError, ValueError:
            pass
    return DemoResult(label=result.label, fits=fits, figures=(), meta=meta)


# ---------------------------------------------------------------------------
# Section 5.1 payload (Figures 1-3)
# ---------------------------------------------------------------------------
def _require_section(result: DemoResult, label: str) -> None:
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    if result.label != label:
        raise ValueError(
            f"summarize_{label.replace('.', '_')} needs a {label} result, "
            f"got label {result.label!r}"
        )
    if not result.fits:
        raise ValueError(f"the {label} result holds no fits")


def summarize_5_1(result: DemoResult) -> dict[str, Any]:
    """Compute the Figure 1-3 quantities from a Section 5.1 result.

    Every value is computed from the stored chains or functionals; no value is
    read from a results note.  Per-prior rows are lists in the result's fit
    order, which is the order of ``result.meta["prior_indices"]``.
    """
    _require_section(result, "5.1")
    targets = ACCEPTANCE_TARGETS["5.1"]
    fits = list(result.fits.values())
    labels = [fit.label for fit in fits]

    prior_means: list[float] = []
    posterior_means: list[float] = []
    posterior_95: list[list[float]] = []
    prior_widths: list[float] = []
    posterior_widths: list[float] = []
    posterior_99: list[float] = []
    n_star_means: list[float] = []
    n_star_probabilities: list[float] = []
    n_star_ranges: list[list[int]] = []
    learning: list[bool] = []

    survival_at_median: list[float] = []
    density_mass: list[float] = []
    second_mode_density: list[float] = []
    first_mode_density: list[float] = []
    change_points: list[list[float]] = []
    band_width_near: list[float] = []
    band_width_far: list[float] = []
    band_ratio: list[float] = []

    for fit in fits:
        nu = _pooled(fit, "nu")
        n_star = _pooled(fit, "n_star")
        prior_mean = float(fit.prior.a_nu / fit.prior.b_nu)
        mean = float(np.mean(nu))
        lo, hi = np.percentile(nu, [2.5, 97.5])
        prior_lo, prior_hi = gamma_dist.ppf(
            [0.025, 0.975], a=fit.prior.a_nu, scale=1.0 / fit.prior.b_nu
        )
        prior_width = float(prior_hi - prior_lo)
        posterior_width = float(hi - lo)

        prior_means.append(prior_mean)
        posterior_means.append(mean)
        posterior_95.append([float(lo), float(hi)])
        prior_widths.append(prior_width)
        posterior_widths.append(posterior_width)
        posterior_99.append(float(np.percentile(nu, 99.0)))
        n_star_means.append(float(np.mean(n_star)))
        low, high = targets["N_STAR_RANGE"]
        n_star_probabilities.append(float(np.mean((n_star >= low) & (n_star <= high))))
        n_star_ranges.append([int(np.min(n_star)), int(np.max(n_star))])
        learning.append(bool(posterior_width < prior_width))

        survival_curve = _mean_curve(fit.functionals.S)
        density_curve = _mean_curve(fit.functionals.f)
        position = _nearest(fit.t_grid, targets["POPULATION_MEDIAN"])
        survival_at_median.append(float(survival_curve[position]))
        density_mass.append(float(np.trapezoid(density_curve, fit.t_grid)))
        second_mode_density.append(
            float(density_curve[_nearest(fit.t_grid, targets["SECOND_DENSITY_MODE"])])
        )
        first_mode_density.append(
            float(density_curve[_nearest(fit.t_grid, targets["FIRST_MODE_NEIGHBOUR"])])
        )

        smoothed = smooth_hazard(
            fit.functionals.h,
            bandwidth=targets["HAZARD_SMOOTHING_BANDWIDTH"],
        )
        change_points.append(
            _sign_change_times(
                fit.t_grid, _mean_curve(smoothed), targets["HAZARD_WINDOW"]
            )
        )
        h_lo, h_hi = _central_interval(smoothed)
        widths = h_hi - h_lo
        near, far = targets["HAZARD_BAND_TIMES"]
        width_near = float(widths[_nearest(fit.t_grid, near)])
        width_far = float(widths[_nearest(fit.t_grid, far)])
        band_width_near.append(width_near)
        band_width_far.append(width_far)
        if width_near > 0.0:
            band_ratio.append(float(width_far / width_near))
        else:
            band_ratio.append(float("inf") if width_far > 0.0 else float("nan"))

    reference = fits[0]
    if reference.prior_band is None:
        prior_band_width: float | None = None
        posterior_band_width: float | None = None
        prior_band_is_wider: bool | None = None
    else:
        bulk = (reference.t_grid > 0.0) & (
            reference.t_grid <= targets["BULK_UPPER_TIME"]
        )
        posterior_lo, posterior_hi = _central_interval(reference.functionals.S)
        prior_lo, prior_hi = _central_interval(reference.prior_band.S)
        posterior_band_width = float(np.mean((posterior_hi - posterior_lo)[bulk]))
        prior_band_width = float(np.mean((prior_hi - prior_lo)[bulk]))
        prior_band_is_wider = bool(prior_band_width >= posterior_band_width)

    payload = {
        "label": result.label,
        "preset": result.meta.get("preset"),
        "prior_labels": labels,
        "figure_1": {
            "nu_prior_mean": prior_means,
            "nu_posterior_mean": posterior_means,
            "nu_posterior_95": posterior_95,
            "nu_prior_95_width": prior_widths,
            "nu_posterior_width": posterior_widths,
            "nu_posterior_99": posterior_99,
            "n_star_posterior_mean": n_star_means,
            "n_star_probability": n_star_probabilities,
            "n_star_observed_range": n_star_ranges,
            "learning": learning,
        },
        "figure_2": {
            "survival_at_median": survival_at_median,
            "density_mass": density_mass,
            "second_mode_density": second_mode_density,
            "first_mode_density": first_mode_density,
            "prior_band_width": prior_band_width,
            "posterior_band_width": posterior_band_width,
            "prior_band_is_wider": prior_band_is_wider,
            "band_prior_label": reference.label,
        },
        "figure_3": {
            "change_points": change_points,
            "band_width_near": band_width_near,
            "band_width_far": band_width_far,
            "band_ratio": band_ratio,
            "smoothing_bandwidth": targets["HAZARD_SMOOTHING_BANDWIDTH"],
        },
    }
    return _floats(payload)


# ---------------------------------------------------------------------------
# Section 5.2 payload (Figures 4-5)
# ---------------------------------------------------------------------------
def _arm_fits(result: DemoResult, arm: str) -> list[tuple[str, Fit]]:
    prefix = f"{arm}: "
    return [(key, fit) for key, fit in result.fits.items() if key.startswith(prefix)]


def _arm_b(result: DemoResult) -> Fit:
    selected = _arm_fits(result, "B")
    if not selected:
        raise ValueError("a Section 5.2 result must contain an arm-B fit")
    return selected[0][1]


def summarize_5_2(result: DemoResult) -> dict[str, Any]:
    """Compute the Figure 4-5 quantities from a Section 5.2 result.

    Arm A's prior-curve agreement is evaluated both directions at every grid
    point of the remission window; the A-versus-B comparison always uses the
    first arm-A fit (``Gamma(2, 0.9)`` in the driver) and the arm-B fit, as in
    the paper and in ``tests/test_paper_5_2.py``.
    """
    _require_section(result, "5.2")
    targets = ACCEPTANCE_TARGETS["5.2"]
    fits_a = [fit for _, fit in _arm_fits(result, "A")]
    if not fits_a:
        raise ValueError("a Section 5.2 result must contain an arm-A fit")
    fit_b = _arm_b(result)
    reference = fits_a[0]
    grid = np.asarray(reference.t_grid, dtype=np.float64)
    window = targets["REMISSION_WINDOW"]
    mask = _window_mask(grid, window)
    if not np.any(mask):
        raise ValueError(f"the grid has no point inside the window {window}")

    agreement: list[list[float]] = []
    for first in fits_a:
        mean = _mean_curve(first.functionals.S)
        row: list[float] = []
        for second in fits_a:
            lo, hi = _central_interval(second.functionals.S)
            inside = (mean[mask] >= lo[mask]) & (mean[mask] <= hi[mask])
            row.append(float(np.mean(inside)))
        agreement.append(row)
    off_diagonal = [
        agreement[i][j]
        for i in range(len(fits_a))
        for j in range(len(fits_a))
        if i != j
    ]
    agreement_min = min(off_diagonal) if off_diagonal else 1.0

    survival_difference = _mean_curve(reference.functionals.S) - _mean_curve(
        fit_b.functionals.S
    )
    survival_difference_min = float(np.min(survival_difference[mask]))

    smoothed_b = _mean_curve(
        smooth_hazard(
            fit_b.functionals.h,
            bandwidth=targets["HAZARD_SMOOTHING_BANDWIDTH"],
        )
    )
    finite = np.isfinite(smoothed_b)
    if not np.any(finite):
        raise ValueError("the arm-B smoothed hazard is nowhere finite")
    peak = int(np.nanargmax(np.where(finite, smoothed_b, np.nan)))
    peak_time = float(grid[peak])
    peak_value = float(smoothed_b[peak])
    above = smoothed_b[finite & (grid > targets["B_HAZARD_PEAK_WINDOW"][1])]
    above_max = float(np.max(above)) if above.size else None

    comparison = compare(reference.functionals, fit_b.functionals, grid)
    difference = np.asarray(comparison["difference"], dtype=np.float64)
    difference_mean = _mean_curve(difference)
    difference_probability = np.mean(difference > 0.0, axis=0)
    ratio = np.asarray(comparison["ratio"], dtype=np.float64)
    ratio_mean = _mean_curve(ratio)
    near, far = targets["HAZARD_BAND_TIMES"]
    near_index = _nearest(grid, near)
    far_index = _nearest(grid, far)
    mean_near = float(ratio_mean[near_index])
    mean_far = float(ratio_mean[far_index])
    if np.isfinite(mean_near) and np.isfinite(mean_far) and mean_far > 0.0:
        decline: float | None = float(mean_near / mean_far)
    else:
        decline = None
    paired = ratio[:, [near_index, far_index]]
    paired_finite = np.isfinite(paired).all(axis=1)
    order_probability = (
        float(np.mean(paired[paired_finite, 0] > paired[paired_finite, 1]))
        if np.any(paired_finite)
        else None
    )
    ratio_lo, ratio_hi = _central_interval(ratio)
    ratio_band = [
        [float(ratio_lo[near_index]), float(ratio_hi[near_index])],
        [float(ratio_lo[far_index]), float(ratio_hi[far_index])],
    ]

    payload = {
        "label": result.label,
        "preset": result.meta.get("preset"),
        "prior_labels": [fit.label for fit in fits_a],
        "figure_4": {
            "prior_agreement": agreement,
            "prior_agreement_min": agreement_min,
            "survival_difference_min": survival_difference_min,
            "arm_b_peak_time": peak_time,
            "arm_b_peak_value": peak_value,
            "arm_b_above_peak_max": above_max,
            "remission_window": list(window),
        },
        "figure_5": {
            "prob_eta_a_gt_eta_b": float(comparison["prob_eta_a_gt_eta_b"]),
            "difference_min": float(np.min(difference_mean[mask])),
            "difference_probability_range": [
                float(np.min(difference_probability[mask])),
                float(np.max(difference_probability[mask])),
            ],
            "difference_probability_mean": float(np.mean(difference_probability[mask])),
            "hazard_ratio_mean": [mean_near, mean_far],
            "hazard_ratio_decline": decline,
            "hazard_ratio_order_probability": order_probability,
            "hazard_ratio_band": ratio_band,
            "n_eta_above_grid": dict(
                comparison["meta"]["n_eta_above_grid"],
            ),
            "ratio_times": [near, far],
        },
    }
    return _floats(payload)


# ---------------------------------------------------------------------------
# Diagnostics: full draws and the old note's first-500-draw slice
# ---------------------------------------------------------------------------
def _diagnostics_block(chains: list[DPChains]) -> dict[str, Any]:
    """R-hat and ESS for ``nu``, ``n*`` and the five-time ``S`` trace."""
    if len(chains) < 2:
        raise ValueError("diagnostics need at least two chains per fit")
    block: dict[str, Any] = {"draws_per_chain": chains[0].n_draws}
    for name in ("nu", "n_star"):
        r = r_hat(chains, name)
        e = ess(chains, name)
        block[name] = {"r_hat": float(r[name]), "ess": float(e[name])}
    r_s_all = r_hat(chains, "S")
    e_s_all = ess(chains, "S")
    r_s = np.asarray(r_s_all["S"], dtype=np.float64)
    e_s = np.asarray(e_s_all["S"], dtype=np.float64)
    labels = np.asarray(r_s_all["t"], dtype=np.float64)
    block["S"] = {
        "r_hat": [float(value) for value in r_s],
        "ess": [float(value) for value in e_s],
        "t": [float(value) for value in labels],
    }
    return block


def diagnostics(
    result: DemoResult, *, slice_draws: int = SLICE_DRAWS
) -> dict[str, Any]:
    """Split R-hat and bulk ESS for every fit, full draws and first-500 slice.

    The slice rebuilds each chain with
    ``dataclasses.replace(chain.settings, n_save=500)`` (required by
    ``DPChains.__post_init__``) and reuses the **full trace's**
    ``J_common``/``J_prior``, which are recorded under each fit's ``j``.  The
    slice's own ``r_hat``/``ess`` are then the old note's probe values, not a
    statistic of a new fit (plan §7 Step 2 and §9).
    """
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    if not result.fits:
        raise ValueError("result must hold at least one fit")
    requested = _as_slice_draws(slice_draws)

    fits: dict[str, Any] = {}
    for key, fit in result.fits.items():
        if not fit.chains:
            raise ValueError(f"fit {key!r} holds no chains")
        take = min(requested, min(chain.n_draws for chain in fit.chains))
        sliced = [_slice_chain(chain, take) for chain in fit.chains]
        band = fit.prior_band
        fits[key] = {
            "full": _diagnostics_block(fit.chains),
            "first_500": _diagnostics_block(sliced),
            "j": {
                "J_common": fit.functionals.meta.get("J_common"),
                "J_prior": None if band is None else band.meta.get("J_prior"),
            },
        }

    return _floats(
        {
            "label": result.label,
            "slice_draws": requested,
            "fits": fits,
        }
    )


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------
def load_baseline(path: str | Path | None = None) -> dict[str, Any]:
    """Load a baseline JSON file (the committed note extract by default)."""
    target = BASELINE_PATH if path is None else Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read baseline {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"baseline {target} must hold a JSON object")
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"baseline {target} has schema_version "
            f"{payload.get('schema_version')!r}, expected {BASELINE_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("sections"), dict):
        raise ValueError(f"baseline {target} must hold a 'sections' object")
    return payload


def delta_vs_baseline(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Signed differences of a summary against the old note's probe values.

    ``summary`` is the payload of :func:`summarize_5_1` or
    :func:`summarize_5_2` (normally of :func:`slice_result` with 500 draws per
    chain, because the baseline is the note's first-500 probe).  ``baseline``
    may be the loaded mapping or a path; the committed
    :data:`BASELINE_PATH` is used when it is omitted.

    The returned ``rows`` mirror the baseline's dotted keys.  A row listed
    under the section's ``not_comparable`` entry is still differenced but is
    flagged, because its values cannot match by construction: the §5.1 prior
    band was drawn 200 times in the probe and 1 000 times in the run, and
    ``n_eta_above_grid`` is a count rather than a summary.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if baseline is None or isinstance(baseline, (str, Path)):
        payload = load_baseline(baseline)
    elif isinstance(baseline, Mapping):
        payload = dict(baseline)
    else:
        raise ValueError("baseline must be a mapping or a path")
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            "baseline has schema_version "
            f"{payload.get('schema_version')!r}, expected {BASELINE_SCHEMA_VERSION}"
        )

    label = summary.get("label")
    sections = payload.get("sections")
    if not isinstance(sections, Mapping) or label not in sections:
        raise ValueError(f"baseline has no section for label {label!r}")
    section = sections[label]
    if not isinstance(section, Mapping):
        raise ValueError(f"baseline section {label!r} must be an object")
    values = section.get("values")
    if not isinstance(values, Mapping):
        raise ValueError(f"baseline section {label!r} must hold a 'values' object")
    not_comparable = list(section.get("not_comparable") or [])

    observed = _flatten(summary)
    expected = _flatten(values)
    rows: dict[str, Any] = {}
    for path, base in expected.items():
        if path not in observed:
            raise ValueError(
                f"summary for {label} has no value at {path!r}; the baseline "
                "and the report payload cannot be compared"
            )
        value = observed[path]
        rows[path] = {
            "observed": _floats(value),
            "baseline": _floats(base),
            "delta": _difference(value, base),
            "not_comparable": _not_comparable(path, not_comparable),
        }

    return {
        "label": label,
        "source": payload.get("source"),
        "not_comparable": not_comparable,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Acceptance clauses
# ---------------------------------------------------------------------------
def _in_range(value: Any, bounds: Sequence[float]) -> bool:
    """Whether a finite scalar lies in the closed interval ``bounds``."""
    if value is None:
        return False
    number = float(value)
    return bool(np.isfinite(number)) and bounds[0] <= number <= bounds[1]


def _strictly_increasing(values: Sequence[Any]) -> bool:
    """Whether every entry is strictly below the next one."""
    numbers = [float(value) for value in values]
    return all(left < right for left, right in zip(numbers, numbers[1:], strict=False))


def _clause(
    clause: str,
    observed: Any,
    target: Any,
    passed: bool,
    description: str = "",
) -> dict[str, Any]:
    """One acceptance row: what was measured, what the paper requires, verdict.

    ``observed`` is converted to its JSON form so the row can be written to
    ``summary/acceptance.json`` and to the acceptance table unchanged.
    """
    return {
        "clause": clause,
        "description": description,
        "target": _floats(target),
        "observed": _floats(observed),
        "pass": bool(passed),
    }


def _n_star_floor(index: int) -> float:
    """The amendment-A1 probability floor of the ``index``-th prior."""
    floors = ACCEPTANCE_TARGETS["5.1"]["N_STAR_MIN_PROBABILITIES"]
    return float(floors[min(index, len(floors) - 1)])


def _clauses_5_1(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One acceptance row per Figure 1-3 clause of the paper's Section 5.1."""
    targets = ACCEPTANCE_TARGETS["5.1"]
    labels = list(summary["prior_labels"])
    figure_1 = summary["figure_1"]
    figure_2 = summary["figure_2"]
    figure_3 = summary["figure_3"]
    clauses: list[dict[str, Any]] = []

    means = figure_1["nu_posterior_mean"]
    clauses.append(
        _clause(
            "figure_1.nu_posterior_mean",
            means,
            "strictly increasing",
            _strictly_increasing(means),
            "posterior mean of nu increases with the prior mean",
        )
    )
    clauses.append(
        _clause(
            "figure_1.n_star_mean_increasing",
            figure_1["n_star_posterior_mean"],
            "strictly increasing",
            _strictly_increasing(figure_1["n_star_posterior_mean"]),
            "posterior mean of n* increases with the prior mean of nu",
        )
    )

    low, high = targets["N_STAR_MEAN_RANGE"]
    for index, label in enumerate(labels):
        mean = figure_1["n_star_posterior_mean"][index]
        clauses.append(
            _clause(
                f"figure_1.n_star_posterior_mean[{index}]",
                mean,
                [low, high],
                _in_range(mean, (low, high)),
                f"{label}: posterior mean of n* in [{low:g}, {high:g}]",
            )
        )
        probability = figure_1["n_star_probability"][index]
        floor = _n_star_floor(index)
        clauses.append(
            _clause(
                f"figure_1.n_star_probability[{index}]",
                probability,
                floor,
                _in_range(probability, (0.0, 1.0)) and probability >= floor,
                f"{label}: Pr({targets['N_STAR_RANGE'][0]} <= n* <= "
                f"{targets['N_STAR_RANGE'][1]}) >= {floor:g}",
            )
        )
        learning = bool(figure_1["learning"][index])
        clauses.append(
            _clause(
                f"figure_1.learning[{index}]",
                learning,
                True,
                learning,
                f"{label}: posterior 95% width below the prior's (learning)",
            )
        )

    upper = figure_1["nu_posterior_99"]
    factor = targets["TAIL_DOMINANCE_FACTOR"]
    clauses.append(
        _clause(
            "figure_1.nu_posterior_99",
            upper,
            f"increasing and >= {factor:g}x the narrowest prior's",
            _strictly_increasing(upper) and upper[-1] >= factor * upper[0],
            "the 99th percentile of nu follows the prior scale",
        )
    )

    survival_bounds = targets["SURVIVAL_AT_MEDIAN_RANGE"]
    mass_bounds = targets["DENSITY_MASS_RANGE"]
    for index, label in enumerate(labels):
        survival = figure_2["survival_at_median"][index]
        clauses.append(
            _clause(
                f"figure_2.survival_at_median[{index}]",
                survival,
                list(survival_bounds),
                _in_range(survival, survival_bounds),
                f"{label}: mean S({targets['POPULATION_MEDIAN']}) in "
                f"[{survival_bounds[0]:g}, {survival_bounds[1]:g}]",
            )
        )
        mass = figure_2["density_mass"][index]
        clauses.append(
            _clause(
                f"figure_2.density_mass[{index}]",
                mass,
                list(mass_bounds),
                _in_range(mass, mass_bounds),
                f"{label}: int f dt in [{mass_bounds[0]:g}, {mass_bounds[1]:g}]",
            )
        )
        second = figure_2["second_mode_density"][index]
        first = figure_2["first_mode_density"][index]
        clauses.append(
            _clause(
                f"figure_2.second_mode_density[{index}]",
                second,
                f"> f({targets['FIRST_MODE_NEIGHBOUR']}) = {float(first):.6g}",
                float(second) > float(first),
                f"{label}: f({targets['SECOND_DENSITY_MODE']}) > "
                f"f({targets['FIRST_MODE_NEIGHBOUR']})",
            )
        )

    expected = list(targets["HAZARD_CHANGE_POINTS"])
    tolerances = list(targets["HAZARD_CHANGE_TOLERANCES"])
    for index, label in enumerate(labels):
        changes = [float(value) for value in figure_3["change_points"][index]]
        matched = len(changes) == len(expected) and all(
            abs(found - wanted) <= tolerance
            for found, wanted, tolerance in zip(
                changes, expected, tolerances, strict=True
            )
        )
        clauses.append(
            _clause(
                f"figure_3.change_points[{index}]",
                changes,
                {"points": expected, "tolerances": tolerances},
                matched,
                f"{label}: {len(expected)} smoothed-hazard change points within "
                "tolerance",
            )
        )
        ratio = figure_3["band_ratio"][index]
        expected_ratio = targets["HAZARD_BAND_RATIO"]
        clauses.append(
            _clause(
                f"figure_3.band_ratio[{index}]",
                ratio,
                expected_ratio,
                np.isfinite(ratio) and float(ratio) >= expected_ratio,
                f"{label}: band at t={targets['HAZARD_BAND_TIMES'][1]:g} is "
                f">= {expected_ratio:g}x the band at t="
                f"{targets['HAZARD_BAND_TIMES'][0]:g}",
            )
        )

    wider = figure_2["prior_band_is_wider"]
    clauses.append(
        _clause(
            "figure_2.prior_band_is_wider",
            wider,
            True,
            wider is True,
            f"prior 95% band is at least as wide as the posterior's on "
            f"(0, {targets['BULK_UPPER_TIME']:g}]",
        )
    )
    return clauses


def _clauses_5_2(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One acceptance row per Figure 4-5 clause of the paper's Section 5.2."""
    targets = ACCEPTANCE_TARGETS["5.2"]
    figure_4 = summary["figure_4"]
    figure_5 = summary["figure_5"]
    clauses: list[dict[str, Any]] = []

    fraction = figure_4["prior_agreement_min"]
    required = targets["PRIOR_AGREEMENT_FRACTION"]
    clauses.append(
        _clause(
            "figure_4.prior_agreement_min",
            fraction,
            required,
            _in_range(fraction, (0.0, 1.0)) and fraction >= required,
            "each pair of prior curves agrees within the other's 95% band",
        )
    )
    difference = figure_4["survival_difference_min"]
    clauses.append(
        _clause(
            "figure_4.survival_difference_min",
            difference,
            "> 0",
            float(difference) > 0.0,
            f"mean S_A - S_B is positive on {targets['REMISSION_WINDOW']}",
        )
    )
    peak_time = figure_4["arm_b_peak_time"]
    peak_low, peak_high = targets["B_HAZARD_PEAK_WINDOW"]
    clauses.append(
        _clause(
            "figure_4.arm_b_peak_time",
            peak_time,
            [peak_low, peak_high],
            _in_range(peak_time, (peak_low, peak_high)),
            f"the arm-B smoothed hazard peaks in [{peak_low:g}, {peak_high:g}]",
        )
    )
    above = figure_4["arm_b_above_peak_max"]
    peak_value = figure_4["arm_b_peak_value"]
    clauses.append(
        _clause(
            "figure_4.arm_b_above_peak_max",
            above,
            f"< peak value {float(peak_value):.6g}",
            above is None or float(above) < float(peak_value),
            "the arm-B hazard falls after its peak",
        )
    )

    probability = figure_5["prob_eta_a_gt_eta_b"]
    eta_low, eta_high = targets["ETA_PROBABILITY_RANGE"]
    clauses.append(
        _clause(
            "figure_5.prob_eta_a_gt_eta_b",
            probability,
            [eta_low, eta_high],
            _in_range(probability, (eta_low, eta_high)),
            f"Pr(eta_A > eta_B | D) in [{eta_low:g}, {eta_high:g}]",
        )
    )
    difference_min = figure_5["difference_min"]
    clauses.append(
        _clause(
            "figure_5.difference_min",
            difference_min,
            ">= 0",
            float(difference_min) >= 0.0,
            f"mean F_B - F_A is non-negative on {targets['REMISSION_WINDOW']}",
        )
    )
    difference_range = figure_5["difference_probability_range"]
    floor = targets["DIFFERENCE_MIN_PROBABILITY"]
    clauses.append(
        _clause(
            "figure_5.difference_probability_range",
            difference_range,
            f"minimum >= {floor:g}",
            float(difference_range[0]) >= floor,
            "Pr(F_B > F_A | D) never falls below the amendment-A3 floor",
        )
    )
    difference_mean = figure_5["difference_probability_mean"]
    mean_floor = targets["DIFFERENCE_MEAN_PROBABILITY"]
    clauses.append(
        _clause(
            "figure_5.difference_probability_mean",
            difference_mean,
            mean_floor,
            float(difference_mean) >= mean_floor,
            f"mean Pr(F_B > F_A | D) >= {mean_floor:g}",
        )
    )
    decline = figure_5["hazard_ratio_decline"]
    factor = targets["RATIO_DECLINE_FACTOR"]
    clauses.append(
        _clause(
            "figure_5.hazard_ratio_decline",
            decline,
            factor,
            decline is not None and float(decline) >= factor,
            f"mean hazard ratio at t={targets['HAZARD_BAND_TIMES'][0]:g} is "
            f">= {factor:g}x the one at t={targets['HAZARD_BAND_TIMES'][1]:g}",
        )
    )
    order = figure_5["hazard_ratio_order_probability"]
    order_floor = targets["RATIO_ORDER_PROBABILITY"]
    clauses.append(
        _clause(
            "figure_5.hazard_ratio_order_probability",
            order,
            order_floor,
            order is not None and float(order) >= order_floor,
            f"Pr(ratio(t={targets['HAZARD_BAND_TIMES'][0]:g}) > "
            f"ratio(t={targets['HAZARD_BAND_TIMES'][1]:g}) | D) >= "
            f"{order_floor:g}",
        )
    )
    return clauses


def evaluate_acceptance(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate every acceptance clause of a summary payload.

    The thresholds are :data:`ACCEPTANCE_TARGETS`, the same table the committed
    acceptance tests mirror, so ``summary/acceptance.json`` answers the gate
    question from the artifacts without importing the paper test modules.  The
    returned object names every clause, its target, the observed value and the
    verdict, plus ``n_clauses``/``n_passed``/``passed`` for a one-line check.

    Raises
    ------
    ValueError
        If ``summary`` is not a mapping, has no ``label``, or names a section
        this module has no clauses for.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    label = summary.get("label")
    if label == "5.1":
        clauses = _clauses_5_1(summary)
    elif label == "5.2":
        clauses = _clauses_5_2(summary)
    else:
        raise ValueError(f"no acceptance clauses for section label {label!r}")
    passed = sum(1 for clause in clauses if clause["pass"])
    return {
        "label": label,
        "n_clauses": len(clauses),
        "n_passed": passed,
        "passed": passed == len(clauses),
        "clauses": clauses,
    }


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------
def _settings_payload(result: DemoResult) -> dict[str, Any]:
    """The run settings table: preset, seeds, grid and sampler knobs."""
    meta = dict(result.meta)
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": result.label,
        "preset": meta.get("preset"),
        "seed": meta.get("seed"),
        "data_seed": meta.get("data_seed"),
        "prior_indices": list(meta.get("prior_indices", ())),
        "n_fits": len(result.fits),
        "slice_draws": SLICE_DRAWS,
        "t_max": meta.get("t_max"),
    }
    for name in ("burn_in", "n_save", "thin", "n_chains"):
        if name in meta:
            payload[name] = meta[name]
    if result.fits:
        first = next(iter(result.fits.values()))
        grid = np.asarray(first.t_grid, dtype=np.float64)
        payload["n_grid"] = int(grid.size)
        payload["grid_first"] = float(grid[0])
        payload["grid_last"] = float(grid[-1])
        if first.chains:
            settings = first.chains[0].settings
            payload["fit_settings"] = {
                "burn_in": settings.burn_in,
                "n_save": settings.n_save,
                "thin": settings.thin,
                "n_alpha": settings.n_alpha,
                "epsilon": settings.epsilon,
                "n_chains": settings.n_chains,
                "seed": settings.seed,
                "preset": settings.preset,
            }
    return _floats(payload)


def manifest_summary(save_dir: str | Path) -> dict[str, Any] | None:
    """Summarise ``<run_root>/MANIFEST.json`` for the report's manifest table.

    ``save_dir`` is the section directory, so the run root is its parent.  The
    report normally runs *before* ``finalize_run`` (plan §7 Step 0 orders the
    driver, the report, then the manifest), so an absent manifest is not an
    error: ``None`` is returned and the table says the run is not finalised
    yet.  A manifest that exists but cannot be read or parsed is an error,
    because a corrupt run record must not be rendered as if it were absent.

    Returns
    ------
    dict or None
        ``schema_version``, ``created_at``, ``code_sha``, ``code_dirty``,
        ``pixi_lock_sha256``, ``packages``, ``sections`` (section names only),
        ``file_count`` and ``total_bytes``.  The absolute run root is
        deliberately not part of the summary: it would make a re-rendered
        ``tables.md`` depend on where the artifacts happen to live.
    """
    section = Path(save_dir)
    target = section.parent / _MANIFEST_NAME
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read run manifest {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"run manifest {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"run manifest {target} must hold a JSON object")
    sections = payload.get("sections")
    return _floats(
        {
            "schema_version": payload.get("schema_version"),
            "created_at": payload.get("created_at"),
            "code_sha": payload.get("code_sha"),
            "code_dirty": payload.get("code_dirty"),
            "pixi_lock_sha256": payload.get("pixi_lock_sha256"),
            "packages": payload.get("packages"),
            "sections": sorted(sections) if isinstance(sections, Mapping) else [],
            "file_count": payload.get("file_count"),
            "total_bytes": payload.get("total_bytes"),
        }
    )


def build_payload(
    result: DemoResult,
    *,
    save_dir: str | Path | None = None,
    baseline: Mapping[str, Any] | str | Path | None = None,
    manifest: Mapping[str, Any] | None | object = _UNSET,
) -> dict[str, Any]:
    """Assemble the report payload from one loaded ``DemoResult``.

    The payload carries the two draw sets side by side -- ``full`` (all stored
    draws) and ``first_500`` (the old note's probe, via :func:`slice_result`) --
    plus the diagnostics, the acceptance verdict on the **full** draws and the
    delta of the first-500 summary against the committed baseline.  The full
    draws are the sharper estimate and are what the acceptance table judges;
    the first-500 column is the like-for-like replay of
    ``docs/research/dp_gibbs_results.md`` (plan §7 Step 2).

    ``manifest`` overrides :func:`manifest_summary`; the default sentinel
    derives it from ``save_dir`` when one is given, and stores ``None``
    otherwise.  Everything in the payload is JSON-safe.
    """
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    if result.label not in ("5.1", "5.2"):
        raise ValueError(f"unknown section label {result.label!r}")

    if result.label == "5.1":
        summarize = summarize_5_1
    else:
        summarize = summarize_5_2
    full = summarize(result)
    first_500 = summarize(slice_result(result))

    if manifest is _UNSET:
        manifest_payload = None if save_dir is None else manifest_summary(save_dir)
    elif manifest is None or isinstance(manifest, Mapping):
        manifest_payload = None if manifest is None else _floats(dict(manifest))
    else:
        raise ValueError("manifest must be a mapping or None")

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "label": result.label,
        "settings": _settings_payload(result),
        "full": full,
        "first_500": first_500,
        "diagnostics": diagnostics(result),
        "acceptance": evaluate_acceptance(full),
        "delta": delta_vs_baseline(first_500, baseline),
        "manifest": manifest_payload,
    }
    return _floats(payload)


# ---------------------------------------------------------------------------
# Rendering: the generated tables of report/tables.md
# ---------------------------------------------------------------------------
def _format_number(value: float) -> str:
    """Short, round-trippable rendering of one float (``%.6g``)."""
    number = float(value)
    if not np.isfinite(number):
        if np.isnan(number):
            return "nan"
        return "inf" if number > 0.0 else "-inf"
    return f"{number:.6g}"


def _format_value(value: Any) -> str:
    """Render one table cell; ``None`` is the report's explicit missing mark."""
    if value is None:
        return _MISSING
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_number(float(value))
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _cell(value: Any) -> str:
    """Render one table cell, escaping the column separator."""
    return _format_value(value).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """A GitHub-flavoured Markdown table, one rendered cell per column."""
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return lines


def _heading(title: str, lines: Sequence[str]) -> list[str]:
    """One ``##`` section followed by a blank line."""
    return [f"## {title}", "", *lines, ""]


def _render_settings(payload: Mapping[str, Any]) -> list[str]:
    settings = payload["settings"]
    rows = list(settings.items())
    return _heading("Settings", _table(["key", "value"], rows))


def _render_figure_1(summary: Mapping[str, Any]) -> list[str]:
    targets = ACCEPTANCE_TARGETS["5.1"]
    low, high = targets["N_STAR_RANGE"]
    figure = summary["figure_1"]
    rows = []
    for index, label in enumerate(summary["prior_labels"]):
        rows.append(
            [
                label,
                figure["nu_prior_mean"][index],
                figure["nu_posterior_mean"][index],
                figure["nu_posterior_95"][index],
                figure["nu_prior_95_width"][index],
                figure["nu_posterior_width"][index],
                figure["nu_posterior_99"][index],
                figure["n_star_posterior_mean"][index],
                figure["n_star_probability"][index],
                figure["n_star_observed_range"][index],
                figure["learning"][index],
            ]
        )
    headers = [
        "prior",
        "nu prior mean",
        "nu posterior mean",
        "nu 95% CI",
        "nu prior width",
        "nu posterior width",
        "nu 99th pct",
        "n* mean",
        f"Pr({low} <= n* <= {high})",
        "n* range",
        "learning",
    ]
    return _heading("Figure 1: nu and n* posteriors", _table(headers, rows))


def _render_figure_2(summary: Mapping[str, Any]) -> list[str]:
    targets = ACCEPTANCE_TARGETS["5.1"]
    figure = summary["figure_2"]
    band_label = figure["band_prior_label"]
    rows = []
    for index, label in enumerate(summary["prior_labels"]):
        is_band = label == band_label
        rows.append(
            [
                label,
                figure["survival_at_median"][index],
                figure["density_mass"][index],
                figure["second_mode_density"][index],
                figure["first_mode_density"][index],
                figure["prior_band_width"] if is_band else None,
                figure["posterior_band_width"] if is_band else None,
                figure["prior_band_is_wider"] if is_band else None,
            ]
        )
    headers = [
        "prior",
        f"mean S({targets['POPULATION_MEDIAN']})",
        "int f dt",
        f"f({targets['SECOND_DENSITY_MODE']})",
        f"f({targets['FIRST_MODE_NEIGHBOUR']})",
        "prior band width",
        "posterior band width",
        "prior band wider",
    ]
    note = (
        f"Band widths are the reference prior's (`{band_label}`) on "
        f"(0, {targets['BULK_UPPER_TIME']:g}]."
    )
    return _heading(
        "Figure 2: survival and density", _table(headers, rows) + ["", note]
    )


def _render_figure_3(summary: Mapping[str, Any]) -> list[str]:
    targets = ACCEPTANCE_TARGETS["5.1"]
    figure = summary["figure_3"]
    near, far = targets["HAZARD_BAND_TIMES"]
    rows = [
        [
            label,
            figure["change_points"][index],
            figure["band_width_near"][index],
            figure["band_width_far"][index],
            figure["band_ratio"][index],
        ]
        for index, label in enumerate(summary["prior_labels"])
    ]
    headers = [
        "prior",
        "change points",
        f"band width t={near:g}",
        f"band width t={far:g}",
        "far / near",
    ]
    note = (
        f"Gaussian smoothing with a bandwidth of "
        f"{figure['smoothing_bandwidth']:g} grid points; change points are the "
        f"sign changes of the smoothed mean on "
        f"({targets['HAZARD_WINDOW'][0]:g}, {targets['HAZARD_WINDOW'][1]:g})."
    )
    return _heading("Figure 3: smoothed hazard", _table(headers, rows) + ["", note])


def _render_figure_4(summary: Mapping[str, Any]) -> list[str]:
    figure = summary["figure_4"]
    rows: list[list[Any]] = [
        ["prior agreement minimum", figure["prior_agreement_min"]],
        ["survival difference minimum", figure["survival_difference_min"]],
        ["arm-B peak time", figure["arm_b_peak_time"]],
        ["arm-B peak value", figure["arm_b_peak_value"]],
        ["arm-B maximum above the peak window", figure["arm_b_above_peak_max"]],
        ["remission window", figure["remission_window"]],
    ]
    lines = _heading(
        "Figure 4: prior agreement and arm A versus arm B",
        _table(["quantity", "value"], rows),
    )

    labels = summary["prior_labels"]
    matrix = [
        [labels[first], *figure["prior_agreement"][first]]
        for first in range(len(labels))
    ]
    lines += _heading(
        "Figure 4: prior agreement matrix",
        _table(["prior"] + labels, matrix),
    )
    return lines


def _render_figure_5(summary: Mapping[str, Any]) -> list[str]:
    figure = summary["figure_5"]
    rows: list[list[Any]] = [
        ["Pr(eta_A > eta_B | D)", figure["prob_eta_a_gt_eta_b"]],
        ["minimum mean F_B - F_A", figure["difference_min"]],
        [
            "Pr(F_B > F_A | D) range",
            figure["difference_probability_range"],
        ],
        ["mean Pr(F_B > F_A | D)", figure["difference_probability_mean"]],
        [
            f"mean hazard ratio at t={figure['ratio_times'][0]:g}",
            figure["hazard_ratio_mean"][0],
        ],
        [
            f"mean hazard ratio at t={figure['ratio_times'][1]:g}",
            figure["hazard_ratio_mean"][1],
        ],
        ["hazard-ratio decline (near / far)", figure["hazard_ratio_decline"]],
        [
            "Pr(ratio(near) > ratio(far) | D)",
            figure["hazard_ratio_order_probability"],
        ],
        [
            f"95% ratio band at t={figure['ratio_times'][0]:g}",
            figure["hazard_ratio_band"][0],
        ],
        [
            f"95% ratio band at t={figure['ratio_times'][1]:g}",
            figure["hazard_ratio_band"][1],
        ],
        ["n_eta_above_grid", figure["n_eta_above_grid"]],
    ]
    return _heading(
        "Figure 5: hazard ratio and CDF difference", _table(["quantity", "value"], rows)
    )


def _render_diagnostics(payload: Mapping[str, Any]) -> list[str]:
    diagnostics = payload["diagnostics"]
    rows = []
    for fit, block in diagnostics["fits"].items():
        truncation = block["j"]
        for draw_set in ("full", "first_500"):
            entry = block[draw_set]
            rows.append(
                [
                    fit,
                    draw_set,
                    entry["draws_per_chain"],
                    entry["nu"]["r_hat"],
                    entry["nu"]["ess"],
                    entry["n_star"]["r_hat"],
                    entry["n_star"]["ess"],
                    entry["S"]["r_hat"],
                    entry["S"]["ess"],
                    truncation["J_common"],
                    truncation["J_prior"],
                ]
            )
    headers = [
        "fit",
        "draw set",
        "draws per chain",
        "R-hat nu",
        "ESS nu",
        "R-hat n*",
        "ESS n*",
        "R-hat S",
        "ESS S",
        "J_common",
        "J_prior",
    ]
    note = (
        f"The `first_500` block truncates each chain to "
        f"{diagnostics['slice_draws']} stored draws and reuses the full trace's "
        "truncation level; `R-hat S` and `ESS S` are the five monitored times "
        "in the order `nu` reports them."
    )
    return _heading("Diagnostics", _table(headers, rows) + ["", note])


def _render_acceptance(payload: Mapping[str, Any]) -> list[str]:
    acceptance = payload["acceptance"]
    rows = [
        [
            index + 1,
            clause["clause"],
            clause["description"],
            clause["target"],
            clause["observed"],
            "pass" if clause["pass"] else "fail",
        ]
        for index, clause in enumerate(acceptance["clauses"])
    ]
    summary = (
        f"{acceptance['n_passed']} / {acceptance['n_clauses']} clauses pass "
        f"({'PASS' if acceptance['passed'] else 'FAIL'})."
    )
    return _heading(
        "Acceptance",
        [summary, ""]
        + _table(["#", "clause", "description", "target", "observed", "result"], rows),
    )


def _render_delta(payload: Mapping[str, Any]) -> list[str]:
    delta = payload["delta"]
    rows = [
        [
            path,
            row["observed"],
            row["baseline"],
            row["delta"],
            row["not_comparable"],
        ]
        for path, row in delta["rows"].items()
    ]
    source = delta.get("source") or "the committed baseline"
    note = (
        f"Signed `observed - baseline` against `{source}`. Rows marked "
        "`not comparable` cannot match by construction and are not part of the "
        "gate."
    )
    return _heading(
        "Delta vs baseline (first 500 draws)",
        [note, ""]
        + _table(["row", "observed", "baseline", "delta", "not comparable"], rows),
    )


def _render_manifest(payload: Mapping[str, Any]) -> list[str]:
    manifest = payload.get("manifest")
    if manifest is None:
        return _heading(
            "Manifest",
            _table(
                ["key", "value"],
                [["manifest", "not finalised yet (run finalize_run after the report)"]],
            ),
        )
    rows: list[list[Any]] = []
    for key in (
        "schema_version",
        "created_at",
        "code_sha",
        "code_dirty",
        "pixi_lock_sha256",
        "packages",
        "sections",
        "file_count",
        "total_bytes",
    ):
        rows.append([key, manifest.get(key)])
    return _heading("Manifest", _table(["key", "value"], rows))


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the generated tables of ``report/tables.md``.

    The text is the tables only -- settings, diagnostics, Figures 1-5, the
    acceptance table, the baseline delta and the manifest -- because the
    narrative lives in the committed report.  It is deterministic: no
    timestamp, no absolute path and no dictionary-order dependency, so a
    re-render from the same artifacts is byte-identical and can be compared
    by ``scripts/verify_replication.sh``.

    Raises
    ------
    ValueError
        If ``payload`` is not a mapping, is missing a report key, or names a
        section this module does not know.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    required = (
        "label",
        "settings",
        "full",
        "first_500",
        "diagnostics",
        "acceptance",
        "delta",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"payload is missing the report keys {missing}")
    if payload["label"] not in ("5.1", "5.2"):
        raise ValueError(f"unknown section label {payload['label']!r}")

    full = payload["full"]
    lines = [
        f"# Replication report: section {payload['label']}",
        "",
        "Generated by `python -m src.paper_report` from the saved artifacts.",
        "The acceptance table judges the full draws; the delta table compares "
        "the first-500 slice against the old note.",
        "",
    ]
    lines += _render_settings(payload)
    if payload["label"] == "5.1":
        lines += _render_figure_1(full)
        lines += _render_figure_2(full)
        lines += _render_figure_3(full)
    else:
        lines += _render_figure_4(full)
        lines += _render_figure_5(full)
    lines += _render_diagnostics(payload)
    lines += _render_acceptance(payload)
    lines += _render_delta(payload)
    lines += _render_manifest(payload)
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# CSV payloads
# ---------------------------------------------------------------------------
def _csv_value(value: Any) -> str:
    """One CSV cell; ``None`` is an empty field, non-finite stays readable."""
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (float, np.floating)):
        return _format_number(float(value))
    if isinstance(value, (list, tuple)):
        return " ".join(_csv_value(item) for item in value)
    return str(value)


def _csv_text(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a CSV document with ``\\n`` line endings and no index column."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_csv_value(value) for value in row])
    return buffer.getvalue()


def rhat_ess_rows(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Long-format R-hat/ESS rows of ``summary/rhat_ess.csv``.

    One row per fit, draw set and monitored variable; the five ``S`` times
    share the variable name and carry their time in the ``time`` column.
    """
    rows: list[list[Any]] = []
    for fit, block in payload["diagnostics"]["fits"].items():
        for draw_set in ("full", "first_500"):
            entry = block[draw_set]
            for variable in ("nu", "n_star"):
                rows.append(
                    [
                        fit,
                        draw_set,
                        entry["draws_per_chain"],
                        variable,
                        None,
                        entry[variable]["r_hat"],
                        entry[variable]["ess"],
                    ]
                )
            for time, r_hat_value, ess_value in zip(
                entry["S"]["t"],
                entry["S"]["r_hat"],
                entry["S"]["ess"],
                strict=True,
            ):
                rows.append(
                    [
                        fit,
                        draw_set,
                        entry["draws_per_chain"],
                        "S",
                        time,
                        r_hat_value,
                        ess_value,
                    ]
                )
    return rows


def _split_arm(key: str) -> tuple[str, str]:
    """Split a ``"A: label"`` fit key into its arm and its label."""
    for arm in ("A", "B"):
        prefix = f"{arm}: "
        if key.startswith(prefix):
            return arm, key[len(prefix) :]
    return "", key


def _quantiles(values: np.ndarray) -> list[Any]:
    """Mean and 2.5/50/97.5 percentiles of the finite entries, plus a count."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    censored = int(array.size - finite.size)
    if finite.size == 0:
        return [None, None, None, None, censored]
    mean = float(np.mean(finite))
    low, median, high = (
        float(value) for value in np.percentile(finite, [2.5, 50.0, 97.5])
    )
    return [mean, low, median, high, censored]


def _eta_values(result: DemoResult) -> list[tuple[str, str, np.ndarray]]:
    """The per-draw median ``eta`` of each arm of a §5.2 result."""
    fits_a = _arm_fits(result, "A")
    if not fits_a:
        raise ValueError("a Section 5.2 result must contain an arm-A fit")
    reference_key, reference = fits_a[0]
    comparison = compare(
        reference.functionals, _arm_b(result).functionals, reference.t_grid
    )
    b_key = next(key for key in result.fits if key.startswith("B: "))
    return [
        (reference_key, "A", np.asarray(comparison["eta_a"], dtype=np.float64)),
        (b_key, "B", np.asarray(comparison["eta_b"], dtype=np.float64)),
    ]


def quantiles_rows(result: DemoResult) -> list[list[Any]]:
    """Long-format rows of ``summary/quantiles.csv``.

    Every fit contributes its ``nu`` and ``n*`` mean/2.5/50/97.5 % rows; a
    §5.2 result adds the per-draw median ``eta`` of both arms, computed with
    the same paired ``compare`` call the Figure 5 summary uses (``p = 0.5``).
    """
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    rows: list[list[Any]] = []
    for key, fit in result.fits.items():
        arm, _ = _split_arm(key)
        for quantity, values in (
            ("nu", _pooled(fit, "nu")),
            ("n_star", _pooled(fit, "n_star")),
        ):
            rows.append([key, arm, quantity, *_quantiles(values)])
    if result.label == "5.2":
        for key, arm, values in _eta_values(result):
            rows.append([key, arm, "eta_median", *_quantiles(values)])
    return rows


# ---------------------------------------------------------------------------
# Figures and the report writer
# ---------------------------------------------------------------------------
def _pyplot() -> Any:
    """Import ``matplotlib`` on demand with the headless Agg backend.

    Same contract as :func:`src.paper_examples._pyplot`: importing this module
    (or the sampler modules) must not import ``matplotlib``, so the import is
    delayed until :func:`write_report_artifacts` really renders a figure.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _write_text(target: Path, text: str) -> Path:
    """Write ``text`` atomically (``.tmp`` + ``os.replace``), UTF-8, LF only."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _render_figures(section: Path, result: DemoResult) -> dict[str, Path]:
    """Re-render the section's figures from the loaded functionals.

    ``paper_examples`` is the single owner of the recipes; this function only
    chooses the inputs, the section's ``figures/`` directory and the cleanup.
    Its PNG bytes are reproducible because ``paper_examples._save`` pins the
    DPI, the format and the metadata (plan R7).  A §5.1 fit without a prior
    band has no Figure 2 to draw, so that one panel is skipped rather than
    invented from the posterior alone.
    """
    plt = _pyplot()
    target = section / _FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    try:
        if result.label == "5.1":
            fits = dict(result.fits)
            reference = next(iter(fits.values()))
            written["figure_1"] = figure_1(fits, path=target / "figure_1.png")
            if reference.prior_band is not None:
                written["figure_2"] = figure_2(
                    reference, reference.prior_band, path=target / "figure_2.png"
                )
            written["figure_3"] = figure_3(fits, path=target / "figure_3.png")
        elif result.label == "5.2":
            fits_a = {
                key[3:]: fit
                for key, fit in result.fits.items()
                if key.startswith("A: ")
            }
            if not fits_a:
                raise ValueError("a Section 5.2 result must contain an arm-A fit")
            fit_b = _arm_b(result)
            reference_a = next(iter(fits_a.values()))
            written["figure_4"] = figure_4(fits_a, fit_b, path=target / "figure_4.png")
            written["figure_5"] = figure_5(
                reference_a, fit_b, path=target / "figure_5.png"
            )
        else:  # pragma: no cover - build_payload rejects this first
            raise ValueError(f"unknown section label {result.label!r}")
    finally:
        plt.close("all")
    return written


def write_report_artifacts(
    save_dir: str | Path,
    payload: Mapping[str, Any],
    result: DemoResult,
) -> dict[str, Path]:
    """Write the report half of a section tree (plan §6.3).

    ``save_dir`` is the section directory written by the driver, so the files
    land exactly where the plan §5 layout puts them and nowhere else::

        <section>/summary/stats.json
        <section>/summary/rhat_ess.csv
        <section>/summary/quantiles.csv
        <section>/summary/acceptance.json
        <section>/report/tables.md
        <section>/figures/figure_*.png

    ``stats.json`` is the whole payload, including the full and first-500
    summaries, so the committed report can be transcribed from one file.  The
    figures are re-rendered from ``result`` (plan R7: the report stage is the
    only figure writer), and every write is atomic.

    Returns
    -------
    dict of str to pathlib.Path
        The written paths, keyed by artifact name (``"stats"``, ``"tables"``,
        ``"figure_1"``, ...).

    Raises
    ------
    ValueError
        If the payload is not a mapping, disagrees with ``result`` about the
        section, or lacks the acceptance/diagnostics blocks.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    if not isinstance(result, DemoResult):
        raise ValueError("result must be a DemoResult instance")
    if payload.get("label") != result.label:
        raise ValueError(
            f"payload label {payload.get('label')!r} disagrees with the result "
            f"label {result.label!r}"
        )
    for key in ("diagnostics", "acceptance", "delta", "settings"):
        if key not in payload:
            raise ValueError(f"payload is missing the report key {key!r}")

    section = Path(save_dir)
    summary_dir = section / _SUMMARY_DIR
    report_dir = section / _REPORT_DIR
    written: dict[str, Path] = {
        "stats": _write_text(
            summary_dir / "stats.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        ),
        "rhat_ess": _write_text(
            summary_dir / "rhat_ess.csv",
            _csv_text(
                [
                    "fit",
                    "draw_set",
                    "draws_per_chain",
                    "variable",
                    "time",
                    "r_hat",
                    "ess",
                ],
                rhat_ess_rows(payload),
            ),
        ),
        "quantiles": _write_text(
            summary_dir / "quantiles.csv",
            _csv_text(
                [
                    "fit",
                    "arm",
                    "quantity",
                    "mean",
                    "p2.5",
                    "p50",
                    "p97.5",
                    "n_censored",
                ],
                quantiles_rows(result),
            ),
        ),
        "acceptance": _write_text(
            summary_dir / "acceptance.json",
            json.dumps(payload["acceptance"], indent=2, sort_keys=True) + "\n",
        ),
        "tables": _write_text(report_dir / "tables.md", render_markdown(payload)),
    }
    written.update(_render_figures(section, result))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the report CLI's argument parser."""
    parser = argparse.ArgumentParser(
        prog="paper_report",
        description=(
            "Recompute a Kottas (2002) replication report from saved "
            "artifacts, without fitting anything."
        ),
    )
    parser.add_argument(
        "--from-artifacts",
        type=Path,
        required=True,
        metavar="DIR",
        help="section directory written by the driver (for example output/run/5.1)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "write summary/, report/ and figures/ into the section directory; "
            "without it the generated tables are printed to stdout"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "baseline JSON to compare the first-500 summary against "
            f"(default: {BASELINE_PATH})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Report CLI entry point.  Returns 0 on success, 1 on a report error.

    ``python -m src.paper_report --from-artifacts <section_dir> --write`` is
    the runbook's report stage.  Without ``--write`` the same tables are
    printed to stdout, which is the dry-run form.  An unreadable artifact
    tree, a missing baseline or a payload the artifacts cannot supply prints
    one ``error:`` line and exits 1 instead of a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = load_demo_result(args.from_artifacts)
        payload = build_payload(
            result,
            save_dir=args.from_artifacts,
            baseline=args.baseline,
        )
        if args.write:
            written = write_report_artifacts(args.from_artifacts, payload, result)
        else:
            written = {}
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    acceptance = payload["acceptance"]
    line = (
        f"section {result.label}: {acceptance['n_passed']}/"
        f"{acceptance['n_clauses']} acceptance clauses pass"
    )
    if args.write:
        for name, path in written.items():
            print(f"wrote {name}: {path}")
        print(line)
    else:
        print(render_markdown(payload), end="")
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
