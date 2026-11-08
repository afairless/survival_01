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
* :data:`ACCEPTANCE_TARGETS` -- the target/label table mirrored from
  ``tests/test_paper_5_1.py`` and ``tests/test_paper_5_2.py``.

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
The module only reads containers: it never samples, never evaluates a
functional and never writes a file.  Rendering (``report/tables.md``, the
figures and the CLI) belongs to the next unit; in particular this module does
not import ``matplotlib``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import gamma as gamma_dist

from src.dp_functionals import Functionals, compare
from src.dp_gibbs import DPChains, ess, r_hat
from src.paper_artifacts import _jsonable
from src.paper_examples import REPO_ROOT, DemoResult, Fit, smooth_hazard

__all__ = [
    "ACCEPTANCE_TARGETS",
    "BASELINE_PATH",
    "BASELINE_SCHEMA_VERSION",
    "SLICE_DRAWS",
    "delta_vs_baseline",
    "diagnostics",
    "load_baseline",
    "slice_result",
    "summarize_5_1",
    "summarize_5_2",
]

#: The number of stored draws per chain in the old note's probe (plan §4).
SLICE_DRAWS = 500

#: On-disk schema of the committed baseline JSON.
BASELINE_SCHEMA_VERSION = 1

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
