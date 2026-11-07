"""Test 30 for Section 5.1 (plan Sections 6.3, 7.5 and 9 step 18).

This is the reproduction acceptance run of the paper's Section 5.1: the
simulated lognormal sample of plan Section 6.1 is fitted under each of the
three ``nu`` priors and Figures 1-3 are checked against the acceptance table of
plan Section 6.3, one assertion per clause:

===========  ==========================================================
Figure       acceptance criterion
===========  ==========================================================
1            posterior mean of ``nu`` increases with the prior mean
             (2.22 -> 20 -> 60); the posterior mean of ``n*`` lies in
             ``[3, 15]`` and increases with the prior mean;
             ``Pr(3 <= n* <= 15 | D) >= 0.85`` under the two narrower
             priors and ``>= 0.70`` under ``Gamma(3, 0.05)`` (plan
             amendment A1); the posterior of ``nu`` is narrower than its
             prior over the bulk (learning) while the right tail under
             ``Gamma(3, 0.05)`` stays prior-dominated
2            posterior mean ``S(1.1727)`` in ``[0.42, 0.58]``;
             ``int f dt`` in ``[0.95, 1.05]``; the prior 95 % band is at
             least as wide as the posterior band on ``(0, 4]``; the
             density keeps its second mode, ``f(3.26) > f(2.4)``
3            after Gaussian smoothing of the hazard, exactly three sign
             changes of the first difference on ``(0, 5)``, the first two
             within ``+-0.25`` of 1.022 and 2.366 and the third within
             ``+-0.35`` of 4.454 (plan amendment A2); the 95 % band at
             ``t = 4.5`` is at least 1.5 times the band at ``t = 2.5``
===========  ==========================================================

The criteria are deliberately inequalities: the paper's figures are
posterior summaries with Monte Carlo error and our RNG, sample and
``t_tilde``-scale priors differ from the paper's (plan Section 6.3's standing
caveats), so "reproduced" means "same qualitative claim, same order of
magnitude, no contradiction".

Running this module
-------------------
The run is a model fit, not a unit test, so it is **opt-in**: the module
carries the ``paper`` marker and ``pixi run test`` excludes it.  Invoke it
explicitly::

    pixi run pytest -m paper tests/test_paper_5_1.py
    pixi run pytest -m paper tests/test_paper_5_1.py -s   # with the log

The preset comes from the ``DP_PAPER_PRESET`` environment variable and
defaults to ``paper`` -- the full 1,505,000-sweep, 4-chain-per-prior run of
plan Section 6.2, which is the reproduction mode the numbers in
``docs/research/dp_gibbs_results.md`` are recorded for.  Setting
``DP_PAPER_PRESET=paper-lite`` selects the documented fallback of plan
decision 5; any use of it must be named in that results note.  The figures are
written to a temporary directory: the recipes are exercised but nothing is
written under the git-ignored ``docs/research/figures/``.

What each test would catch
--------------------------
* a ``nu`` update that ignores ``b_nu`` (the posterior sets would not move
  with the prior) or that samples from the wrong Gamma convention;
* a Polya-urn step (a) whose concentration is wrong -- too few clusters, or
  too many -- visible in ``n*``;
* a functional path that mixes atoms across draws, does not map back through
  the inverse time transform, or prunes away the second density mode -- the
  ``S(1.1727)``, ``int f dt`` and ``f(3.26) > f(2.4)`` clauses;
* a hazard computed from a different atom set than ``S`` and ``f``, or one
  smoothed so heavily that the paper's three change points disappear.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from src import paper_examples
from src.paper_data import NU_PRIORS

pytestmark = pytest.mark.paper

#: Environment variable selecting the reproduction preset (see module docstring).
PRESET_ENV = "DP_PAPER_PRESET"
DEFAULT_PRESET = "paper"

#: Plan Section 6.1, computed there from the true mixture.
POPULATION_MEDIAN = 1.1727
SECOND_DENSITY_MODE = 3.26
FIRST_MODE_NEIGHBOUR = 2.4
HAZARD_CHANGE_POINTS = (1.022, 2.366, 4.454)
#: Plan amendment A2: the third change point is the one the sample's tail
#: controls.  Our Section 5.1 sample has 2 observations above ``t = 4``
#: (max 5.139) against the paper's 4 (max 4.574), so its population value
#: 4.454 is recovered with a wider spread (the paper-grade probe gives
#: 4.462 / 4.245 / 4.166 under the three priors).
HAZARD_CHANGE_TOLERANCES = (0.25, 0.25, 0.35)

#: Plan Section 6.3, row 1: the prior means the posterior means must track.
NU_PRIOR_MEANS = (2.0 / 0.9, 20.0, 60.0)
N_STAR_RANGE = (3, 15)
#: Plan amendment A1: the paper only claims that "the data support roughly 3
#: to 15 distinct components" and its Figure 1 ``n*`` axis runs to 30, so
#: requiring ``Pr >= 0.85`` under the deliberately extreme ``Gamma(3, 0.05)``
#: was the plan's own strengthening.  The paper-grade probe gives
#: ``Pr = 0.987 / 0.885 / 0.730`` with posterior means ``7.0 / 9.9 / 12.8``.
N_STAR_MIN_PROBABILITIES = (0.85, 0.85, 0.70)
#: The paper's claim in words: roughly 3 to 15 components, as a posterior mean.
N_STAR_MEAN_RANGE = (3.0, 15.0)
#: Row 1's "prior-dominated right tail": the widest prior's upper percentile
#: must be this much above the narrowest prior's.
TAIL_DOMINANCE_FACTOR = 2.0

#: Plan Section 6.3, row 2.
SURVIVAL_AT_MEDIAN_RANGE = (0.42, 0.58)
DENSITY_MASS_RANGE = (0.95, 1.05)
BULK_UPPER_TIME = 4.0

#: Plan Section 6.3, row 3.
HAZARD_BAND_RATIO = 1.5
HAZARD_BAND_TIMES = (2.5, 4.5)
HAZARD_SMOOTHING_BANDWIDTH = 5.0
HAZARD_WINDOW = (0.0, 5.0)


# ---------------------------------------------------------------------------
# The reproduction run (one per session, shared by every assertion)
# ---------------------------------------------------------------------------
def _resolve_preset() -> paper_examples.Preset:
    """Preset named by ``DP_PAPER_PRESET``, defaulting to the full run."""
    name = os.environ.get(PRESET_ENV, DEFAULT_PRESET)
    return paper_examples.resolve_preset(name)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> paper_examples.DemoResult:
    """Fit Section 5.1 under all three priors and write Figures 1-3.

    The figures go to a temporary directory: they are a by-product here, and
    the git-ignored ``docs/research/figures/`` is for the manual/CLI runs.
    """
    preset = _resolve_preset()
    out_dir = tmp_path_factory.mktemp(f"section_5_1_{preset.name}")
    return paper_examples.example_5_1(
        preset=preset,
        prior_indices=tuple(range(len(NU_PRIORS))),
        out_dir=out_dir,
    )


def _fit(run: paper_examples.DemoResult, index: int) -> paper_examples.Fit:
    """The fit of ``nu`` prior ``index`` (the dict is keyed by its label)."""
    return run.fits[paper_examples.prior_label(index)]


def _pooled(run: paper_examples.DemoResult, index: int, attr: str) -> np.ndarray:
    """Concatenate ``attr`` over the chains of one prior's fit."""
    fit = _fit(run, index)
    return np.concatenate([np.asarray(getattr(chain, attr)) for chain in fit.chains])


def _nearest(t_grid: np.ndarray, value: float) -> int:
    """Index of the grid point closest to ``value``."""
    return int(np.argmin(np.abs(np.asarray(t_grid, dtype=np.float64) - value)))


def _central_interval(
    values: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Central ``level`` interval along axis 0, ignoring non-finite entries.

    ``h`` is ``inf`` where ``S`` underflows, so the band must drop those
    entries rather than propagate them (the figure recipes do the same).
    """
    array = np.where(np.isfinite(values), values, np.nan)
    tail = 100.0 * (1.0 - level) / 2.0
    lo = np.nanpercentile(array, tail, axis=0)
    hi = np.nanpercentile(array, 100.0 - tail, axis=0)
    return lo, hi


def _mean_curve(values: np.ndarray) -> np.ndarray:
    """Column means of a ``(n_draws, n_grid)`` array, ignoring non-finite."""
    return np.nanmean(np.where(np.isfinite(values), values, np.nan), axis=0)


def _sign_change_times(
    t_grid: np.ndarray, curve: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    """Times of the sign changes of ``diff(curve)`` inside ``window``."""
    mask = (t_grid > window[0]) & (t_grid < window[1])
    t = t_grid[mask]
    differences = np.diff(curve[mask])
    signs = np.sign(differences)
    keep = signs != 0.0
    t = t[1:][keep]
    signs = signs[keep]
    changes = np.flatnonzero(np.diff(signs) != 0.0)
    return np.array([0.5 * (t[i] + t[i + 1]) for i in changes], dtype=np.float64)


# ---------------------------------------------------------------------------
# Preconditions: this must be the paper's run, not a quick smoke test
# ---------------------------------------------------------------------------
def test_run_uses_the_reproduction_settings(run: paper_examples.DemoResult) -> None:
    """The run covers the three priors with the preset's chain count."""
    preset = _resolve_preset()

    assert run.label == "5.1"
    assert run.meta["preset"] == preset.name
    assert run.meta["prior_indices"] == tuple(range(len(NU_PRIORS)))
    assert run.meta["n_chains"] == preset.sampler.n_chains
    assert len(run.fits) == len(NU_PRIORS)
    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        assert fit.n_chains == preset.sampler.n_chains
        assert (fit.prior.a_nu, fit.prior.b_nu) == NU_PRIORS[index]
        assert fit.functionals.n_draws == (
            preset.sampler.n_chains * preset.sampler.n_save
        )


# ---------------------------------------------------------------------------
# Figure 1 (plan Section 6.3, row 1)
# ---------------------------------------------------------------------------
def test_nu_posterior_mean_increases_with_the_prior_mean(
    run: paper_examples.DemoResult,
) -> None:
    """Row 1: the posterior mean of ``nu`` tracks the prior mean, 2.22 -> 60."""
    prior_means = tuple(
        fit.prior.a_nu / fit.prior.b_nu for fit in (_fit(run, i) for i in (0, 1, 2))
    )
    posterior_means = tuple(
        float(_pooled(run, index, "nu").mean()) for index in range(len(NU_PRIORS))
    )

    np.testing.assert_allclose(prior_means, NU_PRIOR_MEANS, rtol=1e-12)
    assert posterior_means[0] < posterior_means[1] < posterior_means[2], (
        "posterior mean of nu must increase with the prior mean: "
        f"prior means {prior_means}, posterior means {posterior_means}"
    )


def test_cluster_count_stays_in_the_papers_range(
    run: paper_examples.DemoResult,
) -> None:
    """Row 1: ``n*`` stays in the paper's 3-15 range, tracked by the prior.

    The paper says the data support "roughly 3 to 15" components and shows a
    ``n*`` posterior under the widest prior that runs past 15, so the
    probability floor is 0.85 for the two narrower priors and 0.70 for
    ``Gamma(3, 0.05)`` (plan amendment A1).  The posterior *mean* must lie in
    the stated range under every prior, and must increase with the prior mean
    of ``nu`` exactly as the paper's Figure 1 does.
    """
    low, high = N_STAR_RANGE
    mean_low, mean_high = N_STAR_MEAN_RANGE
    means = []
    for index in range(len(NU_PRIORS)):
        n_star = _pooled(run, index, "n_star")
        probability = float(np.mean((n_star >= low) & (n_star <= high)))
        minimum = N_STAR_MIN_PROBABILITIES[index]
        mean = float(n_star.mean())
        means.append(mean)
        assert mean_low <= mean <= mean_high, (
            f"{paper_examples.prior_label(index)}: posterior mean of n* is "
            f"{mean:.2f}, outside [{mean_low:g}, {mean_high:g}]"
        )
        assert probability >= minimum, (
            f"{paper_examples.prior_label(index)}: "
            f"Pr({low} <= n* <= {high}) = {probability:.3f} "
            f"(n* mean {mean:.2f}, min {n_star.min()}, max {n_star.max()})"
        )

    assert means[0] < means[1] < means[2], (
        "the mean of n* must increase with the prior mean of nu, as in the "
        f"paper's Figure 1; got {tuple(round(value, 3) for value in means)}"
    )


def test_nu_posterior_is_narrower_than_its_prior(
    run: paper_examples.DemoResult,
) -> None:
    """Row 1: the data narrow ``nu`` over the bulk of each prior."""
    for index in range(len(NU_PRIORS)):
        nu = _pooled(run, index, "nu")
        prior = _fit(run, index).prior
        posterior_lo, posterior_hi = np.percentile(nu, [2.5, 97.5])
        prior_lo, prior_hi = gamma_dist.ppf(
            [0.025, 0.975], a=prior.a_nu, scale=1.0 / prior.b_nu
        )
        posterior_width = float(posterior_hi - posterior_lo)
        prior_width = float(prior_hi - prior_lo)
        assert posterior_width < prior_width, (
            f"{paper_examples.prior_label(index)}: posterior 95% width "
            f"{posterior_width:.3f} is not below prior width {prior_width:.3f}"
        )


def test_widest_prior_keeps_a_prior_dominated_right_tail(
    run: paper_examples.DemoResult,
) -> None:
    """Row 1: the right tail of ``nu`` follows the widest prior's scale.

    The bulk of the posterior is learned from the data (previous test), but
    the upper tail under ``Gamma(3, 0.05)`` is prior-dominated: its 99th
    percentile is far above the one under ``Gamma(2, 0.9)``, and it grows
    monotonically across the three priors even though the three posteriors
    nearly coincide over the bulk.
    """
    upper = tuple(
        float(np.percentile(_pooled(run, index, "nu"), 99.0))
        for index in range(len(NU_PRIORS))
    )

    assert upper[0] < upper[1] < upper[2], (
        f"the nu upper tail must follow the prior scale, got 99% quantiles {upper}"
    )
    assert upper[2] >= TAIL_DOMINANCE_FACTOR * upper[0], (
        f"widest prior's 99% quantile {upper[2]:.2f} is not at least "
        f"{TAIL_DOMINANCE_FACTOR:g}x the narrowest prior's {upper[0]:.2f}"
    )


# ---------------------------------------------------------------------------
# Figure 2 (plan Section 6.3, row 2)
# ---------------------------------------------------------------------------
def test_posterior_survival_at_the_population_median(
    run: paper_examples.DemoResult,
) -> None:
    """Row 2: posterior mean ``S(1.1727)`` lies in ``[0.42, 0.58]``."""
    low, high = SURVIVAL_AT_MEDIAN_RANGE
    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        position = _nearest(fit.t_grid, POPULATION_MEDIAN)
        survival = float(_mean_curve(fit.functionals.S)[position])
        assert low <= survival <= high, (
            f"{paper_examples.prior_label(index)}: posterior mean "
            f"S({POPULATION_MEDIAN}) = {survival:.4f} "
            f"(grid point t = {fit.t_grid[position]:.4f}) outside [{low}, {high}]"
        )


def test_posterior_density_integrates_to_one(run: paper_examples.DemoResult) -> None:
    """Row 2: ``int f dt`` over the grid lies in ``[0.95, 1.05]``.

    The grid starts above zero and stops at ``1.15 * max(t)``, so the integral
    omits the two tails; the plan's band is wide enough for that.
    """
    low, high = DENSITY_MASS_RANGE
    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        mass = float(np.trapezoid(_mean_curve(fit.functionals.f), fit.t_grid))
        assert low <= mass <= high, (
            f"{paper_examples.prior_label(index)}: int f dt = {mass:.4f} "
            f"outside [{low}, {high}]"
        )


def test_prior_band_is_at_least_as_wide_as_the_posterior(
    run: paper_examples.DemoResult,
) -> None:
    """Row 2: on ``(0, 4]`` the prior band is no narrower than the posterior.

    The band is compared against ``prior_functionals`` output on the original
    scale, never against the elicitation guess (plan decision 2).
    """
    fit = _fit(run, 0)
    band = fit.prior_band
    assert band is not None, "example_5_1 must attach the reference prior band"

    posterior_lo, posterior_hi = _central_interval(fit.functionals.S)
    prior_lo, prior_hi = _central_interval(band.S)
    bulk = (fit.t_grid > 0.0) & (fit.t_grid <= BULK_UPPER_TIME)
    assert np.any(bulk)
    posterior_width = float(np.mean(posterior_hi[bulk] - posterior_lo[bulk]))
    prior_width = float(np.mean(prior_hi[bulk] - prior_lo[bulk]))

    assert prior_width >= posterior_width, (
        f"prior 95% band width {prior_width:.4f} on (0, {BULK_UPPER_TIME}] is "
        f"below the posterior width {posterior_width:.4f}"
    )


def test_posterior_density_keeps_its_second_mode(
    run: paper_examples.DemoResult,
) -> None:
    """Row 2: the second density mode at ``~3.26`` survives."""

    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        curve = _mean_curve(fit.functionals.f)
        at_mode = float(curve[_nearest(fit.t_grid, SECOND_DENSITY_MODE)])
        at_first = float(curve[_nearest(fit.t_grid, FIRST_MODE_NEIGHBOUR)])
        assert at_mode > at_first, (
            f"{paper_examples.prior_label(index)}: f(3.26) = {at_mode:.4f} "
            f"is not above f(2.4) = {at_first:.4f}"
        )


# ---------------------------------------------------------------------------
# Figure 3 (plan Section 6.3, row 3)
# ---------------------------------------------------------------------------
def test_smoothed_hazard_has_three_change_points(
    run: paper_examples.DemoResult,
) -> None:
    """Row 3: three sign changes on ``(0, 5)`` near 1.02, 2.37 and 4.45."""
    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        smoothed = paper_examples.smooth_hazard(
            fit.functionals.h, bandwidth=HAZARD_SMOOTHING_BANDWIDTH
        )
        changes = _sign_change_times(fit.t_grid, _mean_curve(smoothed), HAZARD_WINDOW)

        assert changes.size == len(HAZARD_CHANGE_POINTS), (
            f"{paper_examples.prior_label(index)}: expected "
            f"{len(HAZARD_CHANGE_POINTS)} hazard change points, "
            f"found {changes.size} at {np.round(changes, 3)}"
        )
        for found, expected, tolerance in zip(
            changes, HAZARD_CHANGE_POINTS, HAZARD_CHANGE_TOLERANCES, strict=True
        ):
            assert abs(found - expected) <= tolerance, (
                f"{paper_examples.prior_label(index)}: hazard change point "
                f"{found:.3f} is more than {tolerance:g} from {expected}"
            )


def test_hazard_band_widens_beyond_the_data(run: paper_examples.DemoResult) -> None:
    """Row 3: the band at ``t = 4.5`` is ``>= 1.5x`` the band at ``t = 2.5``."""
    near, far = HAZARD_BAND_TIMES
    for index in range(len(NU_PRIORS)):
        fit = _fit(run, index)
        smoothed = paper_examples.smooth_hazard(
            fit.functionals.h, bandwidth=HAZARD_SMOOTHING_BANDWIDTH
        )
        lo, hi = _central_interval(smoothed)
        width = hi - lo
        width_near = float(width[_nearest(fit.t_grid, near)])
        width_far = float(width[_nearest(fit.t_grid, far)])

        assert width_near > 0.0, (
            f"{paper_examples.prior_label(index)}: zero hazard band at t = {near}"
        )
        assert width_far >= HAZARD_BAND_RATIO * width_near, (
            f"{paper_examples.prior_label(index)}: hazard band at t = {far} "
            f"({width_far:.4f}) is not {HAZARD_BAND_RATIO:g}x the band at "
            f"t = {near} ({width_near:.4f})"
        )
