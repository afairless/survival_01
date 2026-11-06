"""Test 21b (plan Section 7.4): prior functionals and the elicitation caveat.

``prior_functionals`` draws ``(nu_b, gamma_b, phi_b)`` from their priors,
draws ``G_b ~ DP(nu_b G0(. | phi_b, gamma_b))`` at the paper's *prior* common
truncation level, prunes the stick-breaking tail and evaluates the exact
mixture functionals on the original time scale.

The point of test 21b is a **falsifiable caveat**, not a happy path.  The
paper elicits the prior on the original time scale but applies it to
``t_tilde = 1 + t / t_max`` (plan Section 5).  For the Section 5.1 constants
and ``t_max = 4.5518`` that means the prior is *not* centred where the
elicitation guess is:

* the prior mean of ``S(1.1727)`` is about ``0.43``;
* its 2.5-97.5% band starts at ``0.00`` (a large part of the prior puts the
  population median at or below zero in original time);
* the prior median of the population median is about ``0.61``, **not** the
  elicited ``1.1727``.

If someone later "fixes" the transform or the priors so that the two scales
coincide, the median assertion fails.  That is deliberate: the paper's
convention is kept (plan Section 10, decision 2), and Figure 2's posterior
comparison is against these prior bands, never against the elicitation guess.

The measured upper band endpoint is about ``0.88`` rather than the ``0.94``
recorded in the plan text.  The assertion below pins the reproducible envelope
(``0.85``-``0.96``) instead of a number that a correct implementation cannot
reach; the load-bearing guard is the population-median assertion.

The remaining tests cover the prior common-``J`` rule, the identity of the
production path with :func:`functionals_from_realizations` on the same draws,
seed replayability, the scale-only ``data``/``t_max`` interface, and input
validation.
"""

import numpy as np
import pytest

from src.dp_functionals import (
    FunctionalSettings,
    functionals_from_realizations,
    prior_functionals,
)
from src.dp_gibbs import DPPrior, make_surv_data
from src.dp_weibull import (
    DPRealization,
    sample_dp_prior,
    sample_prior_hyperparameters,
    truncation_number,
)

# Section 5.1 prior constants (plan Sections 5 and 6.2), transcribed.
PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)
T_MAX = 4.5518
T_ELICITED = 1.1727
EPSILON = 1e-4
SEED = 20240517
N_DRAWS = 1_500

# Plan Section 5 caveat targets for this prior and ``t_max``.
PRIOR_MEAN_S = 0.43
PRIOR_MEDIAN_OF_MEDIAN = 0.61


def _prior_realizations(
    prior: DPPrior,
    n_draws: int,
    *,
    epsilon: float = EPSILON,
    atom_tol: float = 1e-6,
    seed: int = SEED,
) -> tuple[int, list[tuple[float, float, float]], list[DPRealization]]:
    """Reproduce the production draw order: hyperparameters, then ``G_b``.

    The hyperparameters are drawn first (one call per replication), the prior
    truncation level is the paper's rule at the largest drawn precision, and
    only then is each ``G_b`` drawn from the same generator.  Duplicating that
    order here is what lets the tests check ``J_prior`` and the identity with
    :func:`functionals_from_realizations`.
    """
    rng = np.random.default_rng(seed)
    hyperparameters = [
        sample_prior_hyperparameters(
            prior.a_nu,
            prior.b_nu,
            prior.a_gamma,
            prior.b_gamma,
            prior.a_phi,
            prior.b_phi,
            rng=rng,
        )
        for _ in range(n_draws)
    ]
    max_nu = max(precision for precision, _, _ in hyperparameters)
    j_prior = truncation_number(max_nu, epsilon)
    realizations = [
        sample_dp_prior(
            precision,
            gamma,
            phi,
            d=prior.d,
            epsilon=epsilon,
            n_terms=j_prior,
            rng=rng,
        ).pruned(atom_tol)
        for precision, gamma, phi in hyperparameters
    ]
    return int(j_prior), hyperparameters, realizations


def _population_median(realization: DPRealization, t_max: float) -> float:
    """Median of one prior realisation on the original time scale.

    The prior mixture median can be negative in original time: ``t_tilde``
    starts at ``1``, so a CDF already above ``1/2`` there has its median below
    the origin.  The bisection is therefore done on the transformed scale,
    where the support is positive, and the affine inverse map is applied once
    at the end.

    The median can also be ``+inf``: the uniform prior on ``alpha`` puts mass
    arbitrarily close to zero, and an atom with ``alpha ~ 0`` has an almost
    flat kernel ``exp(-1 / lambda)``.  When that flat survival exceeds ``1/2``
    the CDF never reaches ``1/2`` at any finite time.  That is a real feature
    of the prior, so it is reported as ``inf`` rather than hidden by a finite
    bracket.
    """

    def cdf(t_tilde: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(realization.cdf(np.array([t_tilde]))[0])

    lo, hi = 1e-9, 1.0
    while cdf(hi) < 0.5:
        lo, hi = hi, hi * 1.7
        if hi > 1e6:
            return np.inf
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return t_max * (0.5 * (lo + hi) - 1.0)


def _bootstrap_median_se(
    values: np.ndarray, *, n_resamples: int = 500, seed: int = 12345
) -> float:
    """Deterministic bootstrap standard error of the sample median."""
    rng = np.random.default_rng(seed)
    resampled = [
        float(np.median(rng.choice(values, size=values.size, replace=True)))
        for _ in range(n_resamples)
    ]
    return float(np.std(resampled, ddof=1))


def test_prior_functionals_reproduce_the_elicitation_caveat() -> None:
    """Test 21b: prior bands are centred where the prior is, not the guess."""
    grid = np.array([T_ELICITED])
    fit = prior_functionals(PRIOR, grid, N_DRAWS, t_max=T_MAX, seed=SEED)
    survival = fit.S[:, 0]

    mean = float(np.mean(survival))
    standard_error = float(np.std(survival, ddof=1) / np.sqrt(N_DRAWS))
    assert abs(mean - PRIOR_MEAN_S) <= 3.0 * standard_error, (
        f"prior mean S({T_ELICITED}) is {mean!r}, expected {PRIOR_MEAN_S} "
        f"within 3 MC standard errors ({3.0 * standard_error!r})"
    )

    lo, hi = np.percentile(survival, [2.5, 97.5])
    assert lo <= 0.01, f"prior 2.5% band endpoint is {lo!r}, expected ~0.00"
    assert 0.85 <= hi <= 0.96, (
        f"prior 97.5% band endpoint is {hi!r}; the reproducible envelope is "
        "0.85-0.96 (plan text records 0.94)"
    )

    _, _, realizations = _prior_realizations(PRIOR, N_DRAWS)
    medians = np.array(
        [_population_median(realization, T_MAX) for realization in realizations]
    )
    median_of_median = float(np.median(medians))
    median_se = _bootstrap_median_se(medians)

    assert abs(median_of_median - PRIOR_MEDIAN_OF_MEDIAN) <= 3.0 * median_se, (
        f"prior median of the population median is {median_of_median!r}, "
        f"expected {PRIOR_MEDIAN_OF_MEDIAN} within 3 bootstrap standard "
        f"errors ({3.0 * median_se!r})"
    )
    assert abs(median_of_median - T_ELICITED) > 3.0 * median_se, (
        "the prior median of the population median is statistically "
        f"indistinguishable from the elicitation guess {T_ELICITED}; the "
        "elicitation-scale caveat of plan Section 5 has been 'fixed' away"
    )

    fraction_below_zero = float(np.mean(medians <= 0.0))
    assert 0.30 <= fraction_below_zero <= 0.55, (
        "about 43% of prior draws should place the population median at or "
        f"below t=0 (the t_tilde shift), got {fraction_below_zero!r}"
    )


def test_prior_common_j_rule_and_engine_identity() -> None:
    """``J_prior`` is the paper's rule at ``max_b nu_b``; arrays match the engine."""
    n_draws = 40
    grid = np.linspace(0.25, 4.0, 7)
    fit = prior_functionals(PRIOR, grid, n_draws, t_max=T_MAX, seed=SEED)

    j_prior, hyperparameters, realizations = _prior_realizations(PRIOR, n_draws)

    assert fit.meta["sample"] == "prior"
    assert fit.meta["method"] == "exact"
    assert fit.meta["J_prior"] == j_prior
    assert fit.meta["max_nu"] == pytest.approx(
        max(precision for precision, _, _ in hyperparameters)
    )
    assert fit.meta["epsilon"] == EPSILON
    assert fit.meta["atom_tol"] == 1e-6
    assert fit.meta["seed"] == SEED
    assert fit.meta["n_draws"] == n_draws
    assert fit.meta["t_max"] == T_MAX

    # The one-J rule must be conservative for every drawn precision.
    for precision, _, _ in hyperparameters:
        assert truncation_number(precision, EPSILON) <= j_prior

    data = make_surv_data([0.5, 2.0, T_MAX], [True, False, True])
    expected = functionals_from_realizations(realizations, grid, data)
    np.testing.assert_allclose(fit.S, expected.S, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit.F, expected.F, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit.f, expected.f, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit.h, expected.h, rtol=0.0, atol=1e-12)


def test_prior_functionals_shapes_and_internal_consistency() -> None:
    """Validated containers, ``F = 1 - S`` and the hazard ratio."""
    grid = np.linspace(0.1, 4.0, 9)
    fit = prior_functionals(PRIOR, grid, 25, t_max=T_MAX, seed=SEED)

    assert fit.S.shape == (25, grid.size)
    assert fit.F.shape == fit.f.shape == fit.h.shape == fit.S.shape
    np.testing.assert_array_equal(fit.t_grid, grid)
    np.testing.assert_allclose(fit.S + fit.F, 1.0, rtol=0.0, atol=1e-15)
    assert np.all((fit.S >= 0.0) & (fit.S <= 1.0))
    assert np.all(fit.f >= 0.0)
    assert np.all(fit.h >= 0.0)
    assert np.all(np.isfinite(fit.S))
    assert np.all(np.isfinite(fit.F))
    assert np.all(np.isfinite(fit.f))

    positive = fit.S > 0.0
    np.testing.assert_allclose(
        fit.h[positive], fit.f[positive] / fit.S[positive], rtol=1e-12
    )


def test_prior_functionals_ignore_observations_but_use_t_max() -> None:
    """The prior draw does not depend on the data, only on the time scale."""
    grid = np.linspace(0.5, 3.0, 5)
    from_t_max = prior_functionals(PRIOR, grid, 30, t_max=T_MAX, seed=SEED)
    data = make_surv_data([0.4, 1.0, 2.0, 4.5518], [True, True, False, True])
    from_data = prior_functionals(PRIOR, grid, 30, data=data, seed=SEED)
    from_both = prior_functionals(PRIOR, grid, 30, data=data, t_max=T_MAX, seed=SEED)

    np.testing.assert_array_equal(from_t_max.S, from_data.S)
    np.testing.assert_array_equal(from_t_max.S, from_both.S)
    np.testing.assert_array_equal(from_t_max.f, from_data.f)
    assert from_data.meta["t_max"] == T_MAX


def test_prior_functionals_are_seeded_and_replayable() -> None:
    """Same seed reproduces the arrays; different seeds differ."""
    grid = np.linspace(0.5, 3.0, 4)
    first = prior_functionals(PRIOR, grid, 20, t_max=T_MAX, seed=SEED)
    second = prior_functionals(PRIOR, grid, 20, t_max=T_MAX, seed=SEED)
    other = prior_functionals(PRIOR, grid, 20, t_max=T_MAX, seed=SEED + 1)
    fresh = prior_functionals(PRIOR, grid, 20, t_max=T_MAX)

    np.testing.assert_array_equal(first.S, second.S)
    np.testing.assert_array_equal(first.f, second.f)
    np.testing.assert_array_equal(first.h, second.h)
    assert not np.array_equal(first.S, other.S)
    assert isinstance(fresh.meta["seed"], int)
    assert fresh.meta["seed"] >= 0


def test_prior_functionals_reject_bad_input() -> None:
    """Every malformed argument fails at the boundary, not mid-draw."""
    grid = np.linspace(0.5, 3.0, 4)

    with pytest.raises(ValueError, match="DPPrior"):
        prior_functionals(object(), grid, 5, t_max=T_MAX)  # type: ignore[arg-type]

    for bad_draws in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError, match="n_draws"):
            prior_functionals(PRIOR, grid, bad_draws, t_max=T_MAX)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="time scale"):
        prior_functionals(PRIOR, grid, 5)

    with pytest.raises(ValueError, match="data must be a SurvData"):
        prior_functionals(PRIOR, grid, 5, data=object())  # type: ignore[arg-type]

    data = make_surv_data([1.0, 2.0], [True, False])
    with pytest.raises(ValueError, match="must agree"):
        prior_functionals(PRIOR, grid, 5, data=data, t_max=3.0)

    for bad_t_max in (0.0, -1.0, np.nan, np.inf, True, "4.5"):
        with pytest.raises(ValueError, match="t_max"):
            prior_functionals(PRIOR, grid, 5, t_max=bad_t_max)  # type: ignore[arg-type]

    for bad_epsilon in (0.0, 1.0, -0.5, np.nan, True):
        with pytest.raises(ValueError, match="epsilon"):
            prior_functionals(PRIOR, grid, 5, t_max=T_MAX, epsilon=bad_epsilon)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="FunctionalSettings"):
        prior_functionals(PRIOR, grid, 5, t_max=T_MAX, settings=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="method must be 'exact'"):
        prior_functionals(
            PRIOR,
            grid,
            5,
            t_max=T_MAX,
            settings=FunctionalSettings(method="mc"),
        )

    for bad_seed in (-1, 2.5, True):
        with pytest.raises(ValueError, match="seed"):
            prior_functionals(PRIOR, grid, 5, t_max=T_MAX, seed=bad_seed)  # type: ignore[arg-type]

    for bad_grid in (np.array([0.0, 1.0]), np.array([]), np.ones((2, 2))):
        with pytest.raises(ValueError, match="t_grid"):
            prior_functionals(PRIOR, bad_grid, 5, t_max=T_MAX)
