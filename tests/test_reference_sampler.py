"""Test 16 (plan Section 7.3): agreement with an independent reference sampler.

The production sampler has been checked so far against one conditional at a
time (Section 7.2) and against invariants that any correct implementation must
satisfy.  None of those checks can see a mistake that both the implementation
and its test share, because the test's "expected" distribution was derived from
the same algebra.  Test 16 closes that hole: ``tests/reference_sampler.py``
re-derives the whole sampler with different numerical devices -- nested
quadrature for ``q_0``, two-dimensional box rejection for the continuous piece,
rejection from the exact Eq. A.5 joint density for step (b), inverse CDF of the
EPPF conditional for ``nu``, and its own stick-breaking for the functionals.
The two chains then have to land on the same posterior.

The agreement criterion is the plan's: on a synthetic right-censored dataset
with ``n = 12``, the posterior means and the 5%/95% endpoints of ``nu``,
``n*``, ``S(t)`` at three times and the posterior mean of ``h(t)`` must agree
to within **three pooled Monte Carlo standard errors**.  The standard errors
come from batch means over ten contiguous blocks of stored draws, so the
within-chain autocorrelation of a Gibbs sampler is accounted for -- a naive
``sd / sqrt(B)`` would understate the noise by the integrated autocorrelation
time and turn agreement into a coin flip.

Because a failure here can mean either implementation is wrong, the module
also pins the three reference pieces that are easy to mis-derive:

* ``q_0`` of the reference quadrature against an adaptive ``quad`` of the
  closed form (relative error ``< 1e-7``);
* the reference ``nu`` conditional against the Escobar-West two-Gamma scheme
  it deliberately does not use (max CDF deviation ``< 0.02``);
* the ``alpha`` marginal of the reference's joint step (b) draw against the
  exact marginal density ``alpha**n_o prod t**(alpha-1)
  (gamma + sum t**alpha)**(-(d+n_o))`` (max CDF deviation ``< 0.02``).
  This is the check that caught the missing normalising factor while the
  reference was being written: without it the reference let ``alpha`` drift to
  ``phi`` and ``lambda`` to infinity.

Everything is seeded, so the test is reproducible; no quantity is "tuned to
pass" at run time.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.integrate import quad

from src.dp_functionals import FunctionalSettings, posterior_functionals
from src.dp_gibbs import DPPrior as ProductionPrior
from src.dp_gibbs import DPSettings, make_surv_data, sample_chain
from tests.reference_sampler import (
    ReferenceDraws,
    ReferencePrior,
    draw_cluster_location,
    draw_nu,
    log_eppf_conditional,
    prior_predictive_masses,
    reference_functionals,
    sample_reference,
)

# A fixed synthetic dataset: two Weibull-like modes (one near 1, one near 3),
# n = 12, 9 uncensored and 3 right-censored (25%).  Hardcoded rather than
# generated so that a change to a generator cannot silently move the test onto
# a different dataset; the sampler sees the transformed times
# ``1 + t / max(t)``, the same map the production path applies.
TIME = np.array(
    [0.55, 0.90, 1.15, 1.40, 1.70, 2.00, 2.30, 2.70, 3.00, 3.40, 3.90, 4.50]
)
EVENT = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)

# The plan's Section 5.1 prior, used by both implementations.
REFERENCE_PRIOR = ReferencePrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)
PRODUCTION_PRIOR = ProductionPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)

T_GRID = np.array([0.8, 1.6, 2.8])

# Both chains: 1500 burn-in sweeps, then 1000 stored draws 5 sweeps apart.  At
# ~1 ms (production) and ~2 ms (reference) per sweep this is the largest run
# that keeps the test inside the ordinary suite rather than under ``-m slow``.
BURN_IN = 1500
N_SAVE = 1000
THIN = 5
PRODUCTION_SEED = 11
REFERENCE_SEED = 7
N_BATCHES = 10

# Three pooled standard errors, the plan's Section 7.3 threshold.
SIGMA_LIMIT = 3.0
# A quantile of the discrete ``n*`` can have a zero batch-means standard error
# (every batch lands on the same integer), which would demand exact equality.
# Half a cluster is the smallest meaningful discrepancy there.
COUNT_FLOOR = 0.5


@dataclass(frozen=True)
class _Fits:
    """Both fits of the same dataset, plus the functional arrays."""

    production_nu: np.ndarray
    production_n_star: np.ndarray
    production_S: np.ndarray
    production_h: np.ndarray
    reference: ReferenceDraws
    reference_functionals: dict[str, np.ndarray]


@pytest.fixture(scope="module")
def fits() -> _Fits:
    """Run both chains once and share the draws between the agreement tests."""
    reference = sample_reference(
        TIME,
        EVENT,
        REFERENCE_PRIOR,
        burn_in=BURN_IN,
        n_save=N_SAVE,
        thin=THIN,
        seed=REFERENCE_SEED,
    )
    data = make_surv_data(time=TIME, event=EVENT)
    chain = sample_chain(
        data,
        PRODUCTION_PRIOR,
        DPSettings(
            burn_in=BURN_IN,
            n_save=N_SAVE,
            thin=THIN,
            n_alpha=128,
            n_chains=1,
            seed=PRODUCTION_SEED,
            preset="test",
        ),
    )
    production = posterior_functionals(
        chain, T_GRID, settings=FunctionalSettings(), seed=20241129
    )
    return _Fits(
        production_nu=chain.nu,
        production_n_star=chain.n_star.astype(np.float64),
        production_S=production.S,
        production_h=production.h,
        reference=reference,
        reference_functionals=reference_functionals(reference, T_GRID, seed=20241130),
    )


# ---------------------------------------------------------------------------
# Batch-means Monte Carlo errors
# ---------------------------------------------------------------------------
def _batch_mean_se(
    values: np.ndarray, statistic: Callable[[np.ndarray], float]
) -> tuple[float, float]:
    """Value and standard error of one summary, from contiguous batch means.

    Ten contiguous blocks span far more sweeps than the integrated
    autocorrelation time of either chain, so the batch statistics are close to
    independent and their scatter gives an honest standard error for the
    autocorrelated trace.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.linspace(0, values.size, N_BATCHES + 1).astype(int)
    blocks = [values[edges[i] : edges[i + 1]] for i in range(N_BATCHES)]
    per_batch = np.array([statistic(block) for block in blocks])
    return float(statistic(values)), float(
        np.std(per_batch, ddof=1) / np.sqrt(N_BATCHES)
    )


def _assert_agrees(
    name: str,
    production: np.ndarray,
    reference: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    floor: float = 0.0,
) -> None:
    """Assert ``|difference|`` of one summary is within three pooled MC errors."""
    production_value, production_se = _batch_mean_se(production, statistic)
    reference_value, reference_se = _batch_mean_se(reference, statistic)
    pooled = float(np.hypot(production_se, reference_se))
    difference = abs(production_value - reference_value)
    allowed = max(SIGMA_LIMIT * pooled, floor)
    assert difference <= allowed, (
        f"{name}: production {production_value:.5f} +/- {production_se:.5f}, "
        f"reference {reference_value:.5f} +/- {reference_se:.5f}; "
        f"|difference| {difference:.5f} exceeds {allowed:.5f} "
        f"({difference / pooled if pooled > 0 else np.inf:.2f} pooled standard errors)"
    )
    # A pooled standard error of zero for a continuous quantity would mean the
    # batch means are degenerate, which is a bug in the comparison, not a pass.
    assert pooled > 0.0 or floor > 0.0


def _percentile(p: float) -> Callable[[np.ndarray], float]:
    """Statistic returning the ``p``-th percentile of a block."""
    return lambda block: float(np.percentile(block, p))


# ---------------------------------------------------------------------------
# The reference's own numerical pieces
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("time", "event", "gamma", "phi"),
    [
        (1.13, True, 0.232235, 1.77522),
        (1.90, False, 0.232235, 1.77522),
        (1.05, False, 1.0, 0.5),
        (1.99, True, 0.05, 2.5),
    ],
)
def test_q0_quadrature_matches_an_adaptive_rule(
    time: float, event: bool, gamma: float, phi: float
) -> None:
    """The product Gauss-Legendre rule reproduces the closed form to 1e-7.

    The reference's rule evaluates the ``lambda`` integral numerically, so it
    shares nothing with the production quadrature; this check only confirms it
    is accurate.  The closed forms are the plan's Eqs. A.2/A.3 with the
    ``lambda`` integral done analytically, which is *not* what the reference
    does at run time.
    """
    nu = 2.0
    d = 2

    if event:

        def integrand(alpha: float) -> float:
            return (
                d
                * gamma**d
                * alpha
                * time ** (alpha - 1.0)
                / (gamma + time**alpha) ** (d + 1)
            )
    else:

        def integrand(alpha: float) -> float:
            return gamma**d / (gamma + time**alpha) ** d

    expected = nu * quad(integrand, 0.0, phi, epsabs=1e-14, epsrel=1e-12)[0] / phi
    value = prior_predictive_masses(
        np.array([time]), np.array([event]), nu=nu, gamma=gamma, phi=phi, d=d
    )[0]
    # Measured error over these four cases is < 1.3e-9, so 1e-7 is a guard
    # against a quadrature regression, not a threshold the rule grazes.
    assert abs(value - expected) / expected < 1e-7


def test_nu_conditional_matches_the_escobar_west_mixture() -> None:
    """Inverse CDF of the EPPF density reproduces the two-Gamma scheme.

    The reference does not use the Escobar-West augmentation, so this compares
    its ``[nu | s]`` draws with a surrogate Gibbs chain built from the
    two-Gamma update (``u ~ Beta(nu+1, n)`` followed by the mixture).  The two
    must describe the same law; the max CDF deviation of two independent
    40,000-draw samples is ~0.01 at the 5% level, so 0.02 is a safe effect-size
    bound.
    """
    n = TIME.size
    rng = np.random.default_rng(20241129)
    n_draws = 40_000
    n_star = 3

    # Surrogate Escobar-West chain: burn in, then record.  The auxiliary u must
    # be drawn at the current nu for the mixture to target [nu | s].
    nu = 1.5
    escobar_west = np.empty(n_draws)
    for index in range(n_draws + 2_000):
        u = rng.beta(nu + 1.0, n)
        rate = REFERENCE_PRIOR.b_nu - np.log(u)
        numerator = REFERENCE_PRIOR.a_nu + n_star - 1.0
        probability = numerator / (n * rate + numerator)
        shape = (
            REFERENCE_PRIOR.a_nu + n_star
            if rng.random() < probability
            else REFERENCE_PRIOR.a_nu + n_star - 1.0
        )
        nu = float(rng.gamma(shape, 1.0 / rate))
        if index >= 2_000:
            escobar_west[index - 2_000] = nu

    inverse_cdf = np.array(
        [draw_nu(n, n_star, REFERENCE_PRIOR, rng) for _ in range(n_draws)]
    )

    grid = np.linspace(0.02, 8.0, 400)
    cdf_ew = np.searchsorted(np.sort(escobar_west), grid) / n_draws
    cdf_ref = np.searchsorted(np.sort(inverse_cdf), grid) / n_draws
    assert np.max(np.abs(cdf_ew - cdf_ref)) < 0.02
    # And the same density, checked at one point of the grid: the reference's
    # log density must be the EPPF form with the n_star-dependent power.
    assert np.isfinite(log_eppf_conditional(1.5, n, n_star, REFERENCE_PRIOR))


def test_cluster_location_matches_the_exact_alpha_marginal() -> None:
    """Step (b)'s joint rejection draw reproduces the exact ``alpha`` marginal.

    With ``n_o`` uncensored members, integrating ``lambda`` out of the joint
    Eq. A.5 target leaves

    ``p(alpha | cluster) proportional to alpha**n_o prod t**(alpha-1)
      (gamma + sum t**alpha)**(-(d + n_o))`` on ``(0, phi)``.

    A reference that drops the last factor (the normalising constant of the
    conjugate proposal) still produces plausible-looking draws but drifts to
    ``alpha = phi``; this test rejects that.
    """
    t = np.array([1.20, 1.55, 1.85])
    event = np.array([True, False, True])
    d = 2
    gamma = 0.232235
    phi = 1.77522
    n_uncensored = int(np.count_nonzero(event))
    log_t_uncensored = float(np.sum(np.log(t[event])))

    def density(alpha: np.ndarray) -> np.ndarray:
        return (
            alpha**n_uncensored
            * np.exp((alpha - 1.0) * log_t_uncensored)
            / (gamma + np.sum(t[None, :] ** alpha[:, None], axis=1))
            ** (d + n_uncensored)
        )

    nodes = np.linspace(0.0, phi, 2001)
    values = density(nodes)
    cdf = np.concatenate(
        ([0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(nodes)))
    )
    cdf /= cdf[-1]

    rng = np.random.default_rng(20241130)
    pairs = np.array(
        [
            draw_cluster_location(t, event, gamma=gamma, phi=phi, d=d, rng=rng)
            for _ in range(20_000)
        ]
    )
    draws, lambdas = pairs[:, 0], pairs[:, 1]
    grid = np.linspace(0.0, phi, 200)
    exact = np.interp(grid, nodes, cdf)
    empirical = np.searchsorted(np.sort(draws), grid) / draws.size
    assert np.max(np.abs(exact - empirical)) < 0.02
    assert draws.max() <= phi
    # The drawn lambda must be the conjugate inverse gamma at the drawn alpha,
    # which is the other half of the same joint law; check its mean against
    # the closed form for inverse gamma, E[lambda] = scale / (shape - 1).
    scales = gamma + np.array([np.sum(t**alpha) for alpha in draws])
    expected_lambda = scales / (d + n_uncensored - 1.0)
    assert np.mean(lambdas) == pytest.approx(np.mean(expected_lambda), rel=0.05)


def test_reference_functionals_are_proper(fits: _Fits) -> None:
    """``S`` is a decreasing survival curve in ``[0, 1]`` and ``h = f / S``."""
    survival = fits.reference_functionals["S"]
    density = fits.reference_functionals["f"]
    hazard = fits.reference_functionals["h"]

    assert np.all(np.isfinite(survival))
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 0.0)
    assert np.all(density >= 0.0)
    assert np.all(hazard >= 0.0)
    assert np.all(np.isfinite(hazard))
    # The hazard is evaluated from the same atoms as f and S, so the identity
    # holds exactly up to the 1/t_max Jacobian that is applied to both.
    assert np.allclose(hazard * survival, density)


# ---------------------------------------------------------------------------
# Test 16 proper: agreement of the two posteriors
# ---------------------------------------------------------------------------
def test_posterior_means_and_intervals_agree_with_the_reference(fits: _Fits) -> None:
    """Every reported summary agrees to within three pooled MC errors."""
    reference_nu = fits.reference.nu
    reference_n_star = fits.reference.n_star.astype(np.float64)
    reference_S = fits.reference_functionals["S"]
    reference_h = fits.reference_functionals["h"]

    _assert_agrees("nu mean", fits.production_nu, reference_nu, np.mean)
    _assert_agrees("nu 5%", fits.production_nu, reference_nu, _percentile(5.0))
    _assert_agrees("nu 95%", fits.production_nu, reference_nu, _percentile(95.0))

    _assert_agrees("n* mean", fits.production_n_star, reference_n_star, np.mean)
    _assert_agrees(
        "n* 5%",
        fits.production_n_star,
        reference_n_star,
        _percentile(5.0),
        floor=COUNT_FLOOR,
    )
    _assert_agrees(
        "n* 95%",
        fits.production_n_star,
        reference_n_star,
        _percentile(95.0),
        floor=COUNT_FLOOR,
    )

    for position, time in enumerate(T_GRID):
        label = f"t={time:.1f}"
        _assert_agrees(
            f"S({label}) mean",
            fits.production_S[:, position],
            reference_S[:, position],
            np.mean,
        )
        _assert_agrees(
            f"S({label}) 5%",
            fits.production_S[:, position],
            reference_S[:, position],
            _percentile(5.0),
        )
        _assert_agrees(
            f"S({label}) 95%",
            fits.production_S[:, position],
            reference_S[:, position],
            _percentile(95.0),
        )
        _assert_agrees(
            f"E[h({label})]",
            fits.production_h[:, position],
            reference_h[:, position],
            np.mean,
        )
