"""Tests 21, 25 and 26 (plan Section 7.4): posterior functionals.

``posterior_functionals`` is the paper's Section 3 estimator in its
Rao-Blackwellised form: for each stored draw it draws one
``G_b ~ [G | theta_b, nu_b, gamma_b, phi_b]``, prunes the negligible
stick-breaking tail, evaluates the mixture survival, CDF, density and hazard
from exactly those atoms, and maps them back to the original time scale.

These tests pin the three things that are easiest to get wrong silently:

* **test 21 -- single-atom exactness.**  A realisation with one atom of weight
  one must reproduce the closed-form Weibull functionals exactly.  The
  expected values are built from :class:`scipy.stats.weibull_min` with
  ``c = alpha`` and ``scale = lambda ** (1 / alpha)``, never from the
  module's own kernels, so the two-lambda confusion cannot cancel.
* **test 25 -- internal consistency.**  On one realisation the four arrays
  must satisfy ``S = 1 - F``, ``h = f / S``, ``integral f dt = 1`` and
  ``-d/dt log S = h`` by finite differences.  A swapped ``f``/``S`` or a
  missing normalisation fails here even though every individual curve looks
  plausible.
* **test 26 -- transformation Jacobian.**  The functional path evaluates on
  ``t_tilde = 1 + t / t_max`` and maps back with :func:`inverse_map`.
  Survival is unchanged, density and hazard carry ``1 / t_max``, and a median
  maps as ``t_max * (eta_tilde - 1)``.  Missing the Jacobian changes ``f`` and
  ``h`` by a constant and is invisible in ``S``; the explicit comparison
  against ``realization.pdf`` catches it.

The module also tests the production ``posterior_functionals`` path itself:
the common truncation level ``J``, the shape and provenance of the result,
seed-based replayability, and rejection of the not-yet-implemented ``"mc"``
method instead of silently computing the exact path under a different label.
"""

from typing import Any

import numpy as np
import pytest
from scipy.stats import weibull_min

from src.dp_functionals import (
    Functionals,
    FunctionalSettings,
    functionals_from_realizations,
    posterior_functionals,
)
from src.dp_gibbs import (
    DPPrior,
    DPSettings,
    SurvData,
    inverse_map,
    make_surv_data,
    sample_chain,
)
from src.dp_weibull import (
    DPRealization,
    truncation_number_for_posterior,
)

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)

TIME = np.array([0.4, 0.8, 1.2, 1.8, 2.4, 3.0, 3.6, 4.0])
EVENT = np.array([True, False, True, True, False, True, True, False])
SEED = 20241023


def _data() -> SurvData:
    """A small mixed-censoring dataset; ``t_max`` is 4.0."""
    return make_surv_data(time=TIME, event=EVENT)


def _settings(**overrides: object) -> DPSettings:
    """Cheap chain settings: a handful of sweeps and draws."""
    base: dict[str, object] = {
        "burn_in": 0,
        "n_save": 3,
        "thin": 1,
        "n_alpha": 32,
        "n_chains": 1,
        "seed": SEED,
        "preset": "test",
    }
    base.update(overrides)
    return DPSettings(**base)  # type: ignore[arg-type]


def _single_atom(alpha: float, lam: float) -> DPRealization:
    """A proper one-atom realisation, ``G = delta_(alpha, lam)``."""
    return DPRealization(
        weights=np.array([1.0]),
        alphas=np.array([alpha]),
        lambdas=np.array([lam]),
        precision=1.0,
    )


def _scipy_weibull(alpha: float, lam: float) -> Any:
    """The same atom as a SciPy distribution (plan's two-lambda conversion)."""
    return weibull_min(c=alpha, scale=lam ** (1.0 / alpha))


def _weibull_mixture_realization() -> DPRealization:
    """A three-atom realisation whose mass lies above ``t_tilde = 1``.

    The inverse map is a shift as well as a scale (``t = t_max *
    (t_tilde - 1)``), so a realisation with appreciable mass below
    ``t_tilde = 1`` maps that mass to negative original times.  The integral
    of ``f`` over positive original times then misses it and cannot be one.
    These atoms have ``alpha > 1`` (so ``f(0) = 0``) and large enough
    ``lambda`` that ``F_tilde(1)`` is about ``1e-4``; the test below is about
    the sum of the mixture weights, not about the prior-scale caveat of plan
    Section 5.
    """
    return DPRealization(
        weights=np.array([0.40, 0.35, 0.25]),
        alphas=np.array([2.5, 3.0, 2.8]),
        lambdas=np.array([8000.0, 10000.0, 12000.0]),
        precision=12.0,
    )


def _consistency_grid() -> np.ndarray:
    """A wide original-scale grid for the integral and finite-difference tests."""
    return np.linspace(1e-6, 1000.0, 10001)


class TestSingleAtomExactness:
    """Test 21: one atom of weight one is the closed-form Weibull."""

    def test_survival_density_hazard_against_scipy(self) -> None:
        """S, F, f and h equal ``weibull_min`` with the paper's conversion."""
        data = _data()
        alpha, lam = 1.6, 2.0
        realization = _single_atom(alpha, lam)
        t_grid = np.linspace(0.01, 8.0, 17)
        t_tilde = 1.0 + t_grid / data.t_max

        fit = functionals_from_realizations([realization], t_grid, data)

        reference = _scipy_weibull(alpha, lam)
        expected_survival = reference.sf(t_tilde)
        expected_density = reference.pdf(t_tilde) / data.t_max
        expected_hazard = reference.pdf(t_tilde) / reference.sf(t_tilde) / data.t_max

        np.testing.assert_allclose(fit.S[0], expected_survival, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            fit.F[0], reference.cdf(t_tilde), rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(fit.f[0], expected_density, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(fit.h[0], expected_hazard, rtol=0.0, atol=1e-13)

        assert fit.n_draws == 1
        assert fit.n_grid == t_grid.size
        assert fit.meta["n_draws"] == 1
        assert fit.meta["t_max"] == pytest.approx(data.t_max)

    def test_two_lambda_conversion_is_not_identity(self) -> None:
        """The SciPy reference is really the paper's parameterisation.

        If the test used ``scale=lambda`` (the standard Weibull convention
        instead of ``lambda ** (1 / alpha)``) it would disagree with the
        closed form; this asserts the conversion matters, so the exactness
        test above is not vacuous.
        """
        data = _data()
        alpha, lam = 1.6, 2.0
        realization = _single_atom(alpha, lam)
        t_grid = np.linspace(0.05, 5.0, 11)
        t_tilde = 1.0 + t_grid / data.t_max

        fit = functionals_from_realizations([realization], t_grid, data)
        wrong_scale = weibull_min(c=alpha, scale=lam).sf(t_tilde)

        assert not np.allclose(fit.S[0], wrong_scale)


class TestInternalConsistency:
    """Test 25: the four functionals are one coherent object."""

    def test_survival_is_one_minus_cdf(self) -> None:
        """``S = 1 - F`` exactly, on every draw and grid point."""
        data = _data()
        realization = _weibull_mixture_realization()
        t_grid = _consistency_grid()

        fit = functionals_from_realizations([realization], t_grid, data)

        np.testing.assert_allclose(fit.S + fit.F, 1.0, rtol=0.0, atol=1e-15)

    def test_density_integrates_to_one(self) -> None:
        """The trapezoid integral of ``f`` over a wide grid is one to 1e-3."""
        data = _data()
        realization = _weibull_mixture_realization()
        t_grid = _consistency_grid()

        fit = functionals_from_realizations([realization], t_grid, data)
        integral = np.trapezoid(fit.f[0], t_grid)

        assert abs(integral - 1.0) < 1e-3

    def test_hazard_is_density_over_survival(self) -> None:
        """``h = f / S`` from the same atom set, not a second realisation."""
        data = _data()
        t_grid = np.linspace(0.1, 8.0, 31)
        realization = _weibull_mixture_realization()

        fit = functionals_from_realizations([realization], t_grid, data)
        expected = fit.f[0] / fit.S[0]

        np.testing.assert_allclose(fit.h[0], expected, rtol=1e-14, atol=1e-14)

    def test_log_survival_derivative_matches_hazard(self) -> None:
        """``-d/dt log S`` agrees with ``h`` by finite differences to 1e-3."""
        data = _data()
        realization = _weibull_mixture_realization()
        t_grid = _consistency_grid()

        fit = functionals_from_realizations([realization], t_grid, data)
        derivative = -np.gradient(np.log(fit.S[0]), t_grid)

        # The two outermost points use one-sided differences; the hazard is
        # also less informative where the survival has underflowed.
        interior = slice(2, -2)
        mask = fit.S[0, interior] > 1e-6
        relative = np.abs(derivative[interior][mask] - fit.h[0, interior][mask])
        relative = relative / np.abs(fit.h[0, interior][mask])

        assert relative.size > 0
        assert np.max(relative) < 1e-3

    def test_direct_construction_validates_shapes_and_bounds(self) -> None:
        """A transposed or negative functional array fails at the boundary."""
        t_grid = np.linspace(0.1, 1.0, 4)
        good = np.ones((2, 4))
        zero = np.zeros((2, 4))

        fit = Functionals(t_grid=t_grid, S=good, F=zero, f=zero, h=zero, meta={})
        assert fit.n_draws == 2 and fit.n_grid == 4

        with pytest.raises(ValueError, match=r"one column per t_grid point"):
            Functionals(t_grid=t_grid, S=good[:, :-1], F=zero, f=zero, h=zero)
        with pytest.raises(ValueError, match=r"t_grid must be strictly positive"):
            Functionals(
                t_grid=np.array([0.0, 0.5]),
                S=np.ones((1, 2)),
                F=np.zeros((1, 2)),
                f=np.zeros((1, 2)),
                h=np.zeros((1, 2)),
            )
        with pytest.raises(ValueError, match=r"S must lie in"):
            Functionals(
                t_grid=t_grid,
                S=good * 2.0,
                F=zero,
                f=zero,
                h=zero,
            )

    def test_rejects_an_unnormalised_realisation(self) -> None:
        """Weights that do not sum to one cannot define a proper CDF."""
        data = _data()
        realization = DPRealization(
            weights=np.array([0.5]),
            alphas=np.array([1.5]),
            lambdas=np.array([1.0]),
            precision=1.0,
        )
        with pytest.raises(ValueError, match=r"weights must sum to one"):
            functionals_from_realizations([realization], np.linspace(0.1, 2.0, 5), data)


class TestTransformationJacobian:
    """Test 26: the four inverse maps of plan Section 2.3."""

    def test_survival_density_and_hazard_map_back(self) -> None:
        """``S`` is unchanged; ``f`` and ``h`` carry exactly ``1 / t_max``."""
        data = _data()
        alpha, lam = 1.7, 3.0
        realization = _single_atom(alpha, lam)
        t_grid = np.linspace(0.25, 6.0, 23)
        t_tilde = 1.0 + t_grid / data.t_max

        fit = functionals_from_realizations([realization], t_grid, data)

        survival_tilde = realization.survival(t_tilde)
        density_tilde = realization.pdf(t_tilde)
        hazard_tilde = density_tilde / survival_tilde

        np.testing.assert_allclose(fit.S[0], survival_tilde, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            fit.f[0], density_tilde / data.t_max, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            fit.h[0], hazard_tilde / data.t_max, rtol=0.0, atol=0.0
        )

        # The Jacobian is not 1 for this dataset; a forgotten factor would
        # make these equal (which is the classic error the test targets).
        assert not np.allclose(fit.f[0], density_tilde)
        assert not np.allclose(fit.h[0], hazard_tilde)

    def test_median_uses_the_affine_inverse_map(self) -> None:
        """``eta = t_max * (eta_tilde - 1)`` for the Weibull median."""
        data = _data()
        alpha, lam = 1.7, 3.0
        eta_tilde = (lam * np.log(2.0)) ** (1.0 / alpha)

        mapped = inverse_map(data, np.array([1.0]), quantiles=np.array([eta_tilde]))[
            "quantiles"
        ]

        assert mapped[0] == pytest.approx(
            data.t_max * (eta_tilde - 1.0), rel=0.0, abs=1e-12
        )

    def test_probability_of_a_larger_median_is_transform_invariant(self) -> None:
        """A monotone affine map cannot change which median is larger."""
        data = _data()
        grid = np.linspace(0.05, 10.0, 4001)
        t_tilde = 1.0 + grid / data.t_max

        fit_a = functionals_from_realizations([_single_atom(1.8, 4.0)], grid, data)
        fit_b = functionals_from_realizations([_single_atom(1.2, 1.0)], grid, data)

        # Invert the CDF on the transformed grid and on the original grid;
        # the second is the first pushed through the affine map.
        p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
        eta_tilde_a = np.interp(p, fit_a.F[0], t_tilde)
        eta_tilde_b = np.interp(p, fit_b.F[0], t_tilde)
        eta_a = np.interp(p, fit_a.F[0], grid)
        eta_b = np.interp(p, fit_b.F[0], grid)

        mapped_a = inverse_map(data, grid, quantiles=eta_tilde_a)["quantiles"]
        mapped_b = inverse_map(data, grid, quantiles=eta_tilde_b)["quantiles"]

        np.testing.assert_allclose(eta_a, mapped_a, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(eta_b, mapped_b, rtol=0.0, atol=1e-12)
        assert np.array_equal(eta_a > eta_b, eta_tilde_a > eta_tilde_b)


class TestPosteriorFunctionals:
    """The production exact path: common J, schema, provenance and replay."""

    def test_common_truncation_level_and_schema(self) -> None:
        """One ``J_common`` for the run drives every ``G_b`` draw."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        t_grid = np.linspace(0.05, 1.2 * data.t_max, 19)

        fit = posterior_functionals(chain, t_grid, seed=1234)

        expected_j = truncation_number_for_posterior(
            float(np.max(chain.nu)), chain.n, chain.settings.epsilon
        )
        assert isinstance(fit, Functionals)
        assert fit.S.shape == (chain.n_draws, t_grid.size)
        assert fit.F.shape == fit.f.shape == fit.h.shape == fit.S.shape
        assert fit.meta["method"] == "exact"
        assert fit.meta["J_common"] == expected_j
        assert fit.meta["n_draws"] == chain.n_draws
        assert fit.meta["n_chains"] == 1
        assert fit.meta["t_max"] == pytest.approx(data.t_max)

    def test_same_seed_reproduces_and_a_different_seed_does_not(self) -> None:
        """The functional DP draws are seeded explicitly, never globally."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        t_grid = np.linspace(0.05, 1.0, 7)

        first = posterior_functionals(chain, t_grid, seed=99)
        replay = posterior_functionals(chain, t_grid, seed=99)
        other = posterior_functionals(chain, t_grid, seed=100)

        np.testing.assert_array_equal(first.S, replay.S)
        np.testing.assert_array_equal(first.f, replay.f)
        np.testing.assert_array_equal(first.h, replay.h)
        # Different DP realisations almost surely differ somewhere on the grid.
        assert not np.array_equal(first.S, other.S)
        assert first.meta["seed"] == 99

    def test_stacks_multiple_chains_in_order(self) -> None:
        """Two chains contribute ``B_1 + B_2`` rows and share one J."""
        data = _data()
        chain_a = sample_chain(data, PRIOR, _settings(seed=SEED))
        chain_b = sample_chain(data, PRIOR, _settings(seed=SEED + 1))
        t_grid = np.linspace(0.05, 1.0, 5)

        fit = posterior_functionals([chain_a, chain_b], t_grid, seed=7)

        expected_j = truncation_number_for_posterior(
            float(max(chain_a.nu.max(), chain_b.nu.max())),
            chain_a.n,
            chain_a.settings.epsilon,
        )
        assert fit.n_draws == chain_a.n_draws + chain_b.n_draws
        assert fit.meta["J_common"] == expected_j
        assert fit.meta["n_chains"] == 2

    def test_rejects_chains_on_different_datasets(self) -> None:
        """A shared original-scale grid requires a shared ``t_max``."""
        data = _data()
        other = make_surv_data(
            time=np.array([0.5, 1.0, 2.0, 3.0]),
            event=np.array([True, True, False, True]),
        )
        chain_a = sample_chain(data, PRIOR, _settings())
        chain_b = sample_chain(other, PRIOR, _settings())
        t_grid = np.linspace(0.05, 1.0, 5)

        with pytest.raises(ValueError, match=r"same number of observations"):
            posterior_functionals([chain_a, chain_b], t_grid, seed=1)

    def test_rejects_the_not_yet_implemented_mc_method(self) -> None:
        """``"mc"`` must not silently compute the exact path."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        t_grid = np.linspace(0.05, 1.0, 5)

        with pytest.raises(NotImplementedError, match=r"method='mc'"):
            posterior_functionals(
                chain, t_grid, settings=FunctionalSettings(method="mc"), seed=1
            )

    def test_rejects_unknown_settings(self) -> None:
        """An unknown method is an input error, not a fallback."""
        with pytest.raises(ValueError, match=r"method must be 'exact' or 'mc'"):
            FunctionalSettings(method="bogus")
        with pytest.raises(ValueError, match=r"atom_tol must lie strictly"):
            FunctionalSettings(atom_tol=1.0)
