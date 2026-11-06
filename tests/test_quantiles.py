"""Test 27 (plan Section 7.4): quantile functionals by CDF inversion.

The quantile functional is not a function of linear functionals (paper,
Section 3), so it cannot be read off the ``S``/``F`` arrays: the CDF of every
draw has to be inverted separately and the affine inverse map applied to the
quantiles only.  These tests pin four things:

* **single-atom exactness.**  With ``G = delta_(alpha, lambda)`` the quantile
  has the closed form ``t_max * ((lambda * log(1 / (1 - p))) ** (1 / alpha)
  - 1)``, which is what ``weibull_min(c=alpha, scale=lambda ** (1 / alpha))``
  returns.  The comparison is in ``t`` at ``1e-12``: an inversion that only
  interpolates the grid CDF is ``O(h**2)`` in ``t`` and fails by orders of
  magnitude even though the curve looks right (the test asserts this, so the
  tolerance is not vacuous).
* **mixture accuracy.**  For a three-atom ``G_b`` the inverted quantiles must
  equal the roots found by ``brentq`` on the same exact mixture CDF, to the
  plan's ``1e-6`` in ``t`` (and in fact much more tightly), and the CDF at the
  inverted value must return ``p``.
* **the production path.**  ``posterior_quantiles`` inverts the CDF of the
  very ``G_b`` draws that ``posterior_functionals`` evaluates (same seed, same
  ``J_common``, same pruning), so the two agree on a grid inversion, the
  result is monotone in ``p``, replayable from the seed, and stacked chain by
  chain.
* **loud failures.**  A probability the grid does not bracket, an unsorted or
  negative grid, a mistyped ``p`` and a Monte Carlo settings object are input
  errors.  Clamping ``p`` to the grid edge, or silently drawing atoms for a
  quantile, would look plausible and be wrong.
"""

from typing import Any, cast

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import weibull_min

from src.dp_functionals import (
    FunctionalSettings,
    posterior_functionals,
    posterior_quantiles,
    quantiles_from_realizations,
)
from src.dp_gibbs import DPPrior, DPSettings, SurvData, make_surv_data, sample_chain
from src.dp_weibull import DPRealization

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

#: Deliberately coarse grid (``h ~ 0.08`` on the original scale): the plan's
#: tolerance argument is about grid inversion error, so a coarse grid shows
#: that the Newton polish, not the grid spacing, is doing the work.
COARSE_GRID = np.linspace(0.0, 24.0, 301)

#: Probabilities reachable from ``t = 0`` for the single atom used below.  The
#: transformed time at the origin is ``1``, not ``0``, so the mixture CDF is
#: already positive there and small ``p`` cannot be bracketed.
P_VALUES = np.array([0.20, 0.40, 0.50, 0.75, 0.90])

#: The three-atom realisation already carries ``F(0) ~ 0.27``, so its
#: reachable probabilities start above that.
MIXTURE_P = np.array([0.40, 0.50, 0.60, 0.75, 0.90])


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


def _three_atom() -> DPRealization:
    """A small multi-atom realisation with mass in the interesting range."""
    return DPRealization(
        weights=np.array([0.50, 0.30, 0.20]),
        alphas=np.array([1.6, 2.2, 2.8]),
        lambdas=np.array([2.0, 5.0, 9.0]),
        precision=10.0,
    )


def _scipy_quantile_tilde(alpha: float, lam: float, p: float) -> float:
    """Exact Weibull quantile on the ``t_tilde`` scale, from SciPy.

    ``weibull_min(c=alpha, scale=lambda ** (1 / alpha))`` is the paper's
    ``exp(-t**alpha / lambda)`` survival, so its ``ppf`` is the exact
    ``eta_tilde``.  Building the reference from SciPy rather than from the
    module keeps the two-lambda conversion from cancelling.
    """
    return float(weibull_min(c=alpha, scale=lam ** (1.0 / alpha)).ppf(p))


def _mixture_quantile_tilde(realization: DPRealization, p: float) -> float:
    """Reference mixture quantile by ``brentq`` on the exact CDF.

    The bracket is the transformed grid, which is where a real call would
    invert; ``brentq`` then converges to machine precision, so this reference
    is independent of any grid interpolation.
    """
    data = _data()
    t_tilde = 1.0 + COARSE_GRID / data.t_max

    def residual(t: float) -> float:
        return float(np.asarray(realization.cdf(np.array([t])))[0]) - p

    root = brentq(residual, float(t_tilde[0]), float(t_tilde[-1]), xtol=1e-15)
    return cast(float, root)


class TestSingleAtomExactness:
    """A one-atom ``G`` inverts exactly, not to grid-interpolation accuracy."""

    @pytest.mark.parametrize("p", P_VALUES)
    def test_quantile_equals_the_closed_form(self, p: float) -> None:
        """``eta_p = t_max * (eta_tilde_p - 1)`` to ``1e-12`` in ``t``."""
        data = _data()
        alpha, lam = 1.7, 8.0
        realization = _single_atom(alpha, lam)

        quantiles = quantiles_from_realizations([realization], COARSE_GRID, p, data)

        expected = data.t_max * (_scipy_quantile_tilde(alpha, lam, p) - 1.0)
        assert quantiles.shape == (1,)
        assert quantiles[0] == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_the_affine_inverse_map_is_applied(self) -> None:
        """The result is on the original scale, ``t_max * (eta_tilde - 1)``."""
        data = _data()
        alpha, lam, p = 1.7, 8.0, 0.5

        quantiles = quantiles_from_realizations(
            [_single_atom(alpha, lam)], COARSE_GRID, p, data
        )
        eta_tilde = _scipy_quantile_tilde(alpha, lam, p)

        assert not np.allclose(quantiles[0], eta_tilde)

    def test_pure_grid_interpolation_would_miss_the_tolerance(self) -> None:
        """Interpolating the grid CDF is ``O(h**2)`` and fails ``1e-6`` in t.

        This keeps the exactness assertion above meaningful: a regression to
        ``np.interp`` on the grid would leave every curve plausible but fail
        the tight bound used there.
        """
        data = _data()
        alpha, lam, p = 1.7, 8.0, 0.5
        realization = _single_atom(alpha, lam)
        t_tilde = 1.0 + COARSE_GRID / data.t_max

        interpolated = np.interp(p, realization.cdf(t_tilde), COARSE_GRID)
        interpolated_tilde = np.interp(p, realization.cdf(t_tilde), t_tilde)
        exact = data.t_max * (_scipy_quantile_tilde(alpha, lam, p) - 1.0)

        assert abs(interpolated - exact) > 1e-6
        assert abs(data.t_max * (interpolated_tilde - 1.0) - exact) > 1e-6


class TestMixtureInversion:
    """A multi-atom CDF inverts to the ``brentq`` root of the same CDF."""

    @pytest.mark.parametrize("p", MIXTURE_P)
    def test_quantile_matches_brentq_on_the_exact_cdf(self, p: float) -> None:
        """Mixture quantiles agree with the exact root to the plan's ``1e-6``."""
        data = _data()
        realization = _three_atom()

        quantiles = quantiles_from_realizations([realization], COARSE_GRID, p, data)
        expected = data.t_max * (_mixture_quantile_tilde(realization, p) - 1.0)

        assert quantiles[0] == pytest.approx(expected, rel=0.0, abs=1e-6)
        # The implementation refines on the exact CDF, so it is far better
        # than the plan's grid-interpolation tolerance.
        assert quantiles[0] == pytest.approx(expected, rel=0.0, abs=1e-9)

    @pytest.mark.parametrize("p", MIXTURE_P)
    def test_the_cdf_at_the_quantile_returns_p(self, p: float) -> None:
        """``F_b(eta_p) = p`` on the transformed scale, to ``1e-12``."""
        data = _data()
        realization = _three_atom()

        eta = quantiles_from_realizations([realization], COARSE_GRID, p, data)[0]
        eta_tilde = 1.0 + eta / data.t_max
        attained = float(np.asarray(realization.cdf(np.array([eta_tilde])))[0])

        assert attained == pytest.approx(p, rel=0.0, abs=1e-12)
        assert eta >= 0.0

    def test_quantiles_are_monotone_in_p(self) -> None:
        """A monotone CDF gives strictly increasing quantiles in ``p``."""
        data = _data()
        realization = _three_atom()

        quantiles = [
            quantiles_from_realizations([realization], COARSE_GRID, p, data)[0]
            for p in MIXTURE_P
        ]

        assert np.all(np.diff(quantiles) > 0.0)

    def test_one_row_per_realisation_in_order(self) -> None:
        """Realisation order is preserved, one quantile per ``G_b``."""
        data = _data()
        atoms = [_single_atom(1.7, 8.0), _three_atom(), _single_atom(2.4, 30.0)]

        quantiles = quantiles_from_realizations(atoms, COARSE_GRID, 0.5, data)

        assert quantiles.shape == (3,)
        for index, realization in enumerate(atoms):
            expected = data.t_max * (_mixture_quantile_tilde(realization, 0.5) - 1.0)
            assert quantiles[index] == pytest.approx(expected, rel=0.0, abs=1e-9)


class TestProductionPath:
    """``posterior_quantiles`` on real chains: schema, replay and consistency."""

    def test_schema_and_replay(self) -> None:
        """One quantile per stored draw, reproducible from the seed alone."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.0, 1.15 * data.t_max, 300)

        first = posterior_quantiles(chain, grid, 0.35, seed=1234)
        replay = posterior_quantiles(chain, grid, 0.35, seed=1234)
        other = posterior_quantiles(chain, grid, 0.35, seed=1235)

        assert first.shape == (chain.n_draws,)
        assert first.dtype == np.float64
        np.testing.assert_array_equal(first, replay)
        assert not np.array_equal(first, other)

    def test_matches_the_functional_cdf_of_the_same_draws(self) -> None:
        """The same seed gives the same ``G_b`` as ``posterior_functionals``.

        Both paths draw ``G_b`` with the same ``J_common``, the same pruning
        and the same random stream, so the quantiles invert the very CDF that
        ``Functionals.F`` holds.  A grid interpolation of that CDF must land
        on the same quantiles up to the grid's own ``O(h**2)`` error.
        """
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        quantile_grid = np.linspace(0.0, 1.15 * data.t_max, 300)
        positive_grid = np.array(quantile_grid, dtype=np.float64, copy=True)
        positive_grid[0] = 1e-9
        p = 0.35

        eta = posterior_quantiles(chain, quantile_grid, p, seed=1234)
        fit = posterior_functionals(chain, positive_grid, seed=1234)

        # ``F`` is positive at the origin, so the grid inverse must be taken
        # against the full quantile grid; both call the same ``G_b``.
        interpolated = np.array(
            [np.interp(p, fit.F[row], quantile_grid) for row in range(fit.n_draws)]
        )
        np.testing.assert_allclose(eta, interpolated, rtol=0.0, atol=1e-2)

    def test_quantiles_are_monotone_in_p(self) -> None:
        """Every draw's quantiles increase with the probability."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.0, 1.15 * data.t_max, 300)

        draws = np.vstack(
            [posterior_quantiles(chain, grid, p, seed=99) for p in (0.30, 0.35, 0.40)]
        )

        assert np.all(np.diff(draws, axis=0) > 0.0)

    def test_stacks_multiple_chains_in_order(self) -> None:
        """Two chains contribute ``B_1 + B_2`` quantiles."""
        data = _data()
        chain_a = sample_chain(data, PRIOR, _settings(seed=SEED))
        chain_b = sample_chain(data, PRIOR, _settings(seed=SEED + 1))
        grid = np.linspace(0.0, 1.15 * data.t_max, 300)

        eta = posterior_quantiles([chain_a, chain_b], grid, 0.35, seed=7)

        assert eta.shape == (chain_a.n_draws + chain_b.n_draws,)

    def test_rejects_chains_on_different_datasets(self) -> None:
        """A shared quantile grid requires a shared ``t_max``."""
        data = _data()
        other = make_surv_data(
            time=np.array([0.5, 1.0, 2.0, 3.0]),
            event=np.array([True, True, False, True]),
        )
        grid = np.linspace(0.0, 1.15 * data.t_max, 30)

        with pytest.raises(ValueError, match=r"same number of observations"):
            posterior_quantiles(
                [
                    sample_chain(data, PRIOR, _settings()),
                    sample_chain(other, PRIOR, _settings()),
                ],
                grid,
                0.35,
                seed=1,
            )


class TestValidation:
    """Malformed inputs fail loudly instead of returning a plausible curve."""

    @pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.5, True, [0.5], "0.5"])
    def test_rejects_invalid_probabilities(self, bad_p: Any) -> None:
        """``p`` is a single open-interval probability, not a bool or array."""
        data = _data()
        with pytest.raises(ValueError, match=r"p must"):
            quantiles_from_realizations(
                [_single_atom(1.7, 8.0)], COARSE_GRID, bad_p, data
            )

    def test_rejects_an_unbracketed_probability(self) -> None:
        """``p`` outside the grid CDF range is an error, never a clamp."""
        data = _data()
        grid = np.linspace(0.0, 0.1, 11)

        with pytest.raises(ValueError, match=r"not bracketed by the CDF"):
            quantiles_from_realizations([_single_atom(1.7, 8.0)], grid, 0.9, data)

    def test_rejects_a_negative_grid(self) -> None:
        """The original-scale quantile grid starts at the origin."""
        data = _data()
        with pytest.raises(ValueError, match=r"must be non-negative"):
            quantiles_from_realizations(
                [_single_atom(1.7, 8.0)], np.linspace(-1.0, 5.0, 7), 0.5, data
            )

    @pytest.mark.parametrize(
        "bad_grid",
        [
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 1.0, 3.0]),
            np.array([1.0]),
        ],
    )
    def test_rejects_a_grid_that_cannot_be_searched(self, bad_grid: Any) -> None:
        """CDF inversion needs a strictly increasing grid of at least two points."""
        data = _data()
        with pytest.raises(ValueError, match=r"t_grid must"):
            quantiles_from_realizations([_single_atom(1.7, 8.0)], bad_grid, 0.5, data)

    def test_rejects_an_invalid_settings_object(self) -> None:
        """Settings is a ``FunctionalSettings``, not a mapping or a string."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.0, 1.15 * data.t_max, 30)

        with pytest.raises(ValueError, match=r"settings must be a FunctionalSettings"):
            posterior_quantiles(chain, grid, 0.35, settings={"method": "exact"})  # type: ignore[arg-type]

    def test_rejects_the_monte_carlo_method(self) -> None:
        """Quantiles invert the exact CDF; the MC path draws atoms instead."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.0, 1.15 * data.t_max, 30)

        with pytest.raises(ValueError, match=r"settings.method must be 'exact'"):
            posterior_quantiles(
                chain,
                grid,
                0.35,
                settings=FunctionalSettings(method="mc", L=10),
            )

    def test_rejects_an_unnormalised_realisation(self) -> None:
        """Weights that do not sum to one cannot define a proper CDF."""
        data = _data()
        bad = DPRealization(
            weights=np.array([0.5]),
            alphas=np.array([1.7]),
            lambdas=np.array([8.0]),
            precision=1.0,
        )
        with pytest.raises(ValueError, match=r"weights must sum to one"):
            quantiles_from_realizations([bad], COARSE_GRID, 0.5, data)

    def test_rejects_non_realisations_and_empty_input(self) -> None:
        """The engine validates its iterable, like the functional engines."""
        data = _data()
        with pytest.raises(ValueError, match=r"only DPRealization instances"):
            quantiles_from_realizations([object()], COARSE_GRID, 0.5, data)  # type: ignore[list-item]
        with pytest.raises(ValueError, match=r"at least one realisation"):
            quantiles_from_realizations([], COARSE_GRID, 0.5, data)
