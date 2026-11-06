"""Two-population functional comparisons (plan Section 4.4, test 26 guard).

``compare`` takes the functional summaries of two *independent* fits and
returns the three Figure 5 quantities of the paper's Section 5.2:

* the hazard ratio ``h_B(t) / h_A(t)`` (Figure 5a),
* the CDF difference ``F_B(t) - F_A(t)`` (Figure 5b),
* the posterior probability ``Pr(eta_A > eta_B | D)`` of the per-draw
  quantiles ``eta`` (Figure 5c, paper 0.9135), together with the per-draw
  quantiles themselves.

The tests pin four things that are easy to get wrong silently:

* **the arithmetic and the sign conventions.**  With a CDF that is exactly
  piecewise linear on the grid, grid inversion is exact, so the ratio,
  the difference, the quantiles and the probability are compared against
  hand-computed numbers rather than against another copy of the same code.
  ``ratio`` is B over A and ``difference`` is B minus A: swapping either
  changes every plot in Figure 5 without changing a single curve's shape.
* **closed-form Weibull agreement.**  A single-atom ``G`` has an analytic
  survival, hazard and quantile.  The comparison of two such atoms is checked
  against :mod:`scipy.stats` on the transformed scale, so the two-lambda
  confusion and a missing Jacobian both fail here.
* **pairing and transform invariance.**  Draw ``b`` of A is paired with draw
  ``b`` of B, which is valid only for independent fits of equal length; the
  ranking of the two populations, and hence the probability, must not change
  under the strictly increasing time transform of plan Section 2.3.
* **loud failures.**  Different draw counts, grids or ``t_max`` values, a
  non-monotone CDF, a probability the grid does not bracket and a malformed
  ``p`` are errors.  Silently interpolating, clamping, or comparing weeks with
  months would all produce plausible-looking Figure 5 output.
"""

from typing import Any

import numpy as np
import pytest
from scipy.stats import weibull_min

from src.dp_functionals import Functionals, compare, functionals_from_realizations
from src.dp_gibbs import SurvData, make_surv_data
from src.dp_weibull import DPRealization

#: A fine grid over the original times of the synthetic examples.  Grid
#: inversion is ``O(h**2)``, so the closed-form comparisons use a fine grid;
#: the exact piecewise-linear test below deliberately uses a coarse one.
GRID = np.linspace(0.05, 20.0, 5_001)

#: Coarse grid on which ``F = t / (2 * median)`` is exactly piecewise linear.
LINEAR_GRID = np.linspace(0.1, 40.0, 401)

#: Shared original time scale of the synthetic pairs.
T_MAX = 4.0


def _data(t_max: float = T_MAX) -> SurvData:
    """A one-row dataset whose ``t_max`` defines the inverse map."""
    return make_surv_data(time=np.array([t_max]), event=np.array([True]))


def _atom(alpha: float, lam: float) -> DPRealization:
    """A proper single-atom realisation, ``G = delta_(alpha, lam)``."""
    return DPRealization(
        weights=np.array([1.0]),
        alphas=np.array([alpha]),
        lambdas=np.array([lam]),
        precision=1.0,
    )


def _functionals(
    atoms: list[tuple[float, float]],
    *,
    t_max: float = T_MAX,
    grid: np.ndarray = GRID,
) -> Functionals:
    """Evaluate one ``G_b`` per atom on the original scale."""
    realizations = (_atom(alpha, lam) for alpha, lam in atoms)
    return functionals_from_realizations(realizations, grid, _data(t_max))


def _weibull(alpha: float, lam: float) -> Any:
    """The atom as a SciPy distribution (``scale = lambda ** (1 / alpha)``)."""
    return weibull_min(c=alpha, scale=lam ** (1.0 / alpha))


def _linear_cdf_fit(
    medians: np.ndarray,
    grid: np.ndarray,
    hazard,
    *,
    meta: dict[str, Any] | None = None,
) -> Functionals:
    """A hand-built fit whose CDF is exactly piecewise linear on ``grid``.

    ``F_b(t) = clip(t / (2 * median_b), 0, 1)`` has its median exactly at
    ``median_b`` and is linear in ``t`` between grid points, so
    piecewise-linear inversion reproduces the median to machine precision.
    The hazard is supplied independently: the comparison reads ``h`` only for
    the ratio, and keeping it independent of ``S`` makes each assertion
    test one formula.
    """
    scales = 2.0 * np.asarray(medians, dtype=np.float64)
    F = np.clip(grid[None, :] / scales[:, None], 0.0, 1.0)
    h = np.tile(np.asarray(hazard, dtype=np.float64), (F.shape[0], 1))
    return Functionals(
        t_grid=grid,
        S=1.0 - F,
        F=F,
        f=np.zeros_like(F),
        h=h,
        meta={} if meta is None else meta,
    )


def _fit_from_cdf(F: np.ndarray, grid: np.ndarray, h: np.ndarray) -> Functionals:
    """A hand-built fit from explicit CDF and hazard arrays."""
    F = np.asarray(F, dtype=np.float64)
    return Functionals(t_grid=grid, S=1.0 - F, F=F, f=np.zeros_like(F), h=h, meta={})


class TestExactArithmetic:
    """The three Figure 5 quantities on an exactly invertible synthetic pair."""

    @staticmethod
    def _pair() -> tuple[Functionals, Functionals]:
        medians_a = np.array([10.0, 10.0, 10.0, 10.0])
        medians_b = np.array([8.0, 8.0, 12.5, 12.5])
        hazard_a = 2.0 + 0.1 * LINEAR_GRID
        hazard_b = 3.0 - 0.05 * LINEAR_GRID
        meta_a = {"t_max": T_MAX, "method": "exact", "seed": 11}
        meta_b = {"t_max": T_MAX, "method": "exact", "seed": 22}
        fit_a = _linear_cdf_fit(medians_a, LINEAR_GRID, hazard_a, meta=meta_a)
        fit_b = _linear_cdf_fit(medians_b, LINEAR_GRID, hazard_b, meta=meta_b)
        return fit_a, fit_b

    def test_hazard_ratio_is_b_over_a(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, LINEAR_GRID)

        hazard_a = 2.0 + 0.1 * LINEAR_GRID
        hazard_b = 3.0 - 0.05 * LINEAR_GRID
        expected = np.tile(hazard_b / hazard_a, (fit_a.n_draws, 1))
        np.testing.assert_allclose(result["ratio"], expected, rtol=0.0, atol=0.0)
        assert result["ratio"].shape == (fit_a.n_draws, LINEAR_GRID.size)
        # The convention is B over A, not the reciprocal.
        assert not np.allclose(result["ratio"][0], 1.0 / expected[0])

    def test_cdf_difference_is_b_minus_a(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, LINEAR_GRID)

        np.testing.assert_allclose(
            result["difference"], fit_b.F - fit_a.F, rtol=0.0, atol=0.0
        )
        # Row 0: B's median (8) is smaller, so its CDF is larger, and row 2:
        # B's median (12.5) is larger, so its CDF is smaller.  Compare only
        # where neither CDF has saturated at one; above that both are flat.
        unsaturated = (fit_a.F < 1.0) & (fit_b.F < 1.0)
        assert np.all(result["difference"][0][unsaturated[0]] > 0.0)
        assert np.all(result["difference"][2][unsaturated[2]] < 0.0)

    def test_medians_and_probability_are_exact(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, LINEAR_GRID)

        np.testing.assert_allclose(
            result["eta_a"], np.array([10.0, 10.0, 10.0, 10.0]), rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            result["eta_b"],
            np.array([8.0, 8.0, 12.5, 12.5]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["eta_difference"],
            np.array([2.0, 2.0, -2.5, -2.5]),
            rtol=0.0,
            atol=1e-12,
        )
        # A exceeds B in rows 0 and 1 only.
        assert result["prob_eta_a_gt_eta_b"] == pytest.approx(0.5)

    def test_quantile_level_is_selectable(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, LINEAR_GRID, p=0.25)

        np.testing.assert_allclose(
            result["eta_a"], np.array([5.0, 5.0, 5.0, 5.0]), rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            result["eta_b"], np.array([4.0, 4.0, 6.25, 6.25]), rtol=0.0, atol=1e-12
        )
        assert result["prob_eta_a_gt_eta_b"] == pytest.approx(0.5)
        assert result["p"] == 0.25
        assert result["meta"]["p"] == 0.25

    def test_metadata_records_provenance(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, LINEAR_GRID)

        meta = result["meta"]
        assert meta["n_draws"] == 4
        assert meta["n_grid"] == LINEAR_GRID.size
        assert meta["t_max"] == T_MAX
        assert meta["method_a"] == meta["method_b"] == "exact"
        assert meta["seed_a"] == 11
        assert meta["seed_b"] == 22
        assert "independent" in meta["pairing"]
        np.testing.assert_array_equal(result["t_grid"], LINEAR_GRID)

    def test_hazard_ratio_zero_and_infinite_denominators(self) -> None:
        grid = np.array([0.5, 1.0, 1.5, 2.0])
        cdf = np.array([[0.1, 0.4, 0.7, 0.9]])
        fit_a = _fit_from_cdf(cdf, grid, np.array([[0.0, 0.0, np.inf, np.inf]]))
        fit_b = _fit_from_cdf(cdf, grid, np.array([[2.0, 0.0, 3.0, np.inf]]))

        ratio = compare(fit_a, fit_b, grid)["ratio"]
        assert np.isinf(ratio[0, 0])
        assert np.isnan(ratio[0, 1])
        assert ratio[0, 2] == 0.0
        assert np.isnan(ratio[0, 3])


class TestClosedFormWeibull:
    """Single-atom fits against :mod:`scipy.stats` on the transformed scale."""

    @staticmethod
    def _pair() -> tuple[Functionals, Functionals]:
        # A longer-lived than B: ``lambda_A = 2`` exceeds ``lambda_B = 1.5``.
        fit_a = _functionals([(1.5, 2.0)])
        fit_b = _functionals([(1.5, 1.5)])
        return fit_a, fit_b

    def test_hazard_ratio_matches_scipy(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, GRID)

        t_tilde = 1.0 + GRID / T_MAX
        expected = (
            _weibull(1.5, 1.5).pdf(t_tilde)
            / _weibull(1.5, 1.5).sf(t_tilde)
            / (_weibull(1.5, 2.0).pdf(t_tilde) / _weibull(1.5, 2.0).sf(t_tilde))
        )
        np.testing.assert_allclose(result["ratio"][0], expected, rtol=1e-10)

    def test_cdf_difference_matches_scipy(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, GRID)

        t_tilde = 1.0 + GRID / T_MAX
        expected = _weibull(1.5, 1.5).cdf(t_tilde) - _weibull(1.5, 2.0).cdf(t_tilde)
        np.testing.assert_allclose(result["difference"][0], expected, rtol=1e-10)

    def test_median_matches_scipy_ppf(self) -> None:
        fit_a, fit_b = self._pair()
        result = compare(fit_a, fit_b, GRID)

        t_tilde = float(_weibull(1.5, 2.0).ppf(0.5))
        expected_a = T_MAX * (t_tilde - 1.0)
        # Fine-grid inversion leaves only O(h**2) error in t.
        assert result["eta_a"][0] == pytest.approx(expected_a, abs=1e-6)

        t_tilde = float(_weibull(1.5, 1.5).ppf(0.5))
        expected_b = T_MAX * (t_tilde - 1.0)
        assert result["eta_b"][0] == pytest.approx(expected_b, abs=1e-6)
        assert result["eta_difference"][0] == pytest.approx(
            expected_a - expected_b, abs=2e-6
        )
        # A's atom (lambda 2) has the larger median, so the probability is one.
        assert expected_a > expected_b
        assert result["prob_eta_a_gt_eta_b"] == 1.0


class TestPairingAndInvariance:
    """Equal-length pairing and invariance of the ranking under the transform."""

    @staticmethod
    def _mirrored_atoms() -> tuple[
        list[tuple[float, float]], list[tuple[float, float]]
    ]:
        atoms_a = [(1.5, 2.0), (1.5, 2.5), (1.5, 3.0), (1.5, 3.5)]
        atoms_b = list(reversed(atoms_a))
        return atoms_a, atoms_b

    def test_probability_matches_closed_form_medians(self) -> None:
        atoms_a, atoms_b = self._mirrored_atoms()
        fit_a = _functionals(atoms_a)
        fit_b = _functionals(atoms_b)
        result = compare(fit_a, fit_b, GRID)

        scipy_medians_a = np.array(
            [T_MAX * (float(_weibull(a, lam).ppf(0.5)) - 1.0) for a, lam in atoms_a]
        )
        scipy_medians_b = np.array(
            [T_MAX * (float(_weibull(a, lam).ppf(0.5)) - 1.0) for a, lam in atoms_b]
        )
        expected_probability = float(np.mean(scipy_medians_a > scipy_medians_b))
        assert result["prob_eta_a_gt_eta_b"] == pytest.approx(expected_probability)
        np.testing.assert_allclose(
            result["eta_a"], scipy_medians_a, rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            result["eta_b"], scipy_medians_b, rtol=0.0, atol=1e-6
        )
        # Mirrored atoms make the comparison a perfect coin flip.
        assert expected_probability == 0.5

    def test_probability_is_invariant_to_the_time_transform(self) -> None:
        atoms_a, atoms_b = self._mirrored_atoms()
        fit_a_4 = _functionals(atoms_a, t_max=4.0)
        fit_b_4 = _functionals(atoms_b, t_max=4.0)
        fit_a_17 = _functionals(atoms_a, t_max=17.0)
        fit_b_17 = _functionals(atoms_b, t_max=17.0)

        result_4 = compare(fit_a_4, fit_b_4, GRID)
        result_17 = compare(fit_a_17, fit_b_17, GRID)
        assert result_4["prob_eta_a_gt_eta_b"] == 0.5
        assert result_17["prob_eta_a_gt_eta_b"] == 0.5

        # The quantiles themselves are on different original scales, which is
        # why the comparison must not be confused with a pointwise test.
        assert not np.allclose(result_4["eta_a"], result_17["eta_a"])


class TestValidation:
    """Malformed comparisons fail loudly instead of silently interpolating."""

    def test_rejects_non_functionals(self) -> None:
        fit = _functionals([(1.5, 2.0)])
        with pytest.raises(ValueError, match="fit_a must be a Functionals"):
            compare({"S": fit.S}, fit, GRID)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="fit_b must be a Functionals"):
            compare(fit, [fit], GRID)  # type: ignore[arg-type]

    def test_rejects_comparing_a_fit_with_itself(self) -> None:
        fit = _functionals([(1.5, 2.0)])
        with pytest.raises(ValueError, match="two different fits"):
            compare(fit, fit, GRID)

    def test_rejects_different_draw_counts(self) -> None:
        fit_a = _functionals([(1.5, 2.0), (1.5, 2.5)])
        fit_b = _functionals([(1.5, 2.0)])
        with pytest.raises(ValueError, match="same number of stored draws"):
            compare(fit_a, fit_b, GRID)

    def test_rejects_grids_that_do_not_match_the_fits(self) -> None:
        fit_a = _functionals([(1.5, 2.0)])
        fit_b = _functionals([(1.5, 1.5)])
        other = np.linspace(0.1, 10.0, GRID.size)
        with pytest.raises(ValueError, match="fit_a.t_grid does not match"):
            compare(fit_a, fit_b, other)

        fit_b_other = _functionals([(1.5, 1.5)], grid=other)
        with pytest.raises(ValueError, match="fit_b.t_grid does not match"):
            compare(fit_a, fit_b_other, GRID)

    def test_rejects_non_positive_grid(self) -> None:
        fit_a = _functionals([(1.5, 2.0)])
        fit_b = _functionals([(1.5, 1.5)])
        grid = GRID.copy()
        grid[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            compare(fit_a, fit_b, grid)

    def test_rejects_non_increasing_grid(self) -> None:
        fit_a = _functionals([(1.5, 2.0)])
        fit_b = _functionals([(1.5, 1.5)])
        grid = GRID.copy()
        grid[10] = grid[9]
        with pytest.raises(ValueError, match="strictly increasing"):
            compare(fit_a, fit_b, grid)

    def test_rejects_different_t_max(self) -> None:
        grid = np.linspace(0.5, 2.0, 5)
        F = np.tile(np.linspace(0.1, 0.9, grid.size), (1, 1))
        h = np.ones_like(F)
        fit_a = Functionals(
            t_grid=grid, S=1 - F, F=F, f=np.zeros_like(F), h=h, meta={"t_max": 4.0}
        )
        fit_b = Functionals(
            t_grid=grid, S=1 - F, F=F, f=np.zeros_like(F), h=h, meta={"t_max": 5.0}
        )
        with pytest.raises(ValueError, match="must share t_max"):
            compare(fit_a, fit_b, grid)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5, float("nan"), True, "0.5"])
    def test_rejects_malformed_probability(self, p: object) -> None:
        fit_a = _functionals([(1.5, 2.0)])
        fit_b = _functionals([(1.5, 1.5)])
        with pytest.raises(ValueError, match="p must"):
            compare(fit_a, fit_b, GRID, p=p)  # type: ignore[arg-type]

    def test_rejects_non_monotone_cdf(self) -> None:
        grid = np.array([0.5, 1.0, 1.5, 2.0])
        decreasing = np.array([[0.1, 0.6, 0.5, 0.8]])
        h = np.ones_like(decreasing)
        fit_a = _fit_from_cdf(decreasing, grid, h)
        fit_b = _fit_from_cdf(np.array([[0.1, 0.3, 0.6, 0.8]]), grid, h)
        with pytest.raises(ValueError, match="non-decreasing"):
            compare(fit_a, fit_b, grid)

    def test_rejects_probability_below_the_grid(self) -> None:
        grid = np.array([0.5, 1.0, 1.5, 2.0])
        F = np.tile(np.linspace(0.6, 0.9, grid.size), (1, 1))
        h = np.ones_like(F)
        fit_a = _fit_from_cdf(F, grid, h)
        fit_b = _fit_from_cdf(F, grid, h)
        with pytest.raises(ValueError, match="lies below"):
            compare(fit_a, fit_b, grid)

    def test_rejects_probability_above_the_grid(self) -> None:
        grid = np.array([0.5, 1.0, 1.5, 2.0])
        F = np.tile(np.linspace(0.1, 0.4, grid.size), (1, 1))
        h = np.ones_like(F)
        fit_a = _fit_from_cdf(F, grid, h)
        fit_b = _fit_from_cdf(F, grid, h)
        with pytest.raises(ValueError, match="Extend t_grid to the right"):
            compare(fit_a, fit_b, grid)
