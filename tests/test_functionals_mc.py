"""Test 22 (plan Section 7.4): the paper's Monte Carlo functional estimator.

The default functional path is Rao-Blackwellised: it uses the truncated
weights of ``G_b`` directly and is the ``L -> infinity`` limit of the
estimator in Kottas (2002), Section 3.  ``method="mc"`` reproduces that
estimator literally: draw ``L`` atoms ``theta^0_{bl} ~ G_b`` and average the
``L`` Weibull kernels.  This test file pins four things:

* **the estimator is actually computed.**  For a single-atom ``G_b`` every
  draw returns the same atom, so the MC path must equal the exact one for any
  ``L``; a missing ``1 / L`` or a wrong categorical draw breaks this.
* **the sampling error is the advertised ``1 / sqrt(L)``.**  For a fixed
  ``G_b`` the exact variance of the MC average is available in closed form
  from the atom weights and kernel values; the observed deviation of one
  ``L = 2500`` draw is compared with that standard error.  The hazard is a
  ratio of two averages over the *same* atoms, so it is consistent but only
  ``O(1 / L)``-biased; its standard error is estimated from independent
  replicates instead.
* **the mean over independent ``G_b`` equals the exact value.**  For survival
  and density the MC estimator is conditionally unbiased, so averaging the
  difference over 200 independent realisations must be within a few standard
  errors of zero.  The plan explicitly scopes this claim to ``S`` and ``f``.
* **one shared atom set for the hazard.**  ``h`` must equal ``f / S``
  elementwise; drawing separate atom sets for the numerator and denominator
  would leave every curve plausible but fail this identity.
"""

from typing import Any

import numpy as np
import pytest

from src.dp_functionals import (
    Functionals,
    FunctionalSettings,
    functionals_from_mc_realizations,
    functionals_from_realizations,
    posterior_functionals,
)
from src.dp_gibbs import DPPrior, DPSettings, SurvData, make_surv_data, sample_chain
from src.dp_weibull import (
    DPRealization,
    sample_dp_posterior,
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

#: Original-scale grid that keeps the synthetic mixture in mid-survival, so
#: that the hazard is a well-conditioned ratio rather than 0 / 0 in the tail.
GRID = np.linspace(0.25, 3.0, 9)

#: ``L`` used by the paper (Section 3).  Reduced only where a test says so.
L_PAPER = 2_500


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


def _synthetic_theta(rng: np.random.Generator) -> np.ndarray:
    """Latent ``(alpha, lambda)`` pairs for realistic posterior ``G_b`` draws."""
    return np.column_stack(
        [
            rng.uniform(1.2, 2.8, size=TIME.size),
            rng.uniform(1.0, 6.0, size=TIME.size),
        ]
    )


def _posterior_realizations(
    n_draws: int, rng: np.random.Generator
) -> list[DPRealization]:
    """``n_draws`` i.i.d. pruned posterior realisations of one fixed dataset.

    The latent ``theta`` is fixed so that the only randomness is the DP
    posterior draw itself, which is what test 22 averages over.
    """
    theta = _synthetic_theta(rng)
    j_common = truncation_number_for_posterior(2.0, TIME.size, 1e-4)
    return [
        sample_dp_posterior(
            2.0,
            1.5,
            4.0,
            theta,
            d=2,
            epsilon=1e-4,
            n_terms=j_common,
            rng=rng,
        ).pruned(1e-6)
        for _ in range(n_draws)
    ]


def _kernels_tilde(
    realization: DPRealization, t_tilde: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom survival and density kernels on the ``t_tilde`` scale.

    Built from the Weibull closed form rather than from the module's own
    mixture evaluator, so the variance used as the MC standard error does not
    inherit an implementation error from the code under test.
    """
    alpha = realization.alphas[:, None]
    lam = realization.lambdas[:, None]
    time = t_tilde[None, :]
    z = np.power(time, alpha) / lam
    survival = np.exp(-z)
    density = alpha * np.power(time, alpha - 1.0) / lam * survival
    return survival, density


def _mc_standard_error(
    realization: DPRealization, t_tilde: np.ndarray, L: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact standard error of the MC mean of the ``S`` and ``f`` kernels.

    Under ``G_b = sum_j w_j delta_j`` the drawn kernels are i.i.d. with the
    categorical law ``w``, so ``Var(mean) = Var_w(kernel) / L``.  The variance
    is computed from the same weights, with no Monte Carlo error of its own.
    """
    weights = np.asarray(realization.weights, dtype=np.float64)
    weights = weights / weights.sum()
    survival, density = _kernels_tilde(realization, t_tilde)
    errors = []
    for kernel in (survival, density):
        mean = weights @ kernel
        second_moment = weights @ (kernel**2)
        variance = np.clip(second_moment - mean**2, 0.0, None)
        errors.append(np.sqrt(variance / L))
    return errors[0], errors[1]


class TestSingleAtomExactness:
    """The MC path collapses onto the exact path for a one-atom ``G_b``."""

    @pytest.mark.parametrize("L", [1, 7, L_PAPER])
    def test_single_atom_gives_the_exact_functional(self, L: int) -> None:
        """Every atom draw returns the only atom, so MC equals exact."""
        data = _data()
        realization = _single_atom(1.6, 2.0)

        exact = functionals_from_realizations([realization], GRID, data)
        mc = functionals_from_mc_realizations(
            [realization], GRID, data, L=L, rng=np.random.default_rng(5)
        )

        # ``L`` equal terms are summed, so the tolerance is a float
        # accumulation bound (about 1e-13 relative at L = 2500), not exact
        # equality.
        np.testing.assert_allclose(mc.S, exact.S, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(mc.F, exact.F, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(mc.f, exact.f, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(mc.h, exact.h, rtol=1e-12, atol=1e-13)
        assert mc.meta["method"] == "mc"
        assert mc.meta["L"] == L

    def test_missing_one_over_L_would_change_the_scale(self) -> None:
        """The exact single-atom values are not reproduced by a sum, only a mean."""
        data = _data()
        realization = _single_atom(1.6, 2.0)
        exact = functionals_from_realizations([realization], GRID, data)
        mc = functionals_from_mc_realizations(
            [realization], GRID, data, L=7, rng=np.random.default_rng(6)
        )

        assert not np.allclose(mc.S[0], 7.0 * exact.S[0])


class TestMonteCarloErrorScale:
    """One ``L``-atom draw deviates by the closed-form MC standard error."""

    def _check_draw(self, realization: DPRealization, seed: int) -> None:
        data = _data()
        t_tilde = 1.0 + GRID / data.t_max

        exact = functionals_from_realizations([realization], GRID, data)
        mc = functionals_from_mc_realizations(
            [realization], GRID, data, L=L_PAPER, rng=np.random.default_rng(seed)
        )
        se_survival, se_density = _mc_standard_error(realization, t_tilde, L_PAPER)

        # Undo the 1 / t_max Jacobian: the closed-form standard errors are on
        # the t_tilde scale, where the atom draws live.  ``se`` is a per-point
        # array, so the tolerance is applied elementwise by hand.
        assert np.all(
            np.abs(mc.S[0] - exact.S[0]) <= 4.0 * np.maximum(se_survival, 1e-300)
        )
        assert np.all(
            np.abs((mc.f[0] - exact.f[0]) * data.t_max)
            <= 4.0 * np.maximum(se_density, 1e-300)
        )

        # The hazard is a ratio over the same atoms; its standard error is
        # estimated from independent replicates of the atom draw.
        replicates = np.vstack(
            [
                functionals_from_mc_realizations(
                    [realization],
                    GRID,
                    data,
                    L=L_PAPER,
                    rng=np.random.default_rng(seed + k),
                ).h[0]
                for k in range(1, 25)
            ]
        )
        se_hazard = np.std(replicates, axis=0, ddof=1)
        assert np.all(
            np.abs(mc.h[0] - exact.h[0]) <= 4.0 * np.maximum(se_hazard, 1e-300)
        )

    def test_three_atom_realisation(self) -> None:
        """A multi-atom ``G_b``: each functional stays within a few SE."""
        self._check_draw(_three_atom(), seed=101)

    def test_realistic_posterior_realisation(self) -> None:
        """A pruned posterior ``G_b`` behaves the same way."""
        rng = np.random.default_rng(202)
        realization = _posterior_realizations(1, rng)[0]
        self._check_draw(realization, seed=303)


class TestUnbiasednessOverRealizations:
    """Test 22's headline claim: the mean MC value equals the exact value."""

    def test_mean_over_200_realizations_matches_exact(self) -> None:
        """``mean_b(S_hat_mc - S_exact)`` is zero within Monte Carlo error."""
        data = _data()
        n_draws = 200
        realizations = _posterior_realizations(n_draws, np.random.default_rng(31415))

        exact = functionals_from_realizations(realizations, GRID, data)
        mc = functionals_from_mc_realizations(
            realizations,
            GRID,
            data,
            L=L_PAPER,
            rng=np.random.default_rng(2718),
        )

        assert exact.n_draws == mc.n_draws == n_draws
        # The two estimators really are different per draw; otherwise the
        # agreement test below would be vacuous.
        assert np.any(mc.S != exact.S)

        for name in ("S", "f"):
            difference = getattr(mc, name) - getattr(exact, name)
            standard_error = np.std(difference, axis=0, ddof=1) / np.sqrt(n_draws)
            # 3.5 rather than 3 standard errors: the grid points are smooth
            # in t, but the two functionals give 18 checks, and a literal 3
            # sigma band would fail by chance in about 5% of runs.
            assert np.all(np.abs(np.mean(difference, axis=0)) <= 3.5 * standard_error)
            # Absolute agreement is also meaningful at this scale.
            assert np.max(np.abs(np.mean(difference, axis=0))) < 0.05

    def test_the_two_paths_draw_different_estimates(self) -> None:
        """For a multi-atom ``G_b`` an ``L = 1`` draw differs from the exact value."""
        data = _data()
        exact = functionals_from_realizations([_three_atom()], GRID, data)
        mc = functionals_from_mc_realizations(
            [_three_atom()], GRID, data, L=1, rng=np.random.default_rng(11)
        )

        assert not np.allclose(mc.S, exact.S)
        assert not np.allclose(mc.f, exact.f)


class TestSharedAtomSetForHazard:
    """The hazard is ``f / S`` from one shared atom set, never two draws."""

    def test_hazard_is_density_over_survival_elementwise(self) -> None:
        """``h == f / S`` exactly for the MC path as well."""
        data = _data()
        realizations = _posterior_realizations(2, np.random.default_rng(77))

        mc = functionals_from_mc_realizations(
            realizations, GRID, data, L=400, rng=np.random.default_rng(78)
        )
        expected = mc.f / mc.S

        np.testing.assert_allclose(mc.h, expected, rtol=1e-14, atol=1e-14)

    def test_hazard_from_two_separate_atom_sets_would_not_match(self) -> None:
        """A ratio of independent MC estimates differs from the shared one."""
        data = _data()
        realization = _three_atom()
        first = functionals_from_mc_realizations(
            [realization], GRID, data, L=400, rng=np.random.default_rng(1)
        )
        second = functionals_from_mc_realizations(
            [realization], GRID, data, L=400, rng=np.random.default_rng(2)
        )

        separate = second.f / first.S
        assert not np.allclose(separate, first.h, rtol=1e-12, atol=1e-12)


class TestProductionMcPath:
    """``posterior_functionals(method="mc")``: schema, provenance and replay."""

    def test_method_mc_uses_the_common_truncation_and_records_L(self) -> None:
        """The MC path shares ``J_common`` with the exact path and tags ``L``."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.05, 1.2 * data.t_max, 7)

        fit = posterior_functionals(
            chain,
            grid,
            settings=FunctionalSettings(method="mc", L=500),
            seed=4321,
        )

        expected_j = truncation_number_for_posterior(
            float(np.max(chain.nu)), chain.n, chain.settings.epsilon
        )
        assert isinstance(fit, Functionals)
        assert fit.S.shape == (chain.n_draws, grid.size)
        assert fit.F.shape == fit.f.shape == fit.h.shape == fit.S.shape
        assert fit.meta["method"] == "mc"
        assert fit.meta["L"] == 500
        assert fit.meta["J_common"] == expected_j
        assert fit.meta["n_draws"] == chain.n_draws
        np.testing.assert_allclose(fit.h, fit.f / fit.S, rtol=1e-14, atol=1e-14)

    def test_same_seed_replays_and_a_different_seed_does_not(self) -> None:
        """Both DP draws and atom draws derive deterministically from ``seed``."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.05, 1.0, 5)
        settings = FunctionalSettings(method="mc", L=200)

        first = posterior_functionals(chain, grid, settings=settings, seed=99)
        replay = posterior_functionals(chain, grid, settings=settings, seed=99)
        other = posterior_functionals(chain, grid, settings=settings, seed=100)

        np.testing.assert_array_equal(first.S, replay.S)
        np.testing.assert_array_equal(first.f, replay.f)
        np.testing.assert_array_equal(first.h, replay.h)
        assert not np.array_equal(first.S, other.S)
        assert first.meta["seed"] == 99

    def test_mc_and_exact_differ_but_share_schema(self) -> None:
        """Selecting ``method`` must switch the estimator, not just the label."""
        data = _data()
        chain = sample_chain(data, PRIOR, _settings())
        grid = np.linspace(0.25, 3.0, 9)

        exact = posterior_functionals(
            chain, grid, settings=FunctionalSettings(method="exact"), seed=17
        )
        mc = posterior_functionals(
            chain, grid, settings=FunctionalSettings(method="mc", L=500), seed=17
        )

        assert exact.meta["method"] == "exact"
        assert "L" not in exact.meta
        assert mc.meta["method"] == "mc"
        assert exact.S.shape == mc.S.shape
        assert not np.array_equal(exact.S, mc.S)
        # The exact path must not have drawn atoms: same call twice is exact.
        np.testing.assert_array_equal(
            exact.S,
            posterior_functionals(
                chain, grid, settings=FunctionalSettings(method="exact"), seed=17
            ).S,
        )


class TestValidation:
    """Bad ``L`` values are input errors, not silent fallbacks."""

    @pytest.mark.parametrize("bad_L", [0, -1, True, 1.5, "2500"])
    def test_rejects_invalid_atom_draw_counts(self, bad_L: Any) -> None:
        """``L`` must be a positive integer; ``True`` must not mean 1."""
        data = _data()
        with pytest.raises(ValueError, match=r"L must be"):
            functionals_from_mc_realizations(
                [_single_atom(1.5, 2.0)], GRID, data, L=bad_L
            )

    def test_rejects_a_non_generator_rng(self) -> None:
        """The random source is a ``numpy.random.Generator``, not a seed."""
        data = _data()
        with pytest.raises(ValueError, match=r"rng must be a"):
            functionals_from_mc_realizations(
                [_single_atom(1.5, 2.0)],
                GRID,
                data,
                L=10,
                rng=123,  # type: ignore[arg-type]
            )

    def test_rejects_an_unnormalised_realisation(self) -> None:
        """The underlying engine still enforces proper weights."""
        data = _data()
        bad = DPRealization(
            weights=np.array([0.5]),
            alphas=np.array([1.5]),
            lambdas=np.array([1.0]),
            precision=1.0,
        )
        with pytest.raises(ValueError, match=r"weights must sum to one"):
            functionals_from_mc_realizations(
                [bad], GRID, data, L=10, rng=np.random.default_rng(0)
            )

    def test_settings_rejects_a_zero_L(self) -> None:
        """``FunctionalSettings`` validates ``L`` for both methods."""
        with pytest.raises(ValueError, match=r"L must be at least 1"):
            FunctionalSettings(method="mc", L=0)
