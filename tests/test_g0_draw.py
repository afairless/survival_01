"""Tests 4 and 9 (plan Section 7.2): the continuous-piece base draw.

When step (a) sends an observation to a **new** cluster, its latent parameters
come from the continuous part of the base measure (plan Section 2.2)::

    uncensored:  [alpha | .] proportional to
                     alpha t**alpha / (gamma + t**alpha)**(d + 1)
                 [lambda | alpha, .] = IGamma(d + 1, gamma + t**alpha)

    censored:    [alpha | .] proportional to
                     1 / (gamma + z**alpha)**d
                 [lambda | alpha, .] = IGamma(d, gamma + z**alpha)

Two independent checks live here:

* **Test 4** -- 100,000 inverse-CDF draws for a fixed grid row, compared with
  the exact normalised integrand by total-variation distance over 20 bins.
  This catches ``searchsorted`` side/offset bugs, a row that is not
  normalised, and drawing ``alpha`` off the grid nodes.
* **Test 9** -- ``alpha`` against the exact grid CDF at ``M = 4096`` by
  Kolmogorov--Smirnov, and ``lambda | alpha`` against
  ``scipy.stats.invgamma`` at a fixed ``alpha`` node.  The wrong branch
  (censored row for an uncensored observation), the wrong inverse-gamma shape
  (``d`` instead of ``d + 1``) and the wrong inverse-gamma scale are all shown
  to produce a materially larger KS distance, so the test is not merely
  passing by construction.

The reference densities and CDFs are written out from the equations above and
do not call the module's integrand helpers.  The grid is the production object
because the draw is defined on that grid; the weights are the test's own.
"""

from typing import Any

import numpy as np
import pytest
from scipy.stats import invgamma, ks_1samp

from src.dp_gibbs import (
    AlphaGrid,
    SurvData,
    draw_g0_conditional_alpha,
    make_grid,
)

GAMMA_REF = 0.6
PHI_REF = 1.7752
D_REF = 2
T_REF = 1.5
Z_REF = 1.4


def _unit_data(time: np.ndarray) -> SurvData:
    """Wrap already-transformed times in a ``SurvData``."""
    time = np.asarray(time, dtype=np.float64)
    return SurvData(time=time, event=np.ones(time.size, dtype=bool), t_max=1.0)


def _grid(time: float, n_alpha: int = 256):
    """Build a one-row alpha grid for a transformed time."""
    return make_grid(_unit_data(np.array([time])), n_alpha)


def _alpha_grid_density(
    time: float,
    *,
    event: bool,
    gamma: float = GAMMA_REF,
    phi: float = PHI_REF,
    d: int = D_REF,
    n_alpha: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact normalised ``[alpha | .]`` mass on the grid nodes.

    The density is written from plan Section 2.2 directly: the uncensored row
    is the Eq. A.2 integrand and the censored row is the Eq. A.3 integrand.
    Neither branch calls a module helper, so a shared formula error cannot
    make the reference agree with the sampler.
    """
    grid = _grid(time, n_alpha)
    alpha = phi * grid.u
    if event:
        weights = alpha * time**alpha / (gamma + time**alpha) ** (d + 1)
    else:
        weights = 1.0 / (gamma + time**alpha) ** d
    return alpha, weights / weights.sum()


def _ks_statistic(sample: np.ndarray, cdf) -> float:
    """Return the KS D statistic as a plain float (SciPy stubs omit it)."""
    result: Any = ks_1samp(sample, cdf)
    return float(result.statistic)


def _step_cdf(nodes: np.ndarray, probabilities: np.ndarray):
    """Right-continuous CDF of a discrete law supported on ``nodes``."""
    cumulative = np.cumsum(probabilities)

    def cdf(x: object) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        index = np.searchsorted(nodes, values, side="right") - 1
        return np.where(
            index < 0,
            0.0,
            np.where(
                index >= nodes.size,
                1.0,
                cumulative[np.clip(index, 0, nodes.size - 1)],
            ),
        )

    return cdf


def _draw_many(
    time: float,
    *,
    event: bool,
    n: int,
    n_alpha: int,
    seed: int,
    gamma: float = GAMMA_REF,
    phi: float = PHI_REF,
    d: int = D_REF,
) -> tuple[np.ndarray, np.ndarray, AlphaGrid]:
    """Call the production draw ``n`` times and return both components."""
    grid = _grid(time, n_alpha)
    rng = np.random.default_rng(seed)
    alphas = np.empty(n, dtype=np.float64)
    lambdas = np.empty(n, dtype=np.float64)
    for k in range(n):
        alphas[k], lambdas[k] = draw_g0_conditional_alpha(
            time,
            event,
            gamma=gamma,
            phi=phi,
            grid=grid,
            rng=rng,
            d=d,
        )
    return alphas, lambdas, grid


class TestAlphaGridDraw:
    """Test 4: the inverse-CDF draw reproduces the normalised grid density."""

    @pytest.mark.parametrize("event", [True, False], ids=["uncensored", "censored"])
    def test_draw_matches_the_grid_density_by_total_variation(self, event):
        """TV over 20 bins must be below 0.02 at 100,000 draws."""
        time = T_REF if event else Z_REF
        alphas, _, grid = _draw_many(
            time, event=event, n=100_000, n_alpha=256, seed=20240701 + int(event)
        )
        nodes, probabilities = _alpha_grid_density(time, event=event)

        edges = np.linspace(0.0, PHI_REF, 21)
        node_bin = np.clip(np.searchsorted(edges, nodes, side="right") - 1, 0, 19)
        draw_bin = np.clip(np.searchsorted(edges, alphas, side="right") - 1, 0, 19)
        exact = np.bincount(node_bin, weights=probabilities, minlength=20)
        empirical = np.bincount(draw_bin, minlength=20) / alphas.size

        assert exact.sum() == pytest.approx(1.0, rel=1e-15)
        assert empirical.sum() == pytest.approx(1.0, rel=1e-15)
        assert 0.5 * float(np.sum(np.abs(empirical - exact))) < 0.02
        assert np.all(alphas > 0.0) and np.all(alphas <= PHI_REF)
        assert np.isin(alphas, nodes).all()
        assert np.all(grid.u > 0.0)

    @pytest.mark.parametrize("event", [True, False], ids=["uncensored", "censored"])
    def test_the_first_grid_node_is_reachable(self, event):
        """The draw must not start at the second node (an offset bug)."""
        time = T_REF if event else Z_REF
        alphas, _, grid = _draw_many(
            time, event=event, n=5_000, n_alpha=8, seed=20240711 + int(event)
        )
        nodes = PHI_REF * grid.u

        counts = np.bincount(
            np.searchsorted(nodes, alphas, side="left"), minlength=nodes.size
        )
        assert counts[0] > 0
        assert counts[-1] > 0

    def test_same_seed_reproduces_and_different_seed_differs(self):
        """The rng is the only source of randomness, with no global state."""
        first = _draw_many(T_REF, event=True, n=4, n_alpha=16, seed=1234)[:2]
        repeat = _draw_many(T_REF, event=True, n=4, n_alpha=16, seed=1234)[:2]
        other = _draw_many(T_REF, event=True, n=4, n_alpha=16, seed=4321)[:2]

        assert np.array_equal(first[0], repeat[0])
        assert np.array_equal(first[1], repeat[1])
        assert not np.array_equal(first[0], other[0]) or not np.array_equal(
            first[1], other[1]
        )


class TestAlphaAgainstExactCdf:
    """Test 9: the ``alpha`` marginal against the high-resolution grid CDF."""

    @pytest.mark.parametrize(
        ("event", "time"),
        [(True, T_REF), (False, Z_REF)],
        ids=["uncensored", "censored"],
    )
    def test_alpha_ks_distance_is_small(self, event, time):
        """At ``M = 4096`` the exact CDF and the draw must agree to KS < 0.02."""
        alphas, _, _ = _draw_many(
            time,
            event=event,
            n=10_000,
            n_alpha=4096,
            seed=20240721 + int(event),
        )
        nodes, probabilities = _alpha_grid_density(time, event=event, n_alpha=4096)

        statistic = _ks_statistic(alphas, _step_cdf(nodes, probabilities))

        assert statistic < 0.02

    def test_the_censored_row_is_rejected_for_uncensored_data(self):
        """Using the Eq. A.3 row for an uncensored observation must be visible.

        The two densities are not close: the uncensored one carries the extra
        ``alpha * t**alpha / (gamma + t**alpha)`` factor.  Plan Section 7.7
        lists this swap as a mutation the suite must catch.
        """
        alphas, _, _ = _draw_many(
            T_REF, event=True, n=10_000, n_alpha=4096, seed=20240731
        )
        wrong_nodes, wrong_probabilities = _alpha_grid_density(
            T_REF, event=False, n_alpha=4096
        )

        wrong_statistic = _ks_statistic(
            alphas, _step_cdf(wrong_nodes, wrong_probabilities)
        )

        assert wrong_statistic > 0.05


class TestLambdaGivenAlpha:
    """Test 9: the inverse-gamma conditional given the drawn ``alpha``."""

    @pytest.mark.parametrize(
        ("event", "time"),
        [(True, T_REF), (False, Z_REF)],
        ids=["uncensored", "censored"],
    )
    def test_lambda_given_alpha_matches_invgamma(self, event, time):
        """At a fixed grid node, ``lambda`` has the exact conjugate law."""
        n_alpha = 16
        node = 7
        alphas, lambdas, grid = _draw_many(
            time,
            event=event,
            n=48_000,
            n_alpha=n_alpha,
            seed=20240741 + int(event),
        )
        target = PHI_REF * grid.u[node]
        conditional = lambdas[alphas == target]

        assert conditional.size > 2_000
        scale = GAMMA_REF + time**target
        shape = D_REF + 1 if event else D_REF
        statistic = _ks_statistic(conditional, invgamma(a=shape, scale=scale).cdf)

        assert statistic < 0.05

    @pytest.mark.parametrize(
        ("event", "wrong_shape"),
        [(True, D_REF), (False, D_REF + 1)],
        ids=["uncensored-d-instead-of-d-plus-one", "censored-d-plus-one"],
    )
    def test_the_wrong_inverse_gamma_shape_is_rejected(self, event, wrong_shape):
        """Swapping ``d`` and ``d + 1`` must move the KS distance far beyond noise."""
        time = T_REF if event else Z_REF
        n_alpha = 16
        node = 7
        alphas, lambdas, grid = _draw_many(
            time,
            event=event,
            n=48_000,
            n_alpha=n_alpha,
            seed=20240751 + int(event),
        )
        target = PHI_REF * grid.u[node]
        conditional = lambdas[alphas == target]
        scale = GAMMA_REF + time**target

        wrong_statistic = _ks_statistic(
            conditional, invgamma(a=wrong_shape, scale=scale).cdf
        )

        assert wrong_statistic > 0.1

    def test_the_wrong_inverse_gamma_scale_is_rejected(self):
        """The scale is ``gamma + t**alpha``, not ``gamma`` or ``t**alpha``."""
        time = T_REF
        n_alpha = 16
        node = 7
        alphas, lambdas, grid = _draw_many(
            time, event=True, n=48_000, n_alpha=n_alpha, seed=20240761
        )
        target = PHI_REF * grid.u[node]
        conditional = lambdas[alphas == target]
        correct_scale = GAMMA_REF + time**target

        wrong_statistic = _ks_statistic(
            conditional, invgamma(a=D_REF + 1, scale=GAMMA_REF).cdf
        )
        correct_statistic = _ks_statistic(
            conditional, invgamma(a=D_REF + 1, scale=correct_scale).cdf
        )

        assert correct_scale / GAMMA_REF > 1.5
        assert wrong_statistic > 0.1
        assert wrong_statistic > 10.0 * correct_statistic


class TestValidation:
    """Malformed scalar inputs must fail loudly, never silently."""

    def test_time_must_be_a_finite_scalar_greater_than_one(self):
        """The Appendix's transform guarantees ``t > 1``; nothing else is legal."""
        grid = _grid(T_REF, 8)
        rng = np.random.default_rng(0)

        for time in (1.0, 0.5, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="time must be a finite scalar"):
                draw_g0_conditional_alpha(
                    time, True, gamma=0.6, phi=1.4, grid=grid, rng=rng
                )
        with pytest.raises(ValueError, match="time must be a scalar"):
            draw_g0_conditional_alpha(
                np.array([1.5]), True, gamma=0.6, phi=1.4, grid=grid, rng=rng
            )

    def test_time_must_be_a_row_of_the_grid(self):
        """A grid built for other data must not be used silently."""
        grid = _grid(T_REF, 8)

        with pytest.raises(ValueError, match="not a row of the supplied alpha grid"):
            draw_g0_conditional_alpha(
                1.6,
                True,
                gamma=0.6,
                phi=1.4,
                grid=grid,
                rng=np.random.default_rng(0),
            )

    def test_event_must_be_a_scalar_indicator(self):
        """An array or an out-of-range value would mark the wrong likelihood."""
        grid = _grid(T_REF, 8)
        rng = np.random.default_rng(0)

        for event in (2, -1, "yes", np.array([True, False])):
            with pytest.raises(ValueError, match="event must be a scalar"):
                draw_g0_conditional_alpha(
                    T_REF, event, gamma=0.6, phi=1.4, grid=grid, rng=rng
                )

    def test_hyperparameters_and_shape_are_validated(self):
        """``gamma``, ``phi`` and ``d`` come from the model; each must be legal."""
        grid = _grid(T_REF, 8)
        rng = np.random.default_rng(0)

        for gamma, phi in ((0.0, 1.0), (1.0, -1.0), (float("nan"), 1.0)):
            with pytest.raises(ValueError):
                draw_g0_conditional_alpha(
                    T_REF, True, gamma=gamma, phi=phi, grid=grid, rng=rng
                )
        for d in (0, -1, 2.5):
            with pytest.raises(ValueError, match="d must be a positive integer"):
                draw_g0_conditional_alpha(
                    T_REF,
                    True,
                    gamma=0.6,
                    phi=1.4,
                    grid=grid,
                    rng=rng,
                    d=d,  # type: ignore[arg-type]
                )

    @pytest.mark.parametrize("event", [True, False], ids=["uncensored", "censored"])
    def test_extreme_but_legal_inputs_return_finite_parameters(self, event):
        """Large times and tiny/large hyperparameters must not overflow."""
        time = 2.0
        grid = _grid(time, 256)
        rng = np.random.default_rng(0)

        for gamma, phi in ((1e-8, 50.0), (1e8, 1e-3), (1.0, 1.0)):
            alpha, lam = draw_g0_conditional_alpha(
                time, event, gamma=gamma, phi=phi, grid=grid, rng=rng
            )
            assert np.isfinite(alpha) and 0.0 < alpha <= phi
            assert np.isfinite(lam) and lam > 0.0
