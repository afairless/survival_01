"""Tests 1 and 3 (plan Section 7.1): the uncensored prior-predictive mass.

The "new cluster" weight of step (a) for an uncensored observation is the
prior-predictive density of Eq. A.2::

    q0^o(t) = d nu gamma^d / phi * integral_0^phi
              alpha t**(alpha-1) / (gamma + t**alpha)**(d+1) dalpha

Two independent checks live here:

* **Test 1** -- Eq. A.2 against a nested ``quad`` that marginalises the
  inverse-gamma integrand over ``lambda`` *numerically*, i.e. without using
  the analytic ``lambda`` marginal.  Relative tolerance ``1e-10``.  This pins
  every constant of A.2: the ``gamma**d``, the ``d`` versus ``d + 1``
  exponent, the ``1 / phi`` and the ``t**(alpha-1)`` index.  The same 25
  random cases are then evaluated by the module's grid at ``M = 4096`` and
  must agree with quad to ``1e-5`` (the grid's own truncation error, which is
  measured separately in test 3).
* **Test 3** -- the ``M``-point trapezoid rule of plan Section 2.4 against
  ``quad`` for ``M in {32, 64, 128, 256, 512}``: relative error must be below
  ``1e-4`` at ``M = 256``, must decrease monotonically, and the observed
  convergence order must be at least 1.8.  A wrong weight vector (missing
  ``1 / M``, unhalved endpoints, a missing factor of ``phi``) or a grid that
  silently starts at ``alpha = 0`` would break the accuracy, the
  monotonicity, or the order.

The grid lives in the unit interval and is scaled by the *current* ``phi``
(``alpha_m = phi * u_m``), so ``T_im = t_i ** u_m`` can be cached once per
chain.  The structure tests at the end pin that cache, because a stale or
mis-scaled ``T`` would corrupt ``q0`` without any visible symptom in the
chain.
"""

from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

from src.dp_gibbs import (
    AlphaGrid,
    SurvData,
    log_alpha_integrand_uncensored,
    log_integral_alpha,
    log_q0_uncensored,
    make_grid,
    trapezoid_log_weights,
)

# The calibration of plan Section 2.4, used verbatim by test 3.
GAMMA_REF = 0.6
PHI_REF = 1.7752
T_REF = 1.5
D_REF = 2


def _unit_data(time: np.ndarray) -> SurvData:
    """Wrap already-transformed times in a ``SurvData``.

    ``t_max`` only enters the inverse map, so ``1.0`` is a valid stand-in for a
    grid/quadrature test.
    """
    time = np.asarray(time, dtype=np.float64)
    return SurvData(time=time, event=np.ones(time.size, dtype=bool), t_max=1.0)


def _eq_a2(t: float, gamma: float, phi: float, nu: float, d: int) -> float:
    """Eq. A.2 with the ``alpha`` integral evaluated by ``quad``.

    This is the lambda-marginalised closed form; using ``quad`` here only
    resolves the one remaining integral, it does not reuse any code from the
    module under test.
    """

    def integrand(alpha: float) -> float:
        return alpha * t ** (alpha - 1.0) / (gamma + t**alpha) ** (d + 1)

    integral, _ = quad(integrand, 0.0, phi, epsabs=1e-300, epsrel=1e-13, limit=400)
    return d * nu * gamma**d / phi * integral


def _nested_quadrature(t: float, gamma: float, phi: float, nu: float, d: int) -> float:
    """``nu * integral integral k_W(t | alpha, lambda) U(alpha) IGamma(lambda)``.

    Both integrals are done numerically: the inner one over ``lambda`` uses
    the two densities directly, so the analytic ``lambda`` marginal of Eq. A.2
    is *not* assumed.  The inner integral is taken in ``log lambda`` (the
    integrand's peak moves with ``t**alpha``) on a range 40 e-folds either
    side of ``gamma + t**alpha``, where everything outside is below 1e-100 of
    the peak.

    If the algebra of Eq. A.2 were wrong -- a missing ``gamma**d``, a
    ``t**(alpha-1)`` replaced by ``t**alpha``, ``d`` for ``d + 1`` -- this
    value and ``_eq_a2`` would disagree by a factor, not by a rounding error.
    """

    def density_inner(alpha: float) -> float:
        scale = gamma + t**alpha

        def integrand_log_lambda(y: float) -> float:
            lam = np.exp(y)
            # k_W(t | alpha, lam) * IGamma(lam | d, gamma) * lam  (Jacobian)
            log_value = (
                np.log(alpha)
                + (alpha - 1.0) * np.log(t)
                + d * np.log(gamma)
                - gammaln(d)
                - (d + 1) * y
                - scale / lam
            )
            return float(np.exp(min(log_value, 700.0)))

        value, _ = quad(
            integrand_log_lambda,
            np.log(scale) - 40.0,
            np.log(scale) + 40.0,
            points=[np.log(scale / (d + 1))],
            epsabs=1e-300,
            epsrel=1e-13,
            limit=400,
        )
        return value

    integral, _ = quad(density_inner, 0.0, phi, epsabs=1e-300, epsrel=1e-13, limit=400)
    return nu / phi * integral


def _random_cases(n_cases: int) -> list[tuple[float, float, float, float, int]]:
    """25 random ``(t, gamma, phi, nu, d)`` cases with ``t in (1, 2]``."""
    rng = np.random.default_rng(20240611)
    cases = []
    for _ in range(n_cases):
        t = rng.uniform(1.0, 2.0)
        cases.append(
            (
                float(t),
                float(10.0 ** rng.uniform(-2.0, 1.0)),
                float(10.0 ** rng.uniform(-1.0, 1.0)),
                float(10.0 ** rng.uniform(-0.5, 0.5)),
                int(rng.integers(1, 4)),
            )
        )
    return cases


CASES = _random_cases(25)


def _case_id(case: tuple[float, float, float, float, int]) -> str:
    t, gamma, phi, nu, d = case
    return f"t={t:.3f},gamma={gamma:.3g},phi={phi:.3g},nu={nu:.3g},d={d}"


class TestEqA2AgainstNestedQuadrature:
    """Test 1: the closed form (A.2) against a genuinely 2-D quadrature."""

    @pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
    def test_closed_form_matches_nested_quadrature(self, case):
        """A.2 must equal the double integral with ``lambda`` integrated out."""
        t, gamma, phi, nu, d = case

        closed_form = _eq_a2(t, gamma, phi, nu, d)
        nested = _nested_quadrature(t, gamma, phi, nu, d)

        assert closed_form > 0.0
        assert nested == pytest.approx(closed_form, rel=1e-10)

    @pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
    def test_grid_matches_closed_form(self, case):
        """The module's grid quadrature must reproduce A.2 up to its own error."""
        t, gamma, phi, nu, d = case
        grid = make_grid(_unit_data(np.array([t])), 4096)

        log_q0 = log_q0_uncensored(grid, gamma=gamma, phi=phi, nu=nu, d=d)

        assert log_q0.shape == (1,)
        assert np.all(np.isfinite(log_q0))
        assert float(np.exp(log_q0[0])) == pytest.approx(
            _eq_a2(t, gamma, phi, nu, d), rel=1e-5
        )

    def test_quadrature_error_shrinks_with_the_grid(self):
        """A 16x finer grid must be ~256x more accurate, not merely closer."""
        t, gamma, phi, nu, d = CASES[0]
        reference = _eq_a2(t, gamma, phi, nu, d)

        errors = []
        for n_alpha in (64, 1024):
            grid = make_grid(_unit_data(np.array([t])), n_alpha)
            log_q0 = log_q0_uncensored(grid, gamma=gamma, phi=phi, nu=nu, d=d)
            value = float(np.exp(log_q0[0]))
            errors.append(abs(value / reference - 1.0))

        assert errors[1] < errors[0]
        assert errors[1] < errors[0] / 100.0

    def test_rows_are_independent(self):
        """Row ``i`` of ``log q0^o`` must depend on ``t_i`` only."""
        t_values = np.array([1.0 + 1e-6, 1.25, 1.5, 2.0])
        gamma, phi, nu, d = 0.7, 1.4, 1.1, 2

        grid = make_grid(_unit_data(t_values), 512)
        joint = log_q0_uncensored(grid, gamma=gamma, phi=phi, nu=nu, d=d)

        assert joint.shape == t_values.shape
        for i, t in enumerate(t_values):
            single = make_grid(_unit_data(np.array([t])), 512)
            solo = log_q0_uncensored(single, gamma=gamma, phi=phi, nu=nu, d=d)
            assert joint[i] == pytest.approx(float(solo[0]), rel=1e-14, abs=0.0)


class TestGridQuadratureAccuracy:
    """Test 3: the trapezoid rule of plan Section 2.4, measured not assumed."""

    def _relative_errors(self, n_alphas: tuple[int, ...]) -> list[float]:
        def integrand(alpha: float) -> float:
            return alpha * T_REF**alpha / (GAMMA_REF + T_REF**alpha) ** (D_REF + 1)

        reference, _ = quad(
            integrand, 0.0, PHI_REF, epsabs=1e-300, epsrel=1e-13, limit=400
        )
        data = _unit_data(np.array([T_REF]))
        errors = []
        for n_alpha in n_alphas:
            grid = make_grid(data, n_alpha)
            log_integrand = log_alpha_integrand_uncensored(
                grid, gamma=GAMMA_REF, phi=PHI_REF, d=D_REF
            )
            log_integral = log_integral_alpha(log_integrand, grid, phi=PHI_REF)
            errors.append(abs(float(np.exp(log_integral[0])) / reference - 1.0))
        return errors

    def test_accuracy_at_declared_resolution(self):
        """``M = 256`` must reach ``1e-4``; do not demand ``1e-6`` (plan §2.4)."""
        errors = self._relative_errors((256,))

        assert errors[0] < 1e-4

    def test_errors_decrease_monotonically(self):
        """Halving the spacing must always help; a bad weight vector would not."""
        n_alphas = (32, 64, 128, 256, 512)
        errors = self._relative_errors(n_alphas)

        assert all(later < earlier for earlier, later in pairwise(errors))

    def test_observed_convergence_order(self):
        """The omitted first cell makes the error ``O(1/M**2)``: order >= 1.8."""
        n_alphas = (32, 64, 128, 256, 512)
        errors = self._relative_errors(n_alphas)

        orders = [
            float(np.log2(earlier / later)) for earlier, later in pairwise(errors)
        ]
        assert min(orders) >= 1.8
        assert max(orders) <= 2.2

    def test_magnitudes_match_the_recorded_calibration(self):
        """The plan's measured errors (1.8e-3 ... 7.2e-6) must reproduce."""
        errors = self._relative_errors((32, 64, 128, 256, 512))
        documented = (1.8e-3, 4.6e-4, 1.1e-4, 2.9e-5, 7.2e-6)

        for measured, recorded in zip(errors, documented, strict=True):
            assert measured == pytest.approx(recorded, rel=0.05), errors


class TestGridStructure:
    """The cache, the nodes and the weights that the sweep will reuse."""

    @pytest.mark.parametrize("n_alpha", (2, 5, 256))
    def test_nodes_span_the_open_unit_interval(self, n_alpha):
        """``u_m = (m + 1) / M``: the first node is ``1/M``, never ``0``."""
        grid = make_grid(_unit_data(np.array([1.5])), n_alpha)
        expected = np.arange(1, n_alpha + 1) / n_alpha

        assert grid.M == n_alpha
        assert grid.u.shape == (n_alpha,)
        assert np.array_equal(grid.u, expected)
        assert grid.u[0] == pytest.approx(1.0 / n_alpha, rel=0.0, abs=0.0)
        assert grid.u[-1] == 1.0
        assert np.all(grid.u > 0.0)

    def test_trapezoid_weights_halve_the_endpoints(self):
        """Total weight is ``(M - 1) / M``: the rule covers ``[1/M, 1]`` only."""
        for n_alpha in (2, 3, 8, 256):
            weights = np.exp(trapezoid_log_weights(n_alpha))

            assert weights.shape == (n_alpha,)
            assert np.allclose(weights[1:-1], 1.0 / n_alpha, rtol=1e-15, atol=0.0)
            assert weights[0] == pytest.approx(0.5 / n_alpha, rel=1e-15)
            assert weights[-1] == pytest.approx(0.5 / n_alpha, rel=1e-15)
            assert weights.sum() == pytest.approx((n_alpha - 1) / n_alpha, rel=1e-15)

    def test_cache_is_time_to_the_unit_power(self):
        """``T_im = t_i ** u_m``, and ``powers(phi)`` is ``T ** phi``."""
        time = np.array([1.0001, 1.5, 2.0])
        grid = make_grid(_unit_data(time), 16)
        phi = 1.3

        assert grid.T.shape == (time.size, 16)
        assert np.allclose(grid.T, time[:, None] ** grid.u[None, :], rtol=1e-15)
        assert np.allclose(
            grid.powers(phi), time[:, None] ** (phi * grid.u)[None, :], rtol=1e-13
        )
        assert np.all(grid.T > 1.0) and np.all(grid.T <= 2.0)

    def test_log_integrand_is_the_declared_density(self):
        """The integrand must be ``alpha t**alpha / (gamma + t**alpha)**(d+1)``."""
        time = np.array([1.2, 1.9])
        grid = make_grid(_unit_data(time), 64)
        gamma, phi, d = 0.4, 1.1, 2
        alpha = phi * grid.u

        log_integrand = log_alpha_integrand_uncensored(grid, gamma=gamma, phi=phi, d=d)
        direct = np.log(
            alpha[None, :]
            * time[:, None] ** alpha[None, :]
            / (gamma + time[:, None] ** alpha[None, :]) ** (d + 1)
        )

        assert log_integrand.shape == (time.size, 64)
        assert np.allclose(log_integrand, direct, rtol=1e-13, atol=1e-13)

    def test_log_integral_alpha_matches_the_trapezoid_rule_by_hand(self):
        """The quadrature is ``phi * sum_m w_m f(phi u_m)``, nothing else."""
        grid = make_grid(_unit_data(np.array([1.5])), 8)
        gamma, phi, d = 0.6, 1.7, 3
        alpha = phi * grid.u
        values = alpha * 1.5**alpha / (gamma + 1.5**alpha) ** (d + 1)
        weights = np.exp(trapezoid_log_weights(8))

        log_integrand = log_alpha_integrand_uncensored(grid, gamma=gamma, phi=phi, d=d)
        log_integral = log_integral_alpha(log_integrand, grid, phi=phi)

        assert float(np.exp(log_integral[0])) == pytest.approx(
            phi * float(np.sum(weights * values)), rel=1e-14
        )

    def test_log_q0_uncensored_is_finite_over_extreme_inputs(self):
        """No overflow or underflow to zero for hostile-but-legal inputs."""
        time = np.array([1.0 + 1e-9, 2.0])
        grid = make_grid(_unit_data(time), 256)

        for gamma, phi, nu in [
            (1e-6, 50.0, 1e-6),
            (1e6, 1e-3, 1e6),
            (1.0, 1.0, 1.0),
        ]:
            for d in (1, 2, 3):
                log_q0 = log_q0_uncensored(grid, gamma=gamma, phi=phi, nu=nu, d=d)
                assert np.all(np.isfinite(log_q0))


class TestGridValidation:
    """Malformed grids and settings must fail loudly, never silently."""

    def test_n_alpha_below_two_is_rejected(self):
        """A single node has no trapezoid rule and loses the ``delta`` factor."""
        for n_alpha in (1, 0, -3):
            with pytest.raises(ValueError, match="n_alpha must be at least 2"):
                make_grid(_unit_data(np.array([1.5])), n_alpha)
            with pytest.raises(ValueError, match="n_alpha must be at least 2"):
                trapezoid_log_weights(n_alpha)

    def test_two_nodes_are_enough(self):
        """The smallest legal grid still builds and integrates."""
        grid = make_grid(_unit_data(np.array([1.5, 2.0])), 2)

        log_q0 = log_q0_uncensored(grid, gamma=0.6, phi=1.0, nu=1.0, d=2)
        assert grid.M == 2
        assert np.all(np.isfinite(log_q0))

    def test_stale_cache_is_rejected(self):
        """``T`` must be ``time ** u``; a reused cache is a correctness bug."""
        time = np.array([1.5])
        grid = make_grid(_unit_data(time), 4)

        with pytest.raises(ValueError, match="T is not time"):
            AlphaGrid(time=time, u=grid.u, T=grid.T * 1.01, M=grid.M)

    def test_cache_of_untransformed_times_is_rejected(self):
        """``T <= 1`` means the times were never transformed; ``log T`` breaks."""
        time = np.array([0.5, 0.9])
        u = np.array([0.25, 0.5, 0.75, 1.0])

        with pytest.raises(ValueError, match="greater than 1"):
            AlphaGrid(time=time, u=u, T=time[:, None] ** u[None, :], M=4)

    def test_node_and_shape_mismatches_are_rejected(self):
        """``M``, ``u`` and ``T`` must agree; a grid with ``u_0 = 0`` is invalid."""
        time = np.array([1.5])
        u = np.array([0.5, 1.0])
        T = time[:, None] ** u[None, :]

        with pytest.raises(ValueError, match="M must equal the number of grid nodes"):
            AlphaGrid(time=time, u=u, T=T, M=3)
        with pytest.raises(ValueError, match="T must have shape"):
            AlphaGrid(time=time, u=u, T=T[:, :1], M=2)
        with pytest.raises(ValueError, match="u must lie in"):
            AlphaGrid(time=time, u=np.array([0.0, 1.0]), T=T, M=2)
        with pytest.raises(ValueError, match="u must be strictly increasing"):
            AlphaGrid(time=time, u=np.array([1.0, 0.5]), T=T, M=2)
        with pytest.raises(ValueError, match="u must be a 1-D array"):
            AlphaGrid(time=time, u=np.array([[0.5, 1.0]]), T=T, M=2)

    def test_non_positive_hyperparameters_are_rejected(self):
        """``gamma``, ``phi`` and ``nu`` are model constants: positive or error."""
        grid = make_grid(_unit_data(np.array([1.5])), 8)

        for gamma, phi, nu in (
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 0.0),
            (float("nan"), 1.0, 1.0),
        ):
            with pytest.raises(ValueError):
                log_q0_uncensored(grid, gamma=gamma, phi=phi, nu=nu, d=2)

    def test_the_inverse_gamma_shape_must_be_a_positive_integer(self):
        """``d`` indexes the ``lambda`` marginal; ``0`` or ``2.5`` is a bug."""
        grid = make_grid(_unit_data(np.array([1.5])), 8)

        for d in (0, -1, 2.5):
            with pytest.raises(ValueError, match="d must be a positive integer"):
                log_q0_uncensored(grid, gamma=0.6, phi=1.4, nu=1.0, d=d)  # type: ignore[arg-type]

    def test_log_integral_rejects_a_shape_mismatch(self):
        """A transposed or truncated integrand must not be integrated silently."""
        grid = make_grid(_unit_data(np.array([1.5, 1.6])), 8)

        with pytest.raises(ValueError, match="log_integrand must have shape"):
            log_integral_alpha(np.zeros((2, 7)), grid, phi=1.0)
        with pytest.raises(ValueError, match="log_integrand must have shape"):
            log_integral_alpha(np.zeros((8, 2)), grid, phi=1.0)
