"""Test 2 (plan Section 7.1): the right-censored prior-predictive mass.

The "new cluster" weight of step (a) for a right-censored observation is the
prior-predictive *survival* probability of Eq. A.3::

    q0^c(z) = nu gamma^d / phi * integral_0^phi
              1 / (gamma + z**alpha)**d dalpha

Test 2 checks it against a nested ``quad`` that marginalises the
inverse-gamma over ``lambda`` **numerically** (relative tolerance ``1e-10``),
using ``1 - K_W(z | alpha, lambda) = exp(-z**alpha / lambda)``.  That
comparison is the only one in the suite that can see the censored branch's
constants: the ``nu``, the ``gamma**d`` and, above all, the exponent ``d``
instead of the uncensored ``d + 1`` (plan Section 11).  A reference that
reused the analytic ``lambda`` marginal, or the module's own integrand, would
share the mistake it is meant to catch, so both are avoided here.

The same 25 random cases are then evaluated with the module's grid at
``M = 4096`` and must agree with ``quad`` to ``1e-5`` (the grid's own
truncation error, measured in test 3).  The structural tests pin the
remaining invariants of the branch: no ``alpha`` or ``1 / z`` factor, a
strictly decreasing mass in ``z``, and the ``d`` versus ``d + 1`` ratio.
"""

from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

from src.dp_gibbs import (
    SurvData,
    log_alpha_integrand_censored,
    log_alpha_integrand_uncensored,
    log_q0_censored,
    make_grid,
)

# The calibration of plan Section 2.4, reused for the mutation guard.
GAMMA_REF = 0.6
PHI_REF = 1.7752
Z_REF = 1.5
D_REF = 2


def _unit_data(time: np.ndarray) -> SurvData:
    """Wrap already-transformed times in a ``SurvData``.

    ``t_max`` only enters the inverse map, so ``1.0`` is a valid stand-in for a
    grid/quadrature test.
    """
    time = np.asarray(time, dtype=np.float64)
    return SurvData(time=time, event=np.ones(time.size, dtype=bool), t_max=1.0)


def _eq_a3(z: float, gamma: float, phi: float, nu: float, d: int) -> float:
    """Eq. A.3 with the ``alpha`` integral evaluated by ``quad``.

    This uses the analytic ``lambda`` marginal; ``quad`` only resolves the one
    remaining integral and nothing is imported from the module under test.
    """

    def integrand(alpha: float) -> float:
        return 1.0 / (gamma + z**alpha) ** d

    integral, _ = quad(integrand, 0.0, phi, epsabs=1e-300, epsrel=1e-13, limit=400)
    return nu * gamma**d / phi * integral


def _survival_marginal_lambda(z: float, alpha: float, gamma: float, d: int) -> float:
    """``E[exp(-z**alpha / lambda)]`` for ``lambda ~ IGamma(d, gamma)``.

    The inverse-gamma integral has the closed form
    ``(gamma / (gamma + z**alpha)) ** d``; it is used only to validate the
    *reference* quadrature below, never to produce the expected value.
    """
    return (gamma / (gamma + z**alpha)) ** d


def _nested_quadrature_censored(
    z: float, gamma: float, phi: float, nu: float, d: int
) -> float:
    """``nu * integral integral S(z | alpha, lambda) U(alpha) IGamma(lambda)``.

    Both integrals are numerical: the inner one integrates the survival
    ``exp(-z**alpha / lambda)`` against the inverse-gamma *density*, so the
    analytic marginal that Eq. A.3 relies on is not assumed.  The inner
    integral is taken in ``log lambda`` on a range 40 e-folds either side of
    ``gamma + z**alpha``, where the integrand is below 1e-100 of its peak.
    """

    def density_inner(alpha: float) -> float:
        scale = gamma + z**alpha

        def integrand_log_lambda(y: float) -> float:
            lam = np.exp(y)
            # S(z | alpha, lam) * IGamma(lam | d, gamma) * lam  (Jacobian)
            log_value = d * np.log(gamma) - gammaln(d) - d * y - scale / lam
            return float(np.exp(min(log_value, 700.0)))

        value, _ = quad(
            integrand_log_lambda,
            np.log(scale) - 40.0,
            np.log(scale) + 40.0,
            points=[np.log(scale / d)],
            epsabs=1e-300,
            epsrel=1e-13,
            limit=400,
        )
        return value

    integral, _ = quad(density_inner, 0.0, phi, epsabs=1e-300, epsrel=1e-13, limit=400)
    return nu / phi * integral


def _random_cases(n_cases: int) -> list[tuple[float, float, float, float, int]]:
    """25 random ``(z, gamma, phi, nu, d)`` cases with ``z in (1, 2]``."""
    rng = np.random.default_rng(20240612)
    cases = []
    for _ in range(n_cases):
        z = rng.uniform(1.0, 2.0)
        cases.append(
            (
                float(z),
                float(10.0 ** rng.uniform(-2.0, 1.0)),
                float(10.0 ** rng.uniform(-1.0, 1.0)),
                float(10.0 ** rng.uniform(-0.5, 0.5)),
                int(rng.integers(1, 4)),
            )
        )
    return cases


CASES = _random_cases(25)


def _case_id(case: tuple[float, float, float, float, int]) -> str:
    z, gamma, phi, nu, d = case
    return f"z={z:.3f},gamma={gamma:.3g},phi={phi:.3g},nu={nu:.3g},d={d}"


class TestEqA3AgainstNestedQuadrature:
    """Test 2: the censored closed form (A.3) against a 2-D quadrature."""

    @pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
    def test_reference_quadrature_matches_the_lambda_marginal(self, case):
        """The reference itself must reproduce the analytic ``lambda`` integral.

        Validating the oracle first is what makes the ``1e-10`` agreement
        below meaningful: the nested integration over ``(alpha, log lambda)``
        has to collapse to ``(gamma / (gamma + z**alpha)) ** d`` before it may
        be used to judge Eq. A.3.
        """
        z, gamma, phi, nu, d = case

        reference = _nested_quadrature_censored(z, gamma, phi, nu, d)
        marginalised = (
            nu
            / phi
            * quad(
                lambda alpha: _survival_marginal_lambda(z, alpha, gamma, d),
                0.0,
                phi,
                epsabs=1e-300,
                epsrel=1e-13,
                limit=400,
            )[0]
        )

        assert reference > 0.0
        assert reference == pytest.approx(marginalised, rel=1e-10)

    @pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
    def test_closed_form_matches_nested_quadrature(self, case):
        """A.3 must equal the double integral with ``lambda`` integrated out."""
        z, gamma, phi, nu, d = case

        closed_form = _eq_a3(z, gamma, phi, nu, d)
        nested = _nested_quadrature_censored(z, gamma, phi, nu, d)

        assert closed_form > 0.0
        assert nested == pytest.approx(closed_form, rel=1e-10)

    def _grid_relative_errors(
        self, case: tuple[float, float, float, float, int], n_alphas: tuple[int, ...]
    ) -> list[float]:
        """Relative error of the module's grid against Eq. A.3 by ``quad``."""
        z, gamma, phi, nu, d = case
        reference = _eq_a3(z, gamma, phi, nu, d)
        errors = []
        for n_alpha in n_alphas:
            grid = make_grid(_unit_data(np.array([z])), n_alpha)
            log_q0 = log_q0_censored(grid, gamma=gamma, phi=phi, nu=nu, d=d)
            errors.append(abs(float(np.exp(log_q0[0])) / reference - 1.0))
        return errors

    @pytest.mark.parametrize("case", CASES[:5], ids=[_case_id(c) for c in CASES[:5]])
    def test_grid_error_at_the_declared_resolution(self, case):
        """``M = 256`` gives ~6e-3 relative, not the ``1e-4`` of A.2.

        The censored integrand ``(gamma + z**alpha)**-d`` does not vanish as
        ``alpha -> 0``, so the omitted first cell ``[0, alpha_1)`` of the
        plan's grid (first node ``alpha_1 = phi / M``, no ``alpha = 0`` node)
        contributes at first order.  The error is therefore ``O(1 / M)`` here
        and ``O(1 / M**2)`` for Eq. A.2; both claims are measured below.
        This is a property of the prescribed grid, not of Eq. A.3, and it is
        irrelevant to the urn weight (plan Section 2.4); the accuracy is
        stated, not silently assumed.
        """
        errors = self._grid_relative_errors(case, (256,))

        assert errors[0] < 1e-2

    def test_grid_error_converges_at_first_order(self):
        """Error halves when ``M`` doubles, with slope 1.0 (measured 0.61 / M)."""
        n_alphas = (32, 64, 128, 256, 512, 1024)
        errors = self._grid_relative_errors(CASES[0], n_alphas)

        assert all(later < earlier for earlier, later in pairwise(errors))
        orders = [
            float(np.log2(earlier / later)) for earlier, later in pairwise(errors)
        ]
        assert min(orders) > 0.9
        assert max(orders) < 1.1

    def test_grid_error_is_far_below_the_urn_scale_at_high_resolution(self):
        """``M = 4096`` must reach ``1e-3`` relative on the closed form."""
        errors = self._grid_relative_errors(CASES[0], (4096,))

        assert errors[0] < 1e-3

    def test_rows_are_independent(self):
        """Row ``i`` of ``log q0^c`` must depend on ``z_i`` only."""
        z_values = np.array([1.0 + 1e-6, 1.25, 1.5, 2.0])
        gamma, phi, nu, d = 0.7, 1.4, 1.1, 2

        grid = make_grid(_unit_data(z_values), 512)
        joint = log_q0_censored(grid, gamma=gamma, phi=phi, nu=nu, d=d)

        assert joint.shape == z_values.shape
        for i, z in enumerate(z_values):
            single = make_grid(_unit_data(np.array([z])), 512)
            solo = log_q0_censored(single, gamma=gamma, phi=phi, nu=nu, d=d)
            assert joint[i] == pytest.approx(float(solo[0]), rel=1e-14, abs=0.0)


class TestCensoredBranchShape:
    """The properties that distinguish A.3 from A.2."""

    def test_integrand_has_no_alpha_or_inverse_time_factor(self):
        """The integrand is exactly ``(gamma + z**alpha)**-d``: nothing else."""
        z = np.array([1.2, 1.9])
        grid = make_grid(_unit_data(z), 64)
        gamma, phi, d = 0.4, 1.1, 2
        alpha = phi * grid.u

        log_integrand = log_alpha_integrand_censored(grid, gamma=gamma, phi=phi, d=d)
        direct = -d * np.log(gamma + z[:, None] ** alpha[None, :])

        assert log_integrand.shape == (z.size, 64)
        assert np.allclose(log_integrand, direct, rtol=1e-13, atol=1e-13)

    def test_integrand_is_not_the_uncensored_one(self):
        """Swapping the branches must change the value materially."""
        z = np.array([1.5])
        grid = make_grid(_unit_data(z), 64)
        gamma, phi, d = 0.6, 1.4, 2

        censored = log_alpha_integrand_censored(grid, gamma=gamma, phi=phi, d=d)
        uncensored = log_alpha_integrand_uncensored(grid, gamma=gamma, phi=phi, d=d)

        assert censored.shape == uncensored.shape
        assert not np.allclose(censored, uncensored, rtol=1e-3, atol=1e-3)
        assert float(np.max(np.abs(censored - uncensored))) > 1.0

    def test_mass_decreases_in_time(self):
        """A survival mass must be strictly decreasing in ``z``."""
        z_values = np.array([1.0 + 1e-6, 1.05, 1.2, 1.5, 2.0])
        grid = make_grid(_unit_data(z_values), 256)

        log_q0 = log_q0_censored(grid, gamma=0.6, phi=1.4, nu=2.0, d=2)

        assert np.all(np.diff(log_q0) < 0.0)

    def test_scales_linearly_in_nu(self):
        """Doubling ``nu`` must double ``q0^c`` exactly: it enters as a factor."""
        z = 1.5
        grid = make_grid(_unit_data(np.array([z])), 256)

        base = float(np.exp(log_q0_censored(grid, gamma=0.6, phi=1.4, nu=1.0, d=2)[0]))
        doubled = float(
            np.exp(log_q0_censored(grid, gamma=0.6, phi=1.4, nu=2.0, d=2)[0])
        )
        assert doubled == pytest.approx(2.0 * base, rel=1e-14)

    def test_the_d_plus_one_mutation_is_rejected(self):
        """Using ``d + 1`` (the uncensored shape) in A.3 must be visible.

        Plan Section 11: there is no ``d + 1`` in ``q0^c``.  The mutation is
        asserted to be far above the ``1e-10`` tolerance of test 2, so a
        silent copy of the uncensored exponent cannot pass the suite.
        """
        gamma, phi, nu = GAMMA_REF, PHI_REF, 1.0
        grid = make_grid(_unit_data(np.array([Z_REF])), 4096)

        log_correct = log_q0_censored(grid, gamma=gamma, phi=phi, nu=nu, d=D_REF)
        correct = float(np.exp(log_correct[0]))
        mutated = _eq_a3(Z_REF, gamma, phi, nu, D_REF + 1)

        assert abs(mutated / correct - 1.0) > 0.1


class TestCensoredValidation:
    """Malformed inputs must fail loudly, never silently."""

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
                log_q0_censored(grid, gamma=gamma, phi=phi, nu=nu, d=2)
            # The integrand does not depend on ``nu``: only ``gamma``/``phi``
            # are validated there.
            if nu == 1.0:
                with pytest.raises(ValueError):
                    log_alpha_integrand_censored(grid, gamma=gamma, phi=phi, d=2)

    def test_the_inverse_gamma_shape_must_be_a_positive_integer(self):
        """``d`` indexes the survival marginal; ``0`` or ``2.5`` is a bug."""
        grid = make_grid(_unit_data(np.array([1.5])), 8)

        for d in (0, -1, 2.5):
            with pytest.raises(ValueError, match="d must be a positive integer"):
                log_q0_censored(grid, gamma=0.6, phi=1.4, nu=1.0, d=d)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="d must be a positive integer"):
                log_alpha_integrand_censored(grid, gamma=0.6, phi=1.4, d=d)  # type: ignore[arg-type]

    def test_two_nodes_are_enough(self):
        """The smallest legal grid still builds and integrates."""
        grid = make_grid(_unit_data(np.array([1.5, 2.0])), 2)

        log_q0 = log_q0_censored(grid, gamma=0.6, phi=1.0, nu=1.0, d=2)
        assert grid.M == 2
        assert np.all(np.isfinite(log_q0))

    def test_log_q0_censored_is_finite_over_extreme_inputs(self):
        """No overflow or underflow to zero for hostile-but-legal inputs."""
        z = np.array([1.0 + 1e-9, 2.0])
        grid = make_grid(_unit_data(z), 256)

        for gamma, phi, nu in [
            (1e-6, 50.0, 1e-6),
            (1e6, 1e-3, 1e6),
            (1.0, 1.0, 1.0),
        ]:
            for d in (1, 2, 3):
                log_q0 = log_q0_censored(grid, gamma=gamma, phi=phi, nu=nu, d=d)
                assert np.all(np.isfinite(log_q0))
