"""Test 31 (plan Sections 3.7 and 7.5): diagnostics against analytic values.

The three public diagnostics -- ``autocorr_time``, ``r_hat`` and ``ess`` -- are
the only place a stored chain is turned into a convergence claim, and a wrong
formula fails *silently in the safe-looking direction*: a variance ratio taken
over the draw axis instead of the chain axis, or an ESS that forgets the
autocorrelation sum, both return a number near one for any input.  The tests
therefore check the formulas against a process whose stationary variance and
lag-1 correlation are known exactly:

* AR(1) chains ``x_t = rho x_{t-1} + sqrt(1 - rho^2) eps_t`` have stationary
  variance 1 and autocorrelation ``rho`` at every lag, so
  ``tau = (1 + rho) / (1 - rho)`` and ``ESS = n (1 - rho) / (1 + rho)``.
  Four independent chains must give split ``R-hat -> 1`` within 0.05 and the
  bulk ESS within 10% of the analytic value;
* four short chains started at deliberately different values must give
  ``R-hat > 1.2`` -- the between-chain spread is the signal, and a statistic
  computed over draws cannot see it;
* the ``S`` trace is checked against an independent computation of the
  observation-level Weibull mixture in this file, point by point, so a
  transposed ``(draw, point, observation)`` index or a missing inverse map
  fails here rather than in the mixing test of plan Section 7.3.

What each test would catch:

* a variance ratio applied over the wrong axis -- the different-starts test;
* an ESS with no autocorrelation sum -- the AR(1) accuracy test (it would
  return exactly ``n``);
* an autocorrelation time that is not truncated -- the long-lag noise makes
  ``tau`` grow with the trace length, so the iid test fails on a long trace;
* a split that halves the wrong way or drops burn-in twice -- the exact
  equivalence between the DPChains path and the raw-trace path;
* a constant trace silently reported as "converged" -- the explicit
  ``ValueError`` tests (a constant chain is a diagnostic failure, not a
  missing number);
* a malformed input (2-D trace, single chain, empty sequence, mixed kinds,
  non-finite values) -- the None-One-Many validation matrix.
"""

import math

import numpy as np
import pytest

from src.dp_gibbs import (
    DPChains,
    DPPrior,
    DPSettings,
    autocorr_time,
    ess,
    make_surv_data,
    r_hat,
)

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.0,
    d=2,
)

TIME_FRACTIONS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def _ar1(
    rho: float,
    n_draws: int,
    seed: int,
    *,
    start: float | None = None,
    burn_in: int = 1000,
) -> np.ndarray:
    """A stationary AR(1) trace with variance 1 and lag-1 correlation ``rho``.

    ``start=None`` draws the initial value from the stationary ``N(0, 1)``
    law, which is what makes four chains exchangeable for the R-hat test.
    A numeric ``start`` (with ``burn_in=0``) produces the deliberately
    different initial values of the second test.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal() if start is None else float(start)
    innovation = math.sqrt(1.0 - rho * rho)
    for _ in range(burn_in):
        x = rho * x + innovation * rng.standard_normal()
    trace = np.empty(n_draws, dtype=np.float64)
    for t in range(n_draws):
        x = rho * x + innovation * rng.standard_normal()
        trace[t] = x
    return trace


def _make_chain(seed: int, *, n_save: int = 20, t_max: float = 3.0) -> DPChains:
    """Build a valid ``DPChains`` directly, without running the sampler.

    Diagnostics read storage, not the sampler, so the fixture is hand-built:
    a short artificial run whose theta vary enough to produce a finite S
    trace.  ``phi = 1`` is at the prior's lower bound and above every alpha,
    which is what ``DPChains`` validates.
    """
    n = 3
    data = make_surv_data(
        time=np.array([1.0, 0.5 * t_max, t_max]),
        event=np.array([True, False, True]),
    )
    settings = DPSettings(
        burn_in=0,
        n_save=n_save,
        thin=1,
        n_alpha=8,
        n_chains=1,
        seed=seed,
        preset="test",
    )
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.4, 0.9, size=(n_save, n))
    lam = rng.uniform(0.5, 2.5, size=(n_save, n))
    theta = np.stack([alpha, lam], axis=2)
    nu = rng.uniform(1.0, 3.0, size=n_save)
    gamma = rng.uniform(0.5, 2.0, size=n_save)
    phi = 1.0 + rng.uniform(0.0, 0.5, size=n_save)
    n_star = rng.integers(1, n + 1, size=n_save).astype(np.intp)
    log_post = rng.normal(size=n_save)
    return DPChains(
        data=data,
        prior=PRIOR,
        settings=settings,
        theta=theta,
        nu=nu,
        gamma=gamma,
        phi=phi,
        n_star=n_star,
        log_post=log_post,
    )


def _survival_trace(chain: DPChains) -> np.ndarray:
    """Independent recomputation of the observation-level survival mixture.

    Deliberately written out in the test instead of importing a helper: the
    point is to pin the definition ``(1/n) sum_i exp(-t_tilde**alpha_i /
    lambda_i)`` at the five fractions of ``t_max``, including the
    ``t_tilde = 1 + t / t_max`` transform.
    """
    t_tilde = 1.0 + TIME_FRACTIONS
    alpha = chain.theta[:, :, 0]
    lam = chain.theta[:, :, 1]
    kernels = np.exp(-(t_tilde[None, :, None] ** alpha[:, None, :]) / lam[:, None, :])
    return kernels.mean(axis=2)


class TestAutocorrTime:
    """The integrated autocorrelation time (plan Section 3.7)."""

    def test_independent_draws_have_unit_time(self):
        """For iid draws ``tau`` is 1; long-lag noise must not inflate it."""
        rng = np.random.default_rng(20241101)
        trace = rng.standard_normal(20000)
        tau = autocorr_time(trace)
        assert abs(tau - 1.0) < 0.10

    def test_matches_the_ar1_population_value(self):
        """``tau = (1 + rho) / (1 - rho)`` for an AR(1) chain."""
        rho = 0.6
        trace = _ar1(rho, 4000, 101)
        tau = autocorr_time(trace)
        expected = (1.0 + rho) / (1.0 - rho)
        assert abs(tau - expected) / expected < 0.20

    def test_antithetic_chain_has_a_time_below_one(self):
        """Negative lag-1 correlation gives ``tau < 1`` and hence ``ESS > n``."""
        trace = _ar1(-0.5, 4000, 102)
        tau = autocorr_time(trace)
        assert tau < 1.0
        assert ess([trace], "trace")["trace"] > trace.size

    @pytest.mark.parametrize(
        "trace, message",
        [
            (np.ones(20), "constant"),
            (np.zeros((2, 10)), "1-D"),
            (np.array([0.0, np.nan, 1.0, 2.0]), "finite"),
            (np.array([1.0]), "at least two"),
        ],
    )
    def test_rejects_invalid_traces(self, trace, message):
        """None-One-Many over shape, length and finiteness."""
        with pytest.raises(ValueError, match=message):
            autocorr_time(trace)


class TestSplitRHat:
    """Gelman's split potential scale reduction (plan Section 3.7, test 31)."""

    def test_converged_ar1_chains_give_r_hat_near_one(self):
        """Four stationary chains: ``|R-hat - 1| < 0.05`` (test 31)."""
        chains = [_ar1(0.6, 2000, 200 + b) for b in range(4)]
        result = r_hat(chains, "trace")
        value = result["trace"]
        assert isinstance(value, float)
        assert abs(value - 1.0) <= 0.05

    def test_different_initial_values_give_r_hat_above_one(self):
        """Short chains from different starts must be flagged (test 31)."""
        chains = [
            _ar1(0.9, 20, 300 + b, start=start, burn_in=0)
            for b, start in enumerate([-3.0, -1.0, 1.0, 3.0])
        ]
        value = r_hat(chains, "trace")["trace"]
        assert isinstance(value, float)
        assert value > 1.2

    def test_between_chain_shift_is_seen(self):
        """A constant offset added to one chain raises R-hat well above 1."""
        rng = np.random.default_rng(7)
        chains = [rng.standard_normal(200) for _ in range(4)]
        chains[0] = chains[0] + 2.0
        value = r_hat(chains, "trace")["trace"]
        assert isinstance(value, float)
        assert value > 1.2

    def test_two_dimensional_array_is_one_trace_per_row(self):
        """A ``(n_chains, n_draws)`` array is accepted as the chain axis."""
        rng = np.random.default_rng(8)
        matrix = rng.standard_normal((4, 800))
        from_matrix = r_hat(matrix, "trace")["trace"]
        from_list = r_hat([row for row in matrix], "trace")["trace"]
        assert from_matrix == from_list

    def test_agrees_between_the_dpchains_and_raw_trace_paths(self):
        """``DPChains`` extraction is exactly ``[chain.nu]`` as raw traces."""
        chains = [_make_chain(11), _make_chain(12), _make_chain(13)]
        from_chains = r_hat(chains, "nu")["nu"]
        from_raw = r_hat([chain.nu for chain in chains], "nu")["nu"]
        assert from_chains == from_raw

    def test_constant_chains_are_converged_when_they_agree(self):
        """Every draw equal everywhere has ``W = 0 = B``: report 1, not 0/0."""
        chains = [np.full(10, 2.5), np.full(10, 2.5)]
        value = r_hat(chains, "trace")["trace"]
        assert value == 1.0

    def test_constant_chains_are_infinite_when_they_disagree(self):
        """``W = 0`` with a finite between-chain spread is divergence."""
        chains = [np.full(10, 1.0), np.full(10, 2.0)]
        value = r_hat(chains, "trace")["trace"]
        assert isinstance(value, float)
        assert math.isinf(value)

    def test_rejects_unequal_lengths(self):
        """Split R-hat pairs sub-chains of one common half length."""
        with pytest.raises(ValueError, match="equal length"):
            r_hat([np.zeros(10), np.zeros(12)], "trace")

    def test_rejects_too_few_draws(self):
        """Two sub-chains need two draws each, so a chain needs four."""
        with pytest.raises(ValueError, match="at least four"):
            r_hat([np.zeros(3), np.zeros(3)], "trace")


class TestESS:
    """The bulk effective sample size (plan Section 3.7, test 31)."""

    def test_matches_the_ar1_analytic_ess(self):
        """``ESS`` is within 10% of ``n (1 - rho) / (1 + rho)`` (test 31)."""
        rho = 0.6
        n_draws = 4000
        n_chains = 4
        chains = [_ar1(rho, n_draws, 401 + b) for b in range(n_chains)]
        value = ess(chains, "trace")["trace"]
        expected = n_chains * n_draws * (1.0 - rho) / (1.0 + rho)
        assert abs(value - expected) / expected < 0.10

    def test_independent_draws_ess_is_the_draw_count(self):
        """For iid draws the effective sample size is the sample size."""
        rng = np.random.default_rng(9)
        value = ess([rng.standard_normal(4000)], "trace")["trace"]
        assert abs(value - 4000) / 4000 < 0.10

    def test_rejects_a_constant_trace(self):
        """A stuck parameter has no autocorrelation time; do not report ``n``."""
        with pytest.raises(ValueError, match="constant"):
            ess([np.full(20, 1.5)], "trace")


class TestSurvivalTrace:
    """The ``S`` diagnostic at five original-scale times."""

    def test_matches_an_independent_mixture_computation(self):
        """Each point of the ``S`` R-hat is the R-hat of the recomputed trace."""
        chains = [_make_chain(21), _make_chain(22)]
        result = r_hat(chains, "S")
        expected = np.array(
            [
                r_hat([_survival_trace(chain)[:, j] for chain in chains], "x")["x"]
                for j in range(TIME_FRACTIONS.size)
            ]
        )
        assert np.array_equal(np.asarray(result["S"]), expected)

    def test_reports_the_original_scale_times(self):
        """``t`` is the five fractions of ``t_max``, not the transformed grid."""
        chain = _make_chain(23, t_max=4.5518)
        result = r_hat([chain], "S")
        assert np.allclose(np.asarray(result["t"]), TIME_FRACTIONS * 4.5518)

    def test_survival_is_between_zero_and_one(self):
        """A probability cannot leave ``[0, 1]`` at any diagnostic time."""
        chains = [_make_chain(24), _make_chain(25)]
        for chain in chains:
            trace = _survival_trace(chain)
            assert np.all(trace >= 0.0)
            assert np.all(trace <= 1.0)
        assert np.all(np.asarray(r_hat(chains, "S")["S"]) > 0.0)
        assert np.all(np.isfinite(np.asarray(r_hat(chains, "S")["S"])))

    def test_ess_reports_one_value_per_time(self):
        """``ess(..., "S")`` has the same five-point shape as ``r_hat``."""
        chains = [_make_chain(26), _make_chain(27)]
        result = ess(chains, "S")
        assert np.asarray(result["S"]).shape == (TIME_FRACTIONS.size,)
        assert np.all(np.isfinite(np.asarray(result["S"])))
        assert np.all(np.asarray(result["S"]) > 0.0)

    def test_rejects_chains_fitted_to_different_datasets(self):
        """Survival at different ``t_max`` cannot be compared point by point."""
        with pytest.raises(ValueError, match="same dataset"):
            r_hat([_make_chain(28), _make_chain(29, t_max=5.0)], "S")


class TestInputValidation:
    """The None-One-Many matrix around the diagnostics' inputs."""

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValueError, match="at least one chain"):
            r_hat([], "nu")

    def test_rejects_a_bare_chain(self):
        """A single chain would report a single-chain value as multi-chain."""
        with pytest.raises(ValueError, match="single chain"):
            r_hat(_make_chain(31), "nu")

    def test_rejects_a_bare_trace(self):
        with pytest.raises(ValueError, match="single trace"):
            r_hat(np.zeros(10), "nu")

    def test_rejects_mixed_chain_and_trace_inputs(self):
        with pytest.raises(ValueError, match="mixing"):
            r_hat([_make_chain(32), np.zeros(20)], "nu")

    def test_rejects_a_non_string_name(self):
        with pytest.raises(ValueError, match="name must be a string"):
            r_hat([_make_chain(33), _make_chain(34)], 3)  # type: ignore[arg-type]

    def test_rejects_an_unknown_name_for_dpchains(self):
        with pytest.raises(ValueError, match="name must be one of"):
            ess([_make_chain(35), _make_chain(36)], "log_post")
