"""Test 11 (plan Section 7.2): the auxiliary-variable refresh of a cluster shape.

Step (b) of the Appendix refreshes each cluster location.  Its non-conjugate
half is the shape, plan Section 2.2(b), Eqs. A.5-A.6: with
``n_j^o = |{i_o : s_{i_o} = j}|`` the number of *uncensored* members, the
Damien-Wakefield-Walker device draws an auxiliary ``v ~ U(0, C(alpha*_j))``
(``C(alpha) = prod_{i_o} t_{i_o}**alpha``) and one ``w_i`` per member, forms
the bounds ``alpha^-_j`` and ``alpha^+_j``, and then draws

    ``alpha*_j ~ (alpha*_j)**n_j^o``   on ``(alpha^-_j, alpha^+_j)``,

i.e. ``k = n_j^o + 1`` in the inverse-CDF form, **not** uniformly.

The reference used here is the corresponding *unconditional* law, obtained by
integrating the auxiliaries out:

    ``p(alpha*_j) ~ (alpha*_j)**n_j^o C(alpha*_j)
       exp(-lambda*_j**-1 (sum_{i_o} t_{i_o}**alpha*_j
                          + sum_{i_c} z_{i_c}**alpha*_j))``

on ``(0, phi)``.  It is evaluated directly from the plan's formula -- with
plain NumPy and ``scipy.integrate.quad`` for the normalisation -- and shares
no code with the sampler's bounds arithmetic, so an error in ``C``, in a bound
or in the exponent cannot make the reference agree with it.

Comparison rule (the plan states it): the ``1e5`` draws are binned into the 40
equal-mass intervals of the reference, and both a total-variation distance
(``< 0.02``) and a maximum CDF deviation at those edges (``< 0.01``) are
asserted.  A KS p-value would be over-powered at this sample size: it rejects
a deviation of the order of ``1e-2``, which is the size of the effect we are
willing to tolerate.

Cases: ``n_j^o in {0, 1, 2, 5}`` with no censored members, plus mixed
clusters.  The ``n_j^o = 0`` case is the only one whose target is uniform
*conditional on the auxiliaries*; the ``n_j^o = 1`` case is already
triangular.  Every uncensored case would fail if the production code drew
uniformly on the bounds, which is exactly the mutation test 11 exists to
catch.
"""

import math
from typing import Any, cast

import numpy as np
import pytest
from scipy.integrate import quad

from src.dp_gibbs import (
    DPPrior,
    DPState,
    StateInvariantError,
    SurvData,
    alpha_bound_guard_count,
    refresh_cluster,
    refresh_cluster_alpha,
    reset_alpha_bound_guard_count,
    sample_step_b,
)

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
)

NU = 2.5
GAMMA = 0.5
PHI = 2.0
LAM_STAR = 1.4
ALPHA_START = 1.0

N_DRAWS = 100_000
BURN_IN = 1_000
N_BINS = 40
N_REFERENCE_GRID = 4001
TV_TOLERANCE = 0.02
MAX_CDF_TOLERANCE = 0.01

# (uncensored members, censored members)
CASES = [
    pytest.param((0, 3), id="all-censored-n_o=0"),
    pytest.param((1, 0), id="one-uncensored-n_o=1"),
    pytest.param((2, 0), id="two-uncensored-n_o=2"),
    pytest.param((5, 0), id="five-uncensored-n_o=5"),
    pytest.param((2, 3), id="mixed-n_o=2"),
    pytest.param((5, 2), id="mixed-n_o=5"),
]


def _make_case(n_uncensored: int, n_censored: int) -> tuple[SurvData, DPState]:
    """Build one single-cluster dataset and the matching fixed state.

    Times are on the transformed scale (strictly inside ``(1, 2]``) and are
    spread over the interval so that the ``log t`` values in the bounds are
    neither degenerate nor tiny.  ``alpha*_j`` starts at ``1.0``; the refresh
    is a valid Gibbs step from any starting value, and the burn-in below
    removes the start-up transient.
    """
    n = n_uncensored + n_censored
    event = np.array([True] * n_uncensored + [False] * n_censored, dtype=bool)
    data = SurvData(time=np.linspace(1.08, 1.95, n), event=event, t_max=1.0)
    state = DPState(
        alpha=np.full(n, ALPHA_START),
        lam=np.full(n, LAM_STAR),
        s=np.zeros(n, dtype=np.intp),
        alpha_star=np.array([ALPHA_START]),
        lam_star=np.array([LAM_STAR]),
        n_j=np.array([n], dtype=np.intp),
        nu=NU,
        gamma=GAMMA,
        phi=PHI,
    )
    return data, state


def _reference_density(alpha: float, data: SurvData) -> float:
    """Unnormalised Eq. A.5 law of ``alpha*_j`` at ``alpha``.

    Written straight from plan Section 2.2(b): ``alpha**n_j^o`` times
    ``C(alpha) = prod_{i_o} t_{i_o}**alpha`` times the exponential of the
    negative power sum over **all** members divided by ``lambda*_j``.  The
    censored members therefore appear only in the power sum, never in
    ``C(alpha)`` or in the ``alpha**n_j^o`` factor.
    """
    times = data.time
    event = data.event
    n_uncensored = int(np.count_nonzero(event))
    log_c = alpha * float(np.sum(np.log(times[event]))) if n_uncensored else 0.0
    power_sum = float(np.sum(times**alpha))
    return float(alpha**n_uncensored * math.exp(log_c - power_sum / LAM_STAR))


def _reference_summary(data: SurvData) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(alpha_grid, cdf, mean)`` of the reference law.

    The normalising constant is computed with ``quad`` on ``(0, phi)`` as the
    plan requires; the CDF itself is a cumulative trapezoid on a grid fine
    enough (spacing ``5e-4``) that its interpolation error is far below the
    ``0.02`` TV tolerance.  ``mean`` is the exact quadrature mean, used as a
    second, independent check on the draws.
    """
    grid = np.linspace(0.0, PHI, N_REFERENCE_GRID)
    values = np.array([_reference_density(a, data) for a in grid])
    norm: float = quad(lambda a: _reference_density(a, data), 0.0, PHI, limit=200)[0]
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    mean: float = quad(lambda a: a * _reference_density(a, data), 0.0, PHI, limit=200)[
        0
    ]
    return grid, cdf / norm, mean / norm


def _reference_edges(grid: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Return the ``N_BINS`` equal-mass edges of the reference law."""
    probabilities = np.linspace(0.0, 1.0, N_BINS + 1)
    edges = np.interp(probabilities, cdf, grid)
    edges[-1] = PHI
    if not np.all(np.diff(edges) > 0.0):
        raise AssertionError("reference CDF is not strictly increasing")
    return edges


def _production_draws(
    data: SurvData, state: DPState, *, seed: int, n_draws: int = N_DRAWS
) -> np.ndarray:
    """Run the production refresh, discard a burn-in and return the draws.

    ``lambda*_j`` is held fixed across the whole run: that is exactly the
    conditional being tested, ``[alpha*_j | lambda*_j, s, gamma, phi, D]``.
    The update is a two-block Gibbs step on the joint law of
    ``(alpha*_j, v, w)``, so the burn-in drops the transient and the draws are
    from its stationary law -- the A.5 density.
    """
    rng = np.random.default_rng(seed)
    for _ in range(BURN_IN):
        refresh_cluster_alpha(state, data, PRIOR, j=0, rng=rng)
    draws = np.empty(n_draws, dtype=np.float64)
    for k in range(n_draws):
        refresh_cluster_alpha(state, data, PRIOR, j=0, rng=rng)
        draws[k] = state.alpha_star[0]
    return draws


Case = tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]


@pytest.fixture(scope="module", params=CASES)
def case(request: Any) -> Case:
    """Production draws and the reference summary for one composition."""
    n_uncensored, n_censored = request.param
    data, state = _make_case(n_uncensored, n_censored)
    draws = _production_draws(data, state, seed=20260 + n_uncensored * 10 + n_censored)
    grid, cdf, reference_mean = _reference_summary(data)
    return n_uncensored, n_censored, draws, grid, cdf, reference_mean


def _binned_distances(draws: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Return ``(total variation, max CDF deviation)`` on the given edges."""
    counts, _ = np.histogram(draws, bins=edges)
    probabilities = counts / draws.size
    expected = np.full(counts.size, 1.0 / counts.size)
    total_variation = 0.5 * float(np.sum(np.abs(probabilities - expected)))
    empirical_cdf = np.cumsum(probabilities)
    reference_cdf = np.arange(1, counts.size + 1) / counts.size
    return total_variation, float(np.max(np.abs(empirical_cdf - reference_cdf)))


# ---------------------------------------------------------------------------
# Test 11 proper: the draws match the unconditional A.5 law
# ---------------------------------------------------------------------------
class TestAlphaStarConditional:
    def test_matches_the_unconditional_a5_density(self, case: Case):
        n_uncensored, n_censored, draws, grid, cdf, _mean = case
        edges = _reference_edges(grid, cdf)
        total_variation, max_cdf = _binned_distances(draws, edges)
        assert total_variation < TV_TOLERANCE, (
            f"n_j^o={n_uncensored}, n_j^c={n_censored}: TV={total_variation:.4f}"
        )
        assert max_cdf < MAX_CDF_TOLERANCE, (
            f"n_j^o={n_uncensored}, n_j^c={n_censored}: max CDF deviation={max_cdf:.4f}"
        )

    def test_matches_the_reference_mean(self, case: Case):
        n_uncensored, n_censored, draws, _grid, _cdf, reference_mean = case
        empirical_mean = float(np.mean(draws))
        # Autocorrelation is high for large clusters (the bounds are tight);
        # a 5% absolute tolerance is far outside the Monte Carlo error but
        # still rejects the uniform-on-the-bounds mutation, whose mean on
        # these cases sits well above the reference mean.
        assert abs(empirical_mean - reference_mean) < 0.05, (
            f"n_j^o={n_uncensored}, n_j^c={n_censored}: empirical mean "
            f"{empirical_mean:.4f} vs reference {reference_mean:.4f}"
        )

    def test_draws_stay_in_the_parameter_space(self, case: Case):
        n_uncensored, n_censored, draws, _grid, _cdf, _mean = case
        assert np.all(draws > 0.0), (n_uncensored, n_censored)
        # Invariant 3 of plan Section 2.1: alpha*_j <= phi, guaranteed by the
        # min(phi, .) in alpha^+_j.
        assert np.all(draws <= PHI), (n_uncensored, n_censored)

    def test_uniform_draw_on_the_bounds_is_rejected(self, case: Case):
        n_uncensored, n_censored, draws, _grid, _cdf, _mean = case
        if n_uncensored < 2:
            pytest.skip("n_j^o < 2 is too close to the uniform law to separate")
        uniform_edges = np.linspace(0.0, PHI, N_BINS + 1)
        total_variation, _ = _binned_distances(draws, uniform_edges)
        # Drawing uniformly on (alpha^-, alpha^+) -- the A.6 mutation -- would
        # make this distance small; the correct alpha^{n_j^o} draw makes it a
        # multiple of the tolerance.
        assert total_variation > TV_TOLERANCE, (
            f"n_j^o={n_uncensored}, n_j^c={n_censored}: draws are consistent "
            f"with a uniform law (TV={total_variation:.4f})"
        )


# ---------------------------------------------------------------------------
# The degenerate cases the plan singles out
# ---------------------------------------------------------------------------
class TestDegenerateCases:
    def test_all_censored_cluster_matches_the_reference(self):
        """``n_j^o = 0`` means ``k = 1``, but the auxiliaries still bind.

        The target is uniform *conditional on the ``w``'s*; unconditionally it
        is the decreasing density ``exp(-sum_i z_i**alpha / lambda)``.  A
        sampler that skipped the ``w`` bounds would draw uniformly on
        ``(0, phi)`` and miss the reference mean by a wide margin.
        """
        data, state = _make_case(0, 3)
        draws = _production_draws(data, state, seed=99, n_draws=20_000)
        _grid, _cdf, reference_mean = _reference_summary(data)
        assert abs(float(np.mean(draws)) - reference_mean) < 0.04
        assert abs(float(np.mean(draws)) - PHI / 2.0) > 0.1

    def test_single_member_cluster_stays_in_the_parameter_space(self):
        data, state = _make_case(1, 0)
        rng = np.random.default_rng(101)
        for _ in range(200):
            refresh_cluster_alpha(state, data, PRIOR, j=0, rng=rng)
            assert 0.0 < state.alpha_star[0] <= PHI
        state.check_invariants(PRIOR)

    def test_no_guard_clamp_in_normal_operation(self):
        reset_alpha_bound_guard_count()
        for n_uncensored, n_censored in [(0, 3), (1, 0), (2, 0), (5, 0), (2, 3)]:
            data, state = _make_case(n_uncensored, n_censored)
            _production_draws(data, state, seed=7, n_draws=2_000)
        assert alpha_bound_guard_count() == 0


# ---------------------------------------------------------------------------
# The bounds guard: an empty interval is clamped and counted, not sampled
# ---------------------------------------------------------------------------
class _AllOnesRng:
    """Uniform source returning ``1.0``, forcing a degenerate interval."""

    def random(self, size: Any = None) -> Any:
        if size is None:
            return 1.0
        return np.ones(size)


class TestBoundsGuard:
    def test_empty_interval_is_clamped_and_counted(self):
        # alpha*_j = phi and u_v = 1 put alpha^- exactly at phi, while the
        # w bounds give alpha^+ = log(t**phi) / log t = phi too, so the
        # interval is empty and the guard must clamp to the midpoint phi.
        data, state = _make_case(1, 0)
        state.alpha_star[0] = PHI
        state.alpha[:] = PHI
        state.check_invariants(PRIOR)
        before = alpha_bound_guard_count()
        fake = cast(Any, _AllOnesRng())
        refresh_cluster_alpha(state, data, PRIOR, j=0, rng=fake)
        assert alpha_bound_guard_count() == before + 1
        assert state.alpha_star[0] == pytest.approx(PHI)
        assert np.allclose(state.alpha, PHI, rtol=0.0, atol=1e-12)
        state.check_invariants(PRIOR)

    def test_reset_restores_the_counter(self):
        reset_alpha_bound_guard_count()
        assert alpha_bound_guard_count() == 0


# ---------------------------------------------------------------------------
# Bookkeeping and the random-number stream
# ---------------------------------------------------------------------------
class TestRefreshBookkeeping:
    def test_updates_the_cluster_and_its_members_only(self):
        data, state = _make_case(2, 3)
        before_alpha_star = state.alpha_star.copy()
        before_lam_star = state.lam_star.copy()
        before_lam = state.lam.copy()
        before_s = state.s.copy()
        before_n_j = state.n_j.copy()

        refresh_cluster_alpha(state, data, PRIOR, j=0, rng=np.random.default_rng(5))

        assert state.alpha_star[0] != before_alpha_star[0]
        assert np.array_equal(state.alpha, np.full(state.n, state.alpha_star[0]))
        # The shape half must not touch the scale, the labels or the counts.
        assert np.array_equal(state.lam_star, before_lam_star)
        assert np.array_equal(state.lam, before_lam)
        assert np.array_equal(state.s, before_s)
        assert np.array_equal(state.n_j, before_n_j)
        assert state.nu == NU
        assert state.gamma == GAMMA
        assert state.phi == PHI
        state.check_invariants(PRIOR)

    def test_is_reproducible_from_the_seed(self):
        data, first = _make_case(3, 2)
        _data, second = _make_case(3, 2)
        _data, other_seed = _make_case(3, 2)
        refresh_cluster_alpha(first, data, PRIOR, j=0, rng=np.random.default_rng(6))
        refresh_cluster_alpha(second, data, PRIOR, j=0, rng=np.random.default_rng(6))
        refresh_cluster_alpha(
            other_seed, data, PRIOR, j=0, rng=np.random.default_rng(7)
        )
        assert second.alpha_star[0] == first.alpha_star[0]
        assert other_seed.alpha_star[0] != first.alpha_star[0]

    def test_consumes_the_documented_number_of_uniforms(self):
        # No uncensored member: one uniform per member for the w's, then one
        # for the inverse-CDF draw.  With an uncensored member there is one
        # more, for v.  The golden regression of plan Section 7.5 depends on
        # this count, so it is pinned here.
        for n_uncensored, n_censored, n_members in [(0, 3, 3), (2, 0, 2)]:
            data, state = _make_case(n_uncensored, n_censored)
            rng = np.random.default_rng(1234)
            refresh_cluster_alpha(state, data, PRIOR, j=0, rng=rng)
            next_value = float(rng.random())

            expected_uniforms = n_members + 1 + (1 if n_uncensored else 0)
            replay = np.random.default_rng(1234)
            replay.random(expected_uniforms)
            assert float(replay.random()) == pytest.approx(
                next_value, rel=0.0, abs=1e-15
            )

    def test_out_of_range_cluster_raises(self):
        data, state = _make_case(1, 0)
        with pytest.raises(IndexError):
            refresh_cluster_alpha(
                state, data, PRIOR, j=state.n_star, rng=np.random.default_rng(7)
            )
        with pytest.raises(IndexError):
            refresh_cluster_alpha(
                state, data, PRIOR, j=-1, rng=np.random.default_rng(7)
            )

    def test_uncompacted_state_raises(self):
        data, state = _make_case(2, 2)
        state.remove_observation(0)
        with pytest.raises(StateInvariantError):
            refresh_cluster_alpha(state, data, PRIOR, j=0, rng=np.random.default_rng(8))

    def test_mismatched_dataset_raises(self):
        data, state = _make_case(2, 0)
        other = SurvData(
            time=data.time[:1].copy(),
            event=data.event[:1].copy(),
            t_max=1.0,
        )
        with pytest.raises(ValueError):
            refresh_cluster_alpha(
                state, other, PRIOR, j=0, rng=np.random.default_rng(9)
            )


# ---------------------------------------------------------------------------
# The step-(b) driver runs scale then shape, per cluster
# ---------------------------------------------------------------------------
class TestSampleStepB:
    def test_refreshes_both_halves_of_every_cluster(self):
        data, state = _make_case(3, 2)
        before_alpha = state.alpha_star.copy()
        before_lam = state.lam_star.copy()
        sample_step_b(state, data, PRIOR, rng=np.random.default_rng(10))
        assert not np.array_equal(state.alpha_star, before_alpha)
        assert not np.array_equal(state.lam_star, before_lam)
        assert np.array_equal(state.alpha, state.alpha_star[state.s])
        assert np.array_equal(state.lam, state.lam_star[state.s])
        state.check_invariants(PRIOR)

    def test_runs_scale_then_shape_from_one_stream(self):
        # Step (b) draws lambda* first and forms the w bounds at it.  Replaying
        # the two halves from one generator must reproduce both draws exactly;
        # the reversed order would not.
        data, state = _make_case(2, 1)
        sample_step_b(state, data, PRIOR, rng=np.random.default_rng(11))

        _data, replay = _make_case(2, 1)
        rng = np.random.default_rng(11)
        refresh_cluster(replay, data, PRIOR, j=0, rng=rng)
        refresh_cluster_alpha(replay, data, PRIOR, j=0, rng=rng)

        assert replay.alpha_star[0] == pytest.approx(state.alpha_star[0])
        assert replay.lam_star[0] == pytest.approx(state.lam_star[0])
