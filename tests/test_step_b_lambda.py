"""Test 10 (plan Section 7.2): the inverse-gamma refresh of a cluster scale.

Step (b) of the Appendix refreshes each cluster location.  Its conjugate half
is the scale, plan Section 2.2(b)::

    lambda*_j | alpha*_j, s, gamma, D
        ~ IGamma(d + n_j^o,
                 gamma + sum_{i in j} t_i ** alpha*_j),

where ``n_j^o = |{i_o : s_{i_o} = j}|`` counts only the *uncensored* members
while the power sum runs over every member, censored or not, evaluated at the
cluster's own ``alpha*_j``.

The test fixes the state, derives the reference parameters with plain NumPy
from the plan's formula -- never from the sampler's private helper -- and
compares many production draws from ``refresh_cluster`` with
``scipy.stats.invgamma`` by a Kolmogorov-Smirnov test.  The three cases cover
the shapes the plan asks for, ``n_j^o in {0, 1, 3}``, each with censored
members present.

What each assertion would catch:

* the ``n_j^o = 0`` case rejects a shape that adds ``n_j`` (the all-censored
  cluster would get ``d + n_j`` instead of ``d``);
* the ``n_j^o in {1, 3}`` cases reject ``shape = d``, i.e. a dropped
  uncensored count, with the largest effect at ``n_j^o = 3``;
* every case rejects a scale without ``gamma`` and a scale built at a
  neighbouring cluster's shape, so a missing ``gamma`` and a mis-indexed
  ``alpha*_j`` cannot hide behind the conjugate form.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest
from scipy.stats import invgamma, ks_1samp

from src.dp_gibbs import (
    DPPrior,
    DPState,
    StateInvariantError,
    SurvData,
    refresh_cluster,
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

GAMMA = 0.5
PHI = 2.0
NU = 2.5
D = 2

# Eight observations in three clusters:
#   cluster 0 -- three uncensored members and one censored (n_j^o = 3),
#   cluster 1 -- one uncensored member and one censored (n_j^o = 1),
#   cluster 2 -- two censored members only (n_j^o = 0).
# Cluster 2 is the case the plan singles out: its shape is ``d`` alone, so a
# ``d + n_j`` shortcut would give ``d + 2`` there.
DATA = SurvData(
    time=np.array([1.10, 1.25, 1.40, 1.55, 1.70, 1.85, 1.95, 1.50]),
    event=np.array([True, True, True, False, True, False, False, False]),
    t_max=1.0,
)
LABELS = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.intp)
ALPHA_STAR = np.array([0.7, 1.2, 1.6], dtype=np.float64)
LAM_STAR = np.array([0.8, 1.4, 2.2], dtype=np.float64)

N_DRAWS = 20_000

# (cluster, uncensored members, censored members)
CASES = [
    pytest.param(0, 3, 1, id="three-uncensored-one-censored"),
    pytest.param(1, 1, 1, id="one-uncensored-one-censored"),
    pytest.param(2, 0, 2, id="all-censored"),
]


def _make_state() -> DPState:
    """Build the fixed reference state described above."""
    return DPState(
        alpha=ALPHA_STAR[LABELS].copy(),
        lam=LAM_STAR[LABELS].copy(),
        s=LABELS.copy(),
        alpha_star=ALPHA_STAR.copy(),
        lam_star=LAM_STAR.copy(),
        n_j=np.bincount(LABELS, minlength=ALPHA_STAR.size).astype(np.intp),
        nu=NU,
        gamma=GAMMA,
        phi=PHI,
    )


def _reference_parameters(j: int, *, alpha: float | None = None) -> tuple[int, float]:
    """Return ``(d + n_j^o, gamma + sum_i t_i**alpha)`` from the plan.

    Written directly from Section 2.2(b) with NumPy sums over the labelled
    members; it shares no code with the sampler, so a formula error in the
    production helper cannot make the reference agree with it.
    """
    members = LABELS == j
    times = DATA.time[members]
    events = DATA.event[members]
    alpha = ALPHA_STAR[j] if alpha is None else alpha
    shape = D + int(np.count_nonzero(events))
    scale = GAMMA + float(np.sum(times**alpha))
    return shape, scale


def _ks_statistic(sample: np.ndarray, distribution: Any) -> tuple[float, float]:
    """Return the KS statistic and p-value as plain floats."""
    result: Any = ks_1samp(sample, distribution.cdf)
    return float(result.statistic), float(result.pvalue)


@pytest.fixture(scope="module")
def draws_by_cluster() -> dict[int, np.ndarray]:
    """Production ``lambda*_j`` draws for every cluster, from a fixed state.

    The state is fixed except for ``lam_star[j]``/``lam[members]``, which the
    refresh replaces but the next draw does not read (the conditional depends
    on ``alpha*_j``, ``gamma`` and the data only), so one state can be reused
    for all repetitions -- no reset is needed between draws.
    """
    draws: dict[int, np.ndarray] = {}
    for j in range(ALPHA_STAR.size):
        state = _make_state()
        rng = np.random.default_rng(20240 + j)
        sample = np.empty(N_DRAWS, dtype=np.float64)
        for k in range(N_DRAWS):
            refresh_cluster(state, DATA, PRIOR, j=j, rng=rng)
            sample[k] = state.lam_star[j]
        draws[j] = sample
    return draws


# ---------------------------------------------------------------------------
# Test 10 proper: the draws match the exact inverse-gamma conditional
# ---------------------------------------------------------------------------
class TestLambdaStarConditional:
    @pytest.mark.parametrize(("j", "n_uncensored", "n_censored"), CASES)
    def test_matches_inverse_gamma(
        self,
        draws_by_cluster: dict[int, np.ndarray],
        j: int,
        n_uncensored: int,
        n_censored: int,
    ):
        members = LABELS == j
        assert int(np.count_nonzero(DATA.event[members])) == n_uncensored
        assert int(np.count_nonzero(~DATA.event[members])) == n_censored
        shape, scale = _reference_parameters(j)
        statistic, p_value = _ks_statistic(
            draws_by_cluster[j], invgamma(a=shape, scale=scale)
        )
        assert p_value > 1e-3, (
            f"cluster {j} draws reject IGamma({shape}, scale={scale}): "
            f"KS D={statistic:.4f}, p={p_value:.3e}"
        )
        # With 2e4 draws a correct sampler gives D around 1/sqrt(N) ~ 0.007;
        # the effect-size guard keeps an over-powered KS from being the only
        # criterion and documents the scale of a real deviation.
        assert statistic < 0.03

    @pytest.mark.parametrize(("j", "n_uncensored", "n_censored"), CASES)
    def test_dropped_uncensored_count_is_rejected(
        self,
        draws_by_cluster: dict[int, np.ndarray],
        j: int,
        n_uncensored: int,
        n_censored: int,
    ):
        if n_uncensored == 0:
            pytest.skip("shape is d whether or not n_j^o is added")
        _, scale = _reference_parameters(j)
        statistic, p_value = _ks_statistic(
            draws_by_cluster[j], invgamma(a=D, scale=scale)
        )
        assert p_value < 1e-9
        assert statistic > 0.05

    @pytest.mark.parametrize(("j", "n_uncensored", "n_censored"), CASES)
    def test_missing_gamma_is_rejected(
        self,
        draws_by_cluster: dict[int, np.ndarray],
        j: int,
        n_uncensored: int,
        n_censored: int,
    ):
        shape, scale = _reference_parameters(j)
        scaled_without_gamma = scale - GAMMA
        statistic, p_value = _ks_statistic(
            draws_by_cluster[j], invgamma(a=shape, scale=scaled_without_gamma)
        )
        assert p_value < 1e-9
        assert statistic > 0.02

    @pytest.mark.parametrize(("j", "n_uncensored", "n_censored"), CASES)
    def test_neighbour_shape_is_rejected(
        self,
        draws_by_cluster: dict[int, np.ndarray],
        j: int,
        n_uncensored: int,
        n_censored: int,
    ):
        other = (j + 1) % ALPHA_STAR.size
        shape, scale = _reference_parameters(j, alpha=ALPHA_STAR[other])
        statistic, p_value = _ks_statistic(
            draws_by_cluster[j], invgamma(a=shape, scale=scale)
        )
        assert p_value < 1e-6
        assert statistic > 0.02


# ---------------------------------------------------------------------------
# Bookkeeping: the per-observation latent values follow the cluster location
# ---------------------------------------------------------------------------
class TestRefreshBookkeeping:
    def test_updates_the_cluster_and_its_members_only(self):
        state = _make_state()
        before_lam_star = state.lam_star.copy()
        before_alpha_star = state.alpha_star.copy()
        before_alpha = state.alpha.copy()
        refresh_cluster(state, DATA, PRIOR, j=1, rng=np.random.default_rng(3))

        assert state.lam_star[1] != before_lam_star[1]
        untouched = np.array([0, 2])
        assert np.array_equal(state.lam_star[untouched], before_lam_star[untouched])
        # Step (b)'s scale half must not touch the shape; the auxiliary update
        # of Section 2.2(b) is the only thing allowed to change alpha_star.
        assert np.array_equal(state.alpha_star, before_alpha_star)
        assert np.array_equal(state.alpha, before_alpha)
        assert np.array_equal(state.lam, state.lam_star[state.s])
        assert np.array_equal(
            state.lam[LABELS == 1], np.full(int(state.n_j[1]), state.lam_star[1])
        )
        state.check_invariants(PRIOR)

    def test_repeated_refreshes_change_the_scale(self):
        state = _make_state()
        rng = np.random.default_rng(4)
        values = []
        for _ in range(5):
            refresh_cluster(state, DATA, PRIOR, j=2, rng=rng)
            values.append(state.lam_star[2])
        assert len(set(values)) == len(values)
        state.check_invariants(PRIOR)

    def test_out_of_range_cluster_raises(self):
        state = _make_state()
        rng = np.random.default_rng(5)
        with pytest.raises(IndexError):
            refresh_cluster(state, DATA, PRIOR, j=state.n_star, rng=rng)
        with pytest.raises(IndexError):
            refresh_cluster(state, DATA, PRIOR, j=-1, rng=rng)

    def test_uncompacted_state_raises(self):
        state = _make_state()
        state.remove_observation(6)
        state.remove_observation(7)
        with pytest.raises(StateInvariantError):
            refresh_cluster(state, DATA, PRIOR, j=2, rng=np.random.default_rng(6))
        # A removed-but-surviving cluster is the same bookkeeping error: the
        # member list is one longer than the stored count.
        state = _make_state()
        state.remove_observation(4)
        with pytest.raises(StateInvariantError):
            refresh_cluster(state, DATA, PRIOR, j=1, rng=np.random.default_rng(7))

    def test_mismatched_dataset_raises(self):
        state = _make_state()
        other = SurvData(
            time=DATA.time[:6].copy(),
            event=DATA.event[:6].copy(),
            t_max=1.0,
        )
        with pytest.raises(ValueError):
            refresh_cluster(state, other, PRIOR, j=0, rng=np.random.default_rng(8))


# ---------------------------------------------------------------------------
# The step-(b) driver: every cluster, in index order, one variate each
# ---------------------------------------------------------------------------
class TestSampleStepB:
    def test_refreshes_every_cluster(self):
        state = _make_state()
        before = state.lam_star.copy()
        sample_step_b(state, DATA, PRIOR, rng=np.random.default_rng(9))
        assert np.all(state.lam_star != before)
        assert np.array_equal(state.lam, state.lam_star[state.s])
        state.check_invariants(PRIOR)

    def test_consumes_one_variate_per_cluster_in_index_order(self):
        state = _make_state()
        rng = np.random.default_rng(10)
        sample_step_b(state, DATA, PRIOR, rng=rng)

        # Replay the same stream by hand: one gamma variate per cluster, in
        # j = 0..n_star-1 order, from the reference parameters.  This pins the
        # order and the per-cluster consumption, which the golden test of plan
        # Section 7.5 depends on.
        replay = np.random.default_rng(10)
        expected = np.array(
            [
                1.0
                / replay.gamma(
                    shape=_reference_parameters(j)[0],
                    scale=1.0 / _reference_parameters(j)[1],
                )
                for j in range(ALPHA_STAR.size)
            ]
        )
        assert np.allclose(state.lam_star, expected, rtol=1e-15, atol=0.0)

    def test_is_reproducible_from_the_seed(self):
        first = _make_state()
        second = _make_state()
        sample_step_b(first, DATA, PRIOR, rng=np.random.default_rng(11))
        sample_step_b(second, DATA, PRIOR, rng=np.random.default_rng(11))
        assert np.array_equal(first.lam_star, second.lam_star)
        assert np.array_equal(first.lam, second.lam)

    def test_deferred_removal_raises(self):
        state = _make_state()
        state.remove_observation(0)
        with pytest.raises(StateInvariantError):
            sample_step_b(state, DATA, PRIOR, rng=np.random.default_rng(12))

    def test_leaves_the_other_state_fields_alone(self):
        state = _make_state()
        before = dataclasses.replace(
            state,
            alpha=state.alpha.copy(),
            lam=state.lam.copy(),
            s=state.s.copy(),
            alpha_star=state.alpha_star.copy(),
            lam_star=state.lam_star.copy(),
            n_j=state.n_j.copy(),
        )
        sample_step_b(state, DATA, PRIOR, rng=np.random.default_rng(13))
        assert np.array_equal(state.alpha, before.alpha)
        assert np.array_equal(state.alpha_star, before.alpha_star)
        assert np.array_equal(state.s, before.s)
        assert np.array_equal(state.n_j, before.n_j)
        assert state.nu == before.nu
        assert state.gamma == before.gamma
        assert state.phi == before.phi
