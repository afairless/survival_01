"""Test 8 (plan Section 7.2): the Polya-urn weights of step (a).

Step (a) resamples one observation's ``(alpha_i, lambda_i, s_i)`` from the
mixture of plan Section 2.2(a)::

    q0^i h_i(alpha_i, lambda_i | gamma, phi, t_i)
      + sum_j n_j^- q_j^i delta_{(alpha_j^*, lambda_j^*)},
    normalised by q0^i + sum_j n_j^- q_j^i,

where ``q0^i`` is the prior-predictive density ``q0^o`` of Eq. A.2 for an
uncensored observation and the prior-predictive survival mass ``q0^c`` of
Eq. A.3 for a right-censored one, and ``q_j^i`` is the Weibull density (or
survival) at cluster ``j``'s location.  The counts ``n_j^-`` are taken *after*
observation ``i`` is removed.

The test fixes the state, enumerates the category masses from the equations
with ``scipy.integrate.quad`` and the closed-form kernel -- never with the
sampler's grid or its cached ``q0`` -- then applies the production step (a) to
one observation many times from the same starting state and compares the
empirical category frequencies by a chi-square test at ``alpha = 1e-3``.  The
draws come from ``sample_step_a_for_obs`` with ``compact=False``, the
deferred-compaction mode in which ``state.s[i]`` names the sampled category
directly (an old label, or ``n_star_before`` for a freshly appended cluster).

What each assertion would catch:

* the category frequencies reject a missing or misplaced ``n_j^-`` factor
  (tested explicitly below: the same counts reject the ``n_j`` hypothesis);
* the censored case rejects the density-for-survival swap in ``q_j^i``;
* the singleton case rejects a cluster that stays in the candidate set after
  its last observation is removed -- the "join your own singleton cluster"
  path, which has weight ``0 * q_j`` and is therefore invisible unless the
  draw can land there;
* the sweep test rejects a lost or duplicated count, an uncompacted label or a
  stale cached kernel column.
"""

import dataclasses

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

from src.dp_gibbs import (
    DPPrior,
    DPState,
    SurvData,
    make_grid,
    sample_step_a,
    sample_step_a_for_obs,
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

# Five observations, three clusters.  Cluster 0 is uncensored and not a
# singleton; cluster 1 is censored and not a singleton; cluster 2 is an
# uncensored singleton, so resampling observation 4 empties it and its label
# must leave the candidate set.
DATA = SurvData(
    time=np.array([1.10, 1.25, 1.40, 1.55, 1.70], dtype=np.float64),
    event=np.array([True, True, False, False, True]),
    t_max=1.0,
)
LABELS = np.array([0, 0, 1, 1, 2], dtype=np.intp)
ALPHA_STAR = np.array([0.7, 1.2, 1.6], dtype=np.float64)
LAM_STAR = np.array([0.8, 1.4, 2.2], dtype=np.float64)

# Plan Section 7.2 asks for 1e5 repetitions.  The two non-singleton cases use
# the full count; the singleton case needs enough draws for its three
# categories only, and each repetition pays for a full ``q0`` quadrature.
N_DRAWS = 100_000
N_DRAWS_SINGLETON = 50_000
N_ALPHA = 64


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


# ---------------------------------------------------------------------------
# Independent reference masses (quad + closed-form kernel, no grid)
# ---------------------------------------------------------------------------
def _q0_uncensored_reference(t: float) -> float:
    """Eq. A.2 with the alpha integral evaluated by ``quad``."""

    def integrand(alpha: float) -> float:
        return alpha * t ** (alpha - 1.0) / (GAMMA + t**alpha) ** (D + 1)

    integral, _ = quad(integrand, 0.0, PHI, epsabs=1e-14, epsrel=1e-12, limit=200)
    return D * NU * GAMMA**D / PHI * integral


def _q0_censored_reference(z: float) -> float:
    """Eq. A.3 with the alpha integral evaluated by ``quad``."""

    def integrand(alpha: float) -> float:
        return 1.0 / (GAMMA + z**alpha) ** D

    integral, _ = quad(integrand, 0.0, PHI, epsabs=1e-14, epsrel=1e-12, limit=200)
    return NU * GAMMA**D / PHI * integral


def _density_reference(t: float, alpha: float, lam: float) -> float:
    """``k_W(t | alpha, lambda)`` written from the paper's definition."""
    return alpha * t ** (alpha - 1.0) / lam * float(np.exp(-(t**alpha) / lam))


def _survival_reference(z: float, alpha: float, lam: float) -> float:
    """``1 - K_W(z | alpha, lambda) = exp(-z**alpha / lambda)``."""
    return float(np.exp(-(z**alpha) / lam))


def _q0_reference(time: float, event: bool) -> float:
    return _q0_uncensored_reference(time) if event else _q0_censored_reference(time)


def _reference_urn_masses(
    state: DPState,
    data: SurvData,
    i: int,
    *,
    use_removed_counts: bool = True,
    swap_kernel: bool = False,
) -> tuple[np.ndarray, list[int | None]]:
    """Exact category masses and their labels for observation ``i``.

    Category 0 is always the "new cluster" branch.  Category ``k >= 1`` is the
    cluster ``labels[k]``, present only when its count *after removing* ``i``
    is positive.

    Parameters
    ----------
    use_removed_counts : bool
        ``True`` builds the reference from ``n_j^-`` (the specification).  The
        power check passes ``False`` to build the ``n_j`` hypothesis that the
        same draws must reject.
    swap_kernel : bool
        ``True`` uses the density for a censored observation and the survival
        for an uncensored one -- the branch swap the censored test must
        reject.
    """
    event_i = bool(data.event[i])
    counts = state.n_j.astype(np.int64).copy()
    if use_removed_counts:
        counts[int(state.s[i])] -= 1
    masses = [_q0_reference(float(data.time[i]), event_i)]
    labels: list[int | None] = [None]
    for j in range(state.n_star):
        if counts[j] == 0:
            continue
        if swap_kernel:
            use_density = not event_i
        else:
            use_density = event_i
        q = (
            _density_reference(
                float(data.time[i]), state.alpha_star[j], state.lam_star[j]
            )
            if use_density
            else _survival_reference(
                float(data.time[i]), state.alpha_star[j], state.lam_star[j]
            )
        )
        masses.append(float(counts[j]) * q)
        labels.append(j)
    total = float(np.sum(masses))
    return np.array(masses) / total, labels


# ---------------------------------------------------------------------------
# Repeated draws from the fixed state
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class UrnDraws:
    """Empirical category counts plus the reference they are tested against."""

    counts: np.ndarray
    probabilities: np.ndarray
    labels: list[int | None]
    n_draws: int
    observation: int
    removed_label: int


def _snapshot(state: DPState) -> dict[str, np.ndarray]:
    return {
        "alpha": state.alpha.copy(),
        "lam": state.lam.copy(),
        "s": state.s.copy(),
        "alpha_star": state.alpha_star.copy(),
        "lam_star": state.lam_star.copy(),
        "n_j": state.n_j.copy(),
    }


def _state_from_snapshot(state: DPState, snapshot: dict[str, np.ndarray]) -> DPState:
    """Build one validated working state; runs are reset in place afterwards."""
    return DPState(
        alpha=snapshot["alpha"].copy(),
        lam=snapshot["lam"].copy(),
        s=snapshot["s"].copy(),
        alpha_star=snapshot["alpha_star"].copy(),
        lam_star=snapshot["lam_star"].copy(),
        n_j=snapshot["n_j"].copy(),
        nu=state.nu,
        gamma=state.gamma,
        phi=state.phi,
    )


def _reset(state: DPState, snapshot: dict[str, np.ndarray]) -> None:
    """Restore the fixed state in place between repetitions.

    Resetting by attribute assignment rather than rebuilding ``DPState`` keeps
    the 1e5 repetitions affordable; the constructor's validation is exercised
    once by :func:`_state_from_snapshot` and is not part of what this test
    checks.  ``compact=False`` never renumbers the surviving labels, so the
    pre-removal labels of ``snapshot`` stay meaningful.
    """
    for name, values in snapshot.items():
        setattr(state, name, values.copy())


def _draw_urn_frequencies(
    state: DPState,
    data: SurvData,
    i: int,
    *,
    n_draws: int,
    seed: int,
) -> UrnDraws:
    """Draw observation ``i``'s category ``n_draws`` times from a fixed state."""
    grid = make_grid(data, N_ALPHA)
    probabilities, labels = _reference_urn_masses(state, data, i)
    n_star_before = state.n_star
    category_of = {label: k for k, label in enumerate(labels) if label is not None}
    removed_label = int(state.s[i])

    snapshot = _snapshot(state)
    work = _state_from_snapshot(state, snapshot)
    rng = np.random.default_rng(seed)
    counts = np.zeros(probabilities.size, dtype=np.int64)
    for _ in range(n_draws):
        _reset(work, snapshot)
        sample_step_a_for_obs(work, data, PRIOR, i=i, grid=grid, rng=rng, compact=False)
        label = int(work.s[i])
        if label == n_star_before:
            counts[0] += 1
        elif label in category_of:
            counts[category_of[label]] += 1
        else:
            raise AssertionError(
                f"category label {label} is neither a surviving cluster nor the "
                f"appended one; the emptied cluster {removed_label} was still "
                "in the candidate set"
            )
    return UrnDraws(
        counts=counts,
        probabilities=probabilities,
        labels=labels,
        n_draws=n_draws,
        observation=i,
        removed_label=removed_label,
    )


def _chi_square(counts: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Return the chi-square statistic and its upper-tail p-value."""
    expected = probabilities * counts.sum()
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(chi2.sf(statistic, counts.size - 1))
    return statistic, p_value


@pytest.fixture(scope="module")
def uncensored_draws() -> UrnDraws:
    """Observation 0: uncensored, in a cluster of size two."""
    return _draw_urn_frequencies(_make_state(), DATA, 0, n_draws=N_DRAWS, seed=20240601)


@pytest.fixture(scope="module")
def censored_draws() -> UrnDraws:
    """Observation 2: right-censored, in a cluster of size two."""
    return _draw_urn_frequencies(_make_state(), DATA, 2, n_draws=N_DRAWS, seed=20240602)


@pytest.fixture(scope="module")
def singleton_draws() -> UrnDraws:
    """Observation 4: uncensored and alone, so its cluster must vanish."""
    return _draw_urn_frequencies(
        _make_state(), DATA, 4, n_draws=N_DRAWS_SINGLETON, seed=20240603
    )


# ---------------------------------------------------------------------------
# Frequencies against the exact urn probabilities
# ---------------------------------------------------------------------------
class TestUrnCategoryFrequencies:
    """The empirical categories must follow the exact urn probabilities."""

    def test_uncensored_observation(self, uncensored_draws: UrnDraws):
        """Chi-square survives at alpha=1e-3 for an uncensored observation."""
        draws = uncensored_draws
        assert draws.counts.sum() == draws.n_draws
        assert np.all(draws.counts > 0)
        assert draws.labels == [None, 0, 1, 2]

        statistic, p_value = _chi_square(draws.counts, draws.probabilities)
        assert p_value > 1e-3, (
            f"uncensored urn frequencies deviate: statistic={statistic:.1f}, "
            f"p={p_value:.3g}, observed={draws.counts}, "
            f"expected={draws.probabilities * draws.n_draws}"
        )

    def test_uncensored_counts_reject_the_n_j_hypothesis(
        self, uncensored_draws: UrnDraws
    ):
        """The same draws must reject using ``n_j`` instead of ``n_j^-``.

        This is the "what it would catch" statement of test 8: the missing
        ``n_j^-`` shifts the probabilities by O(1-5 percentage points), which
        is far outside the sampling error at this sample size.  If this check
        ever fails, the fixture has lost its power and the frequency test
        above proves nothing.
        """
        draws = uncensored_draws
        wrong, _ = _reference_urn_masses(
            _make_state(), DATA, 0, use_removed_counts=False
        )
        assert np.max(np.abs(wrong - draws.probabilities)) > 0.01
        _, p_value = _chi_square(draws.counts, wrong)
        assert p_value < 1e-3, (
            f"the fixture cannot distinguish n_j from n_j^-; observed={draws.counts}"
        )

    def test_censored_observation(self, censored_draws: UrnDraws):
        """Chi-square survives at alpha=1e-3 for a censored observation."""
        draws = censored_draws
        assert draws.counts.sum() == draws.n_draws
        assert np.all(draws.counts > 0)
        assert draws.labels == [None, 0, 1, 2]

        statistic, p_value = _chi_square(draws.counts, draws.probabilities)
        assert p_value > 1e-3, (
            f"censored urn frequencies deviate: statistic={statistic:.1f}, "
            f"p={p_value:.3g}, observed={draws.counts}, "
            f"expected={draws.probabilities * draws.n_draws}"
        )

    def test_censored_counts_reject_the_swapped_kernel(self, censored_draws: UrnDraws):
        """The censored draws must reject the density-for-survival swap."""
        wrong, _ = _reference_urn_masses(_make_state(), DATA, 2, swap_kernel=True)
        assert np.max(np.abs(wrong - censored_draws.probabilities)) > 0.01
        _, p_value = _chi_square(censored_draws.counts, wrong)
        assert p_value < 1e-3

    def test_singleton_removal_drops_the_old_cluster(self, singleton_draws: UrnDraws):
        """The emptied singleton cluster must not be a candidate any more.

        Observation 4 is alone in cluster 2.  Removing it first means cluster
        2 has count zero and the reference list contains only clusters 0 and
        1; the draw helper raises if any repetition lands on label 2, and the
        chi-square below would otherwise see a category with zero probability.
        """
        draws = singleton_draws
        assert draws.labels == [None, 0, 1]
        assert draws.removed_label == 2
        assert draws.counts.sum() == draws.n_draws
        assert np.all(draws.counts > 0)

        statistic, p_value = _chi_square(draws.counts, draws.probabilities)
        assert p_value > 1e-3, (
            f"singleton urn frequencies deviate: statistic={statistic:.1f}, "
            f"p={p_value:.3g}, observed={draws.counts}, "
            f"expected={draws.probabilities * draws.n_draws}"
        )


# ---------------------------------------------------------------------------
# Bookkeeping of the sweep entry point
# ---------------------------------------------------------------------------
class TestStepABookkeeping:
    """Repeated sweeps must keep the deferred-compaction bookkeeping exact."""

    def test_full_sweeps_preserve_the_invariants(self):
        state = _make_state()
        grid = make_grid(DATA, N_ALPHA)
        rng = np.random.default_rng(20240610)

        for _ in range(40):
            sample_step_a(state, DATA, PRIOR, grid=grid, rng=rng)
            state.check_invariants(PRIOR)
            assert int(state.n_j.sum()) == DATA.time.size
            assert np.array_equal(state.alpha, state.alpha_star[state.s])
            assert np.array_equal(state.lam, state.lam_star[state.s])
            assert np.array_equal(
                state.n_j, np.bincount(state.s, minlength=state.n_star)
            )
            assert state.alpha_star.size == state.n_star
            assert 0 < state.n_star <= DATA.time.size

    def test_single_observation_dataset_always_appends_a_cluster(self):
        """Removing the only observation empties the only cluster."""
        data = SurvData(
            time=np.array([1.5], dtype=np.float64),
            event=np.array([True]),
            t_max=1.0,
        )
        state = DPState(
            alpha=np.array([1.0]),
            lam=np.array([1.0]),
            s=np.array([0], dtype=np.intp),
            alpha_star=np.array([1.0]),
            lam_star=np.array([1.0]),
            n_j=np.array([1], dtype=np.intp),
            nu=NU,
            gamma=GAMMA,
            phi=PHI,
        )
        grid = make_grid(data, N_ALPHA)
        rng = np.random.default_rng(20240611)
        for _ in range(25):
            sample_step_a(state, data, PRIOR, grid=grid, rng=rng)
            state.check_invariants(PRIOR)
            assert state.n_star == 1
            assert int(state.n_j.sum()) == 1
            assert 0.0 < float(state.alpha[0]) <= PHI
            assert float(state.lam[0]) > 0.0

    def test_new_cluster_location_is_drawn_from_the_continuous_piece(self):
        """A new-cluster draw lands on the grid, inside ``(0, phi]``, twice."""
        state = _make_state()
        grid = make_grid(DATA, N_ALPHA)
        rng = np.random.default_rng(20240612)
        nodes = PHI * grid.u
        seen_new = False

        snapshot = _snapshot(state)
        for _ in range(300):
            _reset(state, snapshot)
            n_star_before = state.n_star
            sample_step_a_for_obs(
                state, DATA, PRIOR, i=4, grid=grid, rng=rng, compact=False
            )
            if int(state.s[4]) == n_star_before:
                seen_new = True
                alpha = float(state.alpha_star[-1])
                lam = float(state.lam_star[-1])
                assert np.isclose(alpha, nodes, rtol=0.0, atol=1e-12).any()
                assert 0.0 < alpha <= PHI
                assert lam > 0.0
        assert seen_new

    def test_same_seed_reproduces_and_different_seed_differs(self):
        """The rng argument is the only source of randomness."""
        first = _draw_urn_frequencies(_make_state(), DATA, 0, n_draws=400, seed=99)
        repeat = _draw_urn_frequencies(_make_state(), DATA, 0, n_draws=400, seed=99)
        other = _draw_urn_frequencies(_make_state(), DATA, 0, n_draws=400, seed=100)

        assert np.array_equal(first.counts, repeat.counts)
        assert not np.array_equal(first.counts, other.counts)

    def test_out_of_range_observation_is_rejected(self):
        state = _make_state()
        grid = make_grid(DATA, N_ALPHA)
        rng = np.random.default_rng(1)
        with pytest.raises(IndexError):
            sample_step_a_for_obs(
                state, DATA, PRIOR, i=DATA.time.size, grid=grid, rng=rng
            )

    def test_grid_for_another_dataset_is_rejected(self):
        """A stale grid must not silently feed the wrong ``t ** u`` row."""
        state = _make_state()
        other = SurvData(time=DATA.time + 0.01, event=DATA.event, t_max=DATA.t_max)
        grid = make_grid(other, N_ALPHA)
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError, match="different dataset"):
            sample_step_a(state, DATA, PRIOR, grid=grid, rng=rng)
