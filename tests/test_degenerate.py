"""Test 20 (plan Sections 7.3 and 10): degenerate-data robustness.

The Appendix algorithm has a number of branches that only fire on data shapes
the paper's examples never exhibit, and every one of them divides by something
that is zero for a well-formed dataset:

* **all-censored data.**  No observation contributes a density term, so
  ``q_0^c`` (Eq. A.3) is the only prior-predictive mass used.  In step (b) a
  cluster of censored observations has ``n_j^o = 0``: the ``C(alpha)`` product
  is empty, ``v`` is not drawn, the ``alpha^-`` formula would divide by an
  empty sum, and the shaped draw is uniform (``k = 1``) rather than
  ``alpha**n_j^o``.  Posterior inference on all-censored data is meaningless
  -- the likelihood is flat in the scale beyond ordering -- but the chain must
  run, stay finite and keep every Section 2.1 invariant.
* **all-uncensored data.**  The mirror case: the censored prior-predictive
  mass and the censored terms of ``alpha^+_j`` are never touched.
* **``n = 1`` and ``n = 2``.**  ``n*`` is then 1 (or 1-2), ``n_j`` is a single
  count, and ``sum(n_j)`` has no slack: an off-by-one in the urn bookkeeping
  shows up immediately.
* **duplicated and all-equal times.**  ``log t`` repeats, so the ``alpha^-``
  sum has repeated terms and the ``alpha^+`` minimum has ties.  All-equal
  times make the transformed times all ``2.0``; the sampler must not depend on
  a spread of times to avoid a degenerate interval.

Each case runs 100 sweeps through the public entry points, checking after
every sweep that the state is finite, strictly positive, satisfies all four
Section 2.1 invariants, and that ``n*`` stays in ``1..n``.  A separate
white-box test exercises the all-censored ``alpha^-`` branch directly and
pins its random-number consumption (no ``v`` draw), so the empty-sum path is
covered by an assertion rather than only by "it did not crash".

What each test would catch:

* a ``q_0^c`` or censored-cluster branch that returns ``NaN``, a negative
  shape/scale, or an empty cluster -- the per-sweep invariant checks;
* a division by the empty ``sum_{i_o} log t_{i_o}`` -- the all-censored case,
  which would raise ``ZeroDivisionError`` or produce ``inf``/``NaN``;
* a degenerate ``alpha^+ <= alpha^-`` interval silently clamped on every
  sweep -- the guard counter, which must stay zero;
* an urn bookkeeping error that only shows when ``n* = n`` (``n = 1``) or when
  every observation is drawn into one cluster -- the ``n = 1``/``n = 2`` cases
  and the all-censored chain;
* a chain driver that accepts the data but stores a non-finite or invalid
  draw -- the chain-driver case, whose stored traces are checked explicitly.
"""

import numpy as np
import pytest

from src.dp_gibbs import (
    DPChains,
    DPPrior,
    DPSettings,
    DPState,
    SurvData,
    _alpha_star_bounds,
    alpha_bound_guard_count,
    init_state,
    make_grid,
    make_surv_data,
    refresh_cluster_alpha,
    reset_alpha_bound_guard_count,
    sample_chain,
    sweep,
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

SEED = 20241007
N_ALPHA = 24
N_SWEEPS = 100

_TIMES = np.array([0.7, 1.4, 2.1, 3.3, 4.9, 6.2, 8.5])

CASES = [
    pytest.param(_TIMES, np.ones(_TIMES.size, dtype=bool), id="all-uncensored"),
    pytest.param(_TIMES, np.zeros(_TIMES.size, dtype=bool), id="all-censored"),
    pytest.param(np.array([3.7]), np.array([True]), id="n=1-uncensored"),
    pytest.param(np.array([3.7]), np.array([False]), id="n=1-censored"),
    pytest.param(np.array([1.0, 5.0]), np.array([True, False]), id="n=2"),
    pytest.param(
        np.array([2.0, 2.0, 2.0, 4.0, 4.0, 7.0, 7.0, 7.0]),
        np.array([True, False, True, False, True, True, False, True]),
        id="duplicated-times",
    ),
    pytest.param(
        np.full(6, 4.0),
        np.array([True, False, True, True, False, True]),
        id="all-equal-times",
    ),
    pytest.param(
        np.full(6, 4.0),
        np.zeros(6, dtype=bool),
        id="all-equal-times-all-censored",
    ),
]


def _ingest(time: np.ndarray, event: np.ndarray) -> SurvData:
    """Ingest one degenerate case through the production boundary."""
    return make_surv_data(time, event)


def _settings(**overrides: object) -> DPSettings:
    """Short settings that keep the driver case at 100 sweeps."""
    base: dict[str, object] = {
        "burn_in": 0,
        "n_save": 10,
        "thin": 10,
        "n_alpha": N_ALPHA,
        "n_chains": 1,
        "seed": SEED,
        "preset": "test",
    }
    base.update(overrides)
    return DPSettings(**base)  # type: ignore[arg-type]


def _assert_valid_state(state: DPState, data: SurvData) -> None:
    """All four Section 2.1 invariants plus finiteness, checked explicitly.

    ``check_invariants`` already covers the invariants; the explicit finiteness
    and range assertions are kept separate so a failure names the quantity
    (``alpha``, ``lam``, ``nu``, ...) rather than only the invariant number.
    """
    state.check_invariants(PRIOR)
    assert np.all(np.isfinite(state.alpha)), "alpha is not finite"
    assert np.all(np.isfinite(state.lam)), "lam is not finite"
    assert np.all(np.isfinite(state.alpha_star)), "alpha_star is not finite"
    assert np.all(np.isfinite(state.lam_star)), "lam_star is not finite"
    assert np.all(state.alpha > 0.0) and np.all(state.lam > 0.0)
    assert np.all(state.alpha_star > 0.0) and np.all(state.lam_star > 0.0)
    assert np.all(state.alpha <= state.phi)
    assert np.all(state.alpha_star <= state.phi)
    assert np.isfinite(state.nu) and state.nu > 0.0
    assert np.isfinite(state.gamma) and state.gamma > 0.0
    assert np.isfinite(state.phi) and state.phi >= PRIOR.b_phi
    assert 1 <= state.n_star <= data.time.size
    assert int(state.n_j.sum()) == data.time.size


class TestDegenerateSweeps:
    """One hundred sweeps, invariant-checked after every sweep."""

    @pytest.mark.parametrize(("time", "event"), CASES)
    def test_one_hundred_sweeps_stay_valid(self, time: np.ndarray, event: np.ndarray):
        """No NaN, no negative parameter and no broken invariant, ever."""
        data = _ingest(time, event)
        grid = make_grid(data, N_ALPHA)
        rng = np.random.default_rng(SEED)
        state = init_state(data, PRIOR, grid=grid, rng=rng)
        _assert_valid_state(state, data)

        reset_alpha_bound_guard_count()
        for sweep_index in range(N_SWEEPS):
            score = sweep(state, data, PRIOR, grid=grid, rng=rng)
            assert np.isfinite(score), (
                f"log-posterior is not finite at sweep {sweep_index}"
            )
            _assert_valid_state(state, data)

        # The guard fires only when floating-point rounding collapses an
        # alpha* interval to a point.  On these small, well-separated times and
        # with the empty-sum branch handled explicitly it must never fire: a
        # non-zero count would mean a cluster shape is being frozen rather than
        # sampled.
        assert alpha_bound_guard_count() == 0

    @pytest.mark.parametrize(("time", "event"), CASES)
    def test_chain_driver_stores_valid_draws(self, time: np.ndarray, event: np.ndarray):
        """The driver records 100 sweeps of the same case without corruption."""
        data = _ingest(time, event)
        chain = sample_chain(data, PRIOR, _settings())

        assert isinstance(chain, DPChains)
        assert chain.theta.shape == (10, data.time.size, 2)
        assert np.all(np.isfinite(chain.theta))
        assert np.all(chain.theta > 0.0)
        assert np.all(chain.theta[:, :, 0] <= chain.phi[:, None])
        assert np.all(np.isfinite(chain.nu)) and np.all(chain.nu > 0.0)
        assert np.all(np.isfinite(chain.gamma)) and np.all(chain.gamma > 0.0)
        assert np.all(np.isfinite(chain.phi)) and np.all(chain.phi >= PRIOR.b_phi)
        assert np.all(chain.n_star >= 1) and np.all(chain.n_star <= data.time.size)
        assert np.all(np.isfinite(chain.log_post))


class TestAllCensoredShapeBounds:
    """The ``n_j^o = 0`` branch of the shape update, exercised directly."""

    @staticmethod
    def _single_cluster(
        times: np.ndarray, event: np.ndarray, *, alpha: float = 0.9, lam: float = 1.5
    ) -> tuple[SurvData, DPState]:
        """One cluster holding every observation, with fixed locations."""
        data = SurvData(time=times, event=event, t_max=1.0)
        state = DPState(
            alpha=np.full(times.size, alpha),
            lam=np.full(times.size, lam),
            s=np.zeros(times.size, dtype=np.intp),
            alpha_star=np.array([alpha]),
            lam_star=np.array([lam]),
            n_j=np.array([times.size], dtype=np.intp),
            nu=2.0,
            gamma=0.5,
            phi=2.0,
        )
        return data, state

    @staticmethod
    def _next_uniform(seed: int, n_consumed: int) -> float:
        """The variate a fresh stream would return after ``n_consumed`` draws."""
        return float(np.random.default_rng(seed).random(n_consumed + 1)[-1])

    def test_all_censored_cluster_has_no_lower_bound_and_draws_no_v(self):
        """``C(alpha) = 1`` and the empty sum is never evaluated.

        With ``n_j^o = 0`` the lower bound is exactly ``0`` by definition, and
        that is what the formula returns rather than a division by the empty
        ``sum_{i_o} log t_{i_o}``.  The branch also consumes *no* uniform for
        ``v``: a reader can tell from the random stream that the empty product
        was recognised, not merely that the division happened to survive.
        """
        times = np.linspace(1.1, 1.9, 4)
        event = np.zeros(4, dtype=bool)
        data, state = self._single_cluster(times, event)

        rng = np.random.default_rng(SEED)
        alpha_minus, alpha_plus = _alpha_star_bounds(state, data, PRIOR, j=0, rng=rng)

        assert alpha_minus == 0.0
        assert np.isfinite(alpha_plus)
        assert 0.0 < alpha_plus <= state.phi
        # All four ``w`` auxiliaries were drawn; the next variate is the fifth
        # of the stream, i.e. no ``v`` was consumed before them.
        assert rng.random() == self._next_uniform(SEED, 4)

    def test_an_uncensored_cluster_draws_v_and_can_have_a_lower_bound(self):
        """The contrast: ``n_j^o > 0`` consumes ``v`` and bounds ``alpha^-``."""
        times = np.linspace(1.1, 1.9, 4)
        event = np.array([True, True, False, False])
        data, state = self._single_cluster(times, event, alpha=1.5)

        rng = np.random.default_rng(SEED)
        alpha_minus, alpha_plus = _alpha_star_bounds(state, data, PRIOR, j=0, rng=rng)

        assert 0.0 <= alpha_minus <= 1.5
        assert np.isfinite(alpha_plus)
        assert 0.0 < alpha_plus <= state.phi
        # One ``v`` plus four ``w``'s: the next variate is the sixth.
        assert rng.random() == self._next_uniform(SEED, 5)

    def test_all_censored_cluster_shape_refresh_stays_in_the_support(self):
        """The uniform (``k = 1``) draw on ``(0, alpha^+)`` keeps the invariants."""
        times = np.linspace(1.1, 1.9, 4)
        event = np.zeros(4, dtype=bool)
        data, state = self._single_cluster(times, event)

        reset_alpha_bound_guard_count()
        rng = np.random.default_rng(SEED)
        for _ in range(N_SWEEPS):
            refresh_cluster_alpha(state, data, PRIOR, j=0, rng=rng)
            _assert_valid_state(state, data)
        assert alpha_bound_guard_count() == 0
