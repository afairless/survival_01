"""Plan Section 3.4: ``sweep`` -- one whole Gibbs sweep in (a)-(b)-(c) order.

The three conditionals are tested one at a time elsewhere (plan Section 7.2,
``tests/test_step_a.py``, ``test_step_b_*.py`` and ``test_step_c_*.py``).  This
module tests only the *orchestration*, which those tests cannot see:

* the three steps run exactly once each, in the fixed order (a) -> (b) -> (c),
  on the same state object and through the same random generator;
* the ``(a)-(b)-(c)`` sequence is the algorithm's specification, so the sweep
  result equals an explicit replay of the three calls on the same seed, bit for
  bit -- a swapped (b)/(c) pair would produce a different draw, because step
  (c)'s ``phi`` update reads ``n_star`` and ``max_j alpha*_j`` from step (b),
  and the ``alpha*`` update in step (b) reads the *new* ``lambda*``;
* the value returned is ``log_post`` of the **resulting** state, so the chain
  driver can store it with the draw without evaluating it a second time;
* the sweep consumes one shared stream, so replaying a seed reproduces the
  sweep (no hidden global RNG and no per-step generator);
* the step boundary is where an inconsistent ``(state, data, grid, prior)``
  combination fails -- before any randomness is consumed -- and a sweep is not
  a special case for the initial state of plan Section 3.6.

The smoke tests run the actual sampler for many sweeps on a small dataset and
check that the Section 2.1 invariants, finiteness and the ``n_star`` range hold
throughout: a silent NaN or a count that drifts from the labels would poison
everything downstream, and neither shows up in a single-conditional test.

What each test would catch:

* a missing call (step (b) or (c) forgotten, or a sweep that only runs step
  (a)) -- the replay test fails on the very first draw;
* a swapped order, which is invisible in the conditionals themselves -- the
  replay test diverges and the recorded call order fails;
* a fresh ``default_rng`` inside a step, or a step reading ``state.nu`` from a
  different ``nu`` update -- the shared-stream test fails;
* scoring the pre-sweep state, or returning ``None`` -- the return-value test;
* a state updated in place by a copy -- the identity test;
* an error raised after some steps have already drawn -- the RNG-fingerprint
  assertions in the failure tests.
"""

import numpy as np
import pytest

import src.dp_gibbs as dg
from src.dp_gibbs import (
    AlphaGrid,
    DPPrior,
    DPState,
    StateInvariantError,
    SurvData,
    init_state,
    log_post,
    make_grid,
    make_surv_data,
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

INIT_SEED = 20240918
RNG_SEED = 1729
N_ALPHA = 32

TIME = np.array([0.4, 0.8, 1.0, 1.6, 2.4, 3.0, 4.5, 5.2, 6.1, 7.3, 8.6, 9.9])
EVENT = np.array(
    [True, False, True, True, False, True, True, False, True, True, False, True]
)


def _data(n: int = 8) -> SurvData:
    """A small mixed-censoring dataset on the transformed scale."""
    return make_surv_data(time=TIME[:n], event=EVENT[:n])


def _setup(n: int = 8, n_alpha: int = N_ALPHA) -> tuple[SurvData, AlphaGrid, DPState]:
    """Build data, grid and an initial state for the sweep tests."""
    data = _data(n)
    grid = make_grid(data, n_alpha)
    state = init_state(data, PRIOR, grid=grid, seed=INIT_SEED)
    return data, grid, state


def _clone(state: DPState) -> DPState:
    """Copy a state without sharing any array (``replace`` would not)."""
    return DPState(
        alpha=state.alpha.copy(),
        lam=state.lam.copy(),
        s=state.s.copy(),
        alpha_star=state.alpha_star.copy(),
        lam_star=state.lam_star.copy(),
        n_j=state.n_j.copy(),
        nu=state.nu,
        gamma=state.gamma,
        phi=state.phi,
    )


def _assert_states_equal(left: DPState, right: DPState) -> None:
    """Assert two states carry exactly the same numbers, field by field."""
    assert np.array_equal(left.alpha, right.alpha)
    assert np.array_equal(left.lam, right.lam)
    assert np.array_equal(left.s, right.s)
    assert np.array_equal(left.alpha_star, right.alpha_star)
    assert np.array_equal(left.lam_star, right.lam_star)
    assert np.array_equal(left.n_j, right.n_j)
    assert left.nu == right.nu
    assert left.gamma == right.gamma
    assert left.phi == right.phi


def _rng_fingerprint(rng: np.random.Generator) -> tuple[str, int, int]:
    """Return the raw PCG64 state, so stream alignment is checked exactly."""
    state = rng.bit_generator.state
    return (state["bit_generator"], state["state"]["state"], state["state"]["inc"])


def test_sweep_runs_a_then_b_then_c_and_scores_the_result(monkeypatch) -> None:
    """The order (a) -> (b) -> (c) -> log_post is the specification."""
    data, grid, state = _setup()
    calls: list[dict[str, object]] = []
    original_log_post = dg.log_post

    def fake_step_a(state_arg, data_arg, prior_arg, *, grid, rng):
        calls.append(
            {
                "step": "a",
                "state": state_arg,
                "data": data_arg,
                "prior": prior_arg,
                "grid": grid,
                "rng": rng,
            }
        )

    def fake_step_b(state_arg, data_arg, prior_arg, *, rng):
        calls.append(
            {
                "step": "b",
                "state": state_arg,
                "data": data_arg,
                "prior": prior_arg,
                "rng": rng,
            }
        )

    def fake_step_c(state_arg, data_arg, prior_arg, *, rng):
        calls.append(
            {
                "step": "c",
                "state": state_arg,
                "data": data_arg,
                "prior": prior_arg,
                "rng": rng,
            }
        )

    def fake_log_post(state_arg, data_arg, prior_arg):
        calls.append({"step": "log_post"})
        return original_log_post(state_arg, data_arg, prior_arg)

    monkeypatch.setattr(dg, "sample_step_a", fake_step_a)
    monkeypatch.setattr(dg, "sample_step_b", fake_step_b)
    monkeypatch.setattr(dg, "sample_step_c", fake_step_c)
    monkeypatch.setattr(dg, "log_post", fake_log_post)

    rng = np.random.default_rng(RNG_SEED)
    value = sweep(state, data, PRIOR, grid=grid, rng=rng)

    assert [call["step"] for call in calls] == ["a", "b", "c", "log_post"]
    for call in calls[:3]:
        assert call["state"] is state
        assert call["data"] is data
        assert call["prior"] is PRIOR
        assert call["rng"] is rng
    # Only step (a) consumes the alpha grid; steps (b) and (c) read the state.
    assert calls[0]["grid"] is grid
    assert value == original_log_post(state, data, PRIOR)


def test_sweep_threads_one_rng_stream_through_all_three_steps(monkeypatch) -> None:
    """One supplied generator is shared; no step seeds its own stream."""
    data, grid, state = _setup()
    draws: list[float] = []

    def fake_step_a(state_arg, data_arg, prior_arg, *, grid, rng):
        draws.append(float(rng.random()))

    def fake_step_b(state_arg, data_arg, prior_arg, *, rng):
        draws.append(float(rng.random()))

    def fake_step_c(state_arg, data_arg, prior_arg, *, rng):
        draws.append(float(rng.random()))

    monkeypatch.setattr(dg, "sample_step_a", fake_step_a)
    monkeypatch.setattr(dg, "sample_step_b", fake_step_b)
    monkeypatch.setattr(dg, "sample_step_c", fake_step_c)

    rng = np.random.default_rng(RNG_SEED)
    sweep(state, data, PRIOR, grid=grid, rng=rng)

    reference = np.random.default_rng(RNG_SEED)
    expected = [float(reference.random()) for _ in range(3)]
    assert draws == expected
    assert _rng_fingerprint(rng) == _rng_fingerprint(reference)


def test_sweep_matches_an_explicit_a_b_c_replay() -> None:
    """The joint draw equals a literal replay of the three documented calls."""
    data, grid, state = _setup()
    replay = _clone(state)

    sweep_rng = np.random.default_rng(RNG_SEED)
    value = sweep(state, data, PRIOR, grid=grid, rng=sweep_rng)

    replay_rng = np.random.default_rng(RNG_SEED)
    dg.sample_step_a(replay, data, PRIOR, grid=grid, rng=replay_rng)
    dg.sample_step_b(replay, data, PRIOR, rng=replay_rng)
    dg.sample_step_c(replay, data, PRIOR, rng=replay_rng)
    expected_value = log_post(replay, data, PRIOR)

    _assert_states_equal(state, replay)
    assert value == expected_value
    assert _rng_fingerprint(sweep_rng) == _rng_fingerprint(replay_rng)


def test_sweep_mutates_the_state_in_place_and_returns_its_score() -> None:
    """The state object is reused, all three steps leave a trace, and the
    returned score is the one of the post-sweep state."""
    data, grid, state = _setup()
    state_id = id(state)
    before = _clone(state)
    before_score = log_post(state, data, PRIOR)

    value = sweep(state, data, PRIOR, grid=grid, rng=np.random.default_rng(RNG_SEED))

    assert id(state) == state_id
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert value == log_post(state, data, PRIOR)
    assert value != before_score
    # Step (a) may leave the partition untouched for a fixed seed, so the
    # location-level and hyperparameter checks below are the ones that prove
    # (b) and (c) ran.  Both are continuous draws: exact equality has
    # probability zero under a correct implementation.
    assert not np.array_equal(state.lam_star, before.lam_star)
    assert state.nu != before.nu
    assert state.phi != before.phi
    assert state.gamma != before.gamma
    state.check_invariants(PRIOR)


def test_sweep_is_deterministic_for_a_fixed_seed() -> None:
    """Two independent runs with the same seeds replay the same sweep exactly."""
    data, grid, first = _setup()
    second = _clone(first)

    first_value = sweep(
        first, data, PRIOR, grid=grid, rng=np.random.default_rng(RNG_SEED)
    )
    second_value = sweep(
        second, data, PRIOR, grid=grid, rng=np.random.default_rng(RNG_SEED)
    )

    _assert_states_equal(first, second)
    assert first_value == second_value


def test_sweep_rejects_a_grid_built_for_another_dataset() -> None:
    """A foreign grid must fail at step (a) before any draw is consumed."""
    data, _grid, state = _setup()
    other = make_surv_data(
        time=np.array([0.7, 1.1, 1.5]),
        event=np.array([True, False, True]),
    )
    other_grid = make_grid(other, N_ALPHA)
    rng = np.random.default_rng(RNG_SEED)
    before = _rng_fingerprint(rng)

    with pytest.raises(ValueError, match="grid"):
        sweep(state, data, PRIOR, grid=other_grid, rng=rng)

    assert _rng_fingerprint(rng) == before


def test_sweep_rejects_data_with_a_different_observation_count() -> None:
    """A state fitted to another dataset is rejected at the step boundary."""
    data, grid, state = _setup()
    longer = _data(12)
    rng = np.random.default_rng(RNG_SEED)
    before = _rng_fingerprint(rng)

    with pytest.raises(ValueError, match="same number of observations"):
        sweep(state, longer, PRIOR, grid=grid, rng=rng)

    assert _rng_fingerprint(rng) == before


def test_sweep_rejects_a_state_with_phi_below_the_prior_bound() -> None:
    """Invariant 4 is checked by step (a), before this sweep draws anything."""
    data, grid, state = _setup()
    state.phi = 0.5 * PRIOR.b_phi
    rng = np.random.default_rng(RNG_SEED)
    before = _rng_fingerprint(rng)

    with pytest.raises(StateInvariantError, match="invariant 4"):
        sweep(state, data, PRIOR, grid=grid, rng=rng)

    assert _rng_fingerprint(rng) == before


def test_sweep_smoke_keeps_the_state_valid_over_100_sweeps() -> None:
    """100 real sweeps leave a finite, invariant-clean, mixing chain."""
    data, grid, state = _setup(n=12, n_alpha=32)
    rng = np.random.default_rng(RNG_SEED)
    scores: list[float] = []
    cluster_counts: list[int] = []

    for _ in range(100):
        score = sweep(state, data, PRIOR, grid=grid, rng=rng)
        assert np.isfinite(score)
        scores.append(score)
        state.check_invariants(PRIOR)
        assert 1 <= state.n_star <= state.n
        cluster_counts.append(state.n_star)
        # Every parameter must stay finite and on its support; a single NaN
        # would silently propagate through every later conditional.
        for value in (state.nu, state.gamma, state.phi):
            assert np.isfinite(value) and value > 0.0
        assert np.all(np.isfinite(state.alpha)) and np.all(state.alpha > 0.0)
        assert np.all(np.isfinite(state.lam)) and np.all(state.lam > 0.0)
        assert float(np.max(state.alpha_star)) <= state.phi
        assert state.phi >= PRIOR.b_phi
        assert int(np.sum(state.n_j)) == state.n

    assert len(set(scores)) > 1
    # A chain stuck at n_star == n would pass every invariant check; the
    # smoke run must actually merge the initial singletons.
    assert min(cluster_counts) < state.n


def test_sweep_smoke_merges_the_initial_singletons() -> None:
    """The documented singleton initialisation is not a fixed point of a sweep."""
    data, grid, state = _setup(n=8)
    assert state.n_star == state.n
    rng = np.random.default_rng(RNG_SEED)

    for _ in range(5):
        sweep(state, data, PRIOR, grid=grid, rng=rng)

    state.check_invariants(PRIOR)
    assert state.n_star < state.n
    assert int(np.sum(state.n_j)) == state.n
