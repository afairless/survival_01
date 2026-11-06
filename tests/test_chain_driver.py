"""Plan Sections 3.2, 3.6, 3.7 and 8: the chain driver and its storage.

``sweep`` is tested on its own in ``tests/test_sweep.py``; this module tests
only what the driver adds around it:

* the run length is ``burn_in + thin * n_save`` sweeps and a stored draw is
  the state *after* a block of ``thin`` sweeps, with the burn-in discarded --
  a driver that stores every sweep, or stores before thinning, still produces
  plausible-looking arrays and is invisible downstream;
* initialisation and the sweeps share one generator seeded with
  ``settings.seed + b``, so a whole chain replays bit for bit from the stored
  record and chain ``b`` of a group agrees with a single-chain run at
  ``seed + b``;
* ``DPChains`` stores the documented schema -- ``theta`` of shape
  ``(n_save, n, 2)`` plus one entry per draw for ``nu``, ``gamma``, ``phi``,
  ``n_star`` and ``log_post`` -- and rejects a trace that disagrees with its
  dataset, its settings or a Section 2.1 invariant;
* ``sample_chains`` runs every chain of every settings entry, flat and in
  ``(settings index, chain index)`` order, through a process pool, and the
  parallel result is identical to the serial one.

What each test would catch:

* a total-sweep count that is off by one, a burn-in that is run but stored, a
  thinning applied before burn-in, or the initial state stored as draw 0 --
  the explicit replay test and the ``thin=1, burn_in=0`` first-draw test;
* a second generator for initialisation, a global RNG, or a per-chain seed
  that ignores the chain index -- the replay and ``seed + b`` tests;
* a storage layout that transposes ``(alpha, lam)``, drops a trace, or has
  ``B != n_save`` -- the schema test and the golden-facing shape assertions;
* a driver that silently accepts ``thin=0`` (every stored draw overwritten),
  ``n_save=0`` (empty trace) or ``seed=True`` -- the validation matrix;
* a pool that loses ordering, pickles a mutated state, or lets scheduling
  change the draws -- the parallel-versus-serial equality test.
"""

import dataclasses
import time as clock

import numpy as np
import pytest

from src.dp_gibbs import (
    DPChains,
    DPPrior,
    DPSettings,
    SurvData,
    init_state,
    make_grid,
    make_surv_data,
    sample_chain,
    sample_chains,
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

TIME = np.array([0.4, 0.8, 1.0, 1.6, 2.4, 3.0, 4.5, 5.2, 6.1, 7.3, 8.6, 9.9])
EVENT = np.array(
    [True, False, True, True, False, True, True, False, True, True, False, True]
)

SEED = 20241007
N_ALPHA = 16


def _data(n: int = 10) -> SurvData:
    """A small mixed-censoring dataset on the transformed scale."""
    return make_surv_data(time=TIME[:n], event=EVENT[:n])


def _settings(**overrides: object) -> DPSettings:
    """Short, cheap settings for the driver tests (a few hundred sweeps)."""
    base: dict[str, object] = {
        "burn_in": 6,
        "n_save": 5,
        "thin": 3,
        "n_alpha": N_ALPHA,
        "n_chains": 1,
        "seed": SEED,
        "preset": "test",
    }
    base.update(overrides)
    return DPSettings(**base)  # type: ignore[arg-type]


def _replay(
    data: SurvData, settings: DPSettings, chain_index: int = 0
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Replay a chain by hand: shared generator, burn-in, then thin blocks.

    Deliberately written out here rather than calling ``sample_chain``: the
    point is to pin the *definition* of the run length and of the stored draws,
    so a driver that changes either fails instead of agreeing with itself.
    Returns the per-draw traces and the last log-posterior.
    """
    assert settings.seed is not None, "the driver records the resolved seed"
    seed = int(settings.seed) + chain_index
    grid = make_grid(data, int(settings.n_alpha))
    rng = np.random.default_rng(seed)
    state = init_state(data, prior=PRIOR, grid=grid, rng=rng)
    for _ in range(int(settings.burn_in)):
        sweep(state, data, PRIOR, grid=grid, rng=rng)
    theta = []
    traces = {"nu": [], "gamma": [], "phi": [], "n_star": [], "log_post": []}
    for _ in range(int(settings.n_save)):
        score = np.nan
        for _ in range(int(settings.thin)):
            score = sweep(state, data, PRIOR, grid=grid, rng=rng)
        theta.append(np.column_stack([state.alpha, state.lam]))
        traces["nu"].append(state.nu)
        traces["gamma"].append(state.gamma)
        traces["phi"].append(state.phi)
        traces["n_star"].append(state.n_star)
        traces["log_post"].append(score)
    return np.array(theta), {k: np.array(v) for k, v in traces.items()}


class TestStorage:
    """The ``DPChains`` schema and its validation (plan Section 3.2)."""

    def test_stores_the_documented_schema(self):
        """``theta`` is ``(B, n, 2)`` and every trace has one entry per draw."""
        data = _data(8)
        settings = _settings(n_save=4)
        chain = sample_chain(data, PRIOR, settings)

        assert isinstance(chain, DPChains)
        assert chain.theta.shape == (4, data.time.size, 2)
        assert chain.theta.dtype == np.float64
        assert chain.nu.shape == chain.gamma.shape == chain.phi.shape == (4,)
        assert chain.n_star.shape == (4,)
        assert chain.log_post.shape == (4,)
        assert np.issubdtype(chain.n_star.dtype, np.integer)
        assert chain.n_draws == 4
        assert chain.n == data.time.size

    def test_stores_the_inputs_by_value(self):
        """The dataset, prior and settings travel with the draws."""
        data = _data(8)
        settings = _settings()
        chain = sample_chain(data, PRIOR, settings)

        assert chain.data is data
        assert chain.prior is PRIOR
        assert chain.settings.n_save == settings.n_save
        assert chain.settings.thin == settings.thin
        assert chain.settings.burn_in == settings.burn_in
        assert chain.settings.seed == SEED

    def test_stored_draws_satisfy_the_invariants(self):
        """Every stored draw is a valid Section 2.1 state, not a half-sweep."""
        data = _data(10)
        chain = sample_chain(data, PRIOR, _settings())

        alpha = chain.theta[:, :, 0]
        lam = chain.theta[:, :, 1]
        assert np.all(np.isfinite(chain.theta))
        assert np.all(alpha > 0.0) and np.all(lam > 0.0)
        # Invariant 3 of Section 2.1: alpha <= phi, using this draw's phi.
        assert np.all(alpha <= chain.phi[:, None])
        # Invariant 4: phi is bounded below by the Pareto lower bound.
        assert np.all(chain.phi >= PRIOR.b_phi)
        assert np.all(chain.n_star >= 1) and np.all(chain.n_star <= data.time.size)
        assert np.all(np.isfinite(chain.log_post))

    def test_rejects_theta_of_the_wrong_shape(self):
        """A transposed or 2-D ``theta`` must not reach the functional code."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=2))

        with pytest.raises(ValueError, match=r"theta must have shape"):
            dataclasses.replace(chain, theta=chain.theta[:, :, 0])

    def test_rejects_theta_for_another_dataset(self):
        """``theta`` rows and ``data`` must describe the same observations."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=2))

        with pytest.raises(ValueError, match=r"one row per observation"):
            dataclasses.replace(chain, theta=chain.theta[:, :5, :])

    def test_rejects_a_trace_length_that_disagrees_with_settings(self):
        """``B`` is ``settings.n_save``; a short trace is a driver bug."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=3))

        with pytest.raises(ValueError, match=r"one value per stored draw"):
            dataclasses.replace(chain, nu=chain.nu[:2])

    def test_rejects_non_finite_and_non_positive_traces(self):
        """NaN and a negative lambda are rejected at the storage boundary."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=2))
        nu = np.array(chain.nu, copy=True)
        nu[0] = np.nan
        with pytest.raises(ValueError, match=r"nu must contain only finite"):
            dataclasses.replace(chain, nu=nu)

        theta = np.array(chain.theta, copy=True)
        theta[0, 0, 1] = -1.0
        with pytest.raises(ValueError, match=r"theta must be strictly positive"):
            dataclasses.replace(chain, theta=theta)

    def test_rejects_alpha_above_phi_and_phi_below_the_prior_bound(self):
        """Invariants 3 and 4 are enforced on the stored trace."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=2))
        theta = np.array(chain.theta, copy=True)
        theta[0, 0, 0] = chain.phi[0] + 1.0
        with pytest.raises(ValueError, match=r"alpha must not exceed phi"):
            dataclasses.replace(chain, theta=theta)

        phi = np.array(chain.phi, copy=True)
        phi[1] = PRIOR.b_phi / 2.0
        with pytest.raises(ValueError, match=r"phi must be at least prior.b_phi"):
            dataclasses.replace(chain, phi=phi)

    def test_rejects_an_impossible_cluster_count(self):
        """``n_star`` lies in ``1..n`` and is an integer count."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(n_save=2))
        n_star = np.array(chain.n_star, copy=True)
        n_star[0] = 0
        with pytest.raises(ValueError, match=r"n_star must lie in 1\.\.n"):
            dataclasses.replace(chain, n_star=n_star)

        with pytest.raises(ValueError, match=r"n_star must be an integer array"):
            dataclasses.replace(chain, n_star=chain.n_star.astype(np.float64))


class TestRunLengthAndThinning:
    """``burn_in + thin * n_save`` sweeps, stored after each thin block."""

    def test_replays_the_documented_sweep_schedule(self):
        """Replay by hand equals the driver, trace for trace.

        The replay uses the same definition of the run (burn-in discarded, then
        thin blocks) written out independently, so an off-by-one in the driver's
        loop, a stored burn-in draw, or a ``thin`` applied before burn-in
        diverges on the first draw.
        """
        data = _data(10)
        settings = _settings(burn_in=4, n_save=4, thin=3)
        chain = sample_chain(data, PRIOR, settings)
        theta, traces = _replay(data, chain.settings)

        assert np.array_equal(chain.theta, theta)
        for name in ("nu", "gamma", "phi", "n_star", "log_post"):
            assert np.array_equal(getattr(chain, name), traces[name]), (
                f"trace {name} differs from the manual replay"
            )

    def test_burn_in_is_discarded_and_thin_one_stores_every_sweep(self):
        """With ``thin=1`` draw ``b`` is the state after sweep ``burn_in+1+b``."""
        data = _data(8)
        settings = _settings(burn_in=3, n_save=3, thin=1)
        chain = sample_chain(data, PRIOR, settings)

        grid = make_grid(data, N_ALPHA)
        rng = np.random.default_rng(SEED)
        state = init_state(data, PRIOR, grid=grid, rng=rng)
        for _ in range(3):
            sweep(state, data, PRIOR, grid=grid, rng=rng)
        for b in range(3):
            sweep(state, data, PRIOR, grid=grid, rng=rng)
            assert np.array_equal(state.alpha, chain.theta[b, :, 0])
            assert state.nu == chain.nu[b]

        # The stored draws are not the initial state: burn-in really ran.
        initial = init_state(data, PRIOR, grid=grid, seed=SEED)
        assert not np.array_equal(initial.alpha, chain.theta[0, :, 0])

    def test_zero_burn_in_and_thin_one_starts_after_the_first_sweep(self):
        """``burn_in=0, thin=1`` stores the state one sweep after initialisation."""
        data = _data(8)
        settings = _settings(burn_in=0, n_save=2, thin=1)
        chain = sample_chain(data, PRIOR, settings)
        theta, traces = _replay(data, chain.settings)

        assert np.array_equal(chain.theta, theta)
        assert np.array_equal(chain.log_post, traces["log_post"])
        # The very first stored state is already swept: it is not the
        # init_state draw, which is the failure mode of skipping the sweeps.
        grid = make_grid(data, N_ALPHA)
        unswept = init_state(data, PRIOR, grid=grid, seed=SEED)
        assert not np.array_equal(unswept.alpha, chain.theta[0, :, 0])

    def test_thin_does_not_change_the_stored_draw_positions(self):
        """Thinning changes which sweeps are stored, not how many blocks run."""
        data = _data(8)
        thin_one = sample_chain(data, PRIOR, _settings(burn_in=2, n_save=3, thin=1))
        thin_two = sample_chain(data, PRIOR, _settings(burn_in=2, n_save=3, thin=2))

        assert thin_one.n_draws == thin_two.n_draws == 3
        # Draw 0 of the thin-1 run is the first sweep; of the thin-2 run the
        # second.  They must therefore differ (the sampler moves every sweep).
        assert not np.array_equal(thin_one.theta[0], thin_two.theta[0])


class TestSeedsAndDeterminism:
    """One generator per chain, seeded ``base + chain_index`` (Section 3.6)."""

    def test_same_seed_reproduces_the_chain(self):
        """Two runs of the same settings are bit-for-bit identical."""
        data = _data(8)
        settings = _settings()
        a = sample_chain(data, PRIOR, settings)
        b = sample_chain(data, PRIOR, settings)

        assert np.array_equal(a.theta, b.theta)
        assert np.array_equal(a.nu, b.nu)
        assert np.array_equal(a.log_post, b.log_post)

    def test_a_different_seed_gives_a_different_chain(self):
        """The chain depends on the seed, so the driver is not ignoring it."""
        data = _data(8)
        a = sample_chain(data, PRIOR, _settings(seed=SEED))
        b = sample_chain(data, PRIOR, _settings(seed=SEED + 1000))

        assert not np.array_equal(a.theta, b.theta)

    def test_unseeded_run_records_the_resolved_seed(self):
        """``seed=None`` is resolved once and stored, so the run is replayable."""
        data = _data(6)
        chain = sample_chain(data, PRIOR, _settings(seed=None, n_save=2))

        assert chain.settings.seed is not None
        replay = sample_chain(data, PRIOR, chain.settings)
        assert np.array_equal(chain.theta, replay.theta)

    def test_chain_index_uses_seed_plus_index(self):
        """Chain ``b`` of a group is a single-chain run at ``seed + b``."""
        data = _data(8)
        group = _settings(n_chains=2, seed=SEED)
        chains = sample_chains(data, PRIOR, [group], max_workers=1)

        assert len(chains) == 2
        for b, chain in enumerate(chains):
            alone = sample_chain(
                data,
                PRIOR,
                _settings(n_chains=1, seed=SEED + b),
            )
            assert np.array_equal(chain.theta, alone.theta), (
                f"chain {b} is not the run seeded with seed + {b}"
            )
        assert chains[0].settings.seed == chains[1].settings.seed == SEED
        assert not np.array_equal(chains[0].theta, chains[1].theta)

    def test_does_not_touch_the_global_numpy_rng(self):
        """The driver consumes only its own generator."""
        data = _data(6)
        np.random.seed(12345)
        expected = np.random.random(3)
        np.random.seed(12345)
        sample_chain(data, PRIOR, _settings(n_save=2))

        assert np.array_equal(np.random.random(3), expected)


class TestInputValidation:
    """Invalid knobs fail before the first sweep, with a named message."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"n_save": 0}, r"settings\.n_save must be at least 1"),
            ({"n_save": 2.5}, r"settings\.n_save must be an integer"),
            ({"n_save": True}, r"settings\.n_save must be an integer"),
            ({"thin": 0}, r"settings\.thin must be at least 1"),
            ({"burn_in": -1}, r"settings\.burn_in must be at least 0"),
            ({"n_alpha": 1}, r"settings\.n_alpha must be at least 2"),
            ({"n_chains": 0}, r"settings\.n_chains must be at least 1"),
            ({"seed": -5}, r"settings\.seed must be at least 0"),
            ({"epsilon": 0.0}, r"settings\.epsilon must lie strictly between"),
            ({"epsilon": 1.5}, r"settings\.epsilon must lie strictly between"),
            ({"preset": ""}, r"settings\.preset must be a non-empty string"),
        ],
    )
    def test_rejects_an_invalid_setting(self, overrides, message):
        """Every count knob is validated, not silently accepted."""
        data = _data(6)
        with pytest.raises(ValueError, match=message):
            sample_chain(data, PRIOR, _settings(**overrides))

    def test_rejects_the_wrong_container_types(self):
        """A raw array is not a ``SurvData``, and a bare settings is not a list."""
        data = _data(6)
        settings = _settings()

        with pytest.raises(ValueError, match=r"data must be a SurvData"):
            sample_chain(data.time, PRIOR, settings)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"prior must be a DPPrior"):
            sample_chain(data, object(), settings)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"settings must be a DPSettings"):
            sample_chain(data, PRIOR, object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"settings_list must be a sequence"):
            sample_chains(data, PRIOR, settings)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"at least one DPSettings"):
            sample_chains(data, PRIOR, [], max_workers=1)
        with pytest.raises(ValueError, match=r"max_workers must be at least 1"):
            sample_chains(data, PRIOR, [settings], max_workers=0)

    def test_validation_happens_before_any_sweep(self):
        """An invalid second entry fails the call without running the first."""
        data = _data(6)
        good = _settings(seed=SEED, n_save=1)
        bad = _settings(seed=SEED, n_save=0)

        with pytest.raises(ValueError, match=r"settings\.n_save"):
            sample_chains(data, PRIOR, [good, bad], max_workers=1)


class TestParallelChains:
    """``sample_chains`` across a process pool (plan Section 8, gate 2)."""

    def test_flat_order_is_settings_then_chain(self):
        """Two groups of two chains give four chains in ``(group, chain)`` order."""
        data = _data(8)
        first = _settings(n_chains=2, seed=SEED, n_save=3)
        second = _settings(n_chains=2, seed=SEED + 100, n_save=3)
        chains = sample_chains(data, PRIOR, [first, second], max_workers=2)

        assert len(chains) == 4
        expected = [
            sample_chain(data, PRIOR, _settings(n_chains=1, seed=SEED + b, n_save=3))
            for b in range(2)
        ] + [
            sample_chain(
                data, PRIOR, _settings(n_chains=1, seed=SEED + 100 + b, n_save=3)
            )
            for b in range(2)
        ]
        for chain, reference in zip(chains, expected, strict=True):
            assert np.array_equal(chain.theta, reference.theta)
            assert np.array_equal(chain.log_post, reference.log_post)

    def test_parallel_equals_serial(self):
        """The worker count changes speed only, never the draws."""
        data = _data(8)
        settings = _settings(n_chains=3, seed=SEED, n_save=3)
        serial = sample_chains(data, PRIOR, [settings], max_workers=1)
        parallel = sample_chains(data, PRIOR, [settings], max_workers=3)

        assert len(serial) == len(parallel) == 3
        for a, b in zip(serial, parallel, strict=True):
            assert np.array_equal(a.theta, b.theta)
            assert np.array_equal(a.nu, b.nu)
            assert np.array_equal(a.log_post, b.log_post)
            assert a.settings.seed == b.settings.seed == SEED

    def test_unseeded_groups_get_distinct_recorded_seeds(self):
        """Each unseeded group draws its own base seed in the parent."""
        data = _data(6)
        chains = sample_chains(
            data,
            PRIOR,
            [
                _settings(seed=None, n_chains=1, n_save=1),
                _settings(seed=None, n_chains=1, n_save=1),
            ],
            max_workers=1,
        )

        assert chains[0].settings.seed != chains[1].settings.seed

    def test_single_chain_run_stays_in_process(self):
        """One chain is run without a pool and equals the pooled result."""
        data = _data(6)
        settings = _settings(n_chains=1, seed=SEED, n_save=2)
        chains = sample_chains(data, PRIOR, [settings])
        direct = sample_chain(data, PRIOR, settings)

        assert len(chains) == 1
        assert np.array_equal(chains[0].theta, direct.theta)


class TestIntegrationRun:
    """A longer run on a realistic-sized dataset, checked end to end."""

    def test_quick_sized_run_produces_a_usable_chain(self):
        """``n=20``, a few hundred sweeps: valid draws, finite scores, movement.

        This is the smoke run every later step (diagnostics, functionals,
        golden fixture) builds on: it checks that a chain of realistic length
        stays inside the invariants, that ``log_post`` is finite throughout,
        and that the sampler moves rather than freezing on its initial state.
        """
        rng = np.random.default_rng(7)
        time = np.sort(rng.uniform(0.2, 9.0, size=20))
        event = rng.random(20) > 0.25
        data = make_surv_data(time=time, event=event)
        settings = _settings(burn_in=30, n_save=25, thin=4, seed=SEED, n_alpha=32)

        start = clock.perf_counter()
        chain = sample_chain(data, PRIOR, settings)
        elapsed = clock.perf_counter() - start

        assert chain.n_draws == 25
        assert np.all(np.isfinite(chain.theta))
        assert np.all(np.isfinite(chain.log_post))
        assert np.all(chain.phi >= PRIOR.b_phi)
        assert np.all(chain.theta[:, :, 0] <= chain.phi[:, None])
        assert chain.n_star.min() >= 1
        # The chain moved away from the initial all-singletons state.
        assert np.any(chain.n_star < chain.n)
        # 130 sweeps on n=20 at M=32 must stay inside the Section 8 budget.
        assert elapsed < 10.0
