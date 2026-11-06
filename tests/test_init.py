"""Plan Section 3.6: the documented initial state of one chain.

The paper does not specify initial values, so ``init_state`` fixes them here:

* each observation is its own singleton cluster, ``s_i = i`` and ``n_j = 1``,
  so ``n_star = n``;
* ``(alpha_i, lambda_i)`` comes from the continuous piece of the base measure
  at the prior centre, ``gamma_0 = a_gamma / b_gamma`` and ``phi_0 = b_phi``,
  using the observation's own censoring branch;
* ``nu_0 = a_nu / b_nu``, ``gamma_0 = a_gamma / b_gamma`` and
  ``phi_0 = b_phi``.

The tests below pin the parts that are easy to get wrong silently:

* the state is valid *immediately* (Section 2.1 invariants), including the
  ``max_j alpha*_j <= phi`` bound;
* the locations are exactly ``2 * n`` draws of the production continuous-piece
  sampler, replayed with the same seed and in observation order -- so
  initialisation cannot drift into a second, subtly different base draw, and
  the random-number count stays fixed for the golden regression;
* the ``d`` of the prior reaches the draw (a hard-coded ``d = 2`` would pass a
  default-prior test);
* a censored observation uses the survival branch;
* a grid built for another dataset is rejected rather than fed wrong rows.
"""

import numpy as np
import pytest

from src.dp_gibbs import (
    AlphaGrid,
    DPPrior,
    SurvData,
    draw_g0_conditional_alpha,
    init_state,
    make_grid,
    make_surv_data,
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

SEED = 20240801


def _data() -> SurvData:
    """A small mixed censoring dataset on the transformed scale."""
    return make_surv_data(
        time=np.array([0.4, 0.8, 1.0, 1.6, 2.4, 3.0]),
        event=np.array([True, False, True, True, False, True]),
    )


def _grid(data: SurvData, n_alpha: int = 64) -> AlphaGrid:
    return make_grid(data, n_alpha)


def _manual_draws(
    data: SurvData,
    prior: DPPrior,
    *,
    seed: int = SEED,
    n_alpha: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the continuous-piece draw in observation order."""
    grid = _grid(data, n_alpha)
    rng = np.random.default_rng(seed)
    gamma = prior.a_gamma / prior.b_gamma
    alpha = np.empty(data.time.size, dtype=np.float64)
    lam = np.empty(data.time.size, dtype=np.float64)
    for i in range(data.time.size):
        alpha[i], lam[i] = draw_g0_conditional_alpha(
            data.time[i],
            data.event[i],
            gamma=gamma,
            phi=prior.b_phi,
            grid=grid,
            rng=rng,
            d=prior.d,
        )
    return alpha, lam


class TestInitialStructure:
    """Every observation is a singleton and the invariants hold at once."""

    def test_every_observation_is_its_own_cluster(self):
        """``s = arange(n)``, ``n_j = 1`` and ``n_star = n``."""
        data = _data()
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        assert state.n == data.time.size
        assert state.n_star == data.time.size
        assert np.array_equal(state.s, np.arange(data.time.size))
        assert np.array_equal(state.n_j, np.ones(data.time.size, dtype=np.intp))
        assert np.array_equal(state.alpha, state.alpha_star)
        assert np.array_equal(state.lam, state.lam_star)
        # The full Section 2.1 check, including the ``phi >= b_phi`` bound.
        state.check_invariants(PRIOR)

    def test_locations_lie_in_the_base_measure_support(self):
        """``0 < alpha <= phi`` and ``lambda > 0`` for every start value."""
        data = _data()
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        assert np.all(state.alpha > 0.0)
        assert np.all(state.alpha <= state.phi)
        assert np.all(state.lam > 0.0)
        assert np.all(np.isfinite(state.alpha))
        assert np.all(np.isfinite(state.lam))
        assert state.phi >= PRIOR.b_phi

    def test_hyperparameters_start_at_the_prior_centre(self):
        """``nu``, ``gamma`` and ``phi`` are the documented values."""
        data = _data()
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        assert state.nu == pytest.approx(PRIOR.a_nu / PRIOR.b_nu, rel=1e-15)
        assert state.gamma == pytest.approx(PRIOR.a_gamma / PRIOR.b_gamma, rel=1e-15)
        assert state.phi == pytest.approx(PRIOR.b_phi, rel=1e-15)

    def test_singletons_are_not_duplicates_of_one_draw(self):
        """Identical times still give independent ``(alpha, lambda)`` draws."""
        data = make_surv_data(
            time=np.full(4, 2.0),
            event=np.ones(4, dtype=bool),
        )
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        assert state.n_star == 4
        assert np.unique(state.alpha).size > 1
        assert np.unique(state.lam).size > 1


class TestContinuousPieceReuse:
    """The state locations are the production continuous-piece draws."""

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_locations_are_the_continuous_piece_replayed(self, d):
        """Same seed and order as ``draw_g0_conditional_alpha``, exactly.

        This pins the reuse of the production draw (and its ``d``) and the
        fixed consumption of two variates per observation.
        """
        prior = DPPrior(
            a_nu=PRIOR.a_nu,
            b_nu=PRIOR.b_nu,
            a_gamma=PRIOR.a_gamma,
            b_gamma=PRIOR.b_gamma,
            a_phi=PRIOR.a_phi,
            b_phi=PRIOR.b_phi,
            d=d,
        )
        data = _data()
        state = init_state(data, prior, grid=_grid(data), seed=SEED)
        expected_alpha, expected_lam = _manual_draws(data, prior)

        assert np.array_equal(state.alpha, expected_alpha)
        assert np.array_equal(state.lam, expected_lam)
        assert np.isin(state.alpha, prior.b_phi * _grid(data).u).all()
        assert np.all(state.alpha <= prior.b_phi)

    def test_censored_observations_use_the_survival_branch(self):
        """An all-censored dataset draws the Eq. A.3 row everywhere."""
        data = make_surv_data(
            time=np.array([0.5, 1.0, 2.0]),
            event=np.zeros(3, dtype=bool),
        )
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)
        expected_alpha, expected_lam = _manual_draws(data, PRIOR)

        assert np.array_equal(state.alpha, expected_alpha)
        assert np.array_equal(state.lam, expected_lam)
        state.check_invariants(PRIOR)

    def test_single_observation_is_supported(self):
        """``n = 1`` is a legal initial state with one singleton cluster."""
        data = make_surv_data(time=np.array([1.0]), event=np.array([True]))
        state = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        assert state.n == 1
        assert state.n_star == 1
        assert np.array_equal(state.s, np.array([0]))
        assert np.array_equal(state.n_j, np.array([1]))
        state.check_invariants(PRIOR)

    def test_seedless_call_still_builds_a_valid_state(self):
        """``seed=None`` uses OS entropy but must not break any invariant."""
        data = _data()
        state = init_state(data, PRIOR, grid=_grid(data))

        state.check_invariants(PRIOR)


class TestRandomnessDiscipline:
    """One explicit stream: no global RNG state, reproducible by seed."""

    def test_same_seed_reproduces_the_state(self):
        data = _data()
        first = init_state(data, PRIOR, grid=_grid(data), seed=SEED)
        repeat = init_state(data, PRIOR, grid=_grid(data), seed=SEED)

        for name in ("alpha", "lam", "s", "alpha_star", "lam_star", "n_j"):
            assert np.array_equal(getattr(first, name), getattr(repeat, name))
        assert first.nu == repeat.nu
        assert first.gamma == repeat.gamma
        assert first.phi == repeat.phi

    def test_different_seeds_give_different_locations(self):
        data = _data()
        first = init_state(data, PRIOR, grid=_grid(data), seed=SEED)
        other = init_state(data, PRIOR, grid=_grid(data), seed=SEED + 1)

        assert not np.array_equal(first.alpha, other.alpha)
        assert not np.array_equal(first.lam, other.lam)

    def test_explicit_generator_matches_the_seed_path(self):
        """Passing ``default_rng(seed)`` is exactly the ``seed=`` route."""
        data = _data()
        from_seed = init_state(data, PRIOR, grid=_grid(data), seed=SEED)
        from_rng = init_state(
            data, PRIOR, grid=_grid(data), rng=np.random.default_rng(SEED)
        )

        assert np.array_equal(from_seed.alpha, from_rng.alpha)
        assert np.array_equal(from_seed.lam, from_rng.lam)

    def test_seed_and_rng_together_are_rejected(self):
        data = _data()
        with pytest.raises(ValueError, match="either seed or rng"):
            init_state(
                data,
                PRIOR,
                grid=_grid(data),
                seed=SEED,
                rng=np.random.default_rng(SEED),
            )


class TestValidation:
    """Initialisation is an input boundary, like the data ingestion."""

    def test_a_grid_built_for_another_dataset_is_rejected(self):
        data = _data()
        other = make_surv_data(
            time=np.array([0.7, 1.1, 1.9, 2.6, 3.2, 4.0]),
            event=np.ones(6, dtype=bool),
        )
        with pytest.raises(ValueError, match="another dataset"):
            init_state(data, PRIOR, grid=_grid(other), seed=SEED)

    def test_a_grid_with_the_wrong_row_count_is_rejected(self):
        data = _data()
        shorter = make_surv_data(
            time=np.array([0.7, 1.1, 1.9]),
            event=np.ones(3, dtype=bool),
        )
        with pytest.raises(ValueError, match="same observations"):
            init_state(data, PRIOR, grid=_grid(shorter), seed=SEED)

    @pytest.mark.parametrize(
        "field",
        ["a_nu", "b_nu", "a_gamma", "b_gamma", "a_phi", "b_phi"],
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_hyperparameters_are_rejected(self, field, bad):
        data = _data()
        values = {
            "a_nu": PRIOR.a_nu,
            "b_nu": PRIOR.b_nu,
            "a_gamma": PRIOR.a_gamma,
            "b_gamma": PRIOR.b_gamma,
            "a_phi": PRIOR.a_phi,
            "b_phi": PRIOR.b_phi,
        }
        values[field] = bad
        prior = DPPrior(d=PRIOR.d, **values)

        with pytest.raises(ValueError, match=field.replace(".", r"\.")):
            init_state(data, prior, grid=_grid(data), seed=SEED)

    @pytest.mark.parametrize("d", [0, -2, 1.5])
    def test_invalid_d_is_rejected(self, d):
        data = _data()
        prior = DPPrior(
            a_nu=PRIOR.a_nu,
            b_nu=PRIOR.b_nu,
            a_gamma=PRIOR.a_gamma,
            b_gamma=PRIOR.b_gamma,
            a_phi=PRIOR.a_phi,
            b_phi=PRIOR.b_phi,
            d=d,
        )
        with pytest.raises(ValueError, match="positive integer"):
            init_state(data, prior, grid=_grid(data), seed=SEED)
