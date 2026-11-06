"""Test 5 (plan Section 7.1): ``SurvData`` ingestion and time transformation.

The sampler only works on the Appendix's transformed scale ``t_tilde =
1 + t / max(t)``, where ``log t_tilde > 0`` strictly, and every functional
needs the inverse map with the correct Jacobian.  Both directions are pinned
here, together with the ingestion validation matrix:

* round trip ``t -> t_tilde -> t`` over random data;
* ``t_tilde in (1, 2]`` and ``log t_tilde > 0``;
* ``t_max`` is the maximum of the **original** times, not of the transformed
  ones;
* one test per rejection rule, asserting the exception type and message, with
  zero/one/many observations where the rule allows it;
* ``np.load`` of an ``.npz`` fixture with ``allow_pickle=False``.

What wrong code would break: applying the transform twice (or not at all),
computing ``t_max`` after transforming, omitting the ``np.isfinite`` check so
that a ``NaN`` passes a bare ``time <= 0`` test and poisons ``t_max``, and
dropping the ``1 / t_max`` Jacobian on densities and hazards -- an error that
is invisible in a plot of the survival function.
"""

import dataclasses

import numpy as np
import pytest

from src.dp_gibbs import (
    SurvData,
    inverse_map,
    inverse_time,
    make_surv_data,
    transform_time,
)

ALL_MISSING_EVENT = "event must not be empty"
ALL_MISSING_TIME = "time must not be empty"
BAD_EVENT = "event must be boolean or contain only 0 and 1"
BAD_LENGTH = "time and event must have the same length"
BAD_TMAX = "t_max must be a finite, strictly positive number"
BAD_TIME_FINITE = "time must contain only finite values (no NaN or inf)"
BAD_TIME_POSITIVE = "time must be strictly positive"
BAD_TIME_TRANSFORMED = "time must be greater than 1 on the transformed scale"
NOT_1D = "must be a 1-D array"


class TestTimeTransform:
    """The forward transform, its inverse, and the invariants they induce."""

    @pytest.mark.parametrize("seed", range(10))
    def test_t_max_is_the_original_maximum(self, seed):
        """``t_max`` must come from the original times, before the transform."""
        rng = np.random.default_rng(seed)
        time = rng.uniform(0.01, 100.0, size=rng.integers(1, 50))

        data = make_surv_data(time, np.ones(time.size, dtype=bool))

        assert data.t_max == pytest.approx(float(np.max(time)), rel=0.0, abs=0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_transformed_times_are_in_unit_interval_above_one(self, seed):
        """``t_tilde`` lies in ``(1, 2]`` and has a strictly positive log."""
        rng = np.random.default_rng(100 + seed)
        time = rng.uniform(1e-3, 1e3, size=rng.integers(1, 100))

        data = make_surv_data(time, np.zeros(time.size, dtype=bool))

        assert np.all(data.time > 1.0)
        assert np.all(data.time <= 2.0)
        assert np.all(np.log(data.time) > 0.0)
        assert np.isclose(np.max(data.time), 2.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_round_trip_time(self, seed):
        """``inverse_time(transform(t))`` recovers ``t`` to machine precision."""
        rng = np.random.default_rng(200 + seed)
        time = rng.uniform(1e-3, 1e3, size=rng.integers(1, 100))
        # Include the maximum explicitly so the end point ``t_tilde = 2`` is
        # always covered.
        time = np.append(time, float(np.max(time)))

        data = make_surv_data(time, np.ones(time.size, dtype=bool))
        recovered = inverse_time(data, transform_time(data, time))

        np.testing.assert_allclose(recovered, time, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_round_trip_transformed(self, seed):
        """The other direction, ``transform(inverse(t_tilde))``, also round trips."""
        rng = np.random.default_rng(300 + seed)
        time = rng.uniform(1e-3, 1e3, size=rng.integers(1, 100))

        data = make_surv_data(time, np.ones(time.size, dtype=bool))
        recovered = transform_time(data, inverse_time(data, data.time))

        np.testing.assert_allclose(recovered, data.time, rtol=1e-12, atol=0.0)

    def test_single_observation_maps_to_two(self):
        """A single observation is its own maximum, so ``t_tilde = 2``."""
        data = make_surv_data([7.5], [True])

        assert data.time.tolist() == [2.0]
        assert data.t_max == 7.5
        assert data.event.tolist() == [True]

    @pytest.mark.parametrize("seed", range(10))
    def test_permutation_invariance(self, seed):
        """Ordering the rows only permutes the transformed times."""
        rng = np.random.default_rng(400 + seed)
        n = int(rng.integers(1, 50))
        time = rng.uniform(1e-3, 1e3, size=n)
        event = rng.integers(0, 2, size=n).astype(bool)
        order = rng.permutation(n)

        original = make_surv_data(time, event)
        permuted = make_surv_data(time[order], event[order])

        assert permuted.t_max == pytest.approx(original.t_max, rel=0.0, abs=0.0)
        np.testing.assert_allclose(permuted.time, original.time[order])
        np.testing.assert_array_equal(permuted.event, original.event[order])

    def test_transformed_arrays_are_normalised(self):
        """Ingestion returns ``float64`` times and ``bool`` events."""
        data = make_surv_data([1, 2, 3], [1, 0, 1])

        assert data.time.dtype == np.float64
        assert data.event.dtype == np.bool_
        np.testing.assert_array_equal(data.event, [True, False, True])

    @pytest.mark.parametrize("encoding", [bool, np.int8, np.int64, np.float64])
    def test_event_accepts_boolean_and_zero_one_encodings(self, encoding):
        """Stored indicators may be booleans or exact 0/1 values."""
        event = np.asarray([1, 0, 1], dtype=encoding)

        data = make_surv_data([1.0, 2.0, 3.0], event)

        assert data.event.dtype == np.bool_
        np.testing.assert_array_equal(data.event, [True, False, True])

    def test_container_is_frozen(self):
        """A chain must not be able to rewrite its data in place."""
        data = make_surv_data([1.0, 2.0], [True, False])

        name = "t_max"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(data, name, 3.0)


class TestInverseMap:
    """The four maps of plan Section 2.3, centralised in one helper."""

    def test_survival_is_unchanged_and_density_hazard_get_the_jacobian(self):
        """Only ``f`` and ``h`` carry ``1 / t_max``; ``S`` must not be scaled."""
        data = make_surv_data([1.0, 2.0, 4.0], [True, True, True])
        t_grid = np.array([0.5, 1.0, 2.0])
        survival = np.array([0.9, 0.8, 0.7])
        density = np.array([0.2, 0.3, 0.4])
        hazard = np.array([0.5, 0.6, 0.7])

        mapped = inverse_map(
            data,
            t_grid,
            survival=survival,
            density=density,
            hazard=hazard,
        )

        np.testing.assert_allclose(mapped["survival"], survival)
        np.testing.assert_allclose(mapped["density"], density / data.t_max)
        np.testing.assert_allclose(mapped["hazard"], hazard / data.t_max)
        np.testing.assert_allclose(mapped["time"], 1.0 + t_grid / data.t_max)
        assert float(mapped["jacobian"]) == pytest.approx(1.0 / data.t_max)

    def test_quantiles_use_the_full_inverse_map(self):
        """``eta_p = t_max * (eta_tilde_p - 1)``, not a Jacobian scaling."""
        data = make_surv_data([1.0, 4.0], [True, True])
        quantiles_tilde = np.array([1.0, 1.25, 2.0])

        mapped = inverse_map(data, np.array([0.0]), quantiles=quantiles_tilde)

        np.testing.assert_allclose(
            mapped["quantiles"], data.t_max * (quantiles_tilde - 1.0)
        )

    def test_omitted_functionals_are_absent(self):
        """The helper does not invent arrays the caller did not supply."""
        data = make_surv_data([1.0, 2.0], [True, False])

        mapped = inverse_map(data, np.array([0.0, 1.0]))

        assert set(mapped) == {"time", "jacobian"}


class TestIngestionValidation:
    """One rule per test: exception type and message are part of the contract."""

    @pytest.mark.parametrize(
        ("time", "event", "message"),
        [
            ([], [], ALL_MISSING_TIME),
            ([1.0], [], ALL_MISSING_EVENT),
        ],
    )
    def test_empty_arrays_are_rejected(self, time, event, message):
        """Zero observations would make ``t_max`` and the grid meaningless."""
        with pytest.raises(ValueError) as excinfo:
            make_surv_data(time, event)

        assert str(excinfo.value) == message

    @pytest.mark.parametrize(
        ("time", "event"),
        [
            ([1.0], [True, False]),
            ([1.0, 2.0, 3.0], [True, False]),
        ],
    )
    def test_length_mismatch_is_rejected(self, time, event):
        """The paper's ``time``/``event`` arrays are parallel by definition."""
        with pytest.raises(ValueError) as excinfo:
            make_surv_data(time, event)

        assert str(excinfo.value) == (
            "time and event must have the same length, "
            f"got {len(time)} and {len(event)}"
        )

    @pytest.mark.parametrize(
        "event",
        [
            np.array([2]),
            np.array([0, 2]),
            np.array([0.0, 1.5]),
            np.array([np.nan]),
            np.array(["0", "1"]),
            np.array([[True, False]]),
        ],
    )
    def test_invalid_event_encoding_is_rejected(self, event):
        """A silent coercion would flip censored rows into events."""
        with pytest.raises(ValueError) as excinfo:
            make_surv_data(np.ones(event.size) + 1.0, event)

        expected = NOT_1D if event.ndim != 1 else BAD_EVENT
        assert expected in str(excinfo.value)

    @pytest.mark.parametrize(
        "time",
        [
            [0.0],
            [0.0, 1.0, 2.0],
            [-1.0],
            [-1.0, -2.0, 3.0],
        ],
    )
    def test_non_positive_times_are_rejected(self, time):
        """``t = 0`` gives ``t_tilde = 1`` and ``log t_tilde = 0``."""
        with pytest.raises(ValueError) as excinfo:
            make_surv_data(time, np.ones(len(time), dtype=bool))

        assert str(excinfo.value) == BAD_TIME_POSITIVE

    @pytest.mark.parametrize(
        "bad",
        [np.nan, np.inf, -np.inf],
    )
    @pytest.mark.parametrize("size", [1, 3])
    def test_non_finite_times_are_rejected(self, bad, size):
        """NaN slips past ``time <= 0``; ``isfinite`` must be checked explicitly."""
        time = np.ones(size, dtype=np.float64)
        time[-1] = bad

        with pytest.raises(ValueError) as excinfo:
            make_surv_data(time, np.ones(size, dtype=bool))

        assert str(excinfo.value) == BAD_TIME_FINITE

    @pytest.mark.parametrize(
        "time",
        [
            np.array([[1.0, 2.0]]),
        ],
    )
    def test_non_1d_time_is_rejected(self, time):
        """The sampler indexes a flat observation vector."""
        with pytest.raises(ValueError) as excinfo:
            make_surv_data(time, np.ones(time.size, dtype=bool))

        assert NOT_1D in str(excinfo.value)

    @pytest.mark.parametrize(
        "t_max",
        [0.0, -1.0, np.nan, np.inf, None, "not a number"],
    )
    def test_bad_t_max_is_rejected_by_the_container(self, t_max):
        """A hand-built ``SurvData`` must not carry a bad inverse-map constant."""
        with pytest.raises(ValueError) as excinfo:
            SurvData(time=np.array([1.5]), event=np.array([True]), t_max=t_max)

        assert str(excinfo.value) == BAD_TMAX

    @pytest.mark.parametrize(
        "time",
        [np.array([1.0]), np.array([0.5, 1.5])],
    )
    def test_untransformed_or_transformed_twice_time_is_rejected(self, time):
        """``SurvData`` consumes ``t_tilde > 1``; raw ``t = 1`` and below fail."""
        with pytest.raises(ValueError) as excinfo:
            SurvData(time=time, event=np.ones(time.size, dtype=bool), t_max=5.0)

        assert str(excinfo.value) == BAD_TIME_TRANSFORMED

    def test_container_length_mismatch_is_rejected(self):
        """Direct construction keeps the same parallel-array invariant."""
        with pytest.raises(ValueError) as excinfo:
            SurvData(
                time=np.array([1.5, 1.75]),
                event=np.array([True]),
                t_max=5.0,
            )

        assert str(excinfo.value) == (
            "time and event must have the same length, got 2 and 1"
        )

    def test_container_rejects_non_finite_time(self):
        """The container is a second line of defence behind ``make_surv_data``."""
        with pytest.raises(ValueError) as excinfo:
            SurvData(
                time=np.array([np.nan, 1.5]),
                event=np.array([True, True]),
                t_max=5.0,
            )

        assert str(excinfo.value) == BAD_TIME_FINITE


class TestFixtureLoading:
    """Stored arrays cross the ingestion boundary without pickle."""

    def test_npz_fixture_loads_with_allow_pickle_false(self, tmp_path):
        """``np.load`` must be called with ``allow_pickle=False`` on fixtures."""
        source = np.array([1.0, 2.5, 4.0])
        events = np.array([1, 0, 1], dtype=np.int8)
        path = tmp_path / "surv_data.npz"
        np.savez(path, time=source, event=events)

        with np.load(path, allow_pickle=False) as fixture:
            data = make_surv_data(fixture["time"], fixture["event"])

        assert data.t_max == pytest.approx(4.0)
        np.testing.assert_allclose(inverse_time(data, data.time), source)
        np.testing.assert_array_equal(data.event, [True, False, True])
