"""Section 5 paper data and prior constants (plan Sections 6.1 and 7.6).

The committed fixtures are the paper datasets of record.  They are generated
once by ``tests/fixtures/make_fixtures.py`` and loaded here with
``allow_pickle=False``; no test writes or regenerates a fixture.  The Section
5.1 array is deliberately *not* compared against a fresh SciPy draw, because
that would make the dataset depend on the installed SciPy RNG stream.  The
generator itself is tested separately: it must consume one
``numpy.random.Generator`` for both components, and its variance fields must be
variances (``sqrt``-ed before being passed to ``lognorm(s=...)``).

The Section 5.2 fixture is checked against the paper's transcription, including
the ``+`` censoring flags, and against the plan's corrected counts: arm A has
20 observations with 3 censored, arm B has 20 with 2 censored.  The earlier
"three censored per treatment" claim is wrong for B.

What a wrong fixture or generator would break: the reproduction runs of plan
Sections 6.3 and 9 would silently fit a different dataset, and a reintroduced
shared-seed defect would make the two mixture components dependent while every
marginal statistic still looked plausible.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scipy.stats import lognorm

from src import paper_data
from src.dp_weibull import b_gamma_b_phi_from_weibull, weibull_from_median_iqr
from src.paper_data import TwoLognormalMix

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SECTION_5_1_PATH = FIXTURE_DIR / "section_5_1.npz"
SECTION_5_2_PATH = FIXTURE_DIR / "section_5_2.npz"

# The paper's Section 5.2 lists, transcribed here independently of
# ``src.paper_data``.  ``+`` marks right-censoring.
PAPER_A = (
    "1",
    "3",
    "3",
    "6",
    "7",
    "7",
    "10",
    "12",
    "14",
    "15",
    "18",
    "19",
    "22",
    "26",
    "28+",
    "29",
    "34",
    "40",
    "48+",
    "49+",
)
PAPER_B = (
    "1",
    "1",
    "2",
    "2",
    "3",
    "4",
    "5",
    "8",
    "8",
    "9",
    "11",
    "12",
    "14",
    "16",
    "18",
    "21",
    "27+",
    "31",
    "38+",
    "44",
)

SECTION_5_1_KEYS = {
    "time",
    "event",
    "seed",
    "n",
    "p1_n",
    "p2_n",
    "mu_1",
    "mu_2",
    "var_1",
    "var_2",
    "p",
    "median",
    "iqr",
    "q25",
    "q75",
    "q95",
    "max",
    "t_max",
    "n_above_4",
}
SECTION_5_2_A_KEYS = {
    "a_time",
    "a_event",
    "a_transcription",
    "a_n",
    "a_n_observed",
    "a_n_censored",
    "a_censored_index",
    "a_observed_index",
    "a_t_max",
}
SECTION_5_2_B_KEYS = {
    "b_time",
    "b_event",
    "b_transcription",
    "b_n",
    "b_n_observed",
    "b_n_censored",
    "b_censored_index",
    "b_observed_index",
    "b_t_max",
}


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    """Load a committed ``.npz`` fixture without ever unpickling it."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def parse_transcription(tokens: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Independent ``t`` / ``t+`` parser used for the transcription test."""
    time = np.array([float(token.rstrip("+")) for token in tokens])
    event = np.array([not token.endswith("+") for token in tokens])
    return time, event


# ---------------------------------------------------------------------------
# Section 5.1: fixture structure and realised statistics
# ---------------------------------------------------------------------------
def test_section_5_1_fixture_loads_without_pickle() -> None:
    """The fixture is plain numeric data and needs no pickle support."""
    fixture = load_fixture(SECTION_5_1_PATH)

    assert set(fixture) == SECTION_5_1_KEYS
    assert all(array.dtype != object for array in fixture.values())

    time = fixture["time"]
    assert time.dtype == np.float64
    assert time.shape == (200,)
    assert np.all(np.isfinite(time))
    assert np.all(time > 0.0)

    event = fixture["event"]
    assert event.dtype == np.bool_
    assert event.shape == (200,)
    assert event.all()


def test_section_5_1_fixture_metadata_matches_the_mixture() -> None:
    """Seed, component sizes and mixture parameters are the paper's."""
    fixture = load_fixture(SECTION_5_1_PATH)
    mix = TwoLognormalMix()

    assert int(fixture["seed"]) == mix.seed == 721383
    assert int(fixture["n"]) == mix.n == 200
    assert int(fixture["p1_n"]) == mix.p1_n == 160
    assert int(fixture["p2_n"]) == mix.p2_n == 40

    for name in ("mu_1", "mu_2", "var_1", "var_2", "p"):
        assert float(fixture[name]) == pytest.approx(getattr(mix, name))

    assert float(fixture["mu_1"]) == 0.0
    assert float(fixture["mu_2"]) == 1.2
    assert float(fixture["var_1"]) == 0.25
    assert float(fixture["var_2"]) == 0.02
    assert float(fixture["p"]) == 0.8


def test_section_5_1_fixture_realised_statistics() -> None:
    """The sample has the plan's Section 6.1 median, IQR and maximum."""
    fixture = load_fixture(SECTION_5_1_PATH)
    time = fixture["time"]
    q25, median, q75, q95 = np.quantile(time, [0.25, 0.5, 0.75, 0.95])
    iqr = q75 - q25
    t_max = float(time.max())

    assert median == pytest.approx(1.2172, abs=1e-4)
    assert iqr == pytest.approx(1.5336, abs=1e-4)
    assert q25 == pytest.approx(0.8168, abs=1e-4)
    assert q75 == pytest.approx(2.3505, abs=1e-4)
    assert q95 == pytest.approx(3.7396, abs=1e-4)
    assert t_max == pytest.approx(5.1390, abs=1e-4)
    assert int(np.count_nonzero(time > 4.0)) == 2

    # The stored summaries must describe the stored sample.
    assert float(fixture["median"]) == pytest.approx(median)
    assert float(fixture["iqr"]) == pytest.approx(iqr)
    assert float(fixture["q25"]) == pytest.approx(q25)
    assert float(fixture["q75"]) == pytest.approx(q75)
    assert float(fixture["q95"]) == pytest.approx(q95)
    assert float(fixture["max"]) == pytest.approx(t_max)
    assert float(fixture["t_max"]) == pytest.approx(t_max)
    assert int(fixture["n_above_4"]) == 2


# ---------------------------------------------------------------------------
# Section 5.1: the generator itself
# ---------------------------------------------------------------------------
def test_two_lognormal_mix_uses_one_rng_stream() -> None:
    """Both components consume the same ``Generator``, not the same seed."""
    mix = TwoLognormalMix()
    rng = np.random.default_rng(mix.seed)
    first = lognorm.rvs(
        s=np.sqrt(mix.var_1),
        scale=np.exp(mix.mu_1),
        size=mix.p1_n,
        random_state=rng,
    )
    second = lognorm.rvs(
        s=np.sqrt(mix.var_2),
        scale=np.exp(mix.mu_2),
        size=mix.p2_n,
        random_state=rng,
    )

    np.testing.assert_array_equal(mix.times, np.append(first, second))


def test_two_lognormal_mix_components_are_not_dependent() -> None:
    """The shared-integer-seed defect would give a log-correlation of 1."""
    mix = TwoLognormalMix(p=0.5, n=200)
    log_time = np.log(mix.times)
    correlation = np.corrcoef(log_time[: mix.p1_n], log_time[mix.p1_n :])[0, 1]

    assert np.isfinite(correlation)
    assert abs(correlation) < 0.25


def test_two_lognormal_mix_variances_are_variances() -> None:
    """``var_1``/``var_2`` are variances: the logged components have that sd."""
    mix = TwoLognormalMix(p=0.5, n=4000)
    log_time = np.log(mix.times)

    assert np.std(log_time[: mix.p1_n]) == pytest.approx(np.sqrt(mix.var_1), rel=0.05)
    assert np.std(log_time[mix.p1_n :]) == pytest.approx(np.sqrt(mix.var_2), rel=0.05)


def test_two_lognormal_mix_field_names_use_variances() -> None:
    """The legacy ``sigma_1``/``sigma_2`` names held variances and are gone."""
    names = {field.name for field in dataclasses.fields(TwoLognormalMix)}

    assert {"var_1", "var_2"} <= names
    assert "sigma_1" not in names
    assert "sigma_2" not in names


@pytest.mark.parametrize(
    "kwargs",
    [
        {"var_1": 0.0},
        {"var_2": -1.0},
        {"n": 0},
        {"p": -0.1},
        {"p": 1.1},
        {"mu_1": np.inf},
        {"mu_2": np.nan},
    ],
)
def test_two_lognormal_mix_rejects_invalid_inputs(
    kwargs: dict[str, Any],
) -> None:
    """A broken mixture must fail at construction, not produce a sample."""
    with pytest.raises(ValueError):
        TwoLognormalMix(**kwargs)


# ---------------------------------------------------------------------------
# Section 5.2: transcription and counts
# ---------------------------------------------------------------------------
def test_section_5_2_fixture_loads_without_pickle() -> None:
    """The leukemia fixture is plain arrays and transcribed string tokens."""
    fixture = load_fixture(SECTION_5_2_PATH)

    assert set(fixture) == SECTION_5_2_A_KEYS | SECTION_5_2_B_KEYS
    assert all(array.dtype != object for array in fixture.values())
    for arm in ("a", "b"):
        assert fixture[f"{arm}_time"].dtype == np.float64
        assert fixture[f"{arm}_event"].dtype == np.bool_


def test_section_5_2_transcription_matches_the_paper() -> None:
    """Every time, order and ``+`` flag matches the paper's list."""
    fixture = load_fixture(SECTION_5_2_PATH)

    for arm, paper_tokens in (("a", PAPER_A), ("b", PAPER_B)):
        expected_time, expected_event = parse_transcription(paper_tokens)
        transcription = tuple(str(token) for token in fixture[f"{arm}_transcription"])

        assert transcription == paper_tokens
        np.testing.assert_array_equal(fixture[f"{arm}_time"], expected_time)
        np.testing.assert_array_equal(fixture[f"{arm}_event"], expected_event)

    # The production transcription is the same source of truth.
    assert paper_data.LEUKEMIA_A_TRANSCRIPTION == PAPER_A
    assert paper_data.LEUKEMIA_B_TRANSCRIPTION == PAPER_B


@pytest.mark.parametrize(
    ("arm", "paper_tokens", "n", "n_observed", "n_censored", "censored_index"),
    [
        ("a", PAPER_A, 20, 17, 3, (14, 18, 19)),
        ("b", PAPER_B, 20, 18, 2, (16, 18)),
    ],
)
def test_section_5_2_counts_and_censoring_positions(
    arm: str,
    paper_tokens: tuple[str, ...],
    n: int,
    n_observed: int,
    n_censored: int,
    censored_index: tuple[int, ...],
) -> None:
    """Arm A is 20/17/3 and arm B is 20/18/2; the ``+`` flags are in place."""
    fixture = load_fixture(SECTION_5_2_PATH)
    expected_time, expected_event = parse_transcription(paper_tokens)
    time = fixture[f"{arm}_time"]
    event = fixture[f"{arm}_event"]
    observed_index = np.flatnonzero(event)
    censored_positions = np.flatnonzero(~event)

    assert time.size == int(fixture[f"{arm}_n"]) == n
    assert (
        int(np.count_nonzero(event))
        == int(fixture[f"{arm}_n_observed"])
        == (n_observed)
    )
    assert (
        int(np.count_nonzero(~event))
        == int(fixture[f"{arm}_n_censored"])
        == (n_censored)
    )
    assert tuple(int(index) for index in censored_positions) == censored_index
    np.testing.assert_array_equal(fixture[f"{arm}_censored_index"], censored_positions)
    np.testing.assert_array_equal(fixture[f"{arm}_observed_index"], observed_index)
    np.testing.assert_array_equal(time, expected_time)
    np.testing.assert_array_equal(event, expected_event)
    assert all(paper_tokens[index].endswith("+") for index in censored_index)


def test_section_5_2_t_max_is_the_largest_uncensored_or_censored_time() -> None:
    """The original-scale maximum is 49 weeks for A and 44 weeks for B."""
    fixture = load_fixture(SECTION_5_2_PATH)

    assert float(fixture["a_time"].max()) == float(fixture["a_t_max"]) == 49.0
    assert float(fixture["b_time"].max()) == float(fixture["b_t_max"]) == 44.0


def test_leukemia_5_2_returns_independent_copies() -> None:
    """A caller mutating the returned arrays must not change the constants."""
    arms = paper_data.leukemia_5_2()

    arms["A"][0][0] = -1.0
    arms["A"][1][0] = False

    assert paper_data.LEUKEMIA_A_TIME[0] == 1.0
    assert bool(paper_data.LEUKEMIA_A_EVENT[0]) is True


# ---------------------------------------------------------------------------
# Section 5: prior-guess constants
# ---------------------------------------------------------------------------
def test_prior_guess_values_are_the_plans_population_values() -> None:
    """Section 10 fixes the population median/IQR, not a sample estimate."""
    assert paper_data.PRIOR_MEDIAN == pytest.approx(1.1727)
    assert paper_data.PRIOR_IQR == pytest.approx(1.3679)
    assert paper_data.PRIOR_Q95 == pytest.approx(3.6828)


def test_prior_constants_round_trip_through_the_closed_forms() -> None:
    """The recorded constants are the Weibull/b_gamma/b_phi round trip."""
    alpha_tilde, lambda_tilde = cast(
        tuple[float, float],
        weibull_from_median_iqr(paper_data.PRIOR_MEDIAN, paper_data.PRIOR_IQR),
    )
    b_gamma, b_phi = cast(
        tuple[float, float],
        b_gamma_b_phi_from_weibull(paper_data.ALPHA_TILDE, paper_data.LAMBDA_TILDE),
    )

    assert float(alpha_tilde) == pytest.approx(paper_data.ALPHA_TILDE, rel=1e-4)
    assert float(lambda_tilde) == pytest.approx(paper_data.LAMBDA_TILDE, rel=1e-4)
    assert float(b_gamma) == pytest.approx(paper_data.B_GAMMA, rel=1e-4)
    assert float(b_phi) == pytest.approx(paper_data.B_PHI, rel=1e-4)


def test_nu_priors_are_the_paper_gamma_grid() -> None:
    """Figure 1-3 use Gamma(2,0.9), Gamma(2,0.1) and Gamma(3,0.05)."""
    assert paper_data.NU_PRIORS == ((2.0, 0.9), (2.0, 0.1), (3.0, 0.05))


def test_remaining_model_constants_are_not_estimated() -> None:
    """Section 6.2 fixes d = 2, a_gamma = 1 and a_phi = 2."""
    assert paper_data.D == 2
    assert paper_data.A_GAMMA == 1.0
    assert paper_data.A_PHI == 2.0
