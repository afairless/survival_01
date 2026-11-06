"""Plan Section 7.5 test 28: golden regression over a whole chain.

One fixed seed, tiny ``quick``-like settings (``n = 20`` synthetic censored
observations, 200 sweeps, ``M = 32`` grid nodes) and one stored draw are
compared with the committed fixture ``tests/fixtures/golden_dp_gibbs.json``:

* the discrete, RNG-determined quantities -- the partition ``s`` and the
  cluster count ``n*`` -- exactly;
* the floating summaries -- ``nu``, ``gamma``, ``phi``, the collapsed
  ``log_post`` and ``S(t)`` at three original-scale times -- to ``1e-10``
  relative.

The fixture is produced by ``tests/fixtures/make_fixtures.py``, never by this
test: a test that rewrote its own fixture would turn any change into a pass.
The data generator and the sampler settings are rebuilt here from constants,
so the fixture's input block is checked against the test's own copy rather
than trusted.

What each test would catch:

* a change to the sweep order, to any conditional's random-number consumption,
  or to the initialisation stream -- every floating value and both discrete
  fields move, because the whole sweep sequence shares one generator;
* a partition inconsistency that ``DPChains`` does not check (an
  ``alpha``/``alpha*`` pair no longer shared by a cluster's members) -- the
  recovered first-appearance labels stop matching the fixture;
* a change to the common-``J`` rule, the truncation or the atom pruning used
  by the functional path -- ``S`` moves even when the chain is untouched;
* a fixture regenerated with different data or settings without updating the
  test -- the metadata and data-equality assertions fail first.
"""

from __future__ import annotations

import dataclasses
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.dp_functionals import FunctionalSettings, posterior_functionals
from src.dp_gibbs import DPPrior, DPSettings, make_surv_data, sample_chain

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_dp_gibbs.json"

# The pinned configuration.  These constants are duplicated in
# tests/fixtures/make_fixtures.py on purpose: the fixture generation script is
# not imported by any test (plan Section 7.6), so the test must be able to
# detect a fixture that was regenerated with different inputs.
N_OBSERVATIONS = 20
DATA_SEED = 20241105
CENSOR_TIME = 2.5
LOGNORMAL_MEAN = 0.0
LOGNORMAL_SIGMA = 0.6
N_CENSORED = 3
TOTAL_SWEEPS = 200

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)
SETTINGS = DPSettings(
    burn_in=100,
    n_save=1,
    thin=100,
    n_alpha=32,
    epsilon=1e-4,
    n_chains=1,
    seed=42,
    preset="golden",
)
FUNCTIONAL_SEED = 20241106
T_GRID = (0.5, 1.5, 2.5)

# Floating summaries are compared relatively, and not bitwise: ``pow`` and
# ``logsumexp`` are library calls whose last bits may differ between CPU and
# BLAS paths (plan Section 7.5 test 28).
RELATIVE_TOLERANCE = 1e-10


def _load_fixture() -> dict[str, Any]:
    """Read the committed golden fixture as JSON."""
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def _golden_data() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the synthetic censored dataset of the fixture description."""
    rng = np.random.default_rng(DATA_SEED)
    raw_time = rng.lognormal(
        mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA, size=N_OBSERVATIONS
    )
    event = raw_time <= CENSOR_TIME
    return np.minimum(raw_time, CENSOR_TIME), event


def _canonical_partition(theta: np.ndarray) -> np.ndarray:
    """Recover a stored draw's partition, labelled by first appearance.

    ``DPChains`` stores ``theta[b] = (alpha_i, lambda_i)`` per observation, not
    the labels ``s``; invariant 1 of plan Section 2.1 makes every member of a
    cluster carry exactly the same pair, so grouping the pairs recovers the
    partition.  Labels are assigned in observation order so that the fixture
    does not depend on the arbitrary integer labels the bookkeeping used.
    """
    labels = np.empty(theta.shape[0], dtype=np.intp)
    seen: dict[tuple[float, float], int] = {}
    for i, (alpha, lam) in enumerate(theta):
        labels[i] = seen.setdefault((float(alpha), float(lam)), len(seen))
    return labels


def _assert_close(actual: float, expected: float, name: str) -> None:
    """Assert one floating golden value to the section's relative tolerance."""
    if not np.isclose(actual, expected, rtol=RELATIVE_TOLERANCE, atol=0.0):
        raise AssertionError(
            f"{name} moved: got {actual!r}, fixture holds {expected!r} "
            f"(relative tolerance {RELATIVE_TOLERANCE})"
        )


def test_golden_fixture_describes_the_pinned_configuration() -> None:
    """The fixture's inputs are the tiny settings this module pins."""
    fixture = _load_fixture()

    assert fixture["fixture"] == "golden_dp_gibbs"
    assert fixture["plan"].endswith("test 28")

    data_block = fixture["data"]
    assert data_block["generator"] == "lognormal"
    assert data_block["n"] == N_OBSERVATIONS
    assert data_block["seed"] == DATA_SEED
    assert data_block["censor_time"] == CENSOR_TIME
    assert data_block["t_max"] == CENSOR_TIME

    raw_time, event = _golden_data()
    assert np.array_equal(np.asarray(data_block["time"]), raw_time)
    assert np.array_equal(np.asarray(data_block["event"]), event)
    assert int(np.count_nonzero(~event)) == N_CENSORED

    assert fixture["prior"] == dataclasses.asdict(PRIOR)
    settings_block = fixture["settings"]
    assert settings_block == dataclasses.asdict(SETTINGS)
    assert (
        settings_block["burn_in"] + settings_block["thin"] * settings_block["n_save"]
        == TOTAL_SWEEPS
    )
    assert settings_block["n_alpha"] == 32

    assert fixture["functionals"]["seed"] == FUNCTIONAL_SEED
    assert fixture["functionals"]["t_grid"] == list(T_GRID)


def test_golden_draw_matches_the_fixture() -> None:
    """The chain and its ``S(t)`` reproduce the committed fixture."""
    fixture = _load_fixture()
    golden = fixture["draw"]

    raw_time, event = _golden_data()
    data = make_surv_data(raw_time, event)
    chain = sample_chain(data, PRIOR, SETTINGS)

    assert chain.n_draws == 1
    assert chain.n_star.dtype.kind == "i"

    theta = chain.theta[0]
    assert theta.shape == (N_OBSERVATIONS, 2)
    labels = _canonical_partition(theta)

    # Discrete, RNG-determined quantities: exact equality.
    assert np.array_equal(labels, np.asarray(golden["s"], dtype=np.intp))
    assert int(chain.n_star[0]) == golden["n_star"]
    assert np.unique(labels).size == golden["n_star"]
    assert int(np.bincount(labels).sum()) == N_OBSERVATIONS

    # Floating summaries: relative 1e-10, never bitwise.
    _assert_close(float(chain.nu[0]), golden["nu"], "nu")
    _assert_close(float(chain.gamma[0]), golden["gamma"], "gamma")
    _assert_close(float(chain.phi[0]), golden["phi"], "phi")
    _assert_close(float(chain.log_post[0]), golden["log_post"], "log_post")

    functionals = posterior_functionals(
        chain,
        np.asarray(T_GRID, dtype=np.float64),
        settings=FunctionalSettings(),
        seed=FUNCTIONAL_SEED,
    )
    assert functionals.n_draws == 1
    assert functionals.meta["J_common"] == fixture["functionals"]["J_common"]
    assert np.allclose(
        functionals.S[0],
        np.asarray(golden["S"], dtype=np.float64),
        rtol=RELATIVE_TOLERANCE,
        atol=0.0,
    ), f"S moved: got {functionals.S[0]!r}, fixture holds {golden['S']!r}"

    # A survival curve cannot increase with time; the fixture's three points
    # are ordered, so this also guards against an S/F or index mix-up.
    assert all(later <= earlier for earlier, later in pairwise(functionals.S[0]))
