"""Regenerate the committed test fixtures by hand -- never imported by a test.

Plan Section 7.6 requires the fixture files to be produced by a separate,
manual script invocation: a test must never rewrite its own fixture, otherwise
the "same numbers across SciPy/CPU/BLAS versions" guarantee of the golden
regression would be void.

Run from anywhere with the project environment::

    pixi run python tests/fixtures/make_fixtures.py

The script writes four fixtures next to itself: ``golden_dp_gibbs.json``
(test 28 of plan Section 7.5), ``sbc_seeds.json`` (test 17 of plan Section
7.3), ``section_5_1.npz`` and ``section_5_2.npz`` (plan Section 7.6).  It is
deliberately not imported by ``tests/test_golden.py`` or ``tests/test_sbc.py``;
the tests rebuild the same synthetic data, sampler settings and generation
protocol from their own constants and regenerate the expected numbers only
through the production code path.  The two Section 5 fixtures are checked by
``tests/test_paper_data.py`` against the plan's structural assertions (sample
counts, censoring positions, realised statistics) rather than by regenerating
their arrays.

The SBC fixture holds seeds only -- one generation, one chain and one
functional-inference seed per replicate -- because those are what the plan's
Section 7.3 test 17 asks to keep reproducible.  The generator itself lives in
``tests/test_sbc.py``, so a drift of the generation stream shows up as a rank
change, not as a stale fixture that quietly agrees with itself.

Regenerate only deliberately, and treat the resulting diff as a change to the
sampler's random-number consumption: the discrete field ``s`` and the floating
summaries can only move if a code change alters the stream or the target.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The script is invoked directly, so pytest's root conftest cannot put the
# repository root on sys.path for it.
sys.path.insert(0, str(REPO_ROOT))

from src.dp_functionals import (  # noqa: E402
    FunctionalSettings,
    posterior_functionals,
)
from src.dp_gibbs import (  # noqa: E402
    DPPrior,
    DPSettings,
    make_surv_data,
    sample_chain,
)
from src.paper_data import (  # noqa: E402
    LEUKEMIA_5_2,
    LEUKEMIA_A_TRANSCRIPTION,
    LEUKEMIA_B_TRANSCRIPTION,
    TwoLognormalMix,
)

FIXTURE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = FIXTURE_DIR / "golden_dp_gibbs.json"
SBC_SEEDS_PATH = FIXTURE_DIR / "sbc_seeds.json"
SECTION_5_1_PATH = FIXTURE_DIR / "section_5_1.npz"
SECTION_5_2_PATH = FIXTURE_DIR / "section_5_2.npz"

# Plan Section 7.3 test 17: 100 replicates, each with its own generation,
# chain and functional-inference seed.  One fixed root seed makes the whole
# set reproducible from this script; the child seeds are drawn with numpy's
# ``SeedSequence``, whose output is stable across library versions.
SBC_ROOT_SEED = 20241113
SBC_REPLICATES = 100

# The time unit and censoring-law width of the SBC replicates.  They duplicate
# tests/test_sbc.py on purpose: a fixture regenerated with a different law must
# fail that test's metadata assertions instead of silently moving the ranks.
SBC_T_MAX_ORIGINAL = 5.1390
SBC_CENSOR_MAX_TRANSFORMED = 150.0

# Tiny, deterministic surrogate for the Section 5.1 data: n = 20 lognormal
# times with right-censoring at a fixed threshold.  Only three observations are
# censored, so both likelihood branches of the sampler are exercised.
N_OBSERVATIONS = 20
DATA_SEED = 20241105
CENSOR_TIME = 2.5
LOGNORMAL_MEAN = 0.0
LOGNORMAL_SIGMA = 0.6

# The Section 5.1 prior of plan Section 5 (the same constants the other tests
# use), with 200 sweeps and a 32-node alpha grid: the "quick-like but tiny"
# settings of test 28.
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


def golden_data() -> tuple[np.ndarray, np.ndarray]:
    """Return the raw ``(time, event)`` arrays of the golden dataset."""
    rng = np.random.default_rng(DATA_SEED)
    raw_time = rng.lognormal(
        mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA, size=N_OBSERVATIONS
    )
    event = raw_time <= CENSOR_TIME
    return np.minimum(raw_time, CENSOR_TIME), event


def canonical_partition(theta: np.ndarray) -> np.ndarray:
    """Label the partition of one stored draw by order of first appearance.

    ``DPChains`` deliberately does not store the cluster labels ``s`` (plan
    Section 3.2), but a stored draw keeps them implicitly: invariant 1 of
    Section 2.1 makes ``(alpha_i, lambda_i)`` identical for all members of a
    cluster.  Grouping the exact pairs therefore recovers the partition.  The
    labels of a Gibbs state are arbitrary up to relabelling, so the fixture
    stores the canonical first-appearance labelling, not the raw ``s``.
    """
    labels = np.empty(theta.shape[0], dtype=np.intp)
    seen: dict[tuple[float, float], int] = {}
    for i, (alpha, lam) in enumerate(theta):
        key = (float(alpha), float(lam))
        labels[i] = seen.setdefault(key, len(seen))
    return labels


def build_golden_fixture() -> dict[str, Any]:
    """Run the pinned chain and functional evaluation and describe the result."""
    raw_time, event = golden_data()
    data = make_surv_data(raw_time, event)
    chain = sample_chain(data, PRIOR, SETTINGS)

    theta = chain.theta[0]
    labels = canonical_partition(theta)
    block_sizes = np.bincount(labels)

    functionals = posterior_functionals(
        chain,
        np.asarray(T_GRID, dtype=np.float64),
        settings=FunctionalSettings(),
        seed=FUNCTIONAL_SEED,
    )

    # Structural sanity: a fixture generated from a broken run must not be
    # committed silently.
    assert data.time.size == N_OBSERVATIONS
    assert 0 < int(np.count_nonzero(~event)) < N_OBSERVATIONS
    assert np.isclose(data.t_max, CENSOR_TIME)
    assert labels.size == N_OBSERVATIONS
    assert np.unique(labels).size == int(chain.n_star[0])
    assert int(block_sizes.sum()) == N_OBSERVATIONS
    assert np.all(block_sizes > 0)
    assert np.all(np.isfinite(functionals.S[0]))

    return {
        "fixture": "golden_dp_gibbs",
        "plan": "docs/research/dp_gibbs_plan.md Section 7.5 test 28",
        "generated_by": "python tests/fixtures/make_fixtures.py",
        "data": {
            "generator": "lognormal",
            "n": N_OBSERVATIONS,
            "seed": DATA_SEED,
            "lognormal_mean": LOGNORMAL_MEAN,
            "lognormal_sigma": LOGNORMAL_SIGMA,
            "censor_time": CENSOR_TIME,
            "time": [float(value) for value in raw_time],
            "event": [bool(value) for value in event],
            "t_max": float(data.t_max),
        },
        "prior": dataclasses.asdict(PRIOR),
        "settings": dataclasses.asdict(SETTINGS),
        "functionals": {
            "seed": FUNCTIONAL_SEED,
            "t_grid": [float(value) for value in T_GRID],
            "J_common": int(functionals.meta["J_common"]),
        },
        "draw": {
            "s": [int(value) for value in labels],
            "n_star": int(chain.n_star[0]),
            "nu": float(chain.nu[0]),
            "gamma": float(chain.gamma[0]),
            "phi": float(chain.phi[0]),
            "log_post": float(chain.log_post[0]),
            "S": [float(value) for value in functionals.S[0]],
        },
    }


def sbc_seeds() -> tuple[list[int], list[int], list[int]]:
    """Return the per-replicate generation, chain and functional seeds."""
    children = np.random.SeedSequence(SBC_ROOT_SEED).spawn(3 * SBC_REPLICATES)
    values = [int(child.generate_state(1, np.uint32)[0]) for child in children]
    return (
        values[:SBC_REPLICATES],
        values[SBC_REPLICATES : 2 * SBC_REPLICATES],
        values[2 * SBC_REPLICATES :],
    )


def build_sbc_seed_fixture() -> dict[str, Any]:
    """Describe the seeds and the pinned generation constants of test 17."""
    replicate_seeds, chain_seeds, functional_seeds = sbc_seeds()

    # Structural sanity: a fixture written from a broken seed stream must not
    # be committed silently.  The values duplicate tests/test_sbc.py on
    # purpose (the test is not allowed to import this script).
    assert len(replicate_seeds) == SBC_REPLICATES
    assert len(set(replicate_seeds)) == SBC_REPLICATES
    assert len(set(chain_seeds)) == SBC_REPLICATES
    assert len(set(functional_seeds)) == SBC_REPLICATES

    return {
        "fixture": "sbc_seeds",
        "plan": "docs/research/dp_gibbs_plan.md Section 7.3 test 17",
        "generated_by": "python tests/fixtures/make_fixtures.py",
        "root_seed": SBC_ROOT_SEED,
        "n_replicates": SBC_REPLICATES,
        "n_observations": 15,
        "prior": dataclasses.asdict(PRIOR),
        "generation": {
            "epsilon_tail": 1e-8,
            "t_max_original": SBC_T_MAX_ORIGINAL,
            "censor_max_original": SBC_CENSOR_MAX_TRANSFORMED * SBC_T_MAX_ORIGINAL,
            "censoring_fraction_bounds": [0.15, 0.25],
            "evaluation_times_original": [0.7832, 1.1727, 2.1511],
        },
        "statistics": ["nu", "gamma", "phi", "S_q25", "S_q50", "S_q75"],
        "replicate_seeds": replicate_seeds,
        "chain_seeds": chain_seeds,
        "functional_seeds": functional_seeds,
    }


def build_section_5_1_fixture() -> dict[str, Any]:
    """Return the committed Section 5.1 sample and its realised statistics.

    The generator itself lives in ``src.paper_data`` so the fixture is produced
    through the same production path the drivers use.  The assertions below are
    the plan's Section 6.1 numbers; a regeneration that moves them must fail
    here instead of silently replacing the dataset.
    """
    mix = TwoLognormalMix()
    time = mix.times
    q25, median, q75 = np.quantile(time, [0.25, 0.5, 0.75])
    iqr = q75 - q25
    q95 = np.quantile(time, 0.95)
    t_max = time.max()
    n_above_4 = int(np.count_nonzero(time > 4.0))

    # Structural sanity: plan Sections 6.1 and 7.6.
    assert time.size == mix.n == 200
    assert mix.p1_n == 160
    assert mix.p2_n == 40
    assert np.all(np.isfinite(time))
    assert np.all(time > 0.0)
    assert np.isclose(median, 1.2172, atol=1e-4)
    assert np.isclose(iqr, 1.5336, atol=1e-4)
    assert np.isclose(q25, 0.8168, atol=1e-4)
    assert np.isclose(q75, 2.3505, atol=1e-4)
    assert np.isclose(q95, 3.7396, atol=1e-4)
    assert np.isclose(t_max, 5.1390, atol=1e-4)
    assert n_above_4 == 2

    return {
        "time": time,
        "event": np.ones(time.size, dtype=bool),
        "seed": np.asarray(mix.seed, dtype=np.int64),
        "n": np.asarray(mix.n, dtype=np.int64),
        "p1_n": np.asarray(mix.p1_n, dtype=np.int64),
        "p2_n": np.asarray(mix.p2_n, dtype=np.int64),
        "mu_1": np.asarray(mix.mu_1, dtype=np.float64),
        "mu_2": np.asarray(mix.mu_2, dtype=np.float64),
        "var_1": np.asarray(mix.var_1, dtype=np.float64),
        "var_2": np.asarray(mix.var_2, dtype=np.float64),
        "p": np.asarray(mix.p, dtype=np.float64),
        "median": np.asarray(median, dtype=np.float64),
        "iqr": np.asarray(iqr, dtype=np.float64),
        "q25": np.asarray(q25, dtype=np.float64),
        "q75": np.asarray(q75, dtype=np.float64),
        "q95": np.asarray(q95, dtype=np.float64),
        "max": np.asarray(t_max, dtype=np.float64),
        "t_max": np.asarray(t_max, dtype=np.float64),
        "n_above_4": np.asarray(n_above_4, dtype=np.int64),
    }


def build_section_5_2_fixture() -> dict[str, Any]:
    """Return the leukemia data, its transcription and structural metadata.

    Arm A has 20 observations with 3 censored (28+, 48+, 49+); arm B has 20
    with 2 censored (27+, 38+).  The plan's earlier "three censored per
    treatment" claim was wrong for B and is not asserted here.
    """
    payload: dict[str, Any] = {}
    expected = {
        "a": (LEUKEMIA_A_TRANSCRIPTION, 20, 17, 3, (14, 18, 19), 49.0),
        "b": (LEUKEMIA_B_TRANSCRIPTION, 20, 18, 2, (16, 18), 44.0),
    }
    for arm, (
        tokens,
        n,
        n_observed,
        n_censored,
        censored_index,
        t_max,
    ) in expected.items():
        time, event = LEUKEMIA_5_2[arm.upper()]
        observed_index = np.flatnonzero(event)
        censored_index_fixture = np.flatnonzero(~event)

        # Structural sanity: plan Sections 6.1 and 7.6.
        assert time.size == n
        assert int(np.count_nonzero(event)) == n_observed
        assert int(np.count_nonzero(~event)) == n_censored
        assert tuple(int(i) for i in censored_index_fixture) == censored_index
        assert observed_index.size == n_observed
        assert float(time.max()) == t_max
        assert all(tokens[index].endswith("+") for index in censored_index)

        payload[f"{arm}_time"] = time
        payload[f"{arm}_event"] = event
        payload[f"{arm}_transcription"] = np.asarray(tokens)
        payload[f"{arm}_n"] = np.asarray(n, dtype=np.int64)
        payload[f"{arm}_n_observed"] = np.asarray(n_observed, dtype=np.int64)
        payload[f"{arm}_n_censored"] = np.asarray(n_censored, dtype=np.int64)
        payload[f"{arm}_censored_index"] = censored_index_fixture
        payload[f"{arm}_observed_index"] = observed_index
        payload[f"{arm}_t_max"] = np.asarray(t_max, dtype=np.float64)
    return payload


def main() -> None:
    """Write every fixture and report where it went."""
    payload = build_golden_fixture()
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH.relative_to(REPO_ROOT)}")

    sbc_payload = build_sbc_seed_fixture()
    SBC_SEEDS_PATH.write_text(
        json.dumps(sbc_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {SBC_SEEDS_PATH.relative_to(REPO_ROOT)}")

    section_5_1_payload = build_section_5_1_fixture()
    np.savez(SECTION_5_1_PATH, **section_5_1_payload)
    print(f"wrote {SECTION_5_1_PATH.relative_to(REPO_ROOT)}")

    section_5_2_payload = build_section_5_2_fixture()
    np.savez(SECTION_5_2_PATH, **section_5_2_payload)
    print(f"wrote {SECTION_5_2_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
