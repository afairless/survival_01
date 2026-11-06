"""Test 18 (plan Sections 7.3 and 11): multi-chain mixing on the §5.1 data.

A single chain can look healthy while the sampler is trapped in one region of
the posterior: every within-chain diagnostic is computed from that chain's own
draws, so a chain stuck at ``n* = 1`` or ``n* = n`` has a perfectly smooth
trace.  The only way to see that failure is to start several independent
chains and compare their levels.  This test therefore runs the four chains of
plan Section 3.6 (seeds ``seed + 0..3``) on the simulated §5.1 dataset at the
``quick`` preset and applies the acceptance rule of plan Section 3.7:

* split :math:`\\hat R < 1.05` for ``nu``, ``n*`` and the observation-level
  survival mixture ``S(t)`` at the five diagnostic times;
* every chain visits more than one value of ``n*`` -- a chain frozen at
  ``n* = 1`` or ``n* = n`` fails even if its R-hat happens to be small.

The §5.1 sample is generated here with the corrected one-RNG mixture of plan
Section 6.1 (the legacy ``TwoLognormalMix`` passes the same integer seed to
both components, which correlates the two groups of draws).  The realised
sample statistics are asserted as well, so a change to the generator cannot
silently move the test onto a different dataset.

What each test would catch:

* a sampler that is correct within a chain but mixes poorly across chains --
  the split R-hat assertions;
* a chain that never leaves its initial partition -- the ``n*`` support
  assertion, which R-hat alone can miss when all four chains freeze at the
  same value;
* a data generator that reintroduces the shared-seed defect or changes the
  §5.1 mixture -- the realised median, IQR and maximum.
"""

import numpy as np
from scipy.stats import lognorm

from src.dp_gibbs import (
    DPPrior,
    DPSettings,
    make_surv_data,
    r_hat,
    sample_chains,
)

# Plan Section 6.1: p LN(0, 0.25) + (1-p) LN(1.2, 0.02), n = 200, no censoring.
MIXTURE = {
    "mu_1": 0.0,
    "mu_2": 1.2,
    "var_1": 0.25,
    "var_2": 0.02,
    "p": 0.8,
    "n": 200,
    "seed": 721383,
}

# Plan Section 5.1: Gamma(2, 0.9) on nu, with the prior-guess b_gamma/b_phi.
PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)

# Plan Section 6.2, ``quick`` preset.
QUICK = {
    "burn_in": 1_000,
    "n_save": 1_000,
    "thin": 10,
    "n_alpha": 256,
    "epsilon": 1e-4,
    "n_chains": 4,
    "seed": 20241101,
    "preset": "quick",
}

R_HAT_LIMIT = 1.05


def _section_5_1_sample() -> np.ndarray:
    """Draw the §5.1 lognormal mixture with one shared generator.

    The two components must consume the *same* ``Generator`` object, not the
    same integer seed: the legacy class's shared integer seed makes the second
    component a deterministic function of the first component's draws (plan
    Section 6.1).
    """
    rng = np.random.default_rng(MIXTURE["seed"])
    n_1 = round(MIXTURE["p"] * MIXTURE["n"])
    n_2 = MIXTURE["n"] - n_1
    first = lognorm.rvs(
        s=np.sqrt(MIXTURE["var_1"]),
        scale=np.exp(MIXTURE["mu_1"]),
        size=n_1,
        random_state=rng,
    )
    second = lognorm.rvs(
        s=np.sqrt(MIXTURE["var_2"]),
        scale=np.exp(MIXTURE["mu_2"]),
        size=n_2,
        random_state=rng,
    )
    return np.append(first, second)


def test_section_5_1_sample_matches_the_plan() -> None:
    """The generated sample is the one the plan's §5.1 numbers describe."""
    time = _section_5_1_sample()
    q_25, q_75 = np.percentile(time, [25.0, 75.0])

    assert time.size == MIXTURE["n"]
    assert np.isclose(np.median(time), 1.2172, atol=1e-4)
    assert np.isclose(q_75 - q_25, 1.5336, atol=1e-4)
    assert np.isclose(time.max(), 5.1390, atol=1e-4)


def test_four_chains_mix_on_the_section_5_1_data() -> None:
    """Four ``quick`` chains pass split R-hat and visit several ``n*`` values."""
    time = _section_5_1_sample()
    data = make_surv_data(time=time, event=np.ones(time.size, dtype=bool))
    settings = DPSettings(**QUICK)  # type: ignore[arg-type]

    chains = sample_chains(data, PRIOR, [settings])

    assert len(chains) == QUICK["n_chains"]
    assert all(chain.settings.seed == QUICK["seed"] for chain in chains)

    nu_r_hat = r_hat(chains, "nu")["nu"]
    n_star_r_hat = r_hat(chains, "n_star")["n_star"]
    survival = r_hat(chains, "S")
    survival_r_hat = np.asarray(survival["S"])

    assert isinstance(nu_r_hat, float)
    assert isinstance(n_star_r_hat, float)
    assert nu_r_hat < R_HAT_LIMIT
    assert n_star_r_hat < R_HAT_LIMIT
    assert survival_r_hat.shape == (5,)
    assert np.all(survival_r_hat < R_HAT_LIMIT)

    for chain in chains:
        assert np.unique(chain.n_star).size > 1
