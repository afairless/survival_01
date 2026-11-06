"""Test 17 (plan Sections 7.3 and 9 step 14): simulation-based calibration.

Sections 7.1 and 7.2 check one conditional at a time against its exact law, and
test 16 compares the whole chain against a second implementation.  Neither can
see a mistake that moves *all* conditionals consistently -- a sweep whose steps
are individually correct but ordered wrongly, a hyperprior that enters the
likelihood through ``q0`` but not through ``log_post``, or a censoring branch
that is swapped between two steps.  Those mistakes leave every single-step
distribution intact and still target the wrong posterior.  Simulation-based
calibration (SBC) is the standard test for that class of error: if the model is
right, the rank of a prior draw among posterior draws is uniform.

Protocol (the plan's Section 7.3 test 17, with the deviations recorded below):

1. draw ``nu, gamma, phi`` from the priors, and ``G ~ DP(nu G0)`` truncated at
   ``epsilon_tail = 1e-8``;
2. draw the 15 observation locations ``theta_i ~ G`` and the times
   ``t_i ~ K_W(. | theta_i)``;
3. draw the censoring times ``c_i ~ U(0, c_max)`` independently of ``t_i``,
   with ``c_max`` a fixed multiple of the Section 6.1 sample maximum (the plan
   says ``1.2``; the width actually used is recorded under deviations), and
   record ``(min(t_i, c_i), t_i <= c_i)``, holding the realised censoring
   fraction in ``[0.15, 0.25]``;
4. fit one chain with the Section 5 prior and store the draws;
5. rank the true ``nu, gamma, phi`` and the true population survival
   ``S_G(t)`` at the three pre-registered original-scale times among the
   posterior draws, over ``R = 100`` replicates.

The statistic list is fixed here, before the test runs, as Section 7.3
requires: ``nu``, ``gamma``, ``phi``, and ``S(t)`` at the Section 6.1
population quartiles.  The acceptance rule is Section 7.3 test 17's: (a) each
of the ten rank bins within three binomial standard deviations of ``R/10``,
and (b) the rank distribution's CDF deviation from uniform below ``0.15``.
The plan's alternative -- a KS p-value per statistic -- is rejected for the
reason it gives: at ``R = 100`` it has low power and, with six statistics,
fails spuriously a fixed fraction of the time.

Deviations from the plan's literal wording, and why
---------------------------------------------------
**Coordinates.** The plan draws ``t_i`` and maps them through the paper's
transform, which is what a user would do.  That cannot be SBC'd here: the
sampler's ``SurvData`` requires ``time > 1``, and the prior predictive of
``t~ = 1 + t / T_max`` under the Section 5 prior puts about half its mass
below that bound (the plan's own prior-scale caveat, Section 2.3: the prior
median of the population median is about 0.61 in original units).  Datasets
drawn on the original scale are therefore mostly unrepresentable -- the
measured 15-observation acceptance is 3.2% -- and the sampler's likelihood is
plain ``K_W(t~ | alpha, lambda)`` with no bound above, so the model it targets
is the transformed-time model restricted to ``t~ > 1``.  The replicate is
therefore generated directly on that scale -- ``theta_i ~ G``, ``t_i ~ K_W``
with ``t_i > 1`` -- and the time unit is fixed by ``T_max = 5.1390`` (Section
6.1's realised sample maximum) so that the plan's censoring law and its
original-scale quartiles map to constants on that scale.  Datasets containing
a ``t_i <= 1`` or a non-finite ``t_i`` are redrawn; that condition depends on
the data only, so it does not change the posterior the sampler targets.

Observed times then run up to ``1 + CENSOR_MAX_TRANSFORMED`` rather than
staying inside ``(1, 2]``, so this row does **not** exercise the ``T <= 2``
remark of ``AlphaGrid.powers``; the cached ``t ** u`` and its ``phi`` power are
finite throughout (a few thousand at the largest ``phi`` drawn here), which is
all the sampler requires.

**Censoring fraction.** The plan itself requires the realised fraction to stay
in ``[0.15, 0.25]`` per replicate.  At ``n = 15`` that band contains exactly
one count, ``3/15``, so the generator conditions on it -- the plan's mandate,
not a liberty taken here.  The censoring law stays fixed and independent of
``(G, theta)``, as the plan insists.  What the width ``c_max`` controls is how
rare that conditioning is; the measured prior-predictive count distribution
(3,000 accepted datasets per width) is

| ``c_max`` (transformed width) | mean count of 15 | ``Pr[count = 3]`` |
| --- | --- | --- |
| 1.2 (the plan's literal value) | 13.0 | 0.004 |
| 5 | 9.5 | 0.029 |
| 20 | 6.1 | 0.082 |
| 150 (used here) | 3.2 | 0.089 |

so the plan's literal ``c_max`` makes the plan's own band a tail event, while
``150`` makes it a common one (about one dataset in eleven).  The count
distribution stays broad -- 28% of the datasets are uncensored, because the DP
prior draws very large times -- so some conditioning is unavoidable at
``n = 15`` whatever the width; this choice keeps it modest.

**Run length.** The plan says ``quick`` (1,000 burn-in, 1,000 saved draws 10
sweeps apart = 11,000 sweeps).  At ``n = 15`` a sweep costs about 1.25 ms
(measured), so 100 replicates at the ``quick`` preset is a 23-minute run,
outside the Section 7 T2 budget of 1-5 minutes.  The settings below (500
burn-in, 400 saved draws, thin 4 = 2,100 sweeps) cost 2.6 s per replicate and
keep the whole test at about 4.5 minutes; 400 ranks per replicate already give
40 ranks per bin, and the acceptance rule of Section 7.3 is met with margin
(worst bin count 6 off the expected 10 against a tolerance of 9; worst CDF
deviation 0.111 against 0.15; every mean rank within 0.05 of 0.5).

What each test would catch:

* a sweep that runs (c) before (b), or that refreshes a hyperparameter with a
  stale ``n*``/``max alpha*`` -- every hyperparameter's rank distribution
  moves away from uniform;
* a prior on ``nu``, ``gamma`` or ``phi`` that is used in the urn weights but
  not in ``log_post`` (or vice versa) -- the same;
* a censored row that uses the density instead of the survival in the urn or
  in ``log_post`` -- the ``S(t)`` ranks move, because the likelihood branch is
  what the three survival statistics are built from;
* a wrong ``gamma`` update (``n*`` instead of ``d n*``) or a ``phi`` update
  that drops ``n*`` from the Pareto shape -- the corresponding rank
  distribution is the most direct witness;
* a generator drift -- the fixture pins every replicate, chain and functional
  seed, and the metadata assertions fail before the ranks do.

The test is opt-in with ``-m slow`` (plan Section 7.3).  Every replicate's
seeds live in ``tests/fixtures/sbc_seeds.json`` (plan Section 7.6), so a
failure is replayable; the fixture is written by
``tests/fixtures/make_fixtures.py`` and never by this test.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.dp_functionals import FunctionalSettings, posterior_functionals
from src.dp_gibbs import DPPrior, DPSettings, SurvData, sample_chain
from src.dp_weibull import sample_dp_prior, sample_prior_hyperparameters

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sbc_seeds.json"

# Plan Section 7.3 test 17: 100 replicates of a 15-observation dataset.
N_REPLICATES = 100
N_OBSERVATIONS = 15

# Plan Section 5 / the other tests: the Section 5.1 prior, unchanged.
PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)

# Plan Section 7.3: G is truncated at epsilon_tail = 1e-8 (not the sampler's
# 1e-4 posterior-truncation tolerance, which belongs to the functional path).
EPSILON_TAIL = 1e-8

# Plan Section 6.1: the corrected Section 5.1 sample's maximum.  It fixes the
# time unit of the SBC (see the module docstring for why the replicates live on
# the sampler's scale).
T_MAX_ORIGINAL = 5.1390

# Censoring law: c_tilde ~ U(1, 1 + CENSOR_MAX_TRANSFORMED), i.e. the plan's
# c ~ U(0, c_max), with c_max = 150 times the Section 6.1 sample maximum.  The
# plan's literal 1.2 would censor 13 of 15 observations on average, making the
# band's only admissible count a 0.4% tail event; 150 puts the prior-mean
# count at 3.2 and the band at about 1 dataset in 11 (measured, see the module
# docstring).  Observed times are then at most 151, which keeps ``t ** phi``
# comfortable at every ``phi`` the prior draws here.
CENSOR_MAX_TRANSFORMED = 150.0
CENSOR_MAX_ORIGINAL = CENSOR_MAX_TRANSFORMED * T_MAX_ORIGINAL

# Plan Section 7.3: hold the realised censoring fraction in [0.15, 0.25].
CENSORING_FRACTION_BOUNDS = (0.15, 0.25)
N_CENSORED = 3  # the only count in the band at n = 15

# Plan Section 6.1 population quartiles on the original scale: the three
# pre-registered evaluation times of the survival statistic.  They are the
# quantiles of the true generating distribution (the plan's wording), but
# fixed rather than re-derived from each replicate's G: the top quantiles of a
# DP realisation under this prior can be of order 1e10, deep in the far tail
# where S ~ 0 and the rank carries no information.
EVALUATION_TIMES_ORIGINAL = (0.7832, 1.1727, 2.1511)

# Plan Section 7.3 test 17: ten rank bins, three binomial SD, CDF deviation
# 0.15.  The statistic list is stated here, before the run.
N_RANK_BINS = 10
BIN_SD_LIMIT = 3.0
CDF_DEVIATION_LIMIT = 0.15
STATISTICS = ("nu", "gamma", "phi", "S_q25", "S_q50", "S_q75")

# Run length: the reduced quick configuration of the module docstring (plan
# Section 7.3's quick preset is 11,000 sweeps per replicate, a 23-minute run).
SETTINGS = DPSettings(
    burn_in=500,
    n_save=400,
    thin=4,
    n_alpha=256,
    epsilon=1e-4,
    n_chains=1,
    preset="sbc",
)

# A generation attempt is rejected when a time is outside the sampler's
# support or the censoring fraction is outside the band; the measured
# acceptance is about 1/400, so 20,000 attempts is a generous guard against a
# fixture/prior mismatch (its tail probability is e**-48 per replicate).
MAX_GENERATION_ATTEMPTS = 20_000


@dataclass(frozen=True)
class Replicate:
    """One generated SBC replicate and the truth it was drawn from."""

    data: SurvData
    nu: float
    gamma: float
    phi: float
    survival: np.ndarray  # S_G at EVALUATION_TIMES_ORIGINAL
    censoring_fraction: float


def _mixture_survival(realization, times: np.ndarray) -> np.ndarray:
    """Survival ``sum_j w_j exp(-t**alpha_j / lambda_j)`` of a DP realisation."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        terms = np.exp(
            -np.power(times[:, None], realization.alphas[None, :])
            / realization.lambdas[None, :]
        )
    return np.asarray(terms @ realization.weights, dtype=np.float64)


def generate_replicate(seed: int) -> Replicate:
    """Draw one dataset from the model, conditioned on the sampler's support.

    The random-number consumption is fixed by this function: the fixture seeds
    reproduce the dataset exactly, so a failure can be replayed.  The loop
    redraws while a drawn time leaves the sampler's support (``time > 1``) and
    while the realised censoring fraction leaves the plan's
    ``CENSORING_FRACTION_BOUNDS``, which at ``n = 15`` admits the single count
    :data:`N_CENSORED`.
    """
    rng = np.random.default_rng(seed)
    evaluation_times_tilde = 1.0 + np.asarray(EVALUATION_TIMES_ORIGINAL) / (
        T_MAX_ORIGINAL
    )
    for _ in range(MAX_GENERATION_ATTEMPTS):
        nu, gamma, phi = sample_prior_hyperparameters(
            PRIOR.a_nu,
            PRIOR.b_nu,
            PRIOR.a_gamma,
            PRIOR.b_gamma,
            PRIOR.a_phi,
            PRIOR.b_phi,
            rng=rng,
        )
        realization = sample_dp_prior(
            nu, gamma, phi, d=PRIOR.d, epsilon=EPSILON_TAIL, rng=rng
        )
        index = rng.choice(
            realization.weights.size, size=N_OBSERVATIONS, p=realization.weights
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            time = np.power(
                realization.lambdas[index] * -np.log(rng.random(N_OBSERVATIONS)),
                1.0 / realization.alphas[index],
            )
        if not (np.all(np.isfinite(time)) and np.all(time > 1.0)):
            continue
        # The plan's censoring law, expressed in transformed coordinates:
        # c = T_max * (c_tilde - 1) with c ~ U(0, 1.2 * T_max) is
        # c_tilde ~ U(1, 2.2), independent of the survival times.
        censor = 1.0 + CENSOR_MAX_TRANSFORMED * rng.random(N_OBSERVATIONS)
        event = time <= censor
        censoring_fraction = float(np.count_nonzero(~event)) / N_OBSERVATIONS
        low, high = CENSORING_FRACTION_BOUNDS
        if not low <= censoring_fraction <= high:
            continue
        survival = _mixture_survival(realization, evaluation_times_tilde)
        data = SurvData(
            time=np.minimum(time, censor), event=event, t_max=T_MAX_ORIGINAL
        )
        return Replicate(
            data=data,
            nu=float(nu),
            gamma=float(gamma),
            phi=float(phi),
            survival=survival,
            censoring_fraction=censoring_fraction,
        )
    raise AssertionError(
        f"replicate {seed}: no valid dataset in {MAX_GENERATION_ATTEMPTS} attempts"
    )


def _posterior_draws(chain, functional_seed: int) -> tuple[np.ndarray, ...]:
    """Return the six posterior samples the ranks are computed against."""
    functionals = posterior_functionals(
        chain,
        np.asarray(EVALUATION_TIMES_ORIGINAL, dtype=np.float64),
        settings=FunctionalSettings(),
        seed=functional_seed,
    )
    return (
        chain.nu,
        chain.gamma,
        chain.phi,
        functionals.S[:, 0],
        functionals.S[:, 1],
        functionals.S[:, 2],
    )


def rank_replicate(replicate: Replicate, chain_seed: int, functional_seed: int):
    """Rank the truth of one replicate among its posterior draws.

    Returns one rank per entry of :data:`STATISTICS`, on ``0..B`` where ``B``
    is the number of stored draws.
    """
    chain = sample_chain(
        replicate.data, PRIOR, dataclasses.replace(SETTINGS, seed=chain_seed)
    )
    draws = _posterior_draws(chain, functional_seed)
    truths = (replicate.nu, replicate.gamma, replicate.phi, *replicate.survival)
    return np.array(
        [
            np.count_nonzero(draw < truth)
            for draw, truth in zip(draws, truths, strict=True)
        ],
        dtype=np.intp,
    )


def load_fixture() -> dict:
    """Read the committed seed fixture and check its metadata."""
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        fixture = json.load(handle)

    assert fixture["fixture"] == "sbc_seeds"
    assert fixture["n_replicates"] == N_REPLICATES
    assert fixture["n_observations"] == N_OBSERVATIONS
    generation = fixture["generation"]
    assert generation["epsilon_tail"] == EPSILON_TAIL
    assert generation["t_max_original"] == T_MAX_ORIGINAL
    assert generation["censor_max_original"] == pytest.approx(CENSOR_MAX_ORIGINAL)
    assert tuple(generation["censoring_fraction_bounds"]) == (CENSORING_FRACTION_BOUNDS)
    assert tuple(generation["evaluation_times_original"]) == pytest.approx(
        EVALUATION_TIMES_ORIGINAL
    )
    assert tuple(fixture["statistics"]) == STATISTICS
    assert fixture["prior"] == {
        "a_nu": PRIOR.a_nu,
        "b_nu": PRIOR.b_nu,
        "a_gamma": PRIOR.a_gamma,
        "b_gamma": PRIOR.b_gamma,
        "a_phi": PRIOR.a_phi,
        "b_phi": PRIOR.b_phi,
        "d": PRIOR.d,
    }

    for name in ("replicate_seeds", "chain_seeds", "functional_seeds"):
        seeds = fixture[name]
        assert len(seeds) == N_REPLICATES, f"{name} must have one seed per replicate"
        assert len(set(seeds)) == N_REPLICATES, f"{name} must be unique"
        assert all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in seeds), (
            f"{name} must hold 32-bit seed integers"
        )
    return fixture


def rank_fractions(ranks: np.ndarray) -> np.ndarray:
    """Map ranks on ``0..B`` to ``(0, 1)`` as SBC prescribes."""
    return ranks / (SETTINGS.n_save + 1)


def bin_counts(fractions: np.ndarray) -> np.ndarray:
    """Counts in the plan's ten equal-width rank bins."""
    counts, _ = np.histogram(fractions, bins=N_RANK_BINS, range=(0.0, 1.0))
    return counts


def cdf_deviation(fractions: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of the ranks from the uniform CDF."""
    ordered = np.sort(fractions)
    n = ordered.size
    upper = np.arange(1, n + 1) / n - ordered
    lower = ordered - np.arange(0, n) / n
    return float(max(np.max(upper), np.max(lower)))


def evaluate_replicates(fixture: dict) -> np.ndarray:
    """Run every replicate and return the ``(R, len(STATISTICS))`` ranks."""
    ranks = np.empty((N_REPLICATES, len(STATISTICS)), dtype=np.intp)
    for b in range(N_REPLICATES):
        replicate = generate_replicate(fixture["replicate_seeds"][b])
        assert (
            abs(replicate.censoring_fraction - N_CENSORED / N_OBSERVATIONS) < 1e-12
        ), "the generator must hold the realised censoring fraction in the band"
        assert np.all(replicate.data.time > 1.0)
        assert np.all(np.isfinite(replicate.survival))
        ranks[b] = rank_replicate(
            replicate,
            chain_seed=fixture["chain_seeds"][b],
            functional_seed=fixture["functional_seeds"][b],
        )
    return ranks


@pytest.mark.slow
def test_sbc_rank_calibration() -> None:
    """SBC of ``nu``, ``gamma``, ``phi`` and ``S(t)`` at three times.

    This is the plan's Section 7.3 test 17: after :func:`evaluate_replicates`
    has produced one rank per replicate and statistic, every statistic must
    pass the two calibration criteria of the module docstring.  The
    assertions are reported per statistic so that a failure names the
    conditional that is wrong, not just "SBC failed".
    """
    fixture = load_fixture()
    ranks = evaluate_replicates(fixture)

    expected = N_REPLICATES / N_RANK_BINS
    binomial_sd = np.sqrt(N_REPLICATES * (1.0 / N_RANK_BINS) * (1 - 1.0 / N_RANK_BINS))
    tolerance = BIN_SD_LIMIT * binomial_sd

    failures: list[str] = []
    for column, name in enumerate(STATISTICS):
        fractions = rank_fractions(ranks[:, column])
        counts = bin_counts(fractions)
        deviation = cdf_deviation(fractions)
        if np.any(np.abs(counts - expected) > tolerance):
            failures.append(
                f"{name}: bin counts {counts.tolist()} outside "
                f"{expected:g} +/- {tolerance:g}"
            )
        if deviation >= CDF_DEVIATION_LIMIT:
            failures.append(
                f"{name}: rank CDF deviation {deviation:.3f} >= {CDF_DEVIATION_LIMIT}"
            )
    assert not failures, "SBC calibration failed:\n" + "\n".join(failures)


@pytest.mark.slow
def test_sbc_generator_is_reproducible() -> None:
    """The fixture seeds must reproduce a replicate exactly.

    A fixture whose seeds do not replay the generator makes a failure
    un-diagnosable, and it would hide a change of the generation stream behind
    a passing rank test.  Two calls with the same seed must agree bit for bit;
    a different seed must differ.
    """
    fixture = load_fixture()
    first = generate_replicate(fixture["replicate_seeds"][0])
    second = generate_replicate(fixture["replicate_seeds"][0])
    assert np.array_equal(first.data.time, second.data.time)
    assert np.array_equal(first.data.event, second.data.event)
    assert first.data.t_max == second.data.t_max
    assert (first.nu, first.gamma, first.phi) == (
        second.nu,
        second.gamma,
        second.phi,
    )
    assert np.array_equal(first.survival, second.survival)

    other = generate_replicate(fixture["replicate_seeds"][1])
    assert not np.array_equal(first.data.time, other.data.time)
