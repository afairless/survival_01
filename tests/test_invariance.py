"""Test 19 (plan Sections 7.3 and 10): data-order invariance of the chain.

The collapsed urn of the Appendix updates the observations **sequentially**,
``i = 0 .. n-1``, each time removing observation ``i`` from the partition and
then re-adding it.  A sequential update is legitimate only if the transition
leaves the posterior invariant under a relabelling of the observations: the
data are a set, so permuting their order must not move the posterior.  A bug
that makes the answer depend on the order -- a stale ``q_j^{i}``, a column of
the cached ``log q`` matrix left unpermuted, a cluster count read from the
wrong index, a stale grid row -- still produces smooth, plausible chains and
is invisible to every single-conditional test.

The invariance is statistical, not bitwise.  Permuting the data attaches the
same random numbers to different observations, so the two chains are two
independent draws from the *same* posterior; the test therefore compares
posterior summaries within three Monte Carlo standard errors, exactly as plan
Section 7.3 specifies:

* the posterior means of ``S(t)`` at five times on the original scale;
* the posterior mean of ``nu``;
* the distribution of ``n*``, binned coarsely (see below).

Monte Carlo standard errors are batch means (``N_BATCHES = 20`` blocks of the
stored draws) rather than ``sd / sqrt(B)``: the stored draws are thinned but
still autocorrelated, and a naive standard error would understate the noise
and make the test reject a correct sampler.

``n*`` is compared through bin probabilities, not per-value ones.  A per-value
Wald standard error is unusable where the probability is a few observations
wide (it is noisy and collapses to zero for an unvisited value), and the
support of ``n*`` for ``n = 20`` reaches well past the range either chain
visits.  The bins ``{1,2}, {3,4}, {5,6}, {7,8}, {9,10}, {11,12}, {13,15},
{16,20}`` keep every probability ``O(0.05-0.4)``, so the batch-means standard
error is trustworthy while the shape of the distribution, not only its mean,
is still pinned.  The mean of ``n*`` is asserted as well.

What each test would catch:

* a sampler whose answer depends on the row order of the data -- the survival,
  ``nu`` and ``n*`` comparisons;
* a difference introduced at ingestion rather than in the sampler -- the
  per-observation transform test, which is exact: permuting the raw arrays
  permutes the transformed arrays and leaves every value unchanged;
* a chain that is not deterministic from ``settings.seed`` -- the replay test,
  without which the comparisons above would not even be a well-defined
  property of the code.
"""

import numpy as np
import pytest
from numpy.typing import ArrayLike

from src.dp_gibbs import (
    DPChains,
    DPPrior,
    DPSettings,
    SurvData,
    make_surv_data,
    sample_chain,
    transform_time,
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

# Mixed censoring on the original scale, with repeated and near-repeated
# gaps, so the sequential update has something to get wrong.
TIME = np.array(
    [
        0.4,
        0.8,
        1.0,
        1.6,
        2.4,
        3.0,
        4.5,
        5.2,
        6.1,
        7.3,
        8.6,
        9.9,
        1.2,
        2.0,
        3.6,
        4.1,
        5.8,
        6.7,
        8.0,
        9.2,
    ]
)
EVENT = np.array(
    [
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
)

SEED = 20241007
N_ALPHA = 40
BURN_IN = 400
N_SAVE = 800
THIN = 2

N_BATCHES = 20
MCSE_TOLERANCE = 3.0
SURVIVAL_TIMES = np.array([0.5, 1.5, 3.0, 6.0, 9.0])
N_STAR_BIN_EDGES = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 16.0, 21.0])


def _data() -> SurvData:
    """The reference dataset, ingested once on the transformed scale."""
    return make_surv_data(TIME, EVENT)


def _settings(**overrides: object) -> DPSettings:
    """Long-enough-to-mix, cheap-enough-for-the-fast-suite settings."""
    base: dict[str, object] = {
        "burn_in": BURN_IN,
        "n_save": N_SAVE,
        "thin": THIN,
        "n_alpha": N_ALPHA,
        "n_chains": 1,
        "seed": SEED,
        "preset": "test",
    }
    base.update(overrides)
    return DPSettings(**base)  # type: ignore[arg-type]


def _permutation(seed: int, n: int) -> np.ndarray:
    """A fixed, seeded permutation of ``0..n-1``."""
    return np.random.default_rng(seed).permutation(n)


def _survival_draws(chain: DPChains, times_tilde: np.ndarray) -> np.ndarray:
    """Per-draw ``S(t)`` of the observation-level mixture, ``(B, T)``.

    ``S(t) = n**-1 sum_i exp(-t**alpha_i / lambda_i)``: the functional is
    defined by the mixture of the stored ``theta_i`` (plan Section 4.1), which
    is why no cluster labels are needed.  ``times_tilde`` are on the
    transformed scale; ``S`` itself is unchanged by the transform.
    """
    alpha = chain.theta[:, :, 0]
    lam = chain.theta[:, :, 1]
    kernel = np.exp(
        -(times_tilde[None, None, :] ** alpha[:, :, None]) / lam[:, :, None]
    )
    return kernel.mean(axis=1)


def _batch_mcse(values: np.ndarray, n_batches: int = N_BATCHES) -> np.ndarray:
    """Batch-means Monte Carlo standard error of the mean.

    ``values`` is ``(B,)`` or ``(B, T)``.  The draws are split into
    ``n_batches`` contiguous blocks; the standard deviation of the block means
    divided by ``sqrt(n_batches)`` is the standard error.  Unlike
    ``sd / sqrt(B)`` this stays honest when consecutive draws are correlated:
    a longer autocorrelation time inflates the spread of the block means.
    """
    array = np.asarray(values, dtype=np.float64)
    size = array.shape[0] // n_batches
    if size < 2:
        raise ValueError("not enough draws for batch means; use at least two blocks")
    batches = array[: size * n_batches].reshape((n_batches, size) + array.shape[1:])
    return batches.mean(axis=1).std(axis=0, ddof=1) / np.sqrt(n_batches)


def _difference_in_mcse(
    mean_a: ArrayLike,
    mean_b: ArrayLike,
    mcse_a: ArrayLike,
    mcse_b: ArrayLike,
) -> np.ndarray:
    """``|mean_a - mean_b|`` in units of the two means' combined standard error."""
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    se = np.sqrt(np.square(mcse_a) + np.square(mcse_b))
    difference = np.abs(mean_a - mean_b)
    return np.divide(
        difference,
        se,
        out=np.zeros_like(difference, dtype=np.float64),
        where=se > 0.0,
    )


def _n_star_bin_probabilities(n_star: np.ndarray) -> np.ndarray:
    """Per-draw indicator of each ``n*`` bin, ``(B, n_bins)``."""
    edges = N_STAR_BIN_EDGES
    indices = np.digitize(np.asarray(n_star), edges[1:-1])
    return np.stack(
        [(indices == k).astype(np.float64) for k in range(edges.size - 1)], axis=1
    )


@pytest.fixture(scope="module")
def order_pair() -> tuple[SurvData, SurvData, DPChains, DPChains]:
    """Run the two chains once for every order-invariance assertion.

    Both chains use the *same* seed and the same settings; only the row order
    of the data differs.  Both are expensive (thousands of sweeps), so they
    are built once per module rather than once per test.
    """
    data = _data()
    order = _permutation(SEED + 1, TIME.size)
    permuted = make_surv_data(TIME[order], EVENT[order])
    settings = _settings()
    original = sample_chain(data, PRIOR, settings)
    shuffled = sample_chain(permuted, PRIOR, settings)
    return data, permuted, original, shuffled


class TestIngestionOrderInvariance:
    """Permuting the raw data permutes the transformed data and nothing else."""

    @pytest.mark.parametrize("order_seed", [0, 1, 2, 3, 42])
    def test_permuted_ingestion_maps_each_observation_identically(
        self, order_seed: int
    ):
        """``make_surv_data`` is row-wise: same value, same time, same event.

        This is exact, not statistical, and it isolates the chain comparison:
        any difference between the permuted and original chains comes from the
        sequential sampler, never from the transform, the recorded ``t_max`` or
        a reordering of the censoring indicators.
        """
        data = _data()
        order = _permutation(order_seed, TIME.size)
        permuted = make_surv_data(TIME[order], EVENT[order])

        assert permuted.t_max == data.t_max
        assert np.array_equal(permuted.time, data.time[order])
        assert np.array_equal(permuted.event, data.event[order])
        assert np.array_equal(transform_time(permuted, TIME[order]), data.time[order])


class TestChainOrderInvariance:
    """Posterior summaries agree within three Monte Carlo standard errors."""

    def test_survival_function_means_agree(
        self, order_pair: tuple[SurvData, SurvData, DPChains, DPChains]
    ):
        """``E[S(t)]`` at five times is unaffected by the observation order."""
        data, _, original, shuffled = order_pair
        times_tilde = 1.0 + SURVIVAL_TIMES / data.t_max
        survival_a = _survival_draws(original, times_tilde)
        survival_b = _survival_draws(shuffled, times_tilde)

        z = _difference_in_mcse(
            survival_a.mean(axis=0),
            survival_b.mean(axis=0),
            _batch_mcse(survival_a),
            _batch_mcse(survival_b),
        )
        assert np.all(z <= MCSE_TOLERANCE), (
            "posterior means of S(t) depend on the observation order: "
            f"|difference|/MCSE = {np.round(z, 3).tolist()} at original times "
            f"{SURVIVAL_TIMES.tolist()}"
        )

    def test_nu_mean_agrees(
        self, order_pair: tuple[SurvData, SurvData, DPChains, DPChains]
    ):
        """The DP precision's posterior mean is unaffected by the order."""
        _, _, original, shuffled = order_pair
        z = _difference_in_mcse(
            original.nu.mean(),
            shuffled.nu.mean(),
            _batch_mcse(original.nu),
            _batch_mcse(shuffled.nu),
        )
        assert z <= MCSE_TOLERANCE, (
            "the posterior mean of nu depends on the observation order: "
            f"|difference|/MCSE = {float(z):.3f}, "
            f"means = ({original.nu.mean():.4f}, {shuffled.nu.mean():.4f})"
        )

    def test_n_star_distribution_agrees(
        self, order_pair: tuple[SurvData, SurvData, DPChains, DPChains]
    ):
        """The distribution of ``n*``, not just its mean, is order invariant."""
        _, _, original, shuffled = order_pair
        probabilities_a = _n_star_bin_probabilities(original.n_star)
        probabilities_b = _n_star_bin_probabilities(shuffled.n_star)
        assert np.allclose(probabilities_a.sum(axis=1), 1.0)
        assert np.allclose(probabilities_b.sum(axis=1), 1.0)

        z = _difference_in_mcse(
            probabilities_a.mean(axis=0),
            probabilities_b.mean(axis=0),
            _batch_mcse(probabilities_a),
            _batch_mcse(probabilities_b),
        )
        assert np.all(z <= MCSE_TOLERANCE), (
            "the distribution of n* depends on the observation order: "
            f"|difference|/MCSE per bin = {np.round(z, 3).tolist()} for bins "
            f"{N_STAR_BIN_EDGES.tolist()}"
        )

        mean_z = _difference_in_mcse(
            original.n_star.mean(),
            shuffled.n_star.mean(),
            _batch_mcse(original.n_star.astype(np.float64)),
            _batch_mcse(shuffled.n_star.astype(np.float64)),
        )
        assert mean_z <= MCSE_TOLERANCE, (
            "the posterior mean of n* depends on the observation order: "
            f"|difference|/MCSE = {float(mean_z):.3f}, "
            f"means = ({original.n_star.mean():.4f}, {shuffled.n_star.mean():.4f})"
        )


class TestReplay:
    """The comparisons above must be properties of the code, not of the run."""

    def test_the_permuted_run_replays_bit_for_bit(self):
        """Same permuted data, same seed -> identical stored draws.

        The invariance test difference would be meaningless if a chain
        replayed differently from its own stored ``settings``; this pins the
        determinism the comparison relies on.
        """
        order = _permutation(SEED + 1, TIME.size)
        permuted = make_surv_data(TIME[order], EVENT[order])
        settings = _settings(burn_in=20, n_save=40, thin=1, n_alpha=16)

        first = sample_chain(permuted, PRIOR, settings)
        second = sample_chain(permuted, PRIOR, settings)

        assert np.array_equal(first.theta, second.theta)
        assert np.array_equal(first.nu, second.nu)
        assert np.array_equal(first.gamma, second.gamma)
        assert np.array_equal(first.phi, second.phi)
        assert np.array_equal(first.n_star, second.n_star)
        assert np.array_equal(first.log_post, second.log_post)
