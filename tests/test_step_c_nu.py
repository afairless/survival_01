"""Test 12 (plan Section 7.2): the Escobar-West update of the DP precision.

Step (c) of the Appendix refreshes the hyperparameters in the order ``nu``,
``phi``, ``gamma``.  The first is the only non-conjugate one: with ``a_nu``
and ``b_nu`` the shape and **rate** of the Gamma prior on ``nu``, ``n`` the
number of observations and ``n_star`` the current number of clusters, plan
Section 2.2(c) specifies

    ``u ~ Beta(nu + 1, n)``                       (the current ``nu``)
    ``p = (a_nu + n_star - 1) / (n (b_nu - log u) + a_nu + n_star - 1)``
    ``nu ~ p Gamma(a_nu + n_star, rate = b_nu - log u)
           + (1 - p) Gamma(a_nu + n_star - 1, rate = b_nu - log u)``.

Two exact statements are checked, each against a reference written directly
from those lines with SciPy -- never from the sampler's arithmetic:

1. **The auxiliary.**  At a fixed current ``nu`` the returned ``u`` values
   are a sample from ``Beta(nu + 1, n)``; a Kolmogorov-Smirnov test against
   ``scipy.stats.beta(nu + 1, n)`` rejects a draw made at the prior mean of
   ``nu`` instead of the current value.

2. **The two-Gamma mixture.**  The sample is from the *joint* law of
   ``(u, nu)``, so the mixture CDF is conditional on the ``u`` of the same
   draw.  Evaluating that conditional CDF at ``nu`` defines a probability
   integral transform that is exactly uniform when, and only when, each
   ``nu | u`` follows the two-Gamma mixture above.  A KS test of the transform
   against ``U(0, 1)`` is therefore the exact-mixture-CDF check the plan asks
   for, without a second numerical integration over ``u``.  (Each draw's
   reference CDF uses its own ``u``; using one ``u`` for the whole sample
   would test a different law.)

What each assertion would catch (plan Section 7.7, row "nu: Escobar-West
p-weight or the n* vs n*-1 index"):

* the PIT test rejects a wrong ``p`` weight, the two shapes swapped, a
  ``n_star``/``n_star - 1`` index error in either the weight or a shape, a
  missing ``- log u`` in the rate, and a Gamma rate treated as a scale;
* the auxiliary test rejects ``u ~ Beta(1, n)``, ``Beta(a_nu + 1, n)`` or any
  other shape that does not use the *current* ``nu``;
* the support test rejects a draw that is not finite and positive, which is
  the only way a correct-looking trace can still poison step (a)'s ``q_0``.

Regression: every case uses a fixed seed, so a failure is reproducible.  The
KS thresholds are effect-size decisions at 1e5 draws (a p-value of 1e-3, not
0.05): at this sample size a correct implementation still has a 0.1% chance
per statistic, and the seeds are fixed so the suite is deterministic.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import ks_1samp, uniform

from src.dp_gibbs import DPPrior, DPState, StateInvariantError, refresh_nu

N_DRAWS = 100_000
# At 1e5 draws a KS p-value is not a 5% decision: it rejects deviations of a
# few 1e-3, far below any sampling error we care about.  The plan asks for a
# KS test, so keep one, but set the threshold as an effect-size decision.
KS_P_THRESHOLD = 1e-3


@dataclass(frozen=True)
class NuCase:
    """One fixed state and prior for the conditional ``[nu | u, n, n_star]``."""

    n: int
    n_star: int
    nu: float
    a_nu: float
    b_nu: float


CASES = [
    pytest.param(NuCase(n=200, n_star=15, nu=2.0, a_nu=2.0, b_nu=0.9), id="n200-k15"),
    pytest.param(NuCase(n=20, n_star=3, nu=1.0, a_nu=2.0, b_nu=0.5), id="n20-k3"),
    pytest.param(NuCase(n=1, n_star=1, nu=0.5, a_nu=1.5, b_nu=0.3), id="n1-k1"),
    # nu differs from the prior mean 2 / 0.1 = 20, so using the prior mean in
    # the beta draw instead of the current nu is visible.
    pytest.param(NuCase(n=20, n_star=1, nu=5.0, a_nu=2.0, b_nu=0.1), id="n20-k1"),
    pytest.param(NuCase(n=500, n_star=40, nu=50.0, a_nu=3.0, b_nu=0.1), id="n500-k40"),
    # Small prior shape: both mixture components have comparable weight.
    pytest.param(
        NuCase(n=20, n_star=5, nu=0.7, a_nu=0.1, b_nu=2.5),
        id="n20-k5-small-a",
    ),
]


def _make_state(case: NuCase) -> DPState:
    """Build a valid state with ``n_star`` non-empty clusters.

    Only ``n``, ``n_star`` and ``nu`` enter the update; the locations merely
    have to satisfy the Section 2.1 invariants so that the state can exist.
    """
    labels = np.arange(case.n, dtype=np.intp) % case.n_star
    counts = np.bincount(labels, minlength=case.n_star).astype(np.intp)
    alpha_star = np.linspace(0.4, 1.2, case.n_star, dtype=np.float64)
    lam_star = np.linspace(0.6, 1.8, case.n_star, dtype=np.float64)
    return DPState(
        alpha=alpha_star[labels].copy(),
        lam=lam_star[labels].copy(),
        s=labels.copy(),
        alpha_star=alpha_star,
        lam_star=lam_star,
        n_j=counts,
        nu=case.nu,
        gamma=0.5,
        phi=2.0,
    )


def _prior(case: NuCase) -> DPPrior:
    """The prior with this case's ``nu`` hyperparameters."""
    return DPPrior(
        a_nu=case.a_nu,
        b_nu=case.b_nu,
        a_gamma=1.0,
        b_gamma=0.232235,
        a_phi=2.0,
        b_phi=1.77522,
    )


def _production_draws(case: NuCase) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``N_DRAWS`` pairs ``(u, nu)`` from the conditional.

    ``refresh_nu`` is a Gibbs step, so calling it repeatedly would produce a
    Markov chain rather than iid draws from the conditional.  The current
    ``nu`` is therefore reset before every call: the target being tested is
    ``[nu | u, n, n_star]`` at a fixed ``nu``, and the only field the update
    reads before drawing is ``state.nu``.  The artificial reset makes the
    sample iid, which is what the KS tests assume.
    """
    state = _make_state(case)
    prior = _prior(case)
    rng = np.random.default_rng(20270 + case.n + 31 * case.n_star)
    u_draws = np.empty(N_DRAWS, dtype=np.float64)
    nu_draws = np.empty(N_DRAWS, dtype=np.float64)
    for k in range(N_DRAWS):
        state.nu = case.nu
        u_draws[k] = refresh_nu(state, prior, rng=rng)
        nu_draws[k] = state.nu
    return u_draws, nu_draws


@dataclass(frozen=True)
class Draws:
    """One case's production sample, shared by the assertions below."""

    case: NuCase
    u: np.ndarray
    nu: np.ndarray


@pytest.fixture(scope="module", params=CASES)
def draws(request: Any) -> Draws:
    """Production ``(u, nu)`` draws for one fixed state and prior."""
    case: NuCase = request.param
    u, nu = _production_draws(case)
    return Draws(case=case, u=u, nu=nu)


def _ks_p_value(sample: np.ndarray, cdf: Any) -> float:
    """Return the one-sample KS p-value as a plain float."""
    result: Any = ks_1samp(sample, cdf)
    return float(result.pvalue)


def _conditional_pit(draws: Draws) -> np.ndarray:
    """Evaluate the exact two-Gamma mixture CDF at each ``(u, nu)`` draw.

    Written directly from plan Section 2.2(c), with ``rate = b - log u`` and
    ``p = (a + n_star - 1) / (n * rate + a + n_star - 1)``.  If the production
    code's ``p``, shapes or rate are wrong, the transform is not uniform.
    """
    case = draws.case
    rate = case.b_nu - np.log(draws.u)
    numerator = case.a_nu + case.n_star - 1.0
    probability_high = numerator / (case.n * rate + numerator)
    high = gamma_dist.cdf(draws.nu, a=case.a_nu + case.n_star, scale=1.0 / rate)
    low = gamma_dist.cdf(draws.nu, a=case.a_nu + case.n_star - 1.0, scale=1.0 / rate)
    return probability_high * high + (1.0 - probability_high) * low


# ---------------------------------------------------------------------------
# Test 12 proper: the auxiliary and the two-Gamma mixture
# ---------------------------------------------------------------------------
class TestNuConditional:
    def test_auxiliary_is_beta_at_the_current_nu(self, draws: Draws):
        case = draws.case
        assert np.all((draws.u > 0.0) & (draws.u <= 1.0))
        p_value = _ks_p_value(draws.u, beta_dist(case.nu + 1.0, float(case.n)).cdf)
        assert p_value > KS_P_THRESHOLD, (
            f"n={case.n}, n_star={case.n_star}, nu={case.nu}: u does not "
            f"follow Beta(nu + 1, n) (KS p={p_value:.3e})"
        )

    def test_draws_match_the_exact_two_gamma_mixture(self, draws: Draws):
        case = draws.case
        pit = _conditional_pit(draws)
        assert np.all(np.isfinite(pit))
        assert np.all((pit >= 0.0) & (pit <= 1.0))
        p_value = _ks_p_value(pit, uniform.cdf)
        assert p_value > KS_P_THRESHOLD, (
            f"n={case.n}, n_star={case.n_star}, nu={case.nu}: the PIT of the "
            f"nu draws is not uniform (KS p={p_value:.3e}); the two-Gamma "
            "mixture, its p weight or its rate is wrong"
        )

    def test_probability_integral_transform_is_centred(self, draws: Draws):
        """A uniform PIT has mean 1/2; assert the observed mean is close."""
        pit = _conditional_pit(draws)
        # SD of one uniform is 1/sqrt(12); the mean of 1e5 has SD 9e-4.
        # Four SDs is a loose band, but still rejects a shifted reference CDF
        # (for example a dropped ``-log u``), which moves the mean by >0.02.
        tolerance = 4.0 / np.sqrt(12.0 * N_DRAWS)
        mean = float(np.mean(pit))
        assert abs(mean - 0.5) < tolerance, (
            f"n={draws.case.n}, n_star={draws.case.n_star}: PIT mean "
            f"{mean:.5f} differs from 0.5 by more than {tolerance:.5f}"
        )

    def test_draws_stay_in_the_parameter_space(self, draws: Draws):
        assert np.all(np.isfinite(draws.nu))
        assert np.all(draws.nu > 0.0)
        # A mixture of two Gamma components on a positive rate cannot produce
        # a zero or negative precision; this catches an inf rate or a
        # Gamma(0) component.
        assert np.max(draws.nu) < np.inf


# ---------------------------------------------------------------------------
# State handling: only nu changes, and the update is reproducible
# ---------------------------------------------------------------------------
class TestNuStateHandling:
    def test_refresh_touches_only_nu(self):
        case = NuCase(n=20, n_star=4, nu=2.0, a_nu=2.0, b_nu=0.9)
        state = _make_state(case)
        before = {
            "alpha": state.alpha.copy(),
            "lam": state.lam.copy(),
            "s": state.s.copy(),
            "alpha_star": state.alpha_star.copy(),
            "lam_star": state.lam_star.copy(),
            "n_j": state.n_j.copy(),
            "gamma": state.gamma,
            "phi": state.phi,
        }
        rng = np.random.default_rng(7)
        u = refresh_nu(state, _prior(case), rng=rng)

        assert 0.0 < u <= 1.0
        assert state.nu != case.nu
        for name, value in before.items():
            assert np.array_equal(getattr(state, name), value), (
                f"refresh_nu changed {name}"
            )
        state.check_invariants(_prior(case))

    def test_same_seed_replays_the_same_pair(self):
        case = NuCase(n=50, n_star=6, nu=3.0, a_nu=2.0, b_nu=0.9)
        prior = _prior(case)

        def one_draw(seed: int) -> tuple[float, float]:
            state = _make_state(case)
            rng = np.random.default_rng(seed)
            u = refresh_nu(state, prior, rng=rng)
            return u, state.nu

        assert one_draw(12345) == one_draw(12345)
        assert one_draw(12345) != one_draw(12346)

    @pytest.mark.parametrize("field", ["a_nu", "b_nu"])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_gamma_prior(self, field: str, bad: float):
        case = NuCase(n=20, n_star=2, nu=2.0, a_nu=2.0, b_nu=0.9)
        state = _make_state(case)
        prior = dataclasses.replace(_prior(case), **{field: bad})
        with pytest.raises(ValueError, match=field):
            refresh_nu(state, prior, rng=np.random.default_rng(0))
        # The failed call must not have replaced nu.
        assert state.nu == case.nu

    def test_rejects_non_positive_current_nu(self):
        case = NuCase(n=20, n_star=2, nu=2.0, a_nu=2.0, b_nu=0.9)
        state = _make_state(case)
        state.nu = 0.0
        with pytest.raises(StateInvariantError, match="nu"):
            refresh_nu(state, _prior(case), rng=np.random.default_rng(0))
