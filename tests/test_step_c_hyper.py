"""Tests 13 and 14 (plan Section 7.2): the ``phi`` and ``gamma`` updates.

Step (c) of the Appendix refreshes the hyperparameters in the order ``nu``,
``phi``, ``gamma``; ``nu`` is tested separately by test 12
(``tests/test_step_c_nu.py``).  The two conjugate updates are, from plan
Section 2.2(c) and Eq. (c),

    ``phi | alpha*, n_star ~ Pareto(a_phi + n_star, max(b_phi, max_j alpha*_j))``
    ``gamma | lambda*, n_star
        ~ Gamma(a_gamma + d n_star, rate = b_gamma + sum_j 1/lambda*_j)``,

with ``Pareto(a, b)`` the density ``a b**a / phi**(a+1)`` on ``phi >= b`` and
``Gamma`` the shape-rate convention used throughout the module.

Both tests fix the state, derive the reference parameters with plain NumPy
from the plan's formulas -- never from the sampler's private helpers -- and
compare many production draws with ``scipy.stats`` by a Kolmogorov-Smirnov
test.  Two mistakes are singled out because they are invisible on a trace:

* ``phi``'s Pareto lower bound must be ``max(b_phi, max_j alpha*_j)``.  One
  case has ``max_j alpha*_j < b_phi`` and one has ``max_j alpha*_j > b_phi``,
  so both branches of the maximum are exercised, and every draw is asserted to
  satisfy ``phi >= max_j alpha*_j`` (invariant 3 of plan Section 2.1).
* ``gamma``'s shape must be ``a_gamma + d n_star``, i.e. one ``d`` per
  *cluster*; the ``a_gamma + n_star`` version is tested explicitly as a
  rejected alternative, because it is the natural-looking error.

The ``n_star`` values are chosen so that the wrong alternatives give a
visibly different law: with ``d = 2``, ``n_star = 1`` compares shape ``a + 2``
against ``a + 1``, and the large-``n_star`` case compares ``a + 20`` against
``a + 10``.

Regression: every case uses a fixed seed, so a failure is reproducible.  As in
test 12 the KS threshold is an effect-size decision at 1e5 draws (p > 1e-3),
not a 5% level; a correct sampler still fails at 0.1% per statistic and the
seeds are fixed, so the suite is deterministic.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist
from scipy.stats import ks_1samp
from scipy.stats import pareto as pareto_dist

from src.dp_gibbs import (
    DPPrior,
    DPState,
    StateInvariantError,
    SurvData,
    refresh_gamma,
    refresh_nu,
    refresh_phi,
    sample_step_c,
)

N_DRAWS = 100_000
KS_P_THRESHOLD = 1e-3


# ---------------------------------------------------------------------------
# Fixed states: only the fields a conditional reads are varied
# ---------------------------------------------------------------------------
def _make_state(
    *,
    n: int,
    alpha_star: np.ndarray,
    lam_star: np.ndarray,
    nu: float = 2.0,
    gamma: float = 0.5,
    phi: float = 5.0,
) -> DPState:
    """Build a valid compacted state with the given cluster locations.

    ``n`` observations are spread round-robin over ``len(alpha_star)``
    clusters, so ``sum(n_j) == n`` and ``n_j == bincount(s)``; ``phi`` is
    large enough that invariant 3 holds at construction.
    """
    n_star = int(alpha_star.size)
    labels = np.arange(n, dtype=np.intp) % n_star
    counts = np.bincount(labels, minlength=n_star).astype(np.intp)
    return DPState(
        alpha=alpha_star[labels].copy(),
        lam=lam_star[labels].copy(),
        s=labels.copy(),
        alpha_star=alpha_star.copy(),
        lam_star=lam_star.copy(),
        n_j=counts,
        nu=nu,
        gamma=gamma,
        phi=phi,
    )


def _prior(
    *,
    a_phi: float = 2.0,
    b_phi: float = 0.5,
    a_gamma: float = 1.0,
    b_gamma: float = 0.232235,
    d: int = 2,
) -> DPPrior:
    """A prior whose ``G0`` hyperparameters are the case's own."""
    return DPPrior(
        a_nu=2.0,
        b_nu=0.9,
        a_gamma=a_gamma,
        b_gamma=b_gamma,
        a_phi=a_phi,
        b_phi=b_phi,
        d=d,
    )


@dataclass(frozen=True)
class PhiCase:
    """One fixed state and prior for the conditional ``[phi | alpha*, n_star]``."""

    n_star: int
    a_phi: float
    b_phi: float
    alpha_low: float
    alpha_high: float
    n: int


PHI_CASES = [
    # max_j alpha*_j = 0.4 < b_phi = 0.5: the prior's lower bound binds.
    pytest.param(
        PhiCase(n_star=3, a_phi=2.0, b_phi=0.5, alpha_low=0.2, alpha_high=0.4, n=12),
        id="max-alpha-below-b-phi",
    ),
    # max_j alpha*_j = 1.6 > b_phi = 0.5: the likelihood's bound binds.
    pytest.param(
        PhiCase(n_star=5, a_phi=2.0, b_phi=0.5, alpha_low=0.7, alpha_high=1.6, n=20),
        id="max-alpha-above-b-phi",
    ),
    # A single cluster: shape a_phi + 1, and the lower bound is also its own
    # alpha*_1.  Catches a missing ``n_star`` in the Pareto shape.
    pytest.param(
        PhiCase(n_star=1, a_phi=1.0, b_phi=0.3, alpha_low=1.1, alpha_high=1.1, n=1),
        id="one-cluster",
    ),
    # Many clusters with a small prior shape: both terms of the shape matter.
    pytest.param(
        PhiCase(
            n_star=12, a_phi=0.5, b_phi=1.77522, alpha_low=1.0, alpha_high=1.7, n=40
        ),
        id="twelve-clusters-paper-b-phi",
    ),
]


@dataclass(frozen=True)
class GammaCase:
    """One fixed state and prior for the conditional ``[gamma | lambda*, n_star]``."""

    n_star: int
    a_gamma: float
    b_gamma: float
    d: int
    lam_low: float
    lam_high: float
    n: int


GAMMA_CASES = [
    pytest.param(
        GammaCase(
            n_star=4,
            a_gamma=1.0,
            b_gamma=0.232235,
            d=2,
            lam_low=0.5,
            lam_high=2.0,
            n=12,
        ),
        id="four-clusters-d2",
    ),
    # d = 3 with one cluster: shape a + 3 versus the wrong a + 1 (or a).
    pytest.param(
        GammaCase(
            n_star=1, a_gamma=0.5, b_gamma=0.1, d=3, lam_low=1.3, lam_high=1.3, n=5
        ),
        id="one-cluster-d3",
    ),
    # Many clusters: shape a + 2*10 versus a + 10.
    pytest.param(
        GammaCase(
            n_star=10, a_gamma=2.0, b_gamma=0.4, d=2, lam_low=0.8, lam_high=2.5, n=30
        ),
        id="ten-clusters-d2",
    ),
]


def _alpha_star(case: PhiCase) -> np.ndarray:
    return np.linspace(case.alpha_low, case.alpha_high, case.n_star, dtype=np.float64)


def _lam_star(case: GammaCase) -> np.ndarray:
    return np.linspace(case.lam_low, case.lam_high, case.n_star, dtype=np.float64)


def _ks(sample: np.ndarray, distribution: Any) -> tuple[float, float]:
    """Return the one-sample KS statistic and p-value as plain floats."""
    result: Any = ks_1samp(sample, distribution.cdf)
    return float(result.statistic), float(result.pvalue)


def _reference_phi_lower(case: PhiCase) -> float:
    """``max(b_phi, max_j alpha*_j)``, written straight from plan Section 2.2(c)."""
    return max(case.b_phi, float(np.max(_alpha_star(case))))


def _reference_gamma_rate(case: GammaCase) -> float:
    """``b_gamma + sum_j 1/lambda*_j``, written straight from plan Section 2.2(c)."""
    return case.b_gamma + float(np.sum(1.0 / _lam_star(case)))


# ---------------------------------------------------------------------------
# Test 13: the Pareto conditional of phi
# ---------------------------------------------------------------------------
class TestPhiConditional:
    @pytest.mark.parametrize("case", PHI_CASES)
    def test_matches_pareto(self, case: PhiCase):
        state = _make_state(
            n=case.n,
            alpha_star=_alpha_star(case),
            lam_star=np.linspace(0.5, 1.5, case.n_star, dtype=np.float64),
            phi=float(np.max(_alpha_star(case))),
        )
        prior = _prior(a_phi=case.a_phi, b_phi=case.b_phi)
        rng = np.random.default_rng(21000 + case.n_star + 7 * case.n)

        sample = np.empty(N_DRAWS, dtype=np.float64)
        for k in range(N_DRAWS):
            sample[k] = refresh_phi(state, prior, rng=rng)

        lower = _reference_phi_lower(case)
        shape = case.a_phi + case.n_star
        statistic, p_value = _ks(sample, pareto_dist(b=shape, scale=lower))
        assert p_value > KS_P_THRESHOLD, (
            f"n_star={case.n_star}, a_phi={case.a_phi}, b_phi={case.b_phi}, "
            f"lower={lower}: phi draws reject Pareto({shape}, scale={lower}); "
            f"KS D={statistic:.4f}, p={p_value:.3e}"
        )
        # Effect-size guard: D is around 4e-3 at 1e5 correct draws.
        assert statistic < 0.02

    @pytest.mark.parametrize("case", PHI_CASES)
    def test_every_draw_respects_the_cluster_shapes(self, case: PhiCase):
        state = _make_state(
            n=case.n,
            alpha_star=_alpha_star(case),
            lam_star=np.linspace(0.5, 1.5, case.n_star, dtype=np.float64),
            phi=float(np.max(_alpha_star(case))),
        )
        prior = _prior(a_phi=case.a_phi, b_phi=case.b_phi)
        rng = np.random.default_rng(22000 + case.n_star)

        maximum = float(np.max(_alpha_star(case)))
        for _ in range(N_DRAWS):
            phi = refresh_phi(state, prior, rng=rng)
            assert phi >= maximum
            assert phi >= case.b_phi
        state.check_invariants(prior)

    @pytest.mark.parametrize("case", PHI_CASES)
    def test_wrong_lower_bound_is_rejected(self, case: PhiCase):
        """Using ``b_phi`` alone drops the ``alpha*`` support restriction."""
        lower = _reference_phi_lower(case)
        if lower == case.b_phi:
            pytest.skip("the two lower bounds coincide in this case")
        state = _make_state(
            n=case.n,
            alpha_star=_alpha_star(case),
            lam_star=np.linspace(0.5, 1.5, case.n_star, dtype=np.float64),
            phi=float(np.max(_alpha_star(case))),
        )
        prior = _prior(a_phi=case.a_phi, b_phi=case.b_phi)
        rng = np.random.default_rng(23000 + case.n_star)
        sample = np.array([refresh_phi(state, prior, rng=rng) for _ in range(N_DRAWS)])
        statistic, p_value = _ks(
            sample, pareto_dist(b=case.a_phi + case.n_star, scale=case.b_phi)
        )
        assert p_value < 1e-9
        assert statistic > 0.05

    @pytest.mark.parametrize("case", PHI_CASES)
    def test_missing_n_star_in_the_shape_is_rejected(self, case: PhiCase):
        """The likelihood contributes one ``1/phi`` per cluster."""
        state = _make_state(
            n=case.n,
            alpha_star=_alpha_star(case),
            lam_star=np.linspace(0.5, 1.5, case.n_star, dtype=np.float64),
            phi=float(np.max(_alpha_star(case))),
        )
        prior = _prior(a_phi=case.a_phi, b_phi=case.b_phi)
        rng = np.random.default_rng(24000 + case.n_star)
        sample = np.array([refresh_phi(state, prior, rng=rng) for _ in range(N_DRAWS)])
        statistic, p_value = _ks(
            sample,
            pareto_dist(b=case.a_phi, scale=_reference_phi_lower(case)),
        )
        assert p_value < 1e-9
        assert statistic > 0.02


# ---------------------------------------------------------------------------
# Test 14: the Gamma conditional of gamma
# ---------------------------------------------------------------------------
class TestGammaConditional:
    @pytest.mark.parametrize("case", GAMMA_CASES)
    def test_matches_gamma(self, case: GammaCase):
        state = _make_state(
            n=case.n,
            alpha_star=np.linspace(0.5, 1.2, case.n_star, dtype=np.float64),
            lam_star=_lam_star(case),
        )
        prior = _prior(a_gamma=case.a_gamma, b_gamma=case.b_gamma, d=case.d)
        rng = np.random.default_rng(25000 + case.n_star + 13 * case.d)

        sample = np.empty(N_DRAWS, dtype=np.float64)
        for k in range(N_DRAWS):
            sample[k] = refresh_gamma(state, prior, rng=rng)

        shape = case.a_gamma + case.d * case.n_star
        rate = _reference_gamma_rate(case)
        statistic, p_value = _ks(sample, gamma_dist(a=shape, scale=1.0 / rate))
        assert p_value > KS_P_THRESHOLD, (
            f"n_star={case.n_star}, d={case.d}: gamma draws reject "
            f"Gamma({shape}, rate={rate}); KS D={statistic:.4f}, p={p_value:.3e}"
        )
        assert statistic < 0.02

    @pytest.mark.parametrize("case", GAMMA_CASES)
    def test_n_star_without_d_is_rejected(self, case: GammaCase):
        """The ``d n_star`` versus ``n_star`` error the plan singles out."""
        shape = case.a_gamma + case.n_star
        wrong_shape = case.a_gamma + case.d * case.n_star
        if shape == wrong_shape:
            pytest.skip("d == 1 makes the two shapes identical")
        state = _make_state(
            n=case.n,
            alpha_star=np.linspace(0.5, 1.2, case.n_star, dtype=np.float64),
            lam_star=_lam_star(case),
        )
        prior = _prior(a_gamma=case.a_gamma, b_gamma=case.b_gamma, d=case.d)
        rng = np.random.default_rng(26000 + case.n_star)
        sample = np.array(
            [refresh_gamma(state, prior, rng=rng) for _ in range(N_DRAWS)]
        )
        statistic, p_value = _ks(
            sample,
            gamma_dist(a=shape, scale=1.0 / _reference_gamma_rate(case)),
        )
        assert p_value < 1e-9
        assert statistic > 0.02

    @pytest.mark.parametrize("case", GAMMA_CASES)
    def test_missing_reciprocal_scales_is_rejected(self, case: GammaCase):
        """Dropping ``sum_j 1/lambda*_j`` from the rate changes the law."""
        state = _make_state(
            n=case.n,
            alpha_star=np.linspace(0.5, 1.2, case.n_star, dtype=np.float64),
            lam_star=_lam_star(case),
        )
        prior = _prior(a_gamma=case.a_gamma, b_gamma=case.b_gamma, d=case.d)
        rng = np.random.default_rng(27000 + case.n_star)
        sample = np.array(
            [refresh_gamma(state, prior, rng=rng) for _ in range(N_DRAWS)]
        )
        shape = case.a_gamma + case.d * case.n_star
        statistic, p_value = _ks(sample, gamma_dist(a=shape, scale=1.0 / case.b_gamma))
        assert p_value < 1e-9
        assert statistic > 0.02


# ---------------------------------------------------------------------------
# State handling: the updates touch only their own hyperparameter
# ---------------------------------------------------------------------------
class TestHyperparameterStateHandling:
    def test_refresh_phi_touches_only_phi(self):
        alpha_star = np.array([0.4, 0.9, 1.4], dtype=np.float64)
        state = _make_state(
            n=9,
            alpha_star=alpha_star,
            lam_star=np.array([0.6, 1.1, 1.7], dtype=np.float64),
        )
        before = {
            "alpha": state.alpha.copy(),
            "lam": state.lam.copy(),
            "s": state.s.copy(),
            "alpha_star": state.alpha_star.copy(),
            "lam_star": state.lam_star.copy(),
            "n_j": state.n_j.copy(),
            "nu": state.nu,
            "gamma": state.gamma,
        }
        phi = refresh_phi(state, _prior(), rng=np.random.default_rng(11))

        assert phi == state.phi
        assert state.phi != 5.0
        for name, value in before.items():
            assert np.array_equal(getattr(state, name), value), (
                f"refresh_phi changed {name}"
            )
        state.check_invariants(_prior())

    def test_refresh_gamma_touches_only_gamma(self):
        alpha_star = np.array([0.4, 0.9, 1.4], dtype=np.float64)
        state = _make_state(
            n=9,
            alpha_star=alpha_star,
            lam_star=np.array([0.6, 1.1, 1.7], dtype=np.float64),
        )
        before = {
            "alpha": state.alpha.copy(),
            "lam": state.lam.copy(),
            "s": state.s.copy(),
            "alpha_star": state.alpha_star.copy(),
            "lam_star": state.lam_star.copy(),
            "n_j": state.n_j.copy(),
            "nu": state.nu,
            "phi": state.phi,
        }
        gamma = refresh_gamma(
            state,
            _prior(a_gamma=1.0, b_gamma=0.232235),
            rng=np.random.default_rng(12),
        )

        assert gamma == state.gamma
        assert state.gamma != 0.5
        for name, value in before.items():
            assert np.array_equal(getattr(state, name), value), (
                f"refresh_gamma changed {name}"
            )
        state.check_invariants(_prior())

    def test_same_seed_replays_the_same_draws(self):
        alpha_star = np.array([0.4, 1.6], dtype=np.float64)
        lam_star = np.array([0.7, 2.1], dtype=np.float64)
        prior = _prior()

        def one_phi(seed: int) -> float:
            state = _make_state(n=4, alpha_star=alpha_star, lam_star=lam_star)
            return refresh_phi(state, prior, rng=np.random.default_rng(seed))

        def one_gamma(seed: int) -> float:
            state = _make_state(n=4, alpha_star=alpha_star, lam_star=lam_star)
            return refresh_gamma(state, prior, rng=np.random.default_rng(seed))

        assert one_phi(12345) == one_phi(12345)
        assert one_phi(12345) != one_phi(12346)
        assert one_gamma(12345) == one_gamma(12345)
        assert one_gamma(12345) != one_gamma(12346)

    @pytest.mark.parametrize("field", ["a_phi", "b_phi"])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_pareto_prior(self, field: str, bad: float):
        alpha_star = np.array([0.4, 1.6], dtype=np.float64)
        state = _make_state(
            n=4,
            alpha_star=alpha_star,
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        prior = dataclasses.replace(_prior(), **{field: bad})
        with pytest.raises(ValueError, match=field):
            refresh_phi(state, prior, rng=np.random.default_rng(0))
        # The failed call must not have replaced phi.
        assert state.phi == 5.0

    @pytest.mark.parametrize("field", ["a_gamma", "b_gamma"])
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_gamma_prior(self, field: str, bad: float):
        alpha_star = np.array([0.4, 1.6], dtype=np.float64)
        state = _make_state(
            n=4,
            alpha_star=alpha_star,
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        prior = dataclasses.replace(_prior(), **{field: bad})
        with pytest.raises(ValueError, match=field):
            refresh_gamma(state, prior, rng=np.random.default_rng(0))
        assert state.gamma == 0.5

    def test_rejects_invalid_cluster_locations(self):
        alpha_star = np.array([0.4, 1.6], dtype=np.float64)
        state = _make_state(
            n=4,
            alpha_star=alpha_star,
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        state.lam_star[0] = 0.0
        with pytest.raises(StateInvariantError, match="lam_star"):
            refresh_gamma(state, _prior(), rng=np.random.default_rng(0))

        state = _make_state(
            n=4,
            alpha_star=alpha_star,
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        state.alpha_star[0] = -0.1
        with pytest.raises(StateInvariantError, match="alpha_star"):
            refresh_phi(state, _prior(), rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# The step (c) driver: order, stream and validation
# ---------------------------------------------------------------------------
DATA = SurvData(
    time=np.array([1.10, 1.25, 1.40, 1.55, 1.70, 1.85]),
    event=np.array([True, False, True, True, False, True]),
    t_max=1.0,
)


def _step_c_state() -> DPState:
    """A compacted state matching :data:`DATA` (six observations, two clusters)."""
    alpha_star = np.array([0.8, 1.4], dtype=np.float64)
    lam_star = np.array([0.9, 1.9], dtype=np.float64)
    return _make_state(n=6, alpha_star=alpha_star, lam_star=lam_star)


class TestSampleStepC:
    def test_runs_nu_then_phi_then_gamma(self):
        """Replaying the three refreshes by hand must reproduce the driver.

        This is the order assertion of plan Section 3.4: it fixes the
        random-number stream, so a reordered driver gives different draws from
        the same seed even though every marginal law is unchanged.
        """
        prior = _prior()
        state = _make_state(
            n=6,
            alpha_star=np.array([0.4, 1.6], dtype=np.float64),
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        sample_step_c(state, DATA, prior, rng=np.random.default_rng(31415))

        manual = _make_state(
            n=6,
            alpha_star=np.array([0.4, 1.6], dtype=np.float64),
            lam_star=np.array([0.7, 2.1], dtype=np.float64),
        )
        rng = np.random.default_rng(31415)
        refresh_nu(manual, prior, rng=rng)
        refresh_phi(manual, prior, rng=rng)
        refresh_gamma(manual, prior, rng=rng)

        assert state.nu == manual.nu
        assert state.phi == manual.phi
        assert state.gamma == manual.gamma

    def test_changes_all_three_and_nothing_else(self):
        prior = _prior()
        state = _step_c_state()
        before = {
            "alpha": state.alpha.copy(),
            "lam": state.lam.copy(),
            "s": state.s.copy(),
            "alpha_star": state.alpha_star.copy(),
            "lam_star": state.lam_star.copy(),
            "n_j": state.n_j.copy(),
        }
        sample_step_c(state, DATA, prior, rng=np.random.default_rng(2718))

        assert state.nu != 2.0
        assert state.phi != 5.0
        assert state.gamma != 0.5
        for name, value in before.items():
            assert np.array_equal(getattr(state, name), value), (
                f"sample_step_c changed {name}"
            )
        state.check_invariants(prior)

    def test_same_seed_replays_the_step(self):
        assert _step_triple(seed=7) == _step_triple(seed=7)
        assert _step_triple(seed=7) != _step_triple(seed=8)

    def test_rejects_data_from_another_dataset(self):
        other = SurvData(
            time=np.array([1.10, 1.25, 1.40]),
            event=np.array([True, True, True]),
            t_max=1.0,
        )
        state = _step_c_state()
        with pytest.raises(ValueError, match="same number of observations"):
            sample_step_c(state, other, _prior(), rng=np.random.default_rng(0))
        # Nothing was refreshed: validation runs before the first draw.
        assert state.nu == 2.0
        assert state.phi == 5.0
        assert state.gamma == 0.5


def _step_triple(*, seed: int) -> tuple[float, float, float]:
    """Run step (c) once from a fixed state and return ``(nu, phi, gamma)``."""
    state = _step_c_state()
    sample_step_c(state, DATA, _prior(), rng=np.random.default_rng(seed))
    return state.nu, state.phi, state.gamma
