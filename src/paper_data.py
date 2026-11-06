"""Data and prior constants for the Section 5 examples of Kottas (2002).

This module owns the paper's two datasets and the elicitation constants of
Section 5:

* Section 5.1 -- the simulated lognormal mixture
  ``p LN(mu_1, var_1) + (1 - p) LN(mu_2, var_2)`` with
  ``mu_1 = 0, mu_2 = 1.2, var_1 = 0.25, var_2 = 0.02, p = 0.8, n = 200``.
  The generator uses a single ``numpy.random.Generator`` for both components.
  Passing the same *integer* seed to both ``lognorm.rvs`` calls (the legacy
  behaviour) gives the two components the same driving stream: the second
  component becomes a deterministic function of the first component's first
  ``p2_n`` uniforms and the correlation between the overlapping draws is
  0.977.  Each component is then marginally correct, but the dataset is not
  an i.i.d. sample from the mixture.
* Section 5.2 -- the leukemia remission times of Lawless (1982, p. 351),
  transcribed from the paper as ``A`` and ``B`` arms with right-censoring.
* The prior guess of Section 5: population median ``1.1727`` and IQR
  ``1.3679`` of the Section 5.1 mixture, transformed to the Weibull guess
  ``(alpha_tilde, lambda_tilde)`` and then to the hyperparameters
  ``(b_gamma, b_phi)`` by ``src.dp_weibull``.  The plan of record
  (``docs/research/dp_gibbs_plan.md`` Section 10) fixes the population values
  rather than sample estimates, because they do not depend on a particular
  simulated sample.

Field names follow the variance convention: ``var_1``/``var_2`` are
*variances*, while SciPy's ``lognorm(s=...)`` takes a standard deviation, so
the dataclass takes the square root internally.

The committed fixtures ``tests/fixtures/section_5_1.npz`` and
``tests/fixtures/section_5_2.npz`` are regenerated only by the manual script
``tests/fixtures/make_fixtures.py`` (plan Section 7.6).  Tests load them with
``allow_pickle=False`` and never write or regenerate them.

Out of scope: MCMC, the time transformation, functionals and any plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.stats import lognorm

from src.dp_weibull import b_gamma_b_phi_from_weibull, weibull_from_median_iqr

__all__ = [
    "A_GAMMA",
    "A_PHI",
    "ALPHA_TILDE",
    "B_GAMMA",
    "B_PHI",
    "D",
    "LAMBDA_TILDE",
    "LEUKEMIA_5_2",
    "LEUKEMIA_A_EVENT",
    "LEUKEMIA_A_TIME",
    "LEUKEMIA_A_TRANSCRIPTION",
    "LEUKEMIA_B_EVENT",
    "LEUKEMIA_B_TIME",
    "LEUKEMIA_B_TRANSCRIPTION",
    "NU_PRIORS",
    "PRIOR_IQR",
    "PRIOR_MEDIAN",
    "PRIOR_Q95",
    "TwoLognormalMix",
    "leukemia_5_2",
]


# ---------------------------------------------------------------------------
# Section 5.1: simulated lognormal mixture
# ---------------------------------------------------------------------------
@dataclass
class TwoLognormalMix:
    """Draw an i.i.d. sample from the Section 5.1 lognormal mixture.

    The defaults are exactly the paper's simulation.  ``p1_n = round(p * n)``
    observations come from the first component and the remaining ``p2_n`` from
    the second; both components share one ``numpy.random.Generator`` built
    from ``seed``.
    """

    mu_1: float = 0.0
    mu_2: float = 1.2
    var_1: float = 0.25
    var_2: float = 0.02
    p: float = 0.8
    n: int = 200
    seed: int = 721383

    p1_n: int = field(init=False, compare=False)
    p2_n: int = field(init=False, compare=False)
    times: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        if self.var_1 <= 0.0 or self.var_2 <= 0.0:
            raise ValueError("var_1 and var_2 must be positive")
        if not np.all(np.isfinite([self.mu_1, self.mu_2, self.p])):
            raise ValueError("mu_1, mu_2 and p must be finite")

        self.p1_n = round(self.p * self.n)
        self.p2_n = self.n - self.p1_n

        # One Generator for both components.  Do not replace this with the
        # same integer seed in both calls: that correlates the two samples.
        rng = np.random.default_rng(self.seed)
        first = lognorm.rvs(
            s=np.sqrt(self.var_1),
            scale=np.exp(self.mu_1),
            size=self.p1_n,
            random_state=rng,
        )
        second = lognorm.rvs(
            s=np.sqrt(self.var_2),
            scale=np.exp(self.mu_2),
            size=self.p2_n,
            random_state=rng,
        )
        self.times = np.asarray(np.append(first, second), dtype=np.float64)


# ---------------------------------------------------------------------------
# Section 5.2: leukemia remission times (Lawless 1982, p. 351)
# ---------------------------------------------------------------------------
# The paper's lists, with ``+`` marking a right-censored observation.  These
# tokens are the transcription source of truth: the numeric arrays below are
# parsed from them, and the transcription test compares the committed fixture
# back against these lists element by element.
LEUKEMIA_A_TRANSCRIPTION: tuple[str, ...] = (
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

LEUKEMIA_B_TRANSCRIPTION: tuple[str, ...] = (
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


def _parse_transcription(tokens: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Split ``t`` / ``t+`` tokens into numeric times and a boolean event flag."""
    time = np.empty(len(tokens), dtype=np.float64)
    event = np.empty(len(tokens), dtype=bool)
    for index, token in enumerate(tokens):
        censored = token.endswith("+")
        time[index] = float(token.removesuffix("+"))
        event[index] = not censored
    return time, event


LEUKEMIA_A_TIME, LEUKEMIA_A_EVENT = _parse_transcription(LEUKEMIA_A_TRANSCRIPTION)
LEUKEMIA_B_TIME, LEUKEMIA_B_EVENT = _parse_transcription(LEUKEMIA_B_TRANSCRIPTION)

#: The Section 5.2 data by arm, as ``(time, event)`` arrays.  Arm ``A`` has
#: 20 observations with 3 censored (28+, 48+, 49+); arm ``B`` has 20 with 2
#: censored (27+, 38+).  A previous draft of the plan said "three censored per
#: treatment"; that is wrong for B and must not be asserted.
LEUKEMIA_5_2: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "A": (LEUKEMIA_A_TIME, LEUKEMIA_A_EVENT),
    "B": (LEUKEMIA_B_TIME, LEUKEMIA_B_EVENT),
}


def leukemia_5_2() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return writable copies of the Section 5.2 ``(time, event)`` arrays."""
    return {
        arm: (time.copy(), event.copy()) for arm, (time, event) in LEUKEMIA_5_2.items()
    }


# ---------------------------------------------------------------------------
# Section 5: prior-guess constants
# ---------------------------------------------------------------------------
# Population median, IQR and 95th percentile of the Section 5.1 mixture.
# Section 10 of the plan fixes these population values as the elicitation
# input; the realised statistics of the committed sample are 1.2172, 1.5336
# and 5.1390 (maximum) and are recorded in the results note only.
PRIOR_MEDIAN = 1.1727
PRIOR_IQR = 1.3679
PRIOR_Q95 = 3.6828

# Weibull guess induced by the median/IQR pair (dp_weibull.weibull_from_median_iqr)
# and the hyperparameters that match its medians
# (dp_weibull.b_gamma_b_phi_from_weibull).  The values are the plan's recorded
# four-decimal constants; the round trip is checked in tests/test_paper_data.py.
ALPHA_TILDE = 1.3314
LAMBDA_TILDE = 1.7836
B_GAMMA = 0.232235
B_PHI = 1.77522

# Model constants of Section 6.2 that are not estimated.
A_GAMMA = 1.0
A_PHI = 2.0
D = 2

#: Gamma ``(shape, rate)`` priors on the DP precision ``nu`` used in Section
#: 5.1 (Figure 1-3) and, for Figure 4(a), on Section 5.2.
NU_PRIORS: tuple[tuple[float, float], ...] = (
    (2.0, 0.9),
    (2.0, 0.1),
    (3.0, 0.05),
)

# Verify at import that the recorded constants are the ones the closed-form
# conversions produce.  This is cheap and turns a stale constant into an
# immediate import error instead of a quiet prior change.
_ALPHA_TILDE_CHECK, _LAMBDA_TILDE_CHECK = cast(
    tuple[float, float], weibull_from_median_iqr(PRIOR_MEDIAN, PRIOR_IQR)
)
_B_GAMMA_CHECK, _B_PHI_CHECK = cast(
    tuple[float, float], b_gamma_b_phi_from_weibull(ALPHA_TILDE, LAMBDA_TILDE)
)
if not (
    np.isclose(_ALPHA_TILDE_CHECK, ALPHA_TILDE, rtol=1e-4)
    and np.isclose(_LAMBDA_TILDE_CHECK, LAMBDA_TILDE, rtol=1e-4)
    and np.isclose(_B_GAMMA_CHECK, B_GAMMA, rtol=1e-4)
    and np.isclose(_B_PHI_CHECK, B_PHI, rtol=1e-4)
):
    raise RuntimeError(
        "paper prior constants no longer match the median/IQR round trip; "
        "recompute ALPHA_TILDE, LAMBDA_TILDE, B_GAMMA and B_PHI"
    )
