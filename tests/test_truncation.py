"""Test 24 (§7.1): one common truncation level J for the whole posterior run.

The paper truncates the posterior Sethuraman series at a single level fixed by
the largest precision in the run:

    J = ceil(log(epsilon) / log((n + max_nu) / (n + 1 + max_nu))).

Using ``truncation_number(nu_b + n)`` per draw instead is silently wrong: it
changes the number of atoms between iterations, so the draws target different
truncated posteriors.  This test recomputes the paper's formula independently
of the implementation and asserts that the common J dominates every per-draw J.
"""

import math

import numpy as np
import pytest

from src.dp_weibull import truncation_number, truncation_number_for_posterior

# Five (n, max_nu) pairs, including the §5.1 shape n=200, nu=24.6 -> J=2074.
PAIRS = [
    (1, 0.5),
    (17, 2.0),
    (20, 24.6),
    (200, 24.6),
    (622, 615.0),
]


def paper_formula(n: int, max_nu: float, epsilon: float = 1e-4) -> int:
    """Common-J rule transcribed directly from Kottas (2002) §3."""
    ratio = (n + max_nu) / (n + 1.0 + max_nu)
    return max(math.ceil(math.log(epsilon) / math.log(ratio)), 1)


@pytest.mark.parametrize(("n", "max_nu"), PAIRS)
def test_matches_paper_formula(n: int, max_nu: float) -> None:
    """The helper reproduces the paper's common-J formula exactly."""
    assert truncation_number_for_posterior(max_nu, n) == paper_formula(n, max_nu)


@pytest.mark.parametrize(("n", "max_nu"), PAIRS)
def test_matches_posterior_precision_rule(n: int, max_nu: float) -> None:
    """It is the ordinary truncation rule evaluated at precision n + max_nu."""
    assert truncation_number_for_posterior(max_nu, n) == truncation_number(n + max_nu)


def test_section_5_1_value() -> None:
    """For n=200 and max_nu=24.6 the common J is 2074, not the prior rule's 232."""
    j_common = truncation_number_for_posterior(24.6, 200)

    assert j_common == 2074
    assert j_common == truncation_number(200 + 24.6)
    assert j_common > truncation_number(24.6)  # the prior rule, 232


@pytest.mark.parametrize(("n", "max_nu"), PAIRS)
def test_dominates_every_per_draw_j(n: int, max_nu: float) -> None:
    """Common J is at least as large as every per-draw truncation_number(nu_b + n)."""
    nu_draws = np.linspace(0.0, max_nu, num=50)
    per_draw = [truncation_number(nu_b + n) for nu_b in nu_draws]

    assert truncation_number_for_posterior(max_nu, n) >= max(per_draw)


def test_increasing_in_max_nu() -> None:
    """Fixing J at the largest nu is what makes the rule conservative."""
    js = [truncation_number_for_posterior(nu, 200) for nu in (1.0, 10.0, 24.6)]

    assert js[0] <= js[1] <= js[2]


@pytest.mark.parametrize("epsilon", [1e-1, 1e-2, 1e-4, 1e-6])
def test_epsilon_controls_the_tail(epsilon: float) -> None:
    """A smaller tolerance never shortens the series."""
    assert truncation_number_for_posterior(24.6, 200, epsilon) == paper_formula(
        200, 24.6, epsilon
    )
    assert truncation_number_for_posterior(24.6, 200, epsilon) >= 1


def test_a_loose_tolerance_can_shorten_to_one_atom() -> None:
    """J is floored at 1 even when epsilon is loose enough to ask for 0."""
    assert truncation_number_for_posterior(0.0, 1, epsilon=0.5) == 1


@pytest.mark.parametrize(("max_nu", "n"), [(-0.1, 10), (1.0, -1), (-2.0, -3)])
def test_rejects_negative_inputs(max_nu: float, n: int) -> None:
    """Negative precisions or observation counts are programming errors."""
    with pytest.raises(ValueError):
        truncation_number_for_posterior(max_nu, n)


def test_rejects_empty_problem() -> None:
    """Without data and without prior mass there is no posterior precision."""
    with pytest.raises(ValueError):
        truncation_number_for_posterior(0.0, 0)
