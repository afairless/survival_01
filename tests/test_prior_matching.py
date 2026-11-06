"""Test 7 (§7.1): prior-matching constants reproduce the induced medians.

`b_gamma_b_phi_from_weibull` chooses ``b_gamma`` and ``b_phi`` so that the
medians of the induced marginals of ``[lambda]`` and ``[alpha]`` equal a
Weibull guess ``(alpha_tilde, lambda_tilde)``.  This test rebuilds both
marginal densities numerically (``quad``) and finds their medians
(``brentq``), independently of the closed-form expressions used by the code.

It passes against the current constants and exists to stop the earlier
``(2**(1/3) - 1)`` "correction" (which came from confusing the density
exponent with the CDF exponent of ``[lambda]``) from ever being applied.
"""

from typing import cast

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from src.dp_weibull import b_gamma_b_phi_from_weibull


def lambda_marginal_density(lam: float, b_gamma: float) -> float:
    """Induced marginal density [lambda] = 2 b / (1 + lambda b)**3."""
    return 2.0 * b_gamma / (1.0 + lam * b_gamma) ** 3


def alpha_marginal_density(alpha: float, b_phi: float) -> float:
    """Induced marginal density [alpha] = 2 b**2 / (3 max(alpha, b)**3)."""
    return 2.0 * b_phi**2 / (3.0 * max(alpha, b_phi) ** 3)


def _numerical_median(density, upper: float) -> float:
    """Median of a density on (0, upper), by quadrature plus brentq."""

    def cdf_minus_half(m: float) -> float:
        return quad(density, 0.0, m)[0] - 0.5

    return cast(float, brentq(cdf_minus_half, 0.0, upper))


@pytest.mark.parametrize("lambda_tilde", [0.5, 1.7836, 3.0])
def test_lambda_marginal_median_matches_guess(lambda_tilde: float) -> None:
    """The median of [lambda] is lambda_tilde after the b_gamma round trip."""
    b_gamma, _ = b_gamma_b_phi_from_weibull(1.3314, lambda_tilde)

    median = _numerical_median(
        lambda lam: lambda_marginal_density(lam, b_gamma),
        2.0 * lambda_tilde,
    )

    assert abs(median - lambda_tilde) / lambda_tilde < 1e-6
    assert b_gamma > 0.0


@pytest.mark.parametrize("alpha_tilde", [0.75, 1.3314, 2.5])
def test_alpha_marginal_median_matches_guess(alpha_tilde: float) -> None:
    """The median of [alpha] is alpha_tilde after the b_phi round trip."""
    _, b_phi = b_gamma_b_phi_from_weibull(alpha_tilde, 1.7836)

    median = _numerical_median(
        lambda alpha: alpha_marginal_density(alpha, b_phi),
        2.0 * alpha_tilde,
    )

    assert abs(median - alpha_tilde) / alpha_tilde < 1e-6
    assert b_phi > 0.0


def test_section_5_1_prior_constants() -> None:
    """The §5.1 elicitation round trip yields b_gamma=0.23224, b_phi=1.77522."""
    b_gamma, b_phi = b_gamma_b_phi_from_weibull(alpha_tilde=1.3314, lambda_tilde=1.7836)

    assert np.isclose(b_gamma, 0.23224, rtol=1e-4)
    assert np.isclose(b_phi, 1.77522, rtol=1e-4)
