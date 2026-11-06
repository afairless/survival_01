"""Collapsed Polya-urn Gibbs sampler for the Weibull DP mixture (Kottas 2002).

This module implements the algorithm of Kottas (2002), Section 4.2 and the
Appendix (pp. 22-25), for the model

    t_i | alpha_i, lambda_i  ~  Weibull(alpha_i, lambda_i)
    (alpha_i, lambda_i) | G  ~  G
    G | nu, gamma, phi       ~  DP(nu * G0(.; phi, gamma))
    nu      ~  Gamma(a_nu, b_nu)
    gamma   ~  Gamma(a_gamma, b_gamma)
    phi     ~  Pareto(a_phi, b_phi)

with base measure

    G0(alpha, lambda | phi, gamma)
        = Uniform(alpha | 0, phi) * InvGamma(lambda | d, gamma),   d = 2,

and the paper's Weibull parameterisation
``K_W(t | alpha, lambda) = 1 - exp(-t**alpha / lambda)``.  The Gibbs sampler
is the collapsed urn of the Appendix: it keeps the latent
``(alpha_i, lambda_i, s_i)`` and the cluster locations, not a draw of ``G``.
The base measure primitives and the truncation rule live in
``src.dp_weibull``; functional inference lives in ``src.dp_functionals``.

Scope
-----
In scope: uncensored and right-censored data; the three full conditionals of
the Appendix in their stated order; chain storage and convergence diagnostics.

Out of scope (do not add these here): covariates/regression; interval,
left-censored or truncated data; a marginal (non-collapsed) sampler over
``G``; Bayes factors or model comparison.  Nothing here plots, generates data
or computes functionals.

The time scale caveat
---------------------
The sampler consumes times on the transformed scale
``t_tilde = 1 + t / max(t)`` (Appendix requirement ``t > 1``), while the
``b_gamma``/``b_phi`` prior guesses are elicited on the *original* time scale.
That map is a shift plus a scale, so the prior in force during fitting is not
the elicited prior; this is the paper's own convention and is kept, stated
here and in the results note, and made explicit by the prior-functional tests.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DPPrior", "DPSettings"]


@dataclass(frozen=True)
class DPPrior:
    """Fixed hyperparameters of model (5): the DP precision and ``G0`` priors.

    Attributes
    ----------
    a_nu, b_nu : float
        Shape and rate of the ``Gamma`` prior on the DP precision ``nu``.
        Section 5.1 uses ``Gamma(2, 0.9)``, ``Gamma(2, 0.1)`` and
        ``Gamma(3, 0.05)``.
    a_gamma, b_gamma : float
        Shape and rate of the ``Gamma`` prior on the inverse-gamma scale
        ``gamma``.  ``a_gamma = 1`` and ``b_gamma`` comes from the prior
        guess (``dp_weibull.b_gamma_b_phi_from_weibull``).
    a_phi, b_phi : float
        Shape and scale of the ``Pareto`` prior on the uniform upper bound
        ``phi``.  ``a_phi = 2`` and ``b_phi`` comes from the same prior guess.
    d : int
        Inverse-gamma shape of the base measure's ``lambda`` marginal,
        ``d = 2`` in the paper.

    These are model constants, not sampler knobs: they never change during a
    run, so the container is frozen.
    """

    a_nu: float
    b_nu: float
    a_gamma: float
    b_gamma: float
    a_phi: float
    b_phi: float
    d: int = 2


@dataclass
class DPSettings:
    """Sampler knobs.  Never model parameters, so nothing here is inferred.

    Attributes
    ----------
    burn_in : int
        Sweeps discarded before the first saved draw.
    n_save : int
        Number of saved draws ``B`` after burn-in and thinning.
    thin : int
        Sweeps between saved draws.
    n_alpha : int
        Number of ``alpha`` quadrature nodes ``M`` on the grid
        ``alpha_m = phi * m / M``.
    epsilon : float
        Target residual DP mass for the truncation rule.
    n_chains : int
        Number of chains; chain ``b`` uses ``seed + b``.
    seed : int or None
        Base seed for the run; ``None`` draws one from the OS entropy pool.
    preset : str
        ``"quick"``, ``"paper"`` or ``"custom"``, recorded in the results note
        so that a number can be traced back to the settings that produced it.

    The defaults are the ``paper`` preset of plan Section 6.2.
    """

    burn_in: int = 5_000
    n_save: int = 10_000
    thin: int = 150
    n_alpha: int = 256
    epsilon: float = 1e-4
    n_chains: int = 4
    seed: int | None = None
    preset: str = "custom"
