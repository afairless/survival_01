"""Draws from the Dirichlet process used in Kottas (2002), Section 4.1.

The hierarchical model is

    t_i | alpha_i, lambda_i  ~  Weibull(alpha_i, lambda_i)
    (alpha_i, lambda_i) | G  ~  G
    G | nu, gamma, phi       ~  DP(nu * G0(.; phi, gamma))
    nu                       ~  Gamma(a_nu, b_nu)
    gamma                    ~  Gamma(a_gamma, b_gamma)
    phi                      ~  Pareto(a_phi, b_phi)

with the base measure

    G0(alpha, lambda | phi, gamma)
        = Uniform(alpha | 0, phi) * InvGamma(lambda | d, gamma),   d = 2.

The Weibull kernel uses the paper's parameterisation

    K_W(t | alpha, lambda) = 1 - exp(-t**alpha / lambda)
    k_W(t | alpha, lambda) = alpha * t**(alpha - 1) / lambda
                             * exp(-t**alpha / lambda).

A Dirichlet process is a distribution over probability measures, so it cannot be
represented by a finite vector of parameters.  What we can do is draw a
realisation G, which is almost surely discrete.  This module uses the
Sethuraman (1994) stick-breaking construction

    G = sum_{j=1}^infinity w_j * delta_{theta_j},
    z_j ~ Beta(1, precision),
    w_1 = z_1,  w_j = z_j * prod_{s<j}(1 - z_s),
    theta_j ~ H  i.i.d.,

where H is the base measure and `precision` is nu for the prior or nu + n for
the DP posterior given the latent theta_i's.

The infinite sum is truncated following Section 3 of the paper: choose J such
that the expected remaining mass is at most epsilon,

    (precision / (precision + 1))**J = epsilon,

and set the last weight to the residual mass so that the weights sum to one.

When many G draws are made for different values of nu (for example, one per
MCMC iteration), the paper uses a single J based on max(nu), rather than a
separate J for each draw.  Both sampling functions accept `n_terms` so that
the caller can supply that common truncation level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import invgamma, pareto


# ---------------------------------------------------------------------------
# Weibull kernel (paper parameterisation)
# ---------------------------------------------------------------------------
def weibull_cdf(t, alpha, lam):
    """K_W(t | alpha, lambda) = 1 - exp(-t**alpha / lambda)."""
    t = np.asarray(t, dtype=float)
    return 1.0 - np.exp(-np.power(t, alpha) / lam)


def weibull_sf(t, alpha, lam):
    """1 - K_W(t | alpha, lambda) = exp(-t**alpha / lambda)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-np.power(t, alpha) / lam)


def weibull_pdf(t, alpha, lam):
    """k_W(t | alpha, lambda)."""
    t = np.asarray(t, dtype=float)
    return alpha * np.power(t, alpha - 1.0) / lam * np.exp(-np.power(t, alpha) / lam)


# ---------------------------------------------------------------------------
# Base measure G0 and priors on the DP parameters
# ---------------------------------------------------------------------------
def sample_g0(n, phi, gamma, *, d=2, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Draw n i.i.d. (alpha, lambda) pairs from G0(.; phi, gamma).

    G0(alpha, lambda | phi, gamma)
        = Uniform(alpha | 0, phi) * InvGamma(lambda | d, gamma).

    SciPy's invgamma(a, scale=b) has density
        b**a / Gamma(a) * lambda**(-a-1) * exp(-b/lambda),
    which is the paper's IGamma(lambda | a, b) with a = d and b = gamma.
    """
    rng = np.random.default_rng() if rng is None else rng
    alpha = np.asarray(rng.uniform(0.0, phi, size=n), dtype=float)
    lam = np.asarray(
        invgamma.rvs(a=d, scale=gamma, size=n, random_state=rng), dtype=float
    )
    return alpha, lam


def sample_prior_hyperparameters(
    a_nu,
    b_nu,
    a_gamma,
    b_gamma,
    a_phi,
    b_phi,
    *,
    rng=None,
):
    """Draw (nu, gamma, phi) from their priors.

    The paper writes Gamma(shape, rate), so Gamma(2, 0.9) has mean 2/0.9.
    SciPy uses `scale = 1 / rate`.

    The Pareto prior is Pareto(phi | a_phi, b_phi) with density
        a_phi * b_phi**a_phi / phi**(a_phi + 1),  phi >= b_phi,
    i.e. scipy.stats.pareto(b=a_phi, scale=b_phi).
    """
    rng = np.random.default_rng() if rng is None else rng
    nu = np.asarray(rng.gamma(shape=a_nu, scale=1.0 / b_nu), dtype=float).item()
    gamma = np.asarray(
        rng.gamma(shape=a_gamma, scale=1.0 / b_gamma), dtype=float
    ).item()
    phi = np.asarray(
        pareto.rvs(b=a_phi, scale=b_phi, random_state=rng), dtype=float
    ).item()
    return nu, gamma, phi


def b_gamma_b_phi_from_weibull(alpha_tilde, lambda_tilde):
    """Choose (b_gamma, b_phi) by matching prior medians to a Weibull guess.

    Section 4.3 of the paper marginalises gamma and phi out of the parametric
    version of the model.  With a_gamma = 1 and a_phi = 2 the induced priors
    are

        [lambda] = 2 b_gamma / (1 + lambda b_gamma)**3,
        [alpha]  = 2 b_phi**2 / (3 * max(alpha, b_phi)**3).

    Their medians are (sqrt(2) - 1) / b_gamma and 3 b_phi / 4, respectively.
    Setting these equal to lambda_tilde and alpha_tilde gives

        b_gamma = (sqrt(2) - 1) / lambda_tilde,
        b_phi   = 4 alpha_tilde / 3.
    """
    b_gamma = (np.sqrt(2.0) - 1.0) / lambda_tilde
    b_phi = 4.0 * alpha_tilde / 3.0
    return b_gamma, b_phi


def weibull_from_median_iqr(median, iqr):
    """Find (alpha, lambda) of the Weibull with the given median and IQR.

    Here `lambda` is the paper's scale parameter in
        K_W(t | alpha, lambda) = 1 - exp(-t**alpha / lambda),
    not the standard Weibull scale parameter c in
        1 - exp(-(t / c)**alpha).
    The two are related by c = lambda**(1 / alpha).  The median is
        m = (lambda * log(2)) ** (1 / alpha)
          = c * (log 2) ** (1 / alpha),
    so lambda = m ** alpha / log(2) and c = m / (log(2) ** (1 / alpha)).
    The IQR / median ratio is a function of alpha only.  Find alpha, then
    return lambda in the parameterisation used by the paper.
    """
    if median <= 0.0 or iqr <= 0.0:
        raise ValueError("median and iqr must be positive")

    ratio = iqr / median
    log_ratio_43 = np.log(4.0 / 3.0) / np.log(2.0)

    def f(a):
        return 2.0 ** (1.0 / a) - log_ratio_43 ** (1.0 / a) - ratio

    alpha = brentq(f, 1e-3, 1e3)
    lam = median**alpha / np.log(2.0)
    return alpha, lam


# ---------------------------------------------------------------------------
# Stick-breaking construction
# ---------------------------------------------------------------------------
def truncation_number(precision, epsilon=1e-4):
    """Smallest J with (precision / (precision + 1))**J <= epsilon."""
    if precision <= 0.0:
        raise ValueError("precision must be positive")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0, 1)")
    j = math.ceil(np.log(epsilon) / np.log(precision / (precision + 1.0)))
    return max(j, 1)


def stick_breaking_weights(precision, n_terms, rng):
    """Draw truncated stick-breaking weights.

    w_j = z_j * prod_{s<j}(1 - z_s) for j = 1, ..., J-1
    w_J = prod_{s<J}(1 - z_s)                    (residual mass)

    so that the weights sum to one exactly (up to floating point).
    """
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")
    if n_terms == 1:
        return np.ones(1)

    z = rng.beta(1.0, precision, size=n_terms - 1)
    weights = np.empty(n_terms)
    remaining = 1.0
    for j in range(n_terms - 1):
        weights[j] = z[j] * remaining
        remaining *= 1.0 - z[j]
    weights[-1] = remaining
    return weights


# ---------------------------------------------------------------------------
# A finite realisation of G, with convenience summaries
# ---------------------------------------------------------------------------
@dataclass
class DPRealization:
    """A truncated draw G = sum_j weights[j] * delta_{(alphas[j], lambdas[j])}.

    `precision` is the DP precision used to generate it (nu for the prior,
    nu + n for the posterior).  It is stored only for reference.
    """

    weights: np.ndarray
    alphas: np.ndarray
    lambdas: np.ndarray
    precision: float

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.alphas = np.asarray(self.alphas, dtype=float)
        self.lambdas = np.asarray(self.lambdas, dtype=float)
        if not (self.weights.shape == self.alphas.shape == self.lambdas.shape):
            raise ValueError("weights, alphas and lambdas must have equal shape")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")

    @property
    def n_atoms(self):
        return self.weights.size

    def _scaled_time(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        alpha = self.alphas[:, None]
        lam = self.lambdas[:, None]
        return np.power(t[None, :], alpha) / lam

    # The mixture F(t; G) = sum_j w_j K_W(t | alpha_j, lambda_j), etc.
    def cdf(self, t):
        z = self._scaled_time(t)
        return np.sum(self.weights[:, None] * (1.0 - np.exp(-z)), axis=0)

    def survival(self, t):
        z = self._scaled_time(t)
        return np.sum(self.weights[:, None] * np.exp(-z), axis=0)

    def pdf(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        alpha = self.alphas[:, None]
        lam = self.lambdas[:, None]
        z = np.power(t[None, :], alpha) / lam
        kernel = alpha * np.power(t[None, :], alpha - 1.0) / lam * np.exp(-z)
        return np.sum(self.weights[:, None] * kernel, axis=0)

    def hazard(self, t):
        return self.pdf(t) / self.survival(t)


# ---------------------------------------------------------------------------
# DP prior and DP posterior realisations
# ---------------------------------------------------------------------------
def sample_dp_prior(
    nu,
    gamma,
    phi,
    *,
    d=2,
    epsilon=1e-4,
    n_terms=None,
    rng=None,
):
    """Draw G ~ DP(nu * G0(.; phi, gamma)).

    This is step (i) of Section 3 of the paper when there are no data, i.e.
    the prior draw of the mixing distribution.  For a posterior draw given
    latent (alpha_i, lambda_i), use `sample_dp_posterior`.
    """
    rng = np.random.default_rng() if rng is None else rng
    if n_terms is None:
        n_terms = truncation_number(nu, epsilon=epsilon)
    weights = stick_breaking_weights(nu, n_terms, rng)
    alphas, lambdas = sample_g0(n_terms, phi, gamma, d=d, rng=rng)
    return DPRealization(weights, alphas, lambdas, precision=nu)


def sample_dp_posterior(
    nu,
    gamma,
    phi,
    theta,
    *,
    d=2,
    epsilon=1e-4,
    n_terms=None,
    rng=None,
):
    """Draw G from [G | theta, nu, gamma, phi] for the DP mixture.

    Given latent theta_i = (alpha_i, lambda_i), i = 1, ..., n, the DP
    posterior is again a DP (Ferguson 1973; Antoniak 1974):

        G | theta, nu, gamma, phi
            ~ DP( (nu + n) * (nu G0 + sum_i delta_{theta_i}) / (nu + n) ).

    Its precision is nu + n, and each atom is drawn independently from
    G0 with probability nu / (nu + n), or is one of the observed theta_i
    chosen uniformly with probability n / (nu + n).

    Parameters
    ----------
    theta : array_like, shape (n, 2)
        The latent (alpha_i, lambda_i) draws from the Gibbs sampler.
    """
    rng = np.random.default_rng() if rng is None else rng
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError("theta must have shape (n, 2)")

    n = theta.shape[0]
    precision = nu + n
    if n_terms is None:
        n_terms = truncation_number(precision, epsilon=epsilon)
    weights = stick_breaking_weights(precision, n_terms, rng)

    alphas = np.empty(n_terms)
    lambdas = np.empty(n_terms)

    from_base = rng.random(n_terms) < nu / precision
    n_base = np.count_nonzero(from_base).item()
    if n_base:
        alphas[from_base], lambdas[from_base] = sample_g0(
            n_base, phi, gamma, d=d, rng=rng
        )
    n_data = n_terms - n_base
    if n_data:
        idx = rng.integers(0, n, size=n_data)
        alphas[~from_base] = theta[idx, 0]
        lambdas[~from_base] = theta[idx, 1]

    return DPRealization(weights, alphas, lambdas, precision=precision)


# ---------------------------------------------------------------------------
# Small demonstration
# ---------------------------------------------------------------------------
def _demo():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20240517)

    # Section 5.1 data-generating distribution: 0.8 LN(0, 0.25)
    # + 0.2 LN(1.2, 0.02).  Section 4.3 prior guesses: roughly the
    # median and IQR of that distribution.
    alpha_tilde, lambda_tilde = weibull_from_median_iqr(median=1.1727, iqr=1.3679)
    b_gamma, b_phi = b_gamma_b_phi_from_weibull(alpha_tilde, lambda_tilde)
    print(f"alpha_tilde = {alpha_tilde:.4f}, lambda_tilde = {lambda_tilde:.4f}")
    print(f"b_gamma     = {b_gamma:.4f}, b_phi         = {b_phi:.4f}")

    # The three priors for nu used in Section 5.1.
    nu_priors = [
        ("Gamma(2, 0.9)", 2.0, 0.9),
        ("Gamma(2, 0.1)", 2.0, 0.1),
        ("Gamma(3, 0.05)", 3.0, 0.05),
    ]

    t_grid = np.linspace(1e-3, 6.0, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for label, a_nu, b_nu in nu_priors:
        # One draw of (nu, gamma, phi), then one draw G ~ DP(nu G0).
        # In a full analysis you would repeat this for every posterior
        # (or prior) sample of (nu, gamma, phi).
        nu, gamma, phi = sample_prior_hyperparameters(
            a_nu,
            b_nu,
            a_gamma=1.0,
            b_gamma=b_gamma,
            a_phi=2.0,
            b_phi=b_phi,
            rng=rng,
        )
        G = sample_dp_prior(nu, gamma, phi, epsilon=1e-4, rng=rng)
        print(
            f"{label:14s}: nu={nu:6.3f}, gamma={gamma:6.3f}, "
            f"phi={phi:6.3f}, atoms={G.n_atoms}"
        )

        axes[0].plot(t_grid, G.survival(t_grid), label=label)
        axes[1].plot(t_grid, G.pdf(t_grid), label=label)

    axes[0].set_title("One DP draw: survival function")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("S(t; G)")
    axes[1].set_title("One DP draw: density")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("f(t; G)")
    axes[1].set_ylim(0.0, 2.0)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
