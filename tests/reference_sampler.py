"""An independent, deliberately naive sampler for the Weibull DP mixture.

This module exists for **test 16** of ``docs/research/dp_gibbs_plan.md``
(Section 7.3): it re-implements the Gibbs sampler of the Appendix on a small
dataset so that the production path in ``src/dp_gibbs.py`` can be checked
against a second derivation.  It imports nothing from ``src``: no helper, no
constant, no kernel.  A mistake shared by both implementations would therefore
have to be made twice, independently, from the same paper.

Naive where the production path is clever
-----------------------------------------
Every numerical device of the production sampler is replaced by a slower,
more literal one:

* ``q_0`` (Eqs. A.2/A.3) is a **product Gauss-Legendre rule** over
  ``(alpha, log lambda)``.  The inner ``lambda`` integral is evaluated
  numerically, from the unnormalised ``IGamma(lambda | d, gamma)`` density
  written out by hand; the closed form ``Gamma(d+1)/(gamma+t**alpha)**(d+1)``
  is never used, and no ``alpha`` grid is shared with the production path.
* The continuous piece ``h_i`` of step (a) is drawn by **two-dimensional
  rejection sampling** on the box ``(0, phi) x (0, LAMBDA_MAX)`` with a uniform
  proposal.  Nothing is inverted, no inverse-gamma variate is conditioned on
  ``alpha``; the target density itself is evaluated and accepted or rejected.
* Step (b) draws the *joint* pair ``(alpha*_j, lambda*_j)`` by **rejection from
  the exact Eq. A.5 density** in one step, instead of the production
  auxiliary-variable scheme for ``alpha*`` followed by the conjugate
  inverse-gamma draw for ``lambda*``.  The proposal is ``alpha ~ U(0, phi)``
  with the conjugate ``lambda`` marginal, and the acceptance probability is
  the likelihood ratio ``alpha**n_j^o exp((alpha - 1) sum log t)`` -- that
  ratio is exactly what the auxiliary-variable bounds and the ``alpha**n_j^o``
  inverse-CDF draw are there to avoid, so the two implementations check each
  other.
* ``nu`` is drawn by **inverse CDF of the exact EPPF conditional**,
  ``p(nu | s) proportional to nu**(a_nu+n*-1) exp(-b_nu nu)
  Gamma(nu)/Gamma(nu+n)``, normalised numerically on a fine log grid.  The
  production path reaches the same law through the Escobar-West auxiliary
  variable and a two-Gamma mixture; here that augmentation is not used at all,
  so a wrong rate or mixing weight in the Escobar-West step cannot cancel.
  (The plan's Section 7.3 text names the two-Gamma mixture as the reference;
  inverting the EPPF density directly is the same target law with a *different*
  derivation, which is what the test is for.)
* ``phi`` by the inverse CDF of ``Pareto(a_phi + n*, max(b_phi, max alpha*))``
  and ``gamma`` from ``Gamma(a_gamma + d n*, b_gamma + sum 1/lambda*)``, both
  written from the density rather than through a library's parameterisation.
* The posterior functionals are evaluated from this module's **own
  stick-breaking** draw of ``G | theta``, not through
  ``src.dp_weibull.sample_dp_posterior``.

What is *not* naive, on purpose
-------------------------------
The draws are batched in NumPy, a rejection draw returns the first accepted
member of a vectorised proposal batch, and the stick-breaking uses ``J = 200``
atoms.  None of that changes the law being sampled; it only keeps a 10^4-sweep
chain fast enough to run inside the ordinary test suite rather than under
``-m slow``.

Two documented approximations, both far below the test's Monte Carlo noise
-------------------------------------------------------------------------
* ``LAMBDA_MAX = 50``: the box proposal of the continuous piece truncates the
  inverse-gamma tail.  At ``d = 2`` and the prior scales of Section 5.1,
  ``Pr(lambda > 50)`` is of order ``1e-5``, whose effect on any posterior
  summary is far below the Monte Carlo standard error of the stored
  ``10**3``-draw chain.
* The ``q_0`` quadrature uses fixed orders (``_ALPHA_ORDER`` x
  ``_LAMBDA_ORDER``) on a fixed ``log lambda`` window; the
  ``test_q0_quadrature_matches_an_adaptive_rule`` test pins the resulting
  relative error at better than ``1e-7`` against ``scipy.integrate.quad``.

Public API
----------
``ReferencePrior``, ``sample_reference``, ``ReferenceDraws``,
``reference_functionals``, plus the helpers the agreement test uses directly
(``prior_predictive_masses`` for an independent ``q_0`` check,
``log_eppf_conditional`` for the ``nu`` density, and ``draw_nu`` /
``draw_cluster_location`` for the two reference conditionals that are easiest
to mis-derive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Fixed quadrature rules
# ---------------------------------------------------------------------------
# The q_0 rule is a tensor product over the *unit* alpha interval and over
# ``s = log(lambda / gamma)``.  After the substitution lambda = gamma exp(s)
# the base-measure density becomes (1/Gamma(d)) exp(-d s - exp(-s)) ds, which
# is independent of gamma -- so the ``s`` weights are constants and only the
# ``alpha`` nodes have to be rebuilt when phi moves.  The window is
# asymmetric because in ``u = exp(-s)`` the kernel integrand behaves like
# ``u**(d+m)`` for small ``u`` (right end) but is killed by ``exp(-C u)`` only
# far to the left; ``[-8, 12]`` puts both truncation errors below 1e-10 for
# every ``(gamma, phi, t)`` of the test dataset.
_ALPHA_ORDER = 40
_LAMBDA_ORDER = 112
_S_LO, _S_HI = -8.0, 12.0


def _gauss_legendre(order: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of Gauss-Legendre on ``[lo, hi]``."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (hi - lo)
    return half * nodes + 0.5 * (hi + lo), half * weights


_ALPHA_UNIT, _ALPHA_UNIT_W = _gauss_legendre(_ALPHA_ORDER, 0.0, 1.0)
_LOG_LAMBDA, _LOG_LAMBDA_W = _gauss_legendre(_LAMBDA_ORDER, _S_LO, _S_HI)

# ``exp(-d s - exp(-s)) / Gamma(d)`` folded into the ``s`` weights.  It is a
# function of ``d`` only, so it is rebuilt once per call by ``_log_lambda_weights``.
_LOG_LAMBDA_WEIGHTS_CACHE: dict[int, np.ndarray] = {}


def _log_lambda_weights(d: int) -> np.ndarray:
    """Weights of the ``lambda`` integral after ``lambda = gamma exp(s)``."""
    cached = _LOG_LAMBDA_WEIGHTS_CACHE.get(d)
    if cached is not None:
        return cached
    log_kernel = -d * _LOG_LAMBDA - np.exp(-_LOG_LAMBDA) - gammaln(d)
    weights = np.exp(log_kernel) * _LOG_LAMBDA_W
    _LOG_LAMBDA_WEIGHTS_CACHE[d] = weights
    return weights


# Inverse-CDF grid for the ``nu`` conditional.  The EPPF density is smooth and
# strictly positive on (0, inf); a log-spaced grid over nine decades with 4096
# nodes resolves it to well below the Monte Carlo noise of any chain.
_NU_GRID = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 4096))


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferencePrior:
    """The model-(5) hyperparameters, in the same convention as the sampler.

    ``Gamma(a, b)`` is shape-rate, ``IGamma(a, b)`` is SciPy's
    ``invgamma(a, scale=b)`` and ``Pareto(a, b)`` has density
    ``a b**a / phi**(a+1)`` on ``phi >= b``.
    """

    a_nu: float
    b_nu: float
    a_gamma: float
    b_gamma: float
    a_phi: float
    b_phi: float
    d: int = 2


@dataclass(frozen=True)
class ReferenceDraws:
    """Stored draws of the reference chain, on the transformed time scale.

    Attributes
    ----------
    alpha, lam : np.ndarray
        ``float64[B, n]`` per-observation latent Weibull shape and scale.
    nu, gamma, phi : np.ndarray
        ``float64[B]`` hyperparameter trace.
    n_star : np.ndarray
        ``int64[B]`` number of occupied clusters.
    t_max : float
        Maximum of the **original** times, for the inverse map.
    d : int
        Base-measure inverse-gamma shape, carried for the functional draw
        (the ``lambda`` marginal of ``G_0`` is ``IGamma(d, gamma)``).
    """

    alpha: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    gamma: np.ndarray
    phi: np.ndarray
    n_star: np.ndarray
    t_max: float
    d: int = 2

    @property
    def n_draws(self) -> int:
        """Number of stored draws ``B``."""
        return int(self.nu.size)


# ---------------------------------------------------------------------------
# The prior-predictive masses q_0 (Eqs. A.2 and A.3)
# ---------------------------------------------------------------------------
def prior_predictive_masses(
    time: np.ndarray,
    event: np.ndarray,
    *,
    nu: float,
    gamma: float,
    phi: float,
    d: int,
) -> np.ndarray:
    """``q_0^i`` for every observation, by product Gauss-Legendre rule.

    This evaluates

    ``q_0^i = nu * int_0^phi (1/phi) int_0^inf K_i(t_i | alpha, lambda)
      IGamma(lambda | d, gamma) dlambda dalpha``

    with ``K_i`` the Weibull density (uncensored) or survival (right-censored),
    as a tensor-product quadrature in ``(alpha, log lambda)``.  The
    ``lambda`` integrand is written out (``lambda**(-(d+1)) exp(-gamma/lambda)``
    times the kernel), so neither the closed form of the inner integral nor the
    production ``alpha`` grid is used.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    alpha = phi * _ALPHA_UNIT  # (A,)
    lam = gamma * np.exp(_LOG_LAMBDA)  # (L,)
    weights = np.outer(_ALPHA_UNIT_W, _log_lambda_weights(d))  # (A, L)

    tau = time[:, None, None] ** alpha[None, :, None]  # (n, A, L) = t**alpha
    survival = np.exp(-tau / lam[None, None, :])
    density = (
        alpha[None, :, None]
        * time[:, None, None] ** (alpha[None, :, None] - 1.0)
        / lam[None, None, :]
        * survival
    )
    kernel = np.where(event[:, None, None], density, survival)
    return nu * np.einsum("nal,al->n", kernel, weights)


# ---------------------------------------------------------------------------
# Rejection samplers
# ---------------------------------------------------------------------------
_LAMBDA_MAX = 50.0
_REJECTION_BATCH = 512


def _continuous_piece_log_bound(
    t: float, event: bool, gamma: float, phi: float, d: int
) -> float:
    """Log supremum of the unnormalised continuous-piece density on the box.

    With ``g(alpha, lambda) = lambda**(-(d+1)) exp(-gamma/lambda) K(t | alpha,
    lambda)`` (the constants of ``IGamma`` dropped, since they cancel in the
    acceptance ratio), the ``lambda`` supremum is available in closed form:
    ``lambda**(-m) exp(-A/lambda)`` peaks at ``lambda = A/m`` with value
    ``(m/A)**m exp(-m)``, where ``A = gamma + t**alpha`` and ``m = d + 2`` for
    an uncensored observation (the kernel contributes another ``1/lambda``)
    and ``m = d + 1`` otherwise.  The remaining supremum over
    ``alpha in (0, phi)`` is scanned on a 256-node grid.  Everything is done in
    logs, so a large ``phi`` cannot overflow the bound, and the returned value
    carries a 10% margin so that it really is an upper bound of the supremum --
    a conservative bound keeps the rejection sampler exact, it only lowers the
    acceptance rate.
    """
    alpha = phi * (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    log_t = float(np.log(t))
    log_scale = np.logaddexp(np.log(gamma), alpha * log_t)
    m = d + 2 if event else d + 1
    log_value = m * np.log(m) - m * log_scale - m
    if event:
        log_value = log_value + np.log(alpha) + (alpha - 1.0) * log_t
    return float(np.max(log_value) + np.log(1.1))


def draw_continuous_piece(
    t: float,
    event: bool,
    *,
    gamma: float,
    phi: float,
    d: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw ``(alpha, lambda)`` from the continuous piece ``h_i`` by rejection.

    Proposal: uniform on ``(0, phi) x (0, LAMBDA_MAX)``.  The target is the
    base-measure density times the observation's kernel, normalised by nothing
    -- acceptance against its own supremum is what makes the draw exact.  The
    acceptance rate is low (the uniform ``lambda`` proposal ignores where the
    inverse gamma puts its mass), which is the point of the exercise: no grid,
    no inverse-gamma inversion, no conjugacy.

    The acceptance test is evaluated in log space, so a proposal with a huge
    ``alpha`` (whose kernel is ``exp(-t**alpha/lambda)``, i.e. exactly zero for
    every representable ``lambda``) is rejected rather than turned into a
    ``inf * 0`` NaN.
    """
    log_bound = _continuous_piece_log_bound(t, event, gamma, phi, d)
    log_t = float(np.log(t))
    while True:
        alpha = rng.uniform(0.0, phi, _REJECTION_BATCH)
        # A uniform proposal can return exactly 0 with probability ~2**-53;
        # the floor keeps the log finite and truncates a set of measure zero.
        lam = np.maximum(rng.uniform(0.0, _LAMBDA_MAX, _REJECTION_BATCH), 1e-12)
        log_lam = np.log(lam)
        with np.errstate(over="ignore"):
            log_kernel = -np.exp(alpha * log_t - log_lam)
        if event:
            log_kernel = log_kernel + np.log(alpha) + (alpha - 1.0) * log_t - log_lam
        log_target = -(d + 1.0) * log_lam - gamma / lam + log_kernel
        accepted = log_target >= np.log(rng.random(_REJECTION_BATCH)) + log_bound
        if np.any(accepted):
            index = int(np.argmax(accepted))
            return float(alpha[index]), float(lam[index])


def draw_cluster_location(
    t: np.ndarray,
    event: np.ndarray,
    *,
    gamma: float,
    phi: float,
    d: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw one cluster's ``(alpha*, lambda*)`` from the exact joint A.5 law.

    The target is ``g_0(alpha, lambda | phi, gamma) prod_{i in j} L_i``.  The
    proposal draws ``alpha ~ U(0, phi)`` and ``lambda`` from the conjugate
    ``IGamma(d + n_j^o, gamma + sum_i t_i**alpha)``.  Written out, the
    target-to-proposal ratio is

    ``alpha**n_j^o exp((alpha - 1) sum_{i_o in j} log t_{i_o})
      (gamma + sum_i t_i**alpha)**(-(d + n_j^o))``

    up to a constant.  The last factor is what makes the sampler correct: it
    is the normalising constant of the conjugate proposal, and dropping it
    lets ``alpha`` drift to ``phi`` (and ``lambda`` to infinity) -- the exact
    runaway this reference implementation exhibited while it was being
    written.  The ratio is bounded on ``(0, phi)``; the bound is scanned on a
    256-node grid with a 5% margin, and only the accepted ``alpha`` consumes
    an inverse-gamma variate.
    """
    t = np.asarray(t, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    n_uncensored = int(np.count_nonzero(event))
    log_t = np.log(t)
    log_sum_log_t = float(np.sum(log_t[event])) if n_uncensored else 0.0
    log_gamma = float(np.log(gamma))

    def log_ratio(alpha: np.ndarray) -> np.ndarray:
        log_scale = np.logaddexp(
            log_gamma, np.logaddexp.reduce(alpha[:, None] * log_t[None, :], axis=1)
        )
        return (
            n_uncensored * np.log(alpha)
            + (alpha - 1.0) * log_sum_log_t
            - (d + n_uncensored) * log_scale
        )

    scan = phi * (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    log_bound = float(np.max(log_ratio(scan))) + float(np.log(1.05))
    while True:
        alpha = rng.uniform(0.0, phi, _REJECTION_BATCH)
        accepted = log_ratio(alpha) >= np.log(rng.random(_REJECTION_BATCH)) + log_bound
        if np.any(accepted):
            chosen = float(alpha[int(np.argmax(accepted))])
            scale = gamma + float(np.sum(t**chosen))
            shape = d + n_uncensored
            return chosen, scale / float(rng.gamma(shape=shape))


# ---------------------------------------------------------------------------
# Hyperparameter conditionals
# ---------------------------------------------------------------------------
def log_eppf_conditional(
    nu: np.ndarray | float, n: int, n_star: int, prior: ReferencePrior
) -> np.ndarray:
    """Log density of ``[nu | s]`` up to an additive constant.

    The exchangeable partition probability function contributes
    ``nu**n_star Gamma(nu) / Gamma(nu + n)``; with the ``Gamma(a_nu, b_nu)``
    prior on ``nu`` the density is

    ``p(nu | s) proportional to nu**(a_nu + n_star - 1)
      exp(-b_nu nu) Gamma(nu) / Gamma(nu + n)``.
    """
    nu = np.asarray(nu, dtype=np.float64)
    return (
        (prior.a_nu + n_star - 1.0) * np.log(nu)
        - prior.b_nu * nu
        + gammaln(nu)
        - gammaln(nu + n)
    )


def draw_nu(
    n: int, n_star: int, prior: ReferencePrior, rng: np.random.Generator
) -> float:
    """Draw ``nu`` by inverse CDF of the exact EPPF conditional.

    The normalising constant is a numerical quadrature over the fixed log grid
    ``_NU_GRID``.  The density is weighted by the local grid width
    ``dnu`` before the cumulative sum: on a *log-spaced* grid the node spacing
    grows with ``nu``, so normalising the bare density values would put too
    much mass in the small-``nu`` end (this reference implementation did
    exactly that, which biased ``E[nu]`` down by 30% until it was caught by
    comparing against the Escobar-West two-Gamma scheme).  The draw then
    interpolates linearly inside a grid cell, whose width is ``0.45%`` of
    ``nu`` in relative terms, so the interpolation bias is negligible next to
    the Monte Carlo noise of the chain.
    """
    log_density = log_eppf_conditional(_NU_GRID, n, n_star, prior)
    density = np.exp(log_density - float(np.max(log_density)))
    weight = density * np.gradient(_NU_GRID)
    cdf = np.cumsum(weight)
    cdf /= cdf[-1]
    position = rng.random()
    index = int(np.searchsorted(cdf, position))
    index = min(max(index, 1), _NU_GRID.size - 1)
    low, high = cdf[index - 1], cdf[index]
    fraction = 0.0 if high <= low else (position - low) / (high - low)
    nu_lo, nu_hi = _NU_GRID[index - 1], _NU_GRID[index]
    return float(nu_lo + fraction * (nu_hi - nu_lo))


def draw_phi(
    n_star: int, alpha_star: np.ndarray, prior: ReferencePrior, rng: np.random.Generator
) -> float:
    """``Pareto(a_phi + n_star, max(b_phi, max_j alpha*_j))`` by inverse CDF."""
    lower = max(prior.b_phi, float(np.max(alpha_star)))
    shape = prior.a_phi + n_star
    uniform = rng.random()
    return float(lower * (1.0 - uniform) ** (-1.0 / shape))


def draw_gamma(
    n_star: int, lam_star: np.ndarray, prior: ReferencePrior, rng: np.random.Generator
) -> float:
    """``Gamma(a_gamma + d n_star, rate = b_gamma + sum_j 1/lambda*_j)``."""
    shape = prior.a_gamma + prior.d * n_star
    rate = prior.b_gamma + float(np.sum(1.0 / np.asarray(lam_star, dtype=np.float64)))
    return float(rng.gamma(shape=shape, scale=1.0 / rate))


# ---------------------------------------------------------------------------
# The reference chain
# ---------------------------------------------------------------------------
def _labels_and_locations(
    s: np.ndarray, alpha: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted occupied labels and their locations (the first member's values)."""
    labels = np.unique(s)
    alpha_star = np.empty(labels.size, dtype=np.float64)
    lam_star = np.empty(labels.size, dtype=np.float64)
    sizes = np.empty(labels.size, dtype=np.int64)
    for position, label in enumerate(labels):
        members = np.flatnonzero(s == label)
        alpha_star[position] = alpha[members[0]]
        lam_star[position] = lam[members[0]]
        sizes[position] = members.size
    return labels, alpha_star, lam_star, sizes


def _compact(s: np.ndarray) -> np.ndarray:
    """Relabel the occupied clusters ``0 .. n_star-1`` in sorted label order."""
    _, inverse = np.unique(s, return_inverse=True)
    return inverse.astype(np.intp)


@dataclass
class _ReferenceState:
    """Mutable chain state: per-observation ``(alpha, lambda)`` and labels."""

    alpha: np.ndarray
    lam: np.ndarray
    s: np.ndarray
    nu: float
    gamma: float
    phi: float

    @property
    def n_star(self) -> int:
        """Number of occupied clusters."""
        return int(np.unique(self.s).size)


def _kernels_at(
    t: float, event: bool, alpha_star: np.ndarray, lam_star: np.ndarray
) -> np.ndarray:
    """``q_j^i`` for one observation against every cluster location."""
    tau = t**alpha_star
    survival = np.exp(-tau / lam_star)
    if event:
        return alpha_star * t ** (alpha_star - 1.0) / lam_star * survival
    return survival


def _step_a(
    state: _ReferenceState,
    time: np.ndarray,
    event: np.ndarray,
    prior: ReferencePrior,
    rng: np.random.Generator,
) -> None:
    """Sequential Po'lya-urn resampling of every ``(alpha_i, lambda_i, s_i)``.

    The cluster set is rebuilt from scratch after removing observation ``i`` --
    a singleton therefore disappears with weight zero, exactly as the urn
    mixture requires -- and the labels are compacted after every assignment.
    """
    q0 = prior_predictive_masses(
        time, event, nu=state.nu, gamma=state.gamma, phi=state.phi, d=prior.d
    )
    for i in range(time.size):
        s = state.s.copy()
        s[i] = -1  # observation i is out of the urn while its weights are formed
        labels, alpha_star, lam_star, sizes = _labels_and_locations(
            s[s >= 0], state.alpha[s >= 0], state.lam[s >= 0]
        )
        kernel = _kernels_at(time[i], bool(event[i]), alpha_star, lam_star)
        weights = np.concatenate(([q0[i]], sizes * kernel))
        weights = weights / weights.sum()
        index = int(np.searchsorted(np.cumsum(weights), rng.random()))
        index = min(index, weights.size - 1)
        if index == 0:
            alpha, lam = draw_continuous_piece(
                float(time[i]),
                bool(event[i]),
                gamma=state.gamma,
                phi=state.phi,
                d=prior.d,
                rng=rng,
            )
            state.alpha[i] = alpha
            state.lam[i] = lam
            state.s[i] = -2  # a fresh label; compact below makes it unique
        else:
            state.alpha[i] = alpha_star[index - 1]
            state.lam[i] = lam_star[index - 1]
            state.s[i] = labels[index - 1]
        state.s = _compact(state.s)


def _step_b(
    state: _ReferenceState,
    time: np.ndarray,
    event: np.ndarray,
    prior: ReferencePrior,
    rng: np.random.Generator,
) -> None:
    """Refresh every cluster's ``(alpha*, lambda*)`` from the exact joint law."""
    labels = np.unique(state.s)
    for label in labels:
        members = np.flatnonzero(state.s == label)
        alpha, lam = draw_cluster_location(
            time[members],
            event[members],
            gamma=state.gamma,
            phi=state.phi,
            d=prior.d,
            rng=rng,
        )
        state.alpha[members] = alpha
        state.lam[members] = lam


def _step_c(
    state: _ReferenceState, prior: ReferencePrior, rng: np.random.Generator
) -> None:
    """Refresh ``nu``, then ``phi``, then ``gamma``."""
    n = state.s.size
    n_star = state.n_star
    _, alpha_star, lam_star, _ = _labels_and_locations(state.s, state.alpha, state.lam)
    state.nu = draw_nu(n, n_star, prior, rng)
    state.phi = draw_phi(n_star, alpha_star, prior, rng)
    state.gamma = draw_gamma(n_star, lam_star, prior, rng)


def _init_state(
    time: np.ndarray, prior: ReferencePrior, rng: np.random.Generator
) -> _ReferenceState:
    """Start from singletons with locations drawn from ``G_0`` at its centre."""
    n = time.size
    gamma = prior.a_gamma / prior.b_gamma
    phi = prior.b_phi
    alpha = rng.uniform(0.0, phi, n)
    lam = 1.0 / rng.gamma(shape=prior.d, scale=1.0 / gamma, size=n)
    return _ReferenceState(
        alpha=alpha,
        lam=lam,
        s=np.arange(n, dtype=np.intp),
        nu=prior.a_nu / prior.b_nu,
        gamma=gamma,
        phi=phi,
    )


def sample_reference(
    time: np.ndarray,
    event: np.ndarray,
    prior: ReferencePrior,
    *,
    burn_in: int,
    n_save: int,
    thin: int,
    seed: int,
) -> ReferenceDraws:
    """Run the reference chain and return its stored draws.

    The data are transformed to ``t_tilde = 1 + t / max(t)`` first, exactly as
    ``src.dp_gibbs.make_surv_data`` does (the transform is part of the model,
    not of the implementation being checked).
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if time.size != event.size:
        raise ValueError("time and event must have the same length")
    if time.size == 0 or np.any(time <= 0.0):
        raise ValueError("time must be non-empty and strictly positive")
    t_max = float(np.max(time))
    transformed = 1.0 + time / t_max

    rng = np.random.default_rng(seed)
    state = _init_state(transformed, prior, rng)
    n = time.size

    for _ in range(burn_in):
        _step_a(state, transformed, event, prior, rng)
        _step_b(state, transformed, event, prior, rng)
        _step_c(state, prior, rng)

    alpha = np.empty((n_save, n), dtype=np.float64)
    lam = np.empty((n_save, n), dtype=np.float64)
    nu = np.empty(n_save, dtype=np.float64)
    gamma = np.empty(n_save, dtype=np.float64)
    phi = np.empty(n_save, dtype=np.float64)
    n_star = np.empty(n_save, dtype=np.int64)
    for draw in range(n_save):
        for _ in range(thin):
            _step_a(state, transformed, event, prior, rng)
            _step_b(state, transformed, event, prior, rng)
            _step_c(state, prior, rng)
        alpha[draw] = state.alpha
        lam[draw] = state.lam
        nu[draw] = state.nu
        gamma[draw] = state.gamma
        phi[draw] = state.phi
        n_star[draw] = state.n_star
    return ReferenceDraws(
        alpha=alpha,
        lam=lam,
        nu=nu,
        gamma=gamma,
        phi=phi,
        n_star=n_star,
        t_max=t_max,
        d=prior.d,
    )


# ---------------------------------------------------------------------------
# Posterior functionals from the reference draws
# ---------------------------------------------------------------------------
def reference_functionals(
    draws: ReferenceDraws,
    t_grid: np.ndarray,
    *,
    n_atoms: int = 200,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """``S``, ``f`` and ``h`` of each draw on the original time scale.

    For each stored draw ``G_b ~ [G | theta_b, nu_b, gamma_b, phi_b]`` is a
    ``DP(nu_b + n, G_0')`` with ``G_0' = (nu_b G_0 + sum_i delta_theta_i) /
    (nu_b + n)``, so its first ``n_atoms`` atoms are drawn by this module's own
    stick-breaking: ``v_j ~ Beta(1, nu_b + n)``, and the atom is a fresh
    ``G_0`` draw with probability ``nu_b / (nu_b + n)`` or one of the ``n``
    observations' locations otherwise.  The residual stick mass after
    ``n_atoms`` atoms is below ``1e-6`` for the precisions of this test.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_draws = draws.n_draws
    n = draws.alpha.shape[1]
    rng = np.random.default_rng(seed)

    precision = draws.nu + n  # (B,)
    sticks = rng.beta(1.0, precision[:, None], size=(n_draws, n_atoms))  # (B, J)
    log_sticks = np.log(sticks)
    log_survival = np.concatenate(
        (
            np.zeros((n_draws, 1)),
            np.cumsum(np.log1p(-sticks), axis=1)[:, :-1],
        ),
        axis=1,
    )
    weights = np.exp(log_sticks + log_survival)  # (B, J)

    fresh = rng.random((n_draws, n_atoms)) < (draws.nu / precision)[:, None]
    picked = rng.integers(0, n, size=(n_draws, n_atoms))
    alpha = np.where(
        fresh,
        rng.uniform(0.0, 1.0, (n_draws, n_atoms)) * draws.phi[:, None],
        np.take_along_axis(draws.alpha, picked, axis=1),
    )
    lam_fresh = 1.0 / rng.gamma(
        shape=draws.d,
        scale=1.0 / draws.gamma[:, None],
        size=(n_draws, n_atoms),
    )
    lam = np.where(fresh, lam_fresh, np.take_along_axis(draws.lam, picked, axis=1))

    t_tilde = 1.0 + t_grid / draws.t_max  # (M,)
    tau = t_tilde[None, None, :] ** alpha[:, :, None]  # (B, J, M)
    base = np.exp(-tau / lam[:, :, None])
    survival = np.einsum("bj,bjm->bm", weights, base)
    density = np.einsum(
        "bj,bjm->bm",
        weights,
        alpha[:, :, None]
        / lam[:, :, None]
        * t_tilde[None, None, :] ** (alpha - 1.0)[:, :, None]
        * base,
    )
    return {
        "S": survival,
        "f": density / draws.t_max,
        "h": density / survival / draws.t_max,
    }
