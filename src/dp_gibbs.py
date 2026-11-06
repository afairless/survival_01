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

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "AlphaGrid",
    "DPPrior",
    "DPSettings",
    "DPState",
    "StateInvariantError",
    "SurvData",
    "inverse_map",
    "inverse_time",
    "log_alpha_integrand_censored",
    "log_alpha_integrand_uncensored",
    "log_integral_alpha",
    "draw_g0_conditional_alpha",
    "log_q0_censored",
    "log_q0_uncensored",
    "make_grid",
    "make_surv_data",
    "alpha_bound_guard_count",
    "refresh_cluster",
    "refresh_cluster_alpha",
    "reset_alpha_bound_guard_count",
    "sample_step_a",
    "sample_step_a_for_obs",
    "sample_step_b",
    "trapezoid_log_weights",
    "transform_time",
]


# Diagnostic counter for the ``alpha*`` bounds guard of plan Section 3.5.  It
# is module state rather than a ``DPState`` field because it is a property of
# the sampler's arithmetic, not of a chain: the counter is read by tests and by
# the diagnostics note, and reset explicitly before a run that reports it.
_ALPHA_BOUND_GUARD_CLAMPS = 0


def alpha_bound_guard_count() -> int:
    """Return how many ``alpha*`` draws were clamped by the bounds guard.

    Plan Section 3.5 requires that an empty interval
    ``alpha^-_j < alpha*_j < alpha^+_j`` -- which can only arise from
    floating-point equality at a boundary, never from the mathematics -- be
    clamped to the midpoint rather than fed to a degenerate uniform draw, and
    that the event be counted instead of silently dropped.  A non-zero count is
    a diagnostic, not an error: it says a cluster's shape was frozen for one
    sweep.
    """
    return _ALPHA_BOUND_GUARD_CLAMPS


def reset_alpha_bound_guard_count() -> None:
    """Reset the :func:`alpha_bound_guard_count` diagnostic to zero."""
    global _ALPHA_BOUND_GUARD_CLAMPS
    _ALPHA_BOUND_GUARD_CLAMPS = 0


@dataclass(frozen=True)
class SurvData:
    """Survival data on the transformed time scale, ready for the sampler.

    The Appendix requires ``t > 1`` so that every ``log t`` appearing in the
    ``alpha^-`` bound is strictly positive.  ``make_surv_data`` applies the
    paper's transform

        ``t_tilde = 1 + t / t_max``

    once, at ingestion, and records the original ``t_max`` so that every
    functional can be mapped back (plan Section 2.3).

    Attributes
    ----------
    time : np.ndarray
        ``float64[n]`` transformed times, strictly inside ``(1, 2]`` for
        strictly positive original times.
    event : np.ndarray
        ``bool[n]``; ``True`` marks an uncensored observation (density) and
        ``False`` a right-censored one (survival).
    t_max : float
        Maximum of the **original** times, used by the inverse map.

    The container validates its own invariants so that a hand-built instance
    cannot leak a NaN, a non-positive time or a bad ``t_max`` into the chain.
    Ingest raw data through :func:`make_surv_data`; construct ``SurvData``
    directly only from already-transformed arrays.
    """

    time: np.ndarray
    event: np.ndarray
    t_max: float

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        if time.ndim != 1:
            raise ValueError("time must be a 1-D array")
        if time.size == 0:
            raise ValueError("time must not be empty")
        event = _as_event_array(self.event)
        if time.size != event.size:
            raise ValueError(
                "time and event must have the same length, "
                f"got {time.size} and {event.size}"
            )
        if not np.all(np.isfinite(time)):
            raise ValueError("time must contain only finite values (no NaN or inf)")
        if np.any(time <= 1.0):
            raise ValueError("time must be greater than 1 on the transformed scale")
        try:
            t_max = float(self.t_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "t_max must be a finite, strictly positive number"
            ) from exc
        if not np.isfinite(t_max) or t_max <= 0.0:
            raise ValueError("t_max must be a finite, strictly positive number")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "t_max", t_max)


def _as_event_array(event: object) -> np.ndarray:
    """Normalise a censoring indicator to a 1-D ``bool`` array.

    Booleans are accepted as-is.  Numeric arrays are accepted when every entry
    is exactly ``0`` or ``1`` (the usual encoding in stored data) and are then
    converted to ``bool``.  Everything else -- strings, objects, values such as
    ``2`` or ``NaN`` -- is rejected, because a silently coerced indicator would
    turn a censored observation into an event and corrupt the likelihood.
    """
    event = np.asarray(event)
    if event.ndim != 1:
        raise ValueError("event must be a 1-D array")
    if event.size == 0:
        raise ValueError("event must not be empty")
    if event.dtype == np.bool_:
        return event
    if not np.issubdtype(event.dtype, np.number):
        raise ValueError("event must be boolean or contain only 0 and 1")
    if not np.all(np.isin(event, (0, 1))):
        raise ValueError("event must be boolean or contain only 0 and 1")
    return event.astype(bool)


def transform_time(data: SurvData, time: np.ndarray) -> np.ndarray:
    """Apply ``t_tilde = 1 + t / t_max`` to original-scale times."""
    return 1.0 + np.asarray(time, dtype=np.float64) / data.t_max


def inverse_time(data: SurvData, time_tilde: np.ndarray) -> np.ndarray:
    """Map ``t_tilde = 1 + t / t_max`` back to the original time scale."""
    return data.t_max * (np.asarray(time_tilde, dtype=np.float64) - 1.0)


def make_surv_data(time: object, event: object) -> SurvData:
    """Validate raw survival data and transform the times once.

    This is the ingestion boundary between stored data and the sampler.  It

    1. rejects empty or non-1-D inputs, length mismatches and non-finite times
       (a ``NaN`` would pass a bare ``time <= 0`` check and then poison
       ``t_max``, the alpha grid and the whole chain);
    2. rejects non-positive times, since ``t = 0`` would give ``t_tilde = 1``
       and ``log t_tilde = 0``, which divides by zero in the ``alpha^-`` bound;
    3. normalises the censoring indicator to ``bool`` (booleans or 0/1);
    4. applies ``t_tilde = 1 + t / max(t)`` and records ``max(t)``.

    Parameters
    ----------
    time : array-like
        Original-scale survival or censoring times, ``float64[n]`` after
        conversion, strictly positive and finite.
    event : array-like
        Censoring indicator, ``bool[n]`` or numeric ``0/1``.

    Returns
    -------
    SurvData
        Transformed times strictly inside ``(1, 2]``, boolean events and the
        original maximum needed by the inverse map.

    Raises
    ------
    ValueError
        If any of the rules above is violated.  The message names the input
        and the rule, so a caller can report a malformed dataset directly.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1:
        raise ValueError("time must be a 1-D array")
    if time.size == 0:
        raise ValueError("time must not be empty")
    event = _as_event_array(event)
    if time.size != event.size:
        raise ValueError(
            "time and event must have the same length, "
            f"got {time.size} and {event.size}"
        )
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values (no NaN or inf)")
    if np.any(time <= 0.0):
        raise ValueError("time must be strictly positive")
    t_max = float(np.max(time))
    transformed = 1.0 + time / t_max
    return SurvData(time=transformed, event=event, t_max=t_max)


def inverse_map(
    data: SurvData,
    t_grid: np.ndarray,
    *,
    survival: np.ndarray | None = None,
    density: np.ndarray | None = None,
    hazard: np.ndarray | None = None,
    quantiles: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Map functional output from the ``t_tilde`` scale back to original time.

    The four maps of plan Section 2.3 live here, in one place, so the Jacobian
    is applied exactly once and ``S`` is never scaled by mistake::

        S(s)     = S_tilde(1 + s / t_max)
        f(s)     = f_tilde(1 + s / t_max) / t_max
        h(s)     = h_tilde(1 + s / t_max) / t_max
        eta_p    = t_max * (eta_tilde_p - 1)

    ``survival``, ``density`` and ``hazard`` must already be evaluated at the
    transformed grid ``1 + t_grid / t_max``, which is returned under
    ``"time"``.  Survival and CDF values are unchanged by the transform;
    density and hazard carry the ``1 / t_max`` Jacobian; quantiles go through
    the full inverse map.  Keys for arguments left as ``None`` are omitted.

    Parameters
    ----------
    data : SurvData
        The dataset whose ``t_max`` defines the map.
    t_grid : np.ndarray
        Original-scale time grid at which the functionals were evaluated.
    survival, density, hazard : np.ndarray, optional
        Functional draws evaluated on the transformed grid.
    quantiles : np.ndarray, optional
        Quantile draws on the transformed scale.

    Returns
    -------
    dict of str to np.ndarray
        Always ``"time"`` (the transformed evaluation grid) and
        ``"jacobian"`` (``1 / t_max``), plus each supplied functional under
        ``"survival"``, ``"density"``, ``"hazard"`` and ``"quantiles"``.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    result: dict[str, np.ndarray] = {
        "time": 1.0 + t_grid / data.t_max,
        "jacobian": np.asarray(1.0 / data.t_max),
    }
    if survival is not None:
        result["survival"] = np.asarray(survival, dtype=np.float64)
    if density is not None:
        result["density"] = np.asarray(density, dtype=np.float64) / data.t_max
    if hazard is not None:
        result["hazard"] = np.asarray(hazard, dtype=np.float64) / data.t_max
    if quantiles is not None:
        result["quantiles"] = inverse_time(data, quantiles)
    return result


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


class StateInvariantError(AssertionError):
    """Raised when a :class:`DPState` violates one of the Section 2.1 invariants.

    The class subclasses ``AssertionError`` so that debug ``assert`` calls and
    explicit checks raise the same exception type, and so that a caller can
    catch a state corruption without also swallowing ordinary ``ValueError``
    input-validation failures.
    """


def _as_float_vector(values: object, name: str) -> np.ndarray:
    """Normalise a state vector to an owned, finite ``float64`` 1-D array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return np.array(arr, dtype=np.float64, copy=True)


def _as_index_vector(values: object, name: str) -> np.ndarray:
    """Normalise a non-negative integer state vector to an owned 1-D array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    out = arr.astype(np.intp, copy=True)
    if np.any(out < 0):
        raise ValueError(f"{name} must be non-negative")
    return out


def _as_positive_scalar(value: float, name: str) -> float:
    """Normalise a strictly positive, finite scalar state entry."""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite, strictly positive number")
    return out


@dataclass
class DPState:
    """Latent state of one collapsed Polya-urn chain (plan Section 2.1).

    Fields
    ------
    alpha, lam : np.ndarray
        ``float64[n]`` per-observation Weibull shape and scale (the paper's
        ``lambda`` parameterisation).
    s : np.ndarray
        ``int[n]`` cluster label of each observation, ``0 <= s_i < n_star``.
    alpha_star, lam_star : np.ndarray
        ``float64[n_star]`` cluster location shape and scale.
    n_j : np.ndarray
        ``int[n_star]`` cluster sizes, ``sum(n_j) == n`` whenever every
        observation is assigned.
    nu, gamma, phi : float
        DP precision and the two ``G0`` hyperparameters.

    The Section 2.1 invariants are checked at construction and by
    :meth:`check_invariants`:

    1. ``alpha == alpha_star[s]`` and ``lam == lam_star[s]``;
    2. ``n_j == bincount(s)``, no empty cluster, labels are ``0..n_star-1``;
    3. ``0 < alpha <= phi`` and ``0 < alpha_star <= phi``;
    4. ``lam > 0``, ``lam_star > 0``, ``nu > 0``, ``gamma > 0`` and
       ``phi >= prior.b_phi`` when a prior is supplied.

    Bookkeeping follows the deferred-compaction variant explicitly allowed by
    plan Section 3.4: :meth:`remove_observation` decrements a cluster count but
    leaves the observation labelled (so the caller can form the urn weights
    against the pre-removal labels), and :meth:`compact` performs the single
    relabel pass that drops empty clusters and renumbers the rest.  Invariants
    1 and 4 hold at all times; invariants 2 and 3 hold once every removed
    observation has been reassigned and :meth:`compact` has run.
    """

    alpha: np.ndarray
    lam: np.ndarray
    s: np.ndarray
    alpha_star: np.ndarray
    lam_star: np.ndarray
    n_j: np.ndarray
    nu: float
    gamma: float
    phi: float

    def __post_init__(self) -> None:
        alpha = _as_float_vector(self.alpha, "alpha")
        lam = _as_float_vector(self.lam, "lam")
        s = _as_index_vector(self.s, "s")
        alpha_star = _as_float_vector(self.alpha_star, "alpha_star")
        lam_star = _as_float_vector(self.lam_star, "lam_star")
        n_j = _as_index_vector(self.n_j, "n_j")
        nu = _as_positive_scalar(self.nu, "nu")
        gamma = _as_positive_scalar(self.gamma, "gamma")
        phi = _as_positive_scalar(self.phi, "phi")

        if alpha.size != s.size:
            raise ValueError(
                f"alpha and s must have the same length, got {alpha.size} and {s.size}"
            )
        if lam.size != s.size:
            raise ValueError(
                f"lam and s must have the same length, got {lam.size} and {s.size}"
            )
        if alpha_star.size != lam_star.size:
            raise ValueError(
                "alpha_star and lam_star must have the same length, "
                f"got {alpha_star.size} and {lam_star.size}"
            )
        if alpha_star.size != n_j.size:
            raise ValueError(
                "alpha_star and n_j must have the same length, "
                f"got {alpha_star.size} and {n_j.size}"
            )

        self.alpha = alpha
        self.lam = lam
        self.s = s
        self.alpha_star = alpha_star
        self.lam_star = lam_star
        self.n_j = n_j
        self.nu = nu
        self.gamma = gamma
        self.phi = phi
        self.check_invariants()

    @property
    def n(self) -> int:
        """Number of observations in the chain."""
        return int(self.s.size)

    @property
    def n_star(self) -> int:
        """Number of clusters currently carried by the state."""
        return int(self.alpha_star.size)

    def check_invariants(self, prior: DPPrior | None = None) -> None:
        """Raise :class:`StateInvariantError` unless Section 2.1 holds.

        Parameters
        ----------
        prior : DPPrior, optional
            When supplied, the ``phi >= b_phi`` part of invariant 4 is also
            enforced.  Without a prior the state cannot know the Pareto lower
            bound, so that single bound is skipped.

        Notes
        -----
        This is deliberately a full, explicit check: it is cheap enough to run
        at the end of a sweep in debug mode and its messages name the violated
        invariant, which makes a bookkeeping bug point at the operation that
        caused it rather than at a later ``NaN``.
        """
        if self.alpha.shape != (self.n,) or self.lam.shape != (self.n,):
            raise StateInvariantError(
                "invariant 1 violated: alpha and lam must be 1-D arrays of length n"
            )
        if self.s.shape != (self.n,):
            raise StateInvariantError(
                "invariant 2 violated: s must be a 1-D array of length n"
            )
        if self.alpha_star.shape != (self.n_star,):
            raise StateInvariantError(
                "invariant 1 violated: alpha_star must be 1-D of length n_star"
            )
        if self.lam_star.shape != (self.n_star,) or self.n_j.shape != (self.n_star,):
            raise StateInvariantError(
                "invariant 2 violated: lam_star and n_j must have length n_star"
            )
        if self.n_star == 0:
            raise StateInvariantError("invariant 2 violated: the state has no clusters")
        if np.any(self.s < 0) or np.any(self.s >= self.n_star):
            raise StateInvariantError(
                "invariant 2 violated: cluster labels must lie in 0..n_star-1"
            )
        if not np.array_equal(self.alpha, self.alpha_star[self.s]):
            raise StateInvariantError(
                "invariant 1 violated: alpha is not alpha_star[s]"
            )
        if not np.array_equal(self.lam, self.lam_star[self.s]):
            raise StateInvariantError("invariant 1 violated: lam is not lam_star[s]")
        counts = np.bincount(self.s, minlength=self.n_star)
        if not np.array_equal(self.n_j, counts):
            raise StateInvariantError("invariant 2 violated: n_j is not bincount(s)")
        if np.any(self.n_j == 0):
            raise StateInvariantError(
                "invariant 2 violated: every cluster must be non-empty"
            )
        if int(self.n_j.sum()) != self.n:
            raise StateInvariantError(
                "invariant 2 violated: sum(n_j) must equal the number of observations"
            )
        if np.any(self.alpha <= 0.0) or np.any(self.alpha_star <= 0.0):
            raise StateInvariantError(
                "invariant 3 violated: alpha and alpha_star must be positive"
            )
        if np.any(self.alpha > self.phi) or np.any(self.alpha_star > self.phi):
            raise StateInvariantError(
                "invariant 3 violated: alpha and alpha_star must not exceed phi"
            )
        if np.any(self.lam <= 0.0) or np.any(self.lam_star <= 0.0):
            raise StateInvariantError(
                "invariant 4 violated: lam, lam_star, nu and gamma must be positive"
            )
        finite_scalars = (
            np.isfinite(self.nu) and np.isfinite(self.gamma) and np.isfinite(self.phi)
        )
        if not finite_scalars:
            raise StateInvariantError(
                "invariant 4 violated: nu, gamma and phi must be finite"
            )
        if self.nu <= 0.0 or self.gamma <= 0.0 or self.phi <= 0.0:
            raise StateInvariantError(
                "invariant 4 violated: nu, gamma and phi must be positive"
            )
        if prior is not None and self.phi < prior.b_phi:
            raise StateInvariantError(
                "invariant 4 violated: phi must be at least prior.b_phi"
            )

    def compact(self) -> np.ndarray:
        """Drop empty clusters and relabel the rest to ``0..m-1``.

        Returns
        -------
        np.ndarray
            ``int[n_star_old]`` map from pre-compaction labels to post-
            compaction labels.  A dropped (empty) cluster maps to ``-1``.  A
            caller holding an ``(n, n_star)`` matrix (the cached ``log q``
            columns of plan Section 3.4) permutes its columns with
            ``old_to_new[old_to_new >= 0]``.

        Raises
        ------
        StateInvariantError
            If an observation still carries the label of a cluster whose count
            has already reached zero.  That means compaction ran between
            :meth:`remove_observation` and the matching reassignment, which
            would silently drop a live observation.
        """
        keep = np.flatnonzero(self.n_j > 0)
        old_to_new = np.full(self.n_star, -1, dtype=np.intp)
        old_to_new[keep] = np.arange(keep.size, dtype=np.intp)
        remapped = old_to_new[self.s]
        if np.any(remapped < 0):
            raise StateInvariantError(
                "cannot compact: an observation still labels an empty cluster; "
                "reassign every removed observation first"
            )
        if int(self.n_j.sum()) != self.n:
            raise StateInvariantError(
                "cannot compact: sum(n_j) does not equal the number of observations"
            )
        self.s = remapped
        self.alpha_star = self.alpha_star[keep]
        self.lam_star = self.lam_star[keep]
        self.n_j = self.n_j[keep]
        self.check_invariants()
        return old_to_new

    def remove_observation(self, i: int) -> int:
        """Decrement the count of observation ``i``'s cluster.

        The observation keeps its old label and latent values: this is the
        deferred-compaction variant of plan Section 3.4, so the caller can
        keep indexing the candidate clusters by pre-removal labels while it
        forms the urn weights.  Reassign the observation with
        :meth:`assign_existing` or :meth:`assign_new`, then :meth:`compact`
        once per sweep.

        Returns
        -------
        int
            The observation's old (pre-removal) cluster label.
        """
        if not 0 <= i < self.n:
            raise IndexError(f"observation index {i} is out of range for n={self.n}")
        j = int(self.s[i])
        if self.n_j[j] <= 0:
            raise StateInvariantError(
                f"observation {i} is not counted in cluster {j}; already removed?"
            )
        self.n_j[j] -= 1
        return j

    def assign_existing(self, i: int, j: int) -> int:
        """Assign removed observation ``i`` to existing cluster ``j``.

        Copies the cluster location into the per-observation latent values, so
        invariant 1 holds immediately.  The observation must have been removed
        first; the guard on ``sum(n_j)`` catches the common double-count bug.
        """
        if not 0 <= i < self.n:
            raise IndexError(f"observation index {i} is out of range for n={self.n}")
        if not 0 <= j < self.n_star:
            raise IndexError(
                f"cluster index {j} is out of range for n_star={self.n_star}"
            )
        if int(self.n_j.sum()) >= self.n:
            raise StateInvariantError(
                "assign_existing called before remove_observation"
            )
        self.s[i] = j
        self.alpha[i] = self.alpha_star[j]
        self.lam[i] = self.lam_star[j]
        self.n_j[j] += 1
        return j

    def append_cluster(self, alpha: float, lam: float) -> int:
        """Append an empty cluster at location ``(alpha, lam)``.

        Returns the new label, ``n_star - 1``.  The caller must assign one or
        more observations to it before :meth:`compact`, or the new cluster is
        dropped as empty.
        """
        alpha = _as_positive_scalar(alpha, "alpha")
        lam = _as_positive_scalar(lam, "lam")
        if alpha > self.phi:
            raise StateInvariantError("invariant 3 violated: alpha must not exceed phi")
        self.alpha_star = np.append(self.alpha_star, alpha)
        self.lam_star = np.append(self.lam_star, lam)
        self.n_j = np.append(self.n_j, np.intp(0))
        return self.n_star - 1

    def assign_new(self, i: int, alpha: float, lam: float) -> int:
        """Assign removed observation ``i`` to a fresh cluster.

        Appends the cluster and assigns the observation in one step, returning
        the new label.  This is the step-(a) "new cluster" branch.
        """
        j = self.append_cluster(alpha, lam)
        return self.assign_existing(i, j)


def trapezoid_log_weights(n_alpha: int) -> np.ndarray:
    """Log of the trapezoid weights of the unit-interval alpha grid.

    The grid nodes are ``u_m = m / M`` for ``m = 1..M`` (plan Section 2.4), so
    the trapezoid rule covers ``[1/M, 1]`` rather than ``[0, 1]``: the first
    cell is deliberately omitted because ``alpha = 0`` is not a valid Weibull
    shape.  The dominant quadrature error is therefore the omitted first cell,
    ``O(1/M**2)`` (measured in test 3 of plan Section 7.1).

    Parameters
    ----------
    n_alpha : int
        Number of quadrature nodes ``M``.  Must be at least 2.

    Returns
    -------
    np.ndarray
        ``float64[M]`` log weights ``log(delta)`` with ``delta = 1 / M``, with
        the two endpoints halved.  They integrate over ``u``; the change of
        variable ``alpha = phi * u`` contributes the factor ``phi`` and is
        applied by :func:`log_integral_alpha`, not here.

    Notes
    -----
    The weights are kept in log space because the sweep multiplies them by an
    integrand that can underflow: ``logsumexp`` of ``log integrand + log
    weight`` cannot cancel to zero, whereas summing exponentiated products can.
    """
    n_alpha = int(n_alpha)
    if n_alpha < 2:
        raise ValueError(f"n_alpha must be at least 2, got {n_alpha}")
    weights = np.full(n_alpha, 1.0 / n_alpha, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return np.log(weights)


@dataclass(frozen=True)
class AlphaGrid:
    """Fixed alpha quadrature grid of one chain (plan Sections 2.4 and 3.3).

    The grid lives in the unit interval and is scaled by the *current* ``phi``,
    ``alpha_m = phi * u_m``, so it always stays exactly inside ``(0, phi)`` as
    ``phi`` moves, needs no ``phi`` cap and makes the quadrature domain bounded
    -- the computational reason the paper put a Uniform prior on ``alpha``.

    The expensive part, ``T_im = t_i ** u_m``, does not depend on ``phi`` and
    is cached once per chain.  A sweep then needs only ``t_i ** alpha_m =
    T_im ** phi``, one ``pow`` on an ``(n, M)`` array.  ``T`` is bounded by 2
    because the transformed times lie in ``(1, 2]``.

    Attributes
    ----------
    time : np.ndarray
        ``float64[n]`` transformed times the grid was built from; needed for
        the ``1 / t`` factor of Eq. A.2 and for consistency checks.
    u : np.ndarray
        ``float64[M]`` unit-interval nodes ``u_m = (m + 1) / M``, ``m = 0..M-1``.
    T : np.ndarray
        ``float64[n, M]`` cache ``T_im = time_i ** u_m``.
    M : int
        Number of nodes; kept explicit because the plan's API names it.
    log_w : np.ndarray
        ``float64[M]`` log trapezoid weights from :func:`trapezoid_log_weights`;
        derived, not passed by the caller.

    The constructor re-derives ``T`` from ``time`` and ``u`` and rejects a
    mismatch, so a stale cache (grid built for one dataset and reused with
    another) cannot silently feed wrong prior-predictive masses to step (a).
    """

    time: np.ndarray
    u: np.ndarray
    T: np.ndarray
    M: int
    log_w: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        u = np.asarray(self.u, dtype=np.float64)
        T = np.asarray(self.T, dtype=np.float64)
        M = int(self.M)

        if time.ndim != 1 or time.size == 0:
            raise ValueError("time must be a non-empty 1-D array")
        if not np.all(np.isfinite(time)) or np.any(time <= 1.0):
            raise ValueError("time must be finite and greater than 1")
        if u.ndim != 1 or u.size < 2:
            raise ValueError("u must be a 1-D array with at least two nodes")
        if M != u.size:
            raise ValueError(
                f"M must equal the number of grid nodes, got M={M} and {u.size} nodes"
            )
        if np.any(u <= 0.0) or np.any(u > 1.0):
            raise ValueError("u must lie in (0, 1]")
        if np.any(np.diff(u) <= 0.0):
            raise ValueError("u must be strictly increasing")
        if T.shape != (time.size, u.size):
            raise ValueError(
                f"T must have shape ({time.size}, {u.size}), got {T.shape}"
            )
        if not np.all(np.isfinite(T)) or np.any(T <= 1.0):
            raise ValueError("T must be finite and greater than 1 (times are > 1)")
        expected = np.power(time[:, None], u[None, :])
        if not np.allclose(T, expected, rtol=1e-12, atol=0.0):
            raise ValueError("T is not time ** u; rebuild the grid with make_grid")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "log_w", trapezoid_log_weights(M))

    @property
    def n(self) -> int:
        """Number of observations the grid covers."""
        return int(self.time.size)

    def powers(self, phi: float) -> np.ndarray:
        """Return ``t_i ** alpha_m = T_im ** phi``, shape ``(n, M)``.

        This is the only ``phi``-dependent part of the grid cache: one ``pow``
        call per sweep instead of recomputing ``t ** alpha`` per node, and it
        cannot overflow because ``T <= 2`` and ``phi`` is a fitted scalar of
        order one.
        """
        phi = _as_positive_scalar(phi, "phi")
        return np.power(self.T, phi)


def make_grid(data: SurvData, n_alpha: int) -> AlphaGrid:
    """Build the alpha quadrature grid for one chain.

    Parameters
    ----------
    data : SurvData
        Dataset on the transformed scale; its times define the grid rows.
    n_alpha : int
        Number of nodes ``M``.  ``DPSettings.n_alpha`` defaults to 256, which
        gives a relative quadrature error of order ``3e-5`` for the calibres in
        plan Section 2.4.

    Returns
    -------
    AlphaGrid
        Nodes ``u_m = (m + 1) / M``, the cache ``T = time ** u`` and the log
        trapezoid weights.

    Raises
    ------
    ValueError
        If ``n_alpha < 2``: a single node has no trapezoid rule, and the
        quadrature would silently lose the ``delta`` factor.
    """
    n_alpha = int(n_alpha)
    if n_alpha < 2:
        raise ValueError(f"n_alpha must be at least 2, got {n_alpha}")
    time = np.asarray(data.time, dtype=np.float64)
    u = np.arange(1, n_alpha + 1, dtype=np.float64) / n_alpha
    return AlphaGrid(time=time, u=u, T=np.power(time[:, None], u[None, :]), M=n_alpha)


def log_alpha_integrand_uncensored(
    grid: AlphaGrid, *, gamma: float, phi: float, d: int
) -> np.ndarray:
    """Log of the Eq. A.2 integrand ``alpha * t**alpha / (gamma + t**alpha)**(d+1)``.

    Parameters
    ----------
    grid : AlphaGrid
        Quadrature grid; supplies ``u``, the cache ``T`` and the times.
    gamma, phi : float
        Current ``G0`` hyperparameters, both strictly positive.
    d : int
        Inverse-gamma shape of the base measure (``d = 2`` in the paper).

    Returns
    -------
    np.ndarray
        ``float64[n, M]`` log integrand at ``alpha_m = phi * u_m``.  Row ``i``
        is, up to the omitted normalising constant, the density of
        ``[alpha_i | gamma, phi, t_i]`` for an *uncensored* observation -- the
        single most important implementation fact of the algorithm (plan
        Section 2.2): the same grid row serves both the urn weight ``q0^o``
        and, after normalisation, the continuous-piece draw of ``alpha_i``.

    Notes
    -----
    Eq. A.2 is written as ``alpha t**(alpha-1) / (gamma + t**alpha)**(d+1)``;
    the ``1 / t`` factor is pulled out of the integrand and reapplied in
    :func:`log_q0_uncensored`, which is why ``t**alpha`` (cached as
    ``T**phi``) appears here instead of ``t**(alpha-1)``.  The denominator is
    formed as ``gamma + T**phi`` and never as ``log t**alpha`` exponentiated
    twice, and no quantity is exponentiated before the sum: a large
    ``t**alpha`` can only make the integrand small, but forming it in log
    space keeps the ``logsumexp`` in :func:`log_integral_alpha` exact.
    """
    gamma = _as_positive_scalar(gamma, "gamma")
    phi = _as_positive_scalar(phi, "phi")
    d = _as_shape_parameter(d)
    alpha = phi * grid.u
    tau = grid.powers(phi)
    return np.log(alpha)[None, :] + phi * np.log(grid.T) - (d + 1) * np.log(gamma + tau)


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    """Row-wise ``log(sum(exp(values), axis=1))`` with a max shift.

    Same algorithm as ``scipy.special.logsumexp(..., axis=1)`` -- subtract the
    row maximum before exponentiating and add it back -- without the SciPy
    array-API dispatch, which costs more than the arithmetic on the small rows
    of this module.  Every row must contain at least one finite entry; the
    integrands integrated here always do (they are ``-inf`` only if
    ``gamma + t**alpha`` overflowed, which cannot happen for ``t <= 2`` and a
    fitted ``phi``).
    """
    values = np.asarray(values, dtype=np.float64)
    row_max = np.max(values, axis=1)
    return row_max + np.log(np.sum(np.exp(values - row_max[:, None]), axis=1))


def log_integral_alpha(
    log_integrand: np.ndarray, grid: AlphaGrid, *, phi: float
) -> np.ndarray:
    """Integrate an ``(n, M)`` log integrand over ``alpha`` in ``(0, phi)``.

    Implements the quadrature of plan Section 2.4: the trapezoid rule in
    ``alpha`` with ``logsumexp`` over the nodes, so that the sum of integrand
    values can never underflow to zero.  The change of variable
    ``alpha = phi * u`` is accounted for by the ``log(phi)`` term.

    The row-wise sum uses the same max-shift stabilisation as
    ``scipy.special.logsumexp`` but is written here directly on the 2-D array:
    the SciPy wrapper resolves an array namespace on every call (about 0.15 ms
    for a 16-node row in this environment), and step (a) forms this integral
    once per observation, so it would dominate the sweep cost of plan
    Section 8.  The arithmetic is unchanged -- only the dispatch overhead is
    removed.

    Parameters
    ----------
    log_integrand : np.ndarray
        ``float64[n, M]`` values of the log integrand on the grid nodes.
    grid : AlphaGrid
        Grid supplying the log weights; its shape must match.
    phi : float
        Current uniform upper bound, strictly positive.

    Returns
    -------
    np.ndarray
        ``float64[n]`` log of ``integral_0^phi f(alpha) dalpha``.

    Raises
    ------
    ValueError
        If the integrand shape does not match the grid.
    """
    log_integrand = np.asarray(log_integrand, dtype=np.float64)
    phi = _as_positive_scalar(phi, "phi")
    if log_integrand.shape != (grid.n, grid.M):
        raise ValueError(
            f"log_integrand must have shape ({grid.n}, {grid.M}), "
            f"got {log_integrand.shape}"
        )
    return _log_integral_rows(log_integrand, grid, phi=phi)


def _log_integral_rows(
    log_integrand: np.ndarray, grid: AlphaGrid, *, phi: float
) -> np.ndarray:
    """Quadrature core: integrate already-selected rows, without validation.

    Shared by :func:`log_integral_alpha` (all rows) and by the ``rows``
    subset of :func:`log_q0_uncensored` / :func:`log_q0_censored`, which step
    (a) uses to integrate only the observations of one censoring branch.
    """
    return _logsumexp_rows(log_integrand + grid.log_w[None, :]) + math.log(phi)


def log_q0_uncensored(
    grid: AlphaGrid,
    *,
    gamma: float,
    phi: float,
    nu: float,
    d: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Log prior-predictive density ``q0^o(t)`` of Eq. A.2, per observation.

    This is the "new cluster" mass of the Polya urn for an *uncensored*
    observation::

        q0^o(t) = d nu gamma**d / phi *
                  integral_0^phi alpha t**(alpha-1) / (gamma + t**alpha)**(d+1) dalpha
                = d nu gamma**d / (phi t) *
                  integral_0^phi alpha t**alpha   / (gamma + t**alpha)**(d+1) dalpha

    The second form is what the code evaluates: it reuses the cached grid row
    ``T**phi = t**alpha`` and moves the ``1 / t`` outside the quadrature.  In
    log space the ``phi`` of the change of variable ``alpha = phi u`` cancels
    against the ``1 / phi`` of the base density, so the result is::

        log q0^o(t) = log d + log nu + d log gamma - log t
                      + logsumexp_m[ log(phi u_m) + phi log T_im
                                      - (d + 1) log(gamma + T_im**phi)
                                      + log w_m ]

    Parameters
    ----------
    grid : AlphaGrid
        Grid built from the data whose times are the arguments of ``q0^o``.
    gamma, phi, nu : float
        Current ``G0`` hyperparameters and DP precision, all strictly positive.
    d : int
        Inverse-gamma shape of the base measure (``d = 2`` in the paper).
    rows : np.ndarray, optional
        Subset of grid rows to evaluate.  ``None`` (the default) evaluates
        every row.  Step (a) passes the uncensored rows only, so a
        right-censored observation never pays for the density quadrature it
        cannot use; the result then has ``rows.size`` entries, in the order
        given.

    Returns
    -------
    np.ndarray
        ``float64[n]`` log ``q0^o(t_i)`` for every evaluated row.  Censored
        rows take the survival based mass ``q0^c`` of Eq. A.3 instead, which
        reuses the same grid: see :func:`log_q0_censored`.

    Notes
    -----
    Test 1 of plan Section 7.1 pins this against a nested ``quad`` over
    ``(alpha, lambda)`` that marginalises the inverse-gamma integrand
    numerically.  That comparison catches every constant here: the missing or
    extra ``gamma**d``, the ``1 / phi``, the ``1 / t`` (or wrong
    ``t**(alpha-1)`` index) and the ``d`` versus ``d + 1`` exponent.
    """
    gamma = _as_positive_scalar(gamma, "gamma")
    phi = _as_positive_scalar(phi, "phi")
    nu = _as_positive_scalar(nu, "nu")
    d = _as_shape_parameter(d)
    log_integrand = _select_rows(
        log_alpha_integrand_uncensored(grid, gamma=gamma, phi=phi, d=d), rows
    )
    times = _select_rows(grid.time, rows)
    log_integral = _log_integral_rows(log_integrand, grid, phi=phi)
    return (
        math.log(d)
        + math.log(nu)
        + d * math.log(gamma)
        - math.log(phi)
        - np.log(times)
        + log_integral
    )


def log_alpha_integrand_censored(
    grid: AlphaGrid, *, gamma: float, phi: float, d: int
) -> np.ndarray:
    """Log of the Eq. A.3 integrand ``1 / (gamma + z**alpha)**d``.

    Parameters
    ----------
    grid : AlphaGrid
        Quadrature grid; supplies ``u`` and the cache ``T``.
    gamma, phi : float
        Current ``G0`` hyperparameters, both strictly positive.
    d : int
        Inverse-gamma shape of the base measure (``d = 2`` in the paper).

    Returns
    -------
    np.ndarray
        ``float64[n, M]`` log integrand at ``alpha_m = phi * u_m``.  Row ``i``
        is, up to the omitted normalising constant, the density of
        ``[alpha_i | gamma, phi, z_i]`` for a *right-censored* observation,
        the survival counterpart of :func:`log_alpha_integrand_uncensored`.

    Notes
    -----
    The survival ``1 - K_W(z | alpha, lambda) = exp(-z**alpha / lambda)``
    marginalised over ``lambda ~ IGamma(d, gamma)`` leaves
    ``gamma**d / (gamma + z**alpha)**d``, so the integrand carries the
    exponent ``d`` and **no** ``alpha`` or ``1 / z`` factor: the ``alpha`` of
    Eq. A.2 comes from the Weibull density's shape, and a censored row has no
    density.  This is the one place where ``d + 1`` (the uncensored ``lambda``
    shape) must not be reused (plan Section 11).
    """
    gamma = _as_positive_scalar(gamma, "gamma")
    phi = _as_positive_scalar(phi, "phi")
    d = _as_shape_parameter(d)
    tau = grid.powers(phi)
    return -d * np.log(gamma + tau)


def log_q0_censored(
    grid: AlphaGrid,
    *,
    gamma: float,
    phi: float,
    nu: float,
    d: int,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Log prior-predictive mass ``q0^c(z)`` of Eq. A.3, per observation.

    This is the "new cluster" mass of the Polya urn for a *right-censored*
    observation: the prior-predictive probability that the observation
    exceeds its recorded time::

        q0^c(z) = nu gamma**d / phi *
                  integral_0^phi 1 / (gamma + z**alpha)**d dalpha

    With the grid's change of variable ``alpha = phi u`` the ``phi`` of
    ``dalpha`` cancels the ``1 / phi`` of the base density, exactly as in
    Eq. A.2, so in log space::

        log q0^c(z) = log nu + d log gamma
                      + logsumexp_m[ -d log(gamma + T_im**phi) + log w_m ]

    Parameters
    ----------
    grid : AlphaGrid
        Grid built from the data whose times are the arguments of ``q0^c``.
    gamma, phi, nu : float
        Current ``G0`` hyperparameters and DP precision, all strictly positive.
    d : int
        Inverse-gamma shape of the base measure (``d = 2`` in the paper).
    rows : np.ndarray, optional
        Subset of grid rows to evaluate, as in :func:`log_q0_uncensored`.
        ``None`` (the default) evaluates every row.

    Returns
    -------
    np.ndarray
        ``float64[n]`` log ``q0^c(z_i)``, one entry per evaluated row.  Use
        :func:`log_q0_uncensored` for the uncensored rows; a censored row must
        never be given the density mass (plan Section 11).

    Notes
    -----
    Test 2 of plan Section 7.1 pins this against a nested ``quad`` over
    ``(alpha, lambda)`` that integrates the survival ``exp(-z**alpha/lambda)``
    against the inverse-gamma density numerically, i.e. without using the
    analytic ``lambda`` marginal.  That comparison catches the constants of
    A.3 -- the ``nu``, the ``gamma**d`` and above all the ``d`` instead of
    ``d + 1`` exponent -- none of which the uncensored test can see.

    Grid accuracy differs between the two branches and is stated here rather
    than assumed: ``(gamma + z**alpha)**-d`` does not vanish at ``alpha = 0``,
    so the omitted first cell of the plan's grid (nodes start at
    ``alpha_1 = phi / M``, no ``alpha = 0`` node) costs ``O(1 / M)``, measured
    at ``0.61 / M`` -- ``6e-3`` relative at the default ``M = 256``, versus
    ``3e-5`` and ``O(1 / M**2)`` for Eq. A.2.  That is a property of the
    prescribed quadrature, not of Eq. A.3, and it is negligible for the urn
    weight; a finer grid (never a restructured one) is the plan's remedy.
    """
    gamma = _as_positive_scalar(gamma, "gamma")
    phi = _as_positive_scalar(phi, "phi")
    nu = _as_positive_scalar(nu, "nu")
    d = _as_shape_parameter(d)
    log_integrand = _select_rows(
        log_alpha_integrand_censored(grid, gamma=gamma, phi=phi, d=d), rows
    )
    log_integral = _log_integral_rows(log_integrand, grid, phi=phi)
    return math.log(nu) + d * math.log(gamma) - math.log(phi) + log_integral


def _select_rows(values: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
    """Return ``values`` or the subset named by ``rows``.

    ``rows`` must be a 1-D array of integer indices into ``values``; ``None``
    selects everything.  Used by the ``rows`` option of the ``q0`` helpers so
    that the integrand and the ``1 / t`` factor are subset by the same index
    array.
    """
    if rows is None:
        return np.asarray(values)
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("rows must be a 1-D array of integer indices")
    return np.asarray(values)[rows]


def _as_shape_parameter(value: object, name: str = "d") -> int:
    """Normalise the inverse-gamma shape ``d`` to a positive ``int``."""
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if out != value or out < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out


def _as_scalar_transformed_time(time: object) -> float:
    """Validate one transformed time (``time > 1``) and return it as a float."""
    arr = np.asarray(time)
    if arr.ndim != 0:
        raise ValueError("time must be a scalar")
    try:
        out = float(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError("time must be a finite scalar greater than 1") from exc
    if not np.isfinite(out) or out <= 1.0:
        raise ValueError("time must be a finite scalar greater than 1")
    return out


def _as_scalar_event(event: object) -> bool:
    """Validate one censoring indicator and return it as a bool."""
    arr = np.asarray(event)
    if arr.ndim != 0:
        raise ValueError("event must be a scalar boolean")
    if arr.dtype == np.bool_:
        return bool(arr)
    if np.issubdtype(arr.dtype, np.number) and (arr == 0 or arr == 1):
        return bool(arr)
    raise ValueError("event must be a scalar boolean or 0/1")


def _grid_row_for_time(grid: AlphaGrid, time: float) -> int:
    """Return the row of ``grid`` whose cached time equals ``time``.

    The ``alpha`` draw must reuse the cached ``T_im`` of the observation, not
    rebuild ``t ** u``: step (a) calls this function once per new cluster, and
    recomputing the row would make the ``(n, M)`` cache pointless.  An exact
    match is required, because a grid built for another dataset would give the
    wrong row and therefore the wrong conditional density.
    """
    rows = np.flatnonzero(grid.time == time)
    if rows.size == 0:
        raise ValueError("time is not a row of the supplied alpha grid")
    return int(rows[0])


def draw_g0_conditional_alpha(
    time: object,
    event: object,
    *,
    gamma: float,
    phi: float,
    grid: AlphaGrid,
    rng: np.random.Generator,
    d: int = 2,
) -> tuple[float, float]:
    """Draw ``(alpha, lambda)`` from the continuous piece of the base measure.

    This is the "new cluster" branch of step (a): given that observation ``i``
    is assigned to a fresh atom, its latent parameters come from the
    continuous part ``h_i(alpha, lambda | gamma, phi, t_i)`` of plan
    Section 2.2.  The two factors are

    * uncensored: ``[alpha | .]`` is proportional to
      ``alpha t**alpha / (gamma + t**alpha)**(d + 1)`` and
      ``[lambda | alpha, .]`` is ``IGamma(d + 1, gamma + t**alpha)``;
    * right-censored: ``[alpha | .]`` is proportional to
      ``1 / (gamma + z**alpha)**d`` and ``[lambda | alpha, .]`` is
      ``IGamma(d, gamma + z**alpha)``.

    The uncensored ``alpha`` density is exactly the Eq. A.2 integrand, so
    the same grid row that supplies ``q0`` also supplies this draw; a censored
    observation must use the Eq. A.3 row instead.  ``alpha`` is drawn by
    inverse CDF on the grid (plan Section 2.4(ii)): ``cumsum`` the normalised
    row, draw one uniform, then ``searchsorted``.  ``lambda`` is then drawn
    from its conjugate inverse-gamma conditional, whose scale is
    ``gamma + t**alpha`` and whose shape is ``d + 1`` for an uncensored and
    ``d`` for a right-censored observation.

    Parameters
    ----------
    time : object
        One time on the transformed scale, strictly greater than 1.  It must
        be a row of ``grid`` so that the cached ``t ** u`` row is reused.
    event : object
        Scalar censoring indicator; ``True`` for an uncensored observation
        (density, Eq. A.2) and ``False`` for a right-censored one (survival,
        Eq. A.3).  Numeric ``0``/``1`` is accepted.
    gamma, phi : float
        Current ``G0`` hyperparameters, both finite and strictly positive.
    grid : AlphaGrid
        Grid built from the data containing ``time``; supplies the unit nodes,
        the trapezoid weights and the cached power row.
    rng : np.random.Generator
        Random generator.  Exactly two variates are consumed: one uniform for
        ``alpha`` and one gamma variate for ``lambda``.
    d : int, optional
        Inverse-gamma shape of the base measure.  Defaults to the paper's
        ``d = 2``; exposed because the tests distinguish ``d`` from ``d + 1``.

    Returns
    -------
    tuple of float
        ``(alpha, lambda)`` with ``0 < alpha < phi`` and ``lambda > 0``.

    Raises
    ------
    ValueError
        If ``time``, ``event``, ``gamma``, ``phi`` or ``d`` is invalid, or if
        ``time`` is not a row of ``grid``.

    Notes
    -----
    Drawing ``alpha`` on the grid makes it discrete with spacing
    ``Delta alpha = phi / M``.  This is the paper's own approximation (plan
    Section 2.4(ii)) and is documented rather than removed; the continuous
    ``lambda`` draw is exact.  The inverse-CDF index is intentionally
    ``searchsorted(..., side="right")`` with a final ``min(M - 1, ...)``
    clamp, so a uniform of exactly 1.0 cannot run off the grid.
    """
    t = _as_scalar_transformed_time(time)
    is_event = _as_scalar_event(event)
    gamma = _as_positive_scalar(gamma, "gamma")
    phi = _as_positive_scalar(phi, "phi")
    d = _as_shape_parameter(d)
    row = _grid_row_for_time(grid, t)

    alpha_nodes = phi * grid.u
    tau = np.power(grid.T[row], phi)
    if is_event:
        log_density = (
            np.log(alpha_nodes)
            + phi * np.log(grid.T[row])
            - (d + 1) * np.log(gamma + tau)
        )
        lam_shape = d + 1
    else:
        log_density = -d * np.log(gamma + tau)
        lam_shape = d

    log_weights = log_density + grid.log_w
    # A single row is normalised with a max shift rather than
    # ``scipy.special.logsumexp``: the scipy wrapper costs ~0.15 ms per call
    # on a 16-node array, which would dominate step (a), while this shift is
    # the same stabilisation.  Batch sums in the sweep still use ``logsumexp``.
    weights = np.exp(log_weights - float(np.max(log_weights)))
    weights /= float(np.sum(weights))
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, rng.random(), side="right"))
    index = min(index, grid.M - 1)
    alpha = float(alpha_nodes[index])

    lam_scale = gamma + t**alpha
    lam = float(1.0 / rng.gamma(shape=lam_shape, scale=1.0 / lam_scale))
    return alpha, lam


def _log_q0_by_observation(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    grid: AlphaGrid,
    rows: np.ndarray | None = None,
) -> np.ndarray:
    """Log prior-predictive masses ``q0^i`` for every observation.

    The urn's "new cluster" mass depends on the censoring indicator: an
    uncensored observation contributes a density and uses Eq. A.2
    (:func:`log_q0_uncensored`), a right-censored one contributes a survival
    probability and uses Eq. A.3 (:func:`log_q0_censored`).  Each branch is
    integrated for its own rows only, so an all-uncensored dataset never pays
    for the censored quadrature and a single-observation update (the test
    entry point) integrates one row instead of ``n``.

    Parameters
    ----------
    rows : np.ndarray, optional
        Rows of the grid to evaluate, in the order given.  ``None`` evaluates
        every row and returns an ``(n,)`` array; otherwise the result has
        ``rows.size`` entries.
    """
    event = np.asarray(data.event, dtype=bool)
    selected = (
        np.arange(grid.n, dtype=np.intp)
        if rows is None
        else np.asarray(rows, dtype=np.intp)
    )
    selected_event = event[selected]
    q0 = np.empty(selected.size, dtype=np.float64)
    if np.any(selected_event):
        q0[selected_event] = log_q0_uncensored(
            grid,
            gamma=state.gamma,
            phi=state.phi,
            nu=state.nu,
            d=prior.d,
            rows=selected[selected_event],
        )
    if np.any(~selected_event):
        q0[~selected_event] = log_q0_censored(
            grid,
            gamma=state.gamma,
            phi=state.phi,
            nu=state.nu,
            d=prior.d,
            rows=selected[~selected_event],
        )
    return q0


def _log_cluster_kernel(
    time: np.ndarray,
    event: np.ndarray,
    alpha_star: np.ndarray,
    lam_star: np.ndarray,
) -> np.ndarray:
    """Log ``q_j^i`` for every observation against every cluster location.

    For an uncensored observation ``q_j^i = k_W(t_i | alpha*_j, lambda*_j)``
    (the Weibull density, Eq. A.2's kernel); for a right-censored one it is
    ``1 - K_W(z_i | alpha*_j, lambda*_j) = exp(-z_i**alpha / lambda)`` (the
    survival function).  Swapping the two -- using the density for a censored
    observation -- is one of the bugs test 8 is designed to catch.

    Everything stays in log space: the exponential factor is exactly the log
    survival ``-t**alpha / lambda``, so the density adds the normalising
    ``log(alpha / lambda) + (alpha - 1) log t`` on top of it and no term is
    exponentiated only to be logged again.

    Parameters
    ----------
    time, event : np.ndarray
        ``float64[n]`` transformed times and ``bool[n]`` censoring indicators.
    alpha_star, lam_star : np.ndarray
        ``float64[n_star]`` cluster shapes and scales, all strictly positive.

    Returns
    -------
    np.ndarray
        ``float64[n, n_star]`` log kernel masses.  Column ``j`` refers to the
        cluster ``j`` of the *supplied* location arrays, i.e. the pre-removal
        labelling; step (a) appends columns as clusters are created.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    alpha_star = np.asarray(alpha_star, dtype=np.float64)
    lam_star = np.asarray(lam_star, dtype=np.float64)
    if alpha_star.size == 0:
        return np.empty((time.size, 0), dtype=np.float64)
    tau = np.power(time[:, None], alpha_star[None, :])
    log_survival = -tau / lam_star[None, :]
    log_density = (
        np.log(alpha_star)[None, :]
        + (alpha_star[None, :] - 1.0) * np.log(time)[:, None]
        - np.log(lam_star)[None, :]
        + log_survival
    )
    return np.where(event[:, None], log_density, log_survival)


def _validate_step_a_inputs(
    state: DPState, data: SurvData, prior: DPPrior, grid: AlphaGrid
) -> None:
    """Check that state, data, prior and grid describe the same problem.

    A grid built for another dataset would supply the wrong ``t ** u`` row to
    the continuous-piece draw and the wrong rows to ``q0``, which is silent
    and biases every new cluster; an exact comparison of the times is cheap
    enough to run on every call and turns that into an error.
    """
    _as_shape_parameter(prior.d)
    if state.phi < prior.b_phi:
        raise StateInvariantError(
            "invariant 4 violated: phi must be at least prior.b_phi"
        )
    if data.time.size != state.n:
        raise ValueError(
            "data and state must have the same number of observations, "
            f"got {data.time.size} and {state.n}"
        )
    if grid.n != state.n or not np.array_equal(grid.time, np.asarray(data.time)):
        raise ValueError("the alpha grid was built for a different dataset")


def _sample_urn_category(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    grid: AlphaGrid,
    rng: np.random.Generator,
    i: int,
    log_q0_i: float,
    log_kernel_i: np.ndarray,
) -> bool:
    """Resample observation ``i``'s urn category and apply the assignment.

    Implements the Po'lya urn mixture of plan Section 2.2(a) literally:

    1. remove observation ``i`` first, so that its own cluster contributes
       ``n_j^- q_j^i`` -- zero when it was a singleton, which is what makes a
       singleton's cluster disappear from the candidate set instead of being
       selectable with weight zero;
    2. form ``[log q0^i, log n_j^- + log q_j^i]`` over the *non-empty*
       clusters (deferred-compaction labels) plus the new-cluster category;
    3. draw the category by inverse CDF on the ``logsumexp``-shifted weights;
    4. copy the location (existing cluster) or draw a fresh
       ``(alpha, lambda)`` from the continuous piece and append a cluster.

    The RNG consumption follows the branch: one uniform for the category, plus
    a grid uniform and an inverse-gamma variate when the new cluster is drawn.

    Returns
    -------
    bool
        ``True`` if a new cluster was created.
    """
    state.remove_observation(i)
    candidates = np.flatnonzero(state.n_j > 0)
    log_weights = np.empty(candidates.size + 1, dtype=np.float64)
    log_weights[0] = log_q0_i
    log_weights[1:] = np.log(state.n_j[candidates]) + log_kernel_i[candidates]
    if not np.all(np.isfinite(log_weights)):
        raise StateInvariantError(
            "urn weights are not finite; state, data and grid are inconsistent"
        )
    weights = np.exp(log_weights - float(np.max(log_weights)))
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, rng.random() * weights.sum()))
    index = min(index, weights.size - 1)

    if index == 0:
        alpha, lam = draw_g0_conditional_alpha(
            data.time[i],
            data.event[i],
            gamma=state.gamma,
            phi=state.phi,
            grid=grid,
            rng=rng,
            d=prior.d,
        )
        state.assign_new(i, alpha, lam)
        return True
    state.assign_existing(i, int(candidates[index - 1]))
    return False


def sample_step_a(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    grid: AlphaGrid,
    rng: np.random.Generator,
) -> None:
    """Run step (a): resample every observation's ``(alpha_i, lambda_i, s_i)``.

    This is the Po'lya urn mixture of plan Section 2.2(a), applied
    sequentially over ``i = 0 .. n-1`` in index order.  The order is part of
    the specification -- it fixes the random-number stream -- so the loop is
    deliberately *not* vectorised into one simultaneous update, which would
    change the transition kernel.

    For each ``i`` the full conditional is

    ``q0^i h_i(alpha_i, lambda_i | gamma, phi, t_i)
      + sum_j n_j^- q_j^i delta_{(alpha*_j, lambda*_j)}``

    normalised by ``q0^i + sum_j n_j^- q_j^i``, with ``q0^i`` from Eq. A.2
    (uncensored) or Eq. A.3 (right-censored) and ``q_j^i`` the Weibull density
    or survival at the cluster location.  The ``n_j^-`` are the counts *after*
    removing ``i``: using ``n_j`` is a classic error that makes a singleton
    reselect its own cluster with weight ``q_j`` instead of zero, and it is
    invisible in a trace plot.

    Bookkeeping follows the deferred-compaction variant of plan Section 3.4:
    :meth:`DPState.remove_observation` decrements a count but leaves the label
    in place, so the candidate clusters keep their pre-removal indices while
    the weights are formed; the single :meth:`DPState.compact` pass at the end
    drops any cluster emptied by a removal, renumbers the rest and revalidates
    the Section 2.1 invariants.  New clusters are appended during the loop and
    the cached ``(n, n_star)`` log-kernel matrix grows a column for each, so
    the columns and the state's labels never disagree.

    Parameters
    ----------
    state : DPState
        Mutated in place.  ``nu``, ``gamma`` and ``phi`` are held fixed here;
        they are refreshed by steps (b) and (c).
    data : SurvData
        Dataset on the transformed scale, its ``time`` rows the grid rows.
    prior : DPPrior
        Supplies the base-measure shape ``d``.
    grid : AlphaGrid
        Grid built from ``data``; its cache supplies ``q0`` and the
        continuous-piece ``alpha`` draw.
    rng : np.random.Generator
        Random generator; the only source of randomness.

    Raises
    ------
    ValueError
        If the state, data and grid do not describe the same dataset, or the
        observation index is out of range.
    StateInvariantError
        If the urn weights are not finite or the final state violates a
        Section 2.1 invariant -- a bookkeeping error, never an input error.

    Notes
    -----
    Test 8 of plan Section 7.2 drives :func:`sample_step_a_for_obs` many times
    from a fixed state and compares the empirical category frequencies with
    the reference urn probabilities; the different exits (join an existing
    cluster, create one, join after the old cluster disappeared) are covered
    there.
    """
    _validate_step_a_inputs(state, data, prior, grid)
    log_q0 = _log_q0_by_observation(state, data, prior, grid)
    log_kernel = _log_cluster_kernel(
        data.time, data.event, state.alpha_star, state.lam_star
    )
    for i in range(state.n):
        created = _sample_urn_category(
            state, data, prior, grid, rng, i, log_q0[i], log_kernel[i]
        )
        if created:
            start = log_kernel.shape[1]
            new_columns = _log_cluster_kernel(
                data.time,
                data.event,
                state.alpha_star[start:],
                state.lam_star[start:],
            )
            log_kernel = np.hstack([log_kernel, new_columns])
    # ``compact`` drops the clusters emptied by a removal, renumbers the rest
    # and revalidates the Section 2.1 invariants; the ``phi >= b_phi`` bound
    # (invariant 4) cannot change in step (a) and was checked on entry.
    state.compact()


def sample_step_a_for_obs(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    i: int,
    grid: AlphaGrid,
    rng: np.random.Generator,
    compact: bool = True,
) -> None:
    """Resample one observation's ``(alpha_i, lambda_i, s_i)`` (test entry).

    This is exactly the per-observation update that :func:`sample_step_a`
    applies in its loop, exposed separately so that a test can fix the state,
    draw the category for one ``i`` many times and compare the frequencies
    with the exact urn probabilities (test 8 of plan Section 7.2).  The public
    sweep entry point is :func:`sample_step_a`; production code should not
    call this per observation in a loop, because it recomputes the whole
    ``q0`` vector and kernel matrix on every call.

    By default the state is left compacted and invariant-clean, so the assigned
    category can be read back as ``state.s[i]`` and its location as
    ``(alpha_star[s_i], lam_star[s_i])``; an existing-cluster assignment copies
    the location exactly, while a new-cluster assignment carries a fresh draw
    from the continuous piece.

    ``compact=False`` returns the deferred-compaction state of plan
    Section 3.4 instead: labels are the pre-removal ones and ``s[i]`` names the
    sampled category directly -- the old label of a surviving cluster, or
    ``n_star_before`` when a fresh cluster was appended.  That is what test 8
    needs to count categories without matching locations, and it skips the
    relabel pass on each of its many repetitions.  In that mode the state may
    contain one empty cluster (if ``i`` was a singleton) and is therefore
    *not* invariant-clean until :meth:`DPState.compact` runs; the caller owns
    the state and must either reset or compact it.

    Parameters
    ----------
    state : DPState
        Mutated in place: ``i`` is removed, reassigned and the labels are
        compacted.
    data, prior, grid :
        As in :func:`sample_step_a`.
    i : int
        Observation index in ``0 .. n-1``.
    rng : np.random.Generator
        Random generator; the only source of randomness.
    compact : bool, optional
        Compaction policy; see above.  ``True`` (the default) matches
        :func:`sample_step_a` and leaves a valid state.

    Raises
    ------
    IndexError
        If ``i`` is out of range.
    ValueError
        If the state, data and grid do not describe the same dataset.
    """
    _validate_step_a_inputs(state, data, prior, grid)
    if not 0 <= i < state.n:
        raise IndexError(f"observation index {i} is out of range for n={state.n}")
    log_q0_i = float(
        _log_q0_by_observation(state, data, prior, grid, rows=np.array([i]))[0]
    )
    log_kernel_i = _log_cluster_kernel(
        data.time[i : i + 1],
        data.event[i : i + 1],
        state.alpha_star,
        state.lam_star,
    )[0]
    _sample_urn_category(state, data, prior, grid, rng, i, log_q0_i, log_kernel_i)
    if compact:
        state.compact()


def _validate_step_b_inputs(state: DPState, data: SurvData, prior: DPPrior) -> None:
    """Check that the state and the data describe the same dataset.

    Step (b) indexes the data with the state's labels, so a state carrying
    three clusters and a dataset of five times would either raise an
    IndexError deep in the loop or, worse, silently use the wrong times.  The
    check is an exact length comparison and costs nothing next to the draws.
    """
    _as_shape_parameter(prior.d)
    if data.time.size != state.n:
        raise ValueError(
            "data and state must have the same number of observations, "
            f"got {data.time.size} and {state.n}"
        )
    if data.event.size != state.n:
        raise ValueError(
            "data and state must have the same number of censoring indicators, "
            f"got {data.event.size} and {state.n}"
        )


def _cluster_members(state: DPState, j: int) -> np.ndarray:
    """Return the observation indices labelled ``j``, checking the counts agree.

    Step (b) runs on the compacted state that step (a) leaves behind, so a
    cluster's member list and its stored count must match exactly.  A state
    still carrying the deferred removals of ``remove_observation`` would give
    the wrong ``n_j^o`` and the wrong power sum; that is a bookkeeping error,
    not an input error, so it raises :class:`StateInvariantError`.
    """
    if not 0 <= j < state.n_star:
        raise IndexError(f"cluster index {j} is out of range for n_star={state.n_star}")
    members = np.flatnonzero(state.s == j)
    if members.size == 0:
        raise StateInvariantError(
            f"cluster {j} has no observations; compact the state before step (b)"
        )
    if members.size != int(state.n_j[j]):
        raise StateInvariantError(
            f"cluster {j} holds {members.size} observations but n_j[j] is "
            f"{int(state.n_j[j])}; compact the state before step (b)"
        )
    return members


def _lam_star_parameters(
    state: DPState, data: SurvData, prior: DPPrior, *, j: int
) -> tuple[int, float]:
    """Return the shape and scale of ``[lambda*_j | alpha*_j, gamma, D]``.

    Plan Section 2.2(b):

        ``lambda*_j | alpha*_j, ... ~ IGamma(d + n_j^o,
          gamma + sum_{i_o in j} t_{i_o}**alpha*_j
                + sum_{i_c in j} z_{i_c}**alpha*_j)``

    so the shape is the base-measure shape ``d`` plus the number of
    *uncensored* members (a censored member contributes a survival factor and
    therefore only to the exponential power sum, not to the shape), and the
    scale is ``gamma`` plus the power sum over **all** members at the
    cluster's shape ``alpha*_j``.  Both mistakes are easy to make and silent:
    using ``n_j`` instead of ``n_j^o``, or dropping ``gamma`` from the scale.

    Everything is evaluated at the cluster's own ``alpha*_j``; a neighbour's
    shape would change both the sum and its meaning.
    """
    members = _cluster_members(state, j)
    alpha = float(state.alpha_star[j])
    times = np.asarray(data.time, dtype=np.float64)[members]
    pows = np.power(times, alpha)
    n_uncensored = int(np.count_nonzero(np.asarray(data.event, dtype=bool)[members]))
    shape = _as_shape_parameter(prior.d) + n_uncensored
    scale = float(state.gamma) + float(np.sum(pows))
    if not np.isfinite(scale) or scale <= 0.0:
        raise StateInvariantError(
            "the lambda* conditional of cluster "
            f"{j} has a non-finite scale {scale!r}; state and data are inconsistent"
        )
    return shape, scale


def refresh_cluster(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    j: int,
    rng: np.random.Generator,
) -> None:
    """Refresh cluster ``j``'s scale ``lambda*_j`` from its full conditional.

    This is the first half of step (b) of plan Section 3.4, the
    Bush-MacEachern location refresh.  With
    ``n_j^o = |{i_o : s_{i_o} = j}|`` the number of *uncensored* members,

        ``lambda*_j | alpha*_j ~ IGamma(d + n_j^o,
          gamma + sum_{i in j} t_i**alpha*_j)``,

    where the sum runs over every member, censored or not, and is evaluated at
    the cluster's current shape ``alpha*_j``.  A censored member contributes
    its transformed time ``z`` to the sum but does **not** raise the
    inverse-gamma shape: it is a survival factor, so it appears with exponent
    ``d`` rather than ``d + 1`` in the observation's base-measure density.
    Hence an all-censored cluster (``n_j^o = 0``) has shape ``d`` alone, which
    is exactly the case a ``d + n_j`` shortcut gets right by accident only for
    uncensored data.

    The draw is exact: ``lambda = 1 / G`` with
    ``G ~ Gamma(shape, rate=gamma + sum)``.  One gamma variate is consumed per
    call, and nothing else, so the per-cluster loop in
    :func:`sample_step_b` has a fixed random-number stream.

    After the draw, the cluster location and the per-observation latent values
    of its members are updated together, because invariant 1 of plan
    Section 2.1 requires ``lam == lam_star[s]`` at all times; updating only
    ``lam_star`` would leave the state inconsistent until the caller synced it.

    Parameters
    ----------
    state : DPState
        Mutated in place: ``lam_star[j]`` and ``lam[members]`` are replaced.
        ``alpha_star``, ``nu``, ``gamma`` and ``phi`` are untouched.
    data : SurvData
        Dataset on the transformed scale; supplies the member times and their
        censoring indicators.
    prior : DPPrior
        Supplies the base-measure shape ``d``.
    j : int
        Cluster index in ``0 .. n_star-1``.  This is the test entry point of
        plan Section 3.3: a test fixes the rest of the state, calls this many
        times and compares the draws with ``scipy.stats.invgamma`` (test 10 of
        plan Section 7.2).
    rng : np.random.Generator
        Random generator; the only source of randomness.

    Raises
    ------
    IndexError
        If ``j`` is out of range.
    ValueError
        If the state and the data describe different datasets, or ``prior.d``
        is not a positive integer.
    StateInvariantError
        If cluster ``j`` is empty or ``n_j`` disagrees with its member list --
        i.e. the state was not compacted before step (b).

    Notes
    -----
    ``lambda*_j``'s conditional does not involve ``nu`` or ``phi``, so a
    refresh leaves the ``phi >= max_j alpha*_j`` invariant of plan Section 2.1
    untouched.  The shape ``alpha*_j`` is refreshed by the
    Damien-Wakefield-Walker auxiliary-variable update of plan Section 2.2(b),
    implemented in :func:`refresh_cluster_alpha`, which uses this new
    ``lambda*_j`` and therefore runs *after* this function for each cluster.
    """
    _validate_step_b_inputs(state, data, prior)
    shape, scale = _lam_star_parameters(state, data, prior, j=j)
    lam = float(1.0 / rng.gamma(shape=shape, scale=1.0 / scale))
    if not np.isfinite(lam) or lam <= 0.0:
        raise StateInvariantError(
            f"the lambda* draw for cluster {j} was not a finite positive number"
        )
    members = np.flatnonzero(state.s == j)
    state.lam_star[j] = lam
    state.lam[members] = lam


def _alpha_star_bounds(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    j: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return the auxiliary-variable interval ``(alpha^-_j, alpha^+_j)``.

    Plan Section 2.2(b), Eqs. A.5-A.6.  At the cluster's **current** shape
    ``alpha*_j`` and the **new** scale ``lambda*_j`` the scheme draws
    ``v ~ U(0, C(alpha*_j))`` with ``C(alpha) = prod_{i_o in j}
    t_{i_o}**alpha`` -- a product over *uncensored* members only, so ``C = 1``
    and no draw at all when the cluster has no uncensored member -- and one
    ``u_i`` uniform per member, from which ``w_i = u_i * exp(-t_i^alpha*_j /
    lambda*_j)``.  The bounds are then ``alpha^- = max(0, log v / sum_{i_o}
    log t_{i_o})`` and ``alpha^+ = min(phi, min_i log(-lambda* log w_i) /
    log t_i)`` over every member, with two explicit degenerate branches: the
    ``max(0, .)`` is needed because ``v < 1`` gives a negative log, and the
    empty-sum case ``n_j^o = 0`` sets ``alpha^- = 0`` directly rather than
    dividing by zero (``v`` is not even drawn).  A ``w_i`` of exactly zero
    (probability zero, but a real floating-point value) gives an infinite
    bound and drops out of the minimum.

    The implementation does not form ``v`` or ``w`` themselves, and never
    computes ``C`` by exponentiating a product of powers: with ``v = u C`` and
    ``w = u_w exp(-t**alpha / lambda)`` the expressions simplify to
    ``alpha^- = alpha*_j + log u / sum_i_o log t_i_o`` and
    ``-lambda* log w_i = t_i**alpha*_j - lambda* log u_w``, which is exactly
    what the plan says and free of overflow.  ``log t > 0`` is guaranteed by
    the Appendix requirement ``t > 1`` enforced by :class:`SurvData`.

    Random-number consumption, which the golden regression of plan Section 7.5
    depends on: one uniform for ``v`` when the cluster has at least one
    uncensored member, none otherwise, then one uniform per member for the
    ``w``'s.
    """
    members = _cluster_members(state, j)
    alpha = float(state.alpha_star[j])
    lam = float(state.lam_star[j])
    times = np.asarray(data.time, dtype=np.float64)[members]
    censored_mask = ~np.asarray(data.event, dtype=bool)[members]
    log_times = np.log(times)

    n_uncensored = int(np.count_nonzero(~censored_mask))
    if n_uncensored > 0:
        u_v = float(rng.random())
        if u_v > 0.0:
            log_time_sum = float(np.sum(log_times[~censored_mask]))
            alpha_minus = alpha + math.log(u_v) / log_time_sum
        else:
            # v = 0 imposes no lower constraint on the shape.
            alpha_minus = 0.0
        alpha_minus = max(0.0, alpha_minus)
    else:
        # C(alpha) = 1: the lower bound is unconstrained.  The plan's formula
        # would divide by an empty sum, so it is not evaluated.
        alpha_minus = 0.0

    u_w = np.asarray(rng.random(members.size), dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_u_w = np.log(u_w)
        # ``-lambda* log w_i = t_i**alpha - lambda* log u_i``; the bound is
        # ``log(.) / log t_i``, and ``u_i = 0`` gives ``+inf``, which the
        # minimum correctly ignores (that ``w`` constrains nothing).
        upper_bounds = np.log(np.power(times, alpha) - lam * log_u_w) / log_times
    alpha_plus = min(float(state.phi), float(np.min(upper_bounds)))
    return alpha_minus, alpha_plus


def refresh_cluster_alpha(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    j: int,
    rng: np.random.Generator,
) -> None:
    """Refresh cluster ``j``'s shape ``alpha*_j`` from its full conditional.

    This is the second half of step (b), the Damien-Wakefield-Walker
    auxiliary-variable device of plan Section 2.2(b), Eqs. A.5-A.6.  With
    ``n_j^o = |{i_o : s_{i_o} = j}|`` the number of *uncensored* members and
    ``lambda*_j`` the scale drawn by :func:`refresh_cluster` in this same
    sweep, the target is ``p(alpha*_j | v, w) = k (alpha*_j)**(k - 1) /
    ((alpha^+)**k - (alpha^-)**k)`` on ``(alpha^-_j, alpha^+_j)`` with
    ``k = n_j^o + 1``, i.e. a density proportional to
    ``(alpha*_j)**(k - 1) = (alpha*_j)**n_j^o``, *not* a uniform draw.  The
    inverse-CDF draw is ``alpha*_j = [
    (alpha^-)**k + u ((alpha^+)**k - (alpha^-)**k) ]**(1/k)`` with
    ``u ~ U(0, 1)``.  It is uniform only for an all-censored cluster
    (``n_j^o = 0``, ``k = 1``); for ``n_j^o = 1`` it is already triangular.  A
    uniform draw on the bounds biases every uncensored cluster's shape upward
    (+8% in ``E[alpha*]`` at ``n_j^o = 1``, +29% at ``n_j^o = 5`` on
    ``(0.5, 1.5)``, per plan Section 2.2(b)); test 11 of plan Section 7.2 is
    built to reject it.

    The transition is exact (acceptance probability 1) but is **not** an
    ordinary slice sampler: the auxiliaries are redrawn every sweep from
    ranges determined by the current ``alpha*_j``, and the target is then
    sampled on the resulting interval.  It is implemented literally as
    specified; do not substitute a different sampler.

    Degenerate cases are handled explicitly.  For ``n_j^o = 0``:
    ``alpha^- = 0`` and ``k = 1``, so the target is uniform on
    ``(0, alpha^+)``.  For ``n_j^c = 0`` the censored terms of the second
    minimum are absent, so ``alpha^+`` is ``min(phi, min_{i_o} ...)``.  For
    ``n_j = 1`` both minima are single terms.  If ``alpha^+ <= alpha^-`` --
    possible only by floating-point equality at a boundary -- the shape is
    clamped to the interval's midpoint and the event is counted by
    :func:`alpha_bound_guard_count`; a plain ``uniform`` on a degenerate
    interval would silently freeze the cluster's shape.

    After the draw the cluster location and the per-observation latent shapes
    of its members are updated together, because invariant 1 of plan
    Section 2.1 requires ``alpha == alpha_star[s]`` at all times.

    Parameters
    ----------
    state : DPState
        Mutated in place: ``alpha_star[j]`` and ``alpha[members]`` are replaced.
        ``lam_star``, ``lam``, ``nu``, ``gamma`` and ``phi`` are untouched.
    data : SurvData
        Dataset on the transformed scale; supplies the member times and their
        censoring indicators.
    prior : DPPrior
        Supplies the base-measure shape ``d`` through the input validation; the
        conditional of ``alpha*_j`` does not otherwise involve ``d``.
    j : int
        Cluster index in ``0 .. n_star-1``.  This is the test entry point of plan
        Section 7.2 test 11: a test fixes ``lambda*_j`` and the data, calls this
        many times and compares the draws with the unnormalised A.5 density.
    rng : np.random.Generator
        Random generator; the only source of randomness.

    Raises
    ------
    IndexError
        If ``j`` is out of range.
    ValueError
        If the state and the data describe different datasets, or ``prior.d`` is
        not a positive integer.
    StateInvariantError
        If cluster ``j`` is empty or ``n_j`` disagrees with its member list, or if
        the resulting draw violates ``0 < alpha*_j <= phi``.

    Notes
    -----
    This half must run *after* :func:`refresh_cluster` for the same cluster: the
    bounds are formed at the new ``lambda*_j`` and the old ``alpha*_j``.  Calling
    it first would use the stale scale and bias the cluster's location.
    """
    global _ALPHA_BOUND_GUARD_CLAMPS
    _validate_step_b_inputs(state, data, prior)
    members = _cluster_members(state, j)
    n_uncensored = int(np.count_nonzero(np.asarray(data.event, dtype=bool)[members]))
    alpha_minus, alpha_plus = _alpha_star_bounds(state, data, prior, j=j, rng=rng)

    if alpha_plus > alpha_minus:
        k = n_uncensored + 1
        u = float(rng.random())
        lower = alpha_minus**k
        upper = alpha_plus**k
        alpha = (lower + u * (upper - lower)) ** (1.0 / k)
    else:
        _ALPHA_BOUND_GUARD_CLAMPS += 1
        alpha = 0.5 * (alpha_minus + alpha_plus)

    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > float(state.phi):
        raise StateInvariantError(
            f"the alpha* draw for cluster {j} left (0, phi]: got {alpha!r} "
            f"from bounds ({alpha_minus!r}, {alpha_plus!r})"
        )
    state.alpha_star[j] = alpha
    state.alpha[members] = alpha


def sample_step_b(
    state: DPState,
    data: SurvData,
    prior: DPPrior,
    *,
    rng: np.random.Generator,
) -> None:
    """Run step (b): refresh every cluster location, in ``j`` order.

    Plan Section 3.4 fixes the order ``j = 0 .. n_star-1``; the order is part
    of the specification because it fixes the random-number stream and
    therefore the golden regression test of plan Section 7.5.  Each cluster
    consumes exactly one gamma variate here (the ``lambda*_j`` draw), so a
    sweep's step (b) consumes ``n_star`` variates in a deterministic
    sequence -- a property the caller relies on when it replays a seed.

    Per cluster the two halves of step (b) run in the order the Appendix
    states: the scale ``lambda*_j`` from its inverse-gamma conditional
    (:func:`refresh_cluster`), then the shape ``alpha*_j`` by the
    Damien-Wakefield-Walker auxiliary-variable scheme of plan Section 2.2(b),
    Eqs. A.5-A.6 (:func:`refresh_cluster_alpha`), whose ``w`` bounds are formed
    at the *new* scale and the *old* shape.  Keeping both halves in one entry
    point means the sweep has a single step-(b) call site and the
    ``alpha_i = alpha*_{s_i}`` / ``lambda_i = lambda*_{s_i}`` synchronisation
    happens once per cluster rather than once per parameter.

    The random-number consumption is fixed and documented, because the golden
    regression of plan Section 7.5 depends on it: per cluster one gamma
    variate, one uniform for ``u_v`` if the cluster has at least one
    uncensored member, one uniform per member for the ``w``'s, and one uniform
    for the inverse-CDF shape draw unless the bounds guard clamped the
    interval.

    Parameters
    ----------
    state : DPState
        Mutated in place.  Must be compacted: step (a) leaves it so, and
        :func:`refresh_cluster` rejects a cluster whose member list disagrees
        with ``n_j``.  ``nu``, ``gamma`` and ``phi`` are held fixed here; step
        (c) refreshes them, in that order, after this step.  Invariant 3
        (``0 < alpha*_j <= phi``) is enforced by the shape draw's bound guard,
        which caps ``alpha^+_j`` at ``phi``.
    data, prior :
        As in :func:`refresh_cluster` and :func:`refresh_cluster_alpha`.
    rng : np.random.Generator
        Random generator; the only source of randomness.

    Raises
    ------
    ValueError
        If the state and the data describe different datasets.
    StateInvariantError
        If a cluster is empty or ``n_j`` disagrees with the labels, or the
        final state violates a Section 2.1 invariant.

    Notes
    -----
    ``state.check_invariants(prior)`` runs once at the end of the loop rather
    than after every cluster: the cheap per-cluster consistency checks already
    live in :func:`_cluster_members`, and the full check is what catches a
    desynchronised ``alpha``/``alpha*`` or ``lam``/``lam*`` pair before it
    reaches step (c).
    """
    _validate_step_b_inputs(state, data, prior)
    for j in range(state.n_star):
        refresh_cluster(state, data, prior, j=j, rng=rng)
        refresh_cluster_alpha(state, data, prior, j=j, rng=rng)
    state.check_invariants(prior)
