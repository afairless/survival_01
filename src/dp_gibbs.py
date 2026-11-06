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
from scipy.special import logsumexp

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
    "log_q0_censored",
    "log_q0_uncensored",
    "make_grid",
    "make_surv_data",
    "trapezoid_log_weights",
    "transform_time",
]


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


def log_integral_alpha(
    log_integrand: np.ndarray, grid: AlphaGrid, *, phi: float
) -> np.ndarray:
    """Integrate an ``(n, M)`` log integrand over ``alpha`` in ``(0, phi)``.

    Implements the quadrature of plan Section 2.4: the trapezoid rule in
    ``alpha`` with ``logsumexp`` over the nodes, so that the sum of integrand
    values can never underflow to zero.  The change of variable
    ``alpha = phi * u`` is accounted for by the ``log(phi)`` term.

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
    return logsumexp(log_integrand + grid.log_w[None, :], axis=1) + math.log(phi)


def log_q0_uncensored(
    grid: AlphaGrid, *, gamma: float, phi: float, nu: float, d: int
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

    Returns
    -------
    np.ndarray
        ``float64[n]`` log ``q0^o(t_i)`` for every uncensored row.  Censored
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
    log_integral = log_integral_alpha(
        log_alpha_integrand_uncensored(grid, gamma=gamma, phi=phi, d=d),
        grid,
        phi=phi,
    )
    return (
        math.log(d)
        + math.log(nu)
        + d * math.log(gamma)
        - math.log(phi)
        - np.log(grid.time)
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
    grid: AlphaGrid, *, gamma: float, phi: float, nu: float, d: int
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

    Returns
    -------
    np.ndarray
        ``float64[n]`` log ``q0^c(z_i)``, one entry per row of the grid.  Use
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
    log_integral = log_integral_alpha(
        log_alpha_integrand_censored(grid, gamma=gamma, phi=phi, d=d),
        grid,
        phi=phi,
    )
    return math.log(nu) + d * math.log(gamma) - math.log(phi) + log_integral


def _as_shape_parameter(value: object, name: str = "d") -> int:
    """Normalise the inverse-gamma shape ``d`` to a positive ``int``."""
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if out != value or out < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out
