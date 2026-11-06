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

import numpy as np

__all__ = [
    "DPPrior",
    "DPSettings",
    "SurvData",
    "inverse_map",
    "inverse_time",
    "make_surv_data",
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
