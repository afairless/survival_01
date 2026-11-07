"""Functional inference for the Weibull DP mixture (Kottas 2002, Section 3).

The Gibbs sampler of :mod:`src.dp_gibbs` stores draws of the latent
``theta_i = (alpha_i, lambda_i)`` and of the hyperparameters.  Functional
inference adds the two steps of the paper's Section 3:

1. for each stored draw ``b``, draw a realisation
   ``G_b ~ [G | theta_b, nu_b, gamma_b, phi_b]`` with
   :func:`src.dp_weibull.sample_dp_posterior`;
2. evaluate the mixture functional implied by that one realisation, e.g.

       ``S_b(t) = sum_j w_j exp(-t**alpha_j / lambda_j)``,

   using every atom of ``G_b`` for every functional.

The default ``method="exact"`` is the Rao-Blackwellised limit of the paper's
``L -> infinity`` estimator: it replaces the ``L`` Monte Carlo atom draws of
step (ii) by the truncated weights themselves.  It therefore has no ``L`` noise
and is the path the figures use.  ``method="mc"`` reproduces steps (ii)-(iii)
of the paper literally: for each stored draw it takes one realisation ``G_b``,
draws ``L`` atoms from it, and averages the ``L`` kernels.  Both methods prune
the negligible stick-breaking tail first and evaluate survival, density and
hazard from the *same* atoms, so the MC hazard is the ratio of two averages
over one atom set rather than a ratio of two independent estimates.

Time scale
----------
The sampler consumes ``t_tilde = 1 + t / t_max`` (the Appendix requires
``t > 1``).  Every functional here is returned on the **original** scale, with
the four inverse maps of plan Section 2.3 applied exactly once by
:func:`src.dp_gibbs.inverse_map`:

    ``S(s) = S_tilde(1 + s / t_max)``,
    ``f(s) = f_tilde(1 + s / t_max) / t_max``,
    ``h(s) = h_tilde(1 + s / t_max) / t_max``,
    ``eta_p = t_max * (eta_tilde_p - 1)``.

Survival and CDF values are unchanged by the transform; the density and the
hazard carry the ``1 / t_max`` Jacobian.  Dropping that factor is invisible in
a plot of ``S`` and changes every density and hazard by a constant, which is
why the map lives in one function and test 26 checks it explicitly.

Prior functionals
-----------------
:func:`prior_functionals` applies the same machinery with ``G_b`` drawn from
the **prior**, ``G_b ~ DP(nu_b G0(. | phi_b, gamma_b))`` and
``(nu_b, gamma_b, phi_b)`` drawn from their priors with
:func:`src.dp_weibull.sample_prior_hyperparameters`.  The truncation level
follows the paper's Section 3 rule for the prior,
``J_prior = ceil(log(epsilon) / log(max_b nu_b / (1 + max_b nu_b)))``, i.e.
:func:`src.dp_weibull.truncation_number` at the largest drawn precision; it is
computed once for the run, exactly like the posterior common-``J`` rule.

Prior functionals are the reference for the prior bands of Figure 2, and they
are *not* centred on the elicitation guess: the priors are elicited on the
original time scale but applied to ``t_tilde = 1 + t / t_max``, a shift plus a
scale.  For the Section 5.1 constants and ``t_max = 4.5518`` the prior median
of the population median is about ``0.61``, not the elicited ``1.1727``, and
roughly 43% of prior draws place the population median below zero in original
time.  That is the paper's own convention (plan Section 10, decision 2); test
21b pins it so that a later "fix" of the transform or of the priors cannot
silently make the two scales coincide.

Quantile functionals
--------------------
A quantile is **not** a linear functional, so it cannot be read off the
``S``/``F`` arrays: the paper's Section 3 warns that the quantile functional
must be obtained by inverting the CDF of each draw separately.
:func:`posterior_quantiles` therefore evaluates the mixture CDF of ``G_b`` on
the supplied grid, inverts it per draw and then applies the affine inverse
map ``eta_p = t_max * (eta_tilde_p - 1)``.  Inversion uses the (monotone)
grid CDF for a piecewise-linear first guess and then polishes the root on the
*exact* mixture CDF with a bisection-safeguarded Newton iteration.  The polish
is what makes a single-atom draw exact to machine precision instead of the
``O(h**2)`` error a pure grid interpolation leaves behind; it also means the
inverted quantiles satisfy ``F_b(eta_p^{(b)}) = p`` far more tightly than the
``1e-6`` in ``t`` the plan asks for.

The CDF is only defined on the grid that is passed in, so a probability the
grid does not bracket is an error rather than a silent clamp: at ``t = 0``
the transformed time is ``1`` and the mixture CDF is already positive, which
is exactly why small probabilities may be unreachable on a positive grid.
(The two-population comparison of :func:`compare` is the one exception: it
ranks per-draw medians, and a median below its grid's first point is
interpolated from the origin rather than rejected — see
:func:`_grid_quantiles`.)

Two-population comparisons
--------------------------
:func:`compare` implements plan Section 4.4 for two independent fits of the
same time scale (e.g. treatments A and B of Section 5.2).  It returns, on the
one grid both fits carry,

* ``ratio``: the hazard ratio ``h_B(t) / h_A(t)`` per draw and grid point
  (Figure 5a; the paper calls the hazard ``lambda``);
* ``difference``: ``F_B(t) - F_A(t)`` per draw and grid point (Figure 5b), so
  a positive difference means population B fails earlier at that time;
* ``eta_a``, ``eta_b`` and ``eta_difference = eta_a - eta_b`` per draw, plus
  ``prob_eta_a_gt_eta_b``, the posterior probability that A's quantile is the
  larger one (Figure 5c, the paper's ``Pr(eta_A > eta_B | D) = 0.9135``).

Because the two fits come from independent chains, draw ``b`` of A is paired
with draw ``b`` of B: the product of the two posteriors is a sample from the
joint posterior, so pairing is valid only for **independent fits with the same
number of stored draws**.  The fit objects cannot prove independence, so this
is a documented obligation of the caller rather than a checked invariant.

``eta`` is the quantile of level ``p`` (default the median, ``p = 0.5``).
:class:`Functionals` stores only the evaluated ``S``/``F`` arrays, not the DP
realisations, so the quantile is found by inverting each draw's CDF
*piecewise linearly on ``t_grid``* (the same first guess
:func:`_invert_cdf` starts from, before its Newton polish).  Grid inversion
is ``O(h**2)`` in ``t`` and is what Figure 5c needs -- a per-draw ranking of
two clearly separated populations -- not a substitute for the exact per-draw
inversion of :func:`posterior_quantiles`.  The ranking, and therefore
``Pr(eta_A > eta_B)``, is invariant under any strictly increasing change of
time scale, which is why the comparison is meaningful on the original scale.

Scope
-----
In scope: posterior survival, CDF, density and hazard functionals, by exact
mixture evaluation over a pruned DP realisation (``method="exact"``) or by
the paper's ``L``-atom Monte Carlo estimator (``method="mc"``); quantile
functionals by inverting the exact mixture CDF per draw; prior functionals
from prior DP realisations, used as the Figure 2 prior bands; and
two-population comparisons of those functionals.

Out of scope (deliberately absent): plotting, model comparison, and any
re-sampling of ``theta``.  Nothing here draws from the posterior of the
sampler; it consumes :class:`DPChains` produced by :mod:`src.dp_gibbs`, or a
:class:`DPPrior` when there are no data.  Two-population comparisons are in
scope through :func:`compare`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.dp_gibbs import DPChains, DPPrior, SurvData, inverse_map, transform_time
from src.dp_weibull import (
    DPRealization,
    sample_dp_posterior,
    sample_dp_prior,
    sample_prior_hyperparameters,
    truncation_number,
    truncation_number_for_posterior,
)

__all__ = [
    "FunctionalSettings",
    "Functionals",
    "compare",
    "functionals_from_mc_realizations",
    "functionals_from_realizations",
    "posterior_functionals",
    "posterior_quantiles",
    "prior_functionals",
    "quantiles_from_realizations",
]


@dataclass
class FunctionalSettings:
    """Knobs of the functional-inference path, never model parameters.

    Attributes
    ----------
    method : str
        ``"exact"`` (default) evaluates the mixture functional from the
        truncated weights of one ``G_b``, the Rao-Blackwellised limit of the
        paper's ``L``-draw estimator.  ``"mc"`` reproduces the paper's steps
        (ii)-(iii) literally: draw ``L`` atoms from ``G_b`` and average their
        kernels.  The exact path is used for the figures; the MC path exists
        to check that the paper's estimator is being computed at all.
    L : int
        Number of atom draws ``L`` for ``method="mc"``.  Ignored by the exact
        path; kept in the container so that one settings object describes
        either method.
    atom_tol : float
        Tolerance passed to :meth:`src.dp_weibull.DPRealization.pruned`.
        The dropped tail carries at most this much mass, so every mixture
        functional changes by at most ``atom_tol``.
    n_t_grid : int
        Suggested number of points for a caller-built evaluation grid.  The
        functional functions take the grid explicitly; this value only
        documents and validates the conventional 300-point grid.
    """

    method: str = "exact"
    L: int = 2_500
    atom_tol: float = 1e-6
    n_t_grid: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string")
        if self.method not in {"exact", "mc"}:
            raise ValueError(f"method must be 'exact' or 'mc', got {self.method!r}")
        if isinstance(self.L, bool) or not isinstance(self.L, (int, np.integer)):
            raise ValueError(f"L must be an integer, got {self.L!r}")
        if int(self.L) < 1:
            raise ValueError(f"L must be at least 1, got {self.L!r}")
        if isinstance(self.n_t_grid, bool) or not isinstance(
            self.n_t_grid, (int, np.integer)
        ):
            raise ValueError(f"n_t_grid must be an integer, got {self.n_t_grid!r}")
        if int(self.n_t_grid) < 2:
            raise ValueError(f"n_t_grid must be at least 2, got {self.n_t_grid!r}")
        if isinstance(self.atom_tol, bool) or not isinstance(
            self.atom_tol, (int, float, np.floating)
        ):
            raise ValueError(f"atom_tol must be a number, got {self.atom_tol!r}")
        atom_tol = float(self.atom_tol)
        if not np.isfinite(atom_tol) or not 0.0 < atom_tol < 1.0:
            raise ValueError(
                f"atom_tol must lie strictly between 0 and 1, got {self.atom_tol!r}"
            )
        self.L = int(self.L)
        self.n_t_grid = int(self.n_t_grid)
        self.atom_tol = atom_tol


@dataclass
class Functionals:
    """Per-draw functional values on the original time scale.

    One row per stored draw ``b`` and one column per grid point.  The full
    ``(B, M)`` arrays are kept rather than online means because the quantile
    functional added by a later step needs the whole grid, and because a
    percentile band is a function of the whole cross-section, not of its mean.

    Attributes
    ----------
    t_grid : np.ndarray
        ``float64[M]`` original-scale evaluation times, strictly positive and
        finite.
    S, F, f, h : np.ndarray
        ``float64[B, M]`` survival, CDF, density and hazard of each draw on
        the original scale.  ``F = 1 - S``; ``h`` is evaluated from the same
        atom set as ``S`` and ``f`` (never by mixing two realisations).
        ``h`` is ``inf`` where ``S`` underflows to zero while ``f > 0``, the
        correct limit of the ratio.
    meta : dict
        Provenance of the run: ``method``, ``J_common``, ``atom_tol``,
        ``n_draws``, ``seed``, the atom-draw count ``L`` on the Monte Carlo
        path and, for :func:`posterior_functionals`, the chain settings seeds
        and ``t_max``.

    The container validates its own shape and bounds so that a transposed
    array or a wrong number of grid points fails here, at the storage
    boundary, rather than inside a plotting or summary routine.
    """

    t_grid: np.ndarray
    S: np.ndarray
    F: np.ndarray
    f: np.ndarray
    h: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t_grid = _as_t_grid(self.t_grid)
        arrays: dict[str, np.ndarray] = {}
        n_draws = n_grid = -1
        for name in ("S", "F", "f", "h"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(
                    f"{name} must be a 2-D (n_draws, n_grid) array, "
                    f"got shape {values.shape}"
                )
            if n_draws < 0:
                n_draws, n_grid = int(values.shape[0]), int(values.shape[1])
                if n_draws < 1:
                    raise ValueError("functional arrays must hold at least one draw")
                if n_grid != t_grid.size:
                    raise ValueError(
                        "functional arrays must have one column per t_grid point, "
                        f"got {n_grid} columns for {t_grid.size} points"
                    )
            if values.shape != (n_draws, n_grid):
                raise ValueError(
                    f"{name} must have shape {(n_draws, n_grid)}, got {values.shape}"
                )
            arrays[name] = np.array(values, dtype=np.float64, copy=True)

        for name in ("S", "F", "f"):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(
                    f"{name} must contain only finite values (no NaN or inf)"
                )
        if np.any(arrays["S"] < 0.0) or np.any(arrays["S"] > 1.0):
            raise ValueError("S must lie in [0, 1]")
        if np.any(arrays["F"] < 0.0) or np.any(arrays["F"] > 1.0):
            raise ValueError("F must lie in [0, 1]")
        if np.any(arrays["f"] < 0.0):
            raise ValueError("f must be non-negative")
        if np.any(arrays["h"] < 0.0):
            raise ValueError("h must be non-negative")

        self.t_grid = t_grid
        self.S = arrays["S"]
        self.F = arrays["F"]
        self.f = arrays["f"]
        self.h = arrays["h"]
        self.meta = dict(self.meta) if self.meta is not None else {}

    @property
    def n_draws(self) -> int:
        """Number of stored draws ``B``."""
        return int(self.S.shape[0])

    @property
    def n_grid(self) -> int:
        """Number of evaluation points ``M``."""
        return int(self.S.shape[1])


def _as_t_grid(t_grid: object) -> np.ndarray:
    """Validate an original-scale evaluation grid.

    A positive, finite grid is required because this module returns the
    density and the hazard as well as the survival function: at ``t = 0`` a
    Weibull density is infinite for ``alpha < 1`` and the hazard is then not a
    usable number.  Quantile inversion uses the CDF only, so
    :func:`_as_quantile_grid` allows the zero endpoint and only insists on a
    strictly increasing grid.
    """
    grid = np.asarray(t_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError(f"t_grid must be a 1-D array, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("t_grid must not be empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("t_grid must contain only finite values (no NaN or inf)")
    if np.any(grid <= 0.0):
        raise ValueError(
            "t_grid must be strictly positive on the original time scale; "
            "the density and hazard are not finite at t=0 for every atom"
        )
    return np.array(grid, dtype=np.float64, copy=True)


def _as_quantile_grid(t_grid: object) -> np.ndarray:
    """Validate a strictly increasing, non-negative original-scale grid.

    Quantile inversion differs from functional evaluation in two ways.  The
    zero endpoint is allowed, because the CDF (unlike the density and the
    hazard) is finite and equal to ``1 - exp(-1 / lambda**alpha)`` there on
    the transformed scale.  And the grid must be *strictly increasing*, since
    the inversion locates ``p`` in the grid CDF with a search and interpolates
    between neighbouring points; an unsorted or duplicated grid would make
    that search meaningless.
    """
    grid = np.asarray(t_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError(f"t_grid must be a 1-D array, got shape {grid.shape}")
    if grid.size < 2:
        raise ValueError("t_grid must hold at least two points to invert a CDF")
    if not np.all(np.isfinite(grid)):
        raise ValueError("t_grid must contain only finite values (no NaN or inf)")
    if np.any(grid < 0.0):
        raise ValueError(
            "t_grid must be non-negative on the original time scale; the "
            "quantile grid starts at the origin"
        )
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(
            "t_grid must be strictly increasing for CDF inversion, got a "
            "duplicate or descending pair"
        )
    return np.array(grid, dtype=np.float64, copy=True)


def _as_probability(p: object) -> float:
    """Validate a single probability in the open interval ``(0, 1)``.

    ``p = 0`` and ``p = 1`` are rejected rather than clamped: the ``0 %`` and
    ``100 %`` points of a Weibull mixture are ``-inf``/``+inf`` on the
    transformed scale, so the inverse map would produce nonsense (or
    ``t = -t_max``) instead of a quantile.  Booleans are rejected explicitly
    because ``True`` is an ``int`` in Python and would silently mean ``p = 1``.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError(f"p must be a single probability number, got {p!r}")
    value = float(p)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"p must lie strictly between 0 and 1, got {p!r}")
    return value


def _as_chain_sequence(chains: object) -> list[DPChains]:
    """Normalise one chain or a sequence of chains and check compatibility.

    All chains must describe the same number of observations and the same
    ``t_max``: the functionals are returned on one shared original-scale grid,
    so mixing datasets on different time scales would silently compare
    different units.  Different posterior draws of the same data are fine and
    are the intended multi-chain use.
    """
    if isinstance(chains, DPChains):
        return [chains]
    try:
        sequence = list(chains)  # type: ignore[call-overload]
    except TypeError as exc:
        raise ValueError(
            "chains must be a DPChains instance or a sequence of DPChains"
        ) from exc
    if not sequence:
        raise ValueError("chains must contain at least one DPChains instance")
    for chain in sequence:
        if not isinstance(chain, DPChains):
            raise ValueError(
                "chains must contain only DPChains instances, "
                f"got {type(chain).__name__}"
            )
    first = sequence[0]
    for chain in sequence[1:]:
        if chain.n != first.n:
            raise ValueError(
                "all chains must have the same number of observations, got "
                f"{chain.n} and {first.n}"
            )
        if chain.data.t_max != first.data.t_max:
            raise ValueError(
                "all chains must share the same t_max so that the original-scale "
                f"grid is comparable, got {chain.data.t_max} and {first.data.t_max}"
            )
    return sequence


def _resolve_seed(seed: int | None) -> int:
    """Return a non-negative integer seed, drawing one when ``None``."""
    if seed is None:
        return int(np.random.default_rng().integers(0, 2**63 - 1))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")
    if int(seed) < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    return int(seed)


def _evaluate_realization(
    realization: DPRealization, t_tilde: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``S``, ``f`` and ``h`` of one realisation on the t-tilde scale.

    The survival and density come from the same realisation, so the hazard
    ``f / S`` is a functional of one ``G_b``.  Where ``S`` underflows to zero
    the ratio is set to ``inf`` when ``f > 0`` (the correct limit) and to zero
    only when both are exactly zero.
    """
    survival = np.asarray(realization.survival(t_tilde), dtype=np.float64)
    density = np.asarray(realization.pdf(t_tilde), dtype=np.float64)
    hazard = np.divide(
        density,
        survival,
        out=np.full_like(density, np.inf),
        where=survival > 0.0,
    )
    both_zero = (survival == 0.0) & (density == 0.0)
    if np.any(both_zero):
        hazard[both_zero] = 0.0
    return survival, density, hazard


def functionals_from_realizations(
    realizations: Iterable[DPRealization],
    t_grid: object,
    data: SurvData,
    *,
    meta: dict[str, Any] | None = None,
) -> Functionals:
    """Evaluate mixture functionals from already-drawn DP realisations.

    This is the lower-level engine shared by :func:`posterior_functionals`
    and by tests that need a synthetic, known ``G``.  It does **not** prune:
    the caller decides whether a realisation's tail should be dropped, so a
    single-atom test realisation with weight one is evaluated exactly as
    supplied.

    For each realisation it computes, on the transformed grid
    ``t_tilde = 1 + t_grid / t_max``,

        ``S_tilde(t) = sum_j w_j exp(-t**alpha_j / lambda_j)``,
        ``f_tilde(t) = sum_j w_j alpha_j t**(alpha_j-1) / lambda_j
                       exp(-t**alpha_j / lambda_j)``,
        ``h_tilde(t) = f_tilde(t) / S_tilde(t)``,

    and then maps ``S`` and ``f``/``h`` back to the original scale with
    :func:`src.dp_gibbs.inverse_map` (``F = 1 - S``).  The hazard uses the
    same atoms as the survival and density; it is never formed from two
    different draws.

    Parameters
    ----------
    realizations : iterable of DPRealization
        The realisations to evaluate, one functional row per item.  Each must
        have weights that sum to one, so that ``S`` and ``F`` are proper.
        A generator is consumed once and is the memory-efficient form for a
        long chain of draws.
    t_grid : array-like
        Original-scale times, 1-D, finite and strictly positive.
    data : SurvData
        The dataset whose ``t_max`` defines the inverse map.
    meta : dict, optional
        Provenance to store on the result.  ``n_draws`` and ``t_max`` are
        filled in (or overwritten) here.

    Returns
    -------
    Functionals
        Arrays of shape ``(n_draws, len(t_grid))`` on the original scale.
    """
    if not isinstance(data, SurvData):
        raise ValueError("data must be a SurvData instance")
    grid = _as_t_grid(t_grid)
    t_tilde = transform_time(data, grid)

    survival_rows: list[np.ndarray] = []
    density_rows: list[np.ndarray] = []
    hazard_rows: list[np.ndarray] = []
    for realization in realizations:
        if not isinstance(realization, DPRealization):
            raise ValueError(
                "realizations must contain only DPRealization instances, "
                f"got {type(realization).__name__}"
            )
        total = float(np.sum(realization.weights))
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(
                "realisation weights must sum to one, got "
                f"{total!r}; call DPRealization.pruned or use a proper draw"
            )
        survival, density, hazard = _evaluate_realization(realization, t_tilde)
        survival_rows.append(survival)
        density_rows.append(density)
        hazard_rows.append(hazard)

    if not survival_rows:
        raise ValueError("realizations must contain at least one realisation")

    survival_tilde = np.vstack(survival_rows)
    density_tilde = np.vstack(density_rows)
    hazard_tilde = np.vstack(hazard_rows)

    mapped = inverse_map(
        data,
        grid,
        survival=survival_tilde,
        density=density_tilde,
        hazard=hazard_tilde,
    )
    survival = mapped["survival"]
    density = mapped["density"]
    hazard = mapped["hazard"]

    out_meta: dict[str, Any] = dict(meta) if meta is not None else {}
    out_meta["n_draws"] = int(survival.shape[0])
    out_meta["t_max"] = float(data.t_max)
    return Functionals(
        t_grid=grid,
        S=survival,
        F=1.0 - survival,
        f=density,
        h=hazard,
        meta=out_meta,
    )


def _as_atom_draw_count(L: object, name: str = "L") -> int:
    """Validate the Monte Carlo atom-draw count ``L``.

    ``L`` must be a positive integer.  Booleans are rejected explicitly
    because ``True`` is an ``int`` in Python and would silently mean ``L = 1``.
    """
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {L!r}")
    count = int(L)
    if count < 1:
        raise ValueError(f"{name} must be at least 1, got {L!r}")
    return count


def _draw_atoms_from_realization(
    realization: DPRealization, n_atoms: int, rng: np.random.Generator
) -> DPRealization:
    """Draw ``n_atoms`` i.i.d. atoms from a realisation, as equal weights.

    Step (ii) of the paper's Section 3 draws ``theta^0_{bl} ~ G_b``.  For a
    truncated ``G_b = sum_j w_j delta_(alpha_j, lambda_j)`` that is a
    categorical draw over the atom indices; the result is represented as a
    :class:`DPRealization` whose atoms carry equal weight ``1 / n_atoms``, so
    that evaluating it gives exactly the estimator

        ``H_b = (1 / L) sum_l H(K_W(. | theta^0_{bl}))``

    of step (iii).  Keeping one shared set of drawn atoms is what makes the
    hazard ``f / S`` a ratio over the *same* ``L`` atoms rather than a ratio
    of two independent Monte Carlo averages.
    """
    weights = np.asarray(realization.weights, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "realisation weights must carry positive total mass to draw atoms"
        )
    probabilities = weights / total
    indices = rng.choice(probabilities.size, size=n_atoms, p=probabilities)
    atom_weights = np.full(n_atoms, 1.0 / n_atoms, dtype=np.float64)
    atom_weights /= np.sum(atom_weights)
    return DPRealization(
        weights=atom_weights,
        alphas=np.asarray(realization.alphas)[indices],
        lambdas=np.asarray(realization.lambdas)[indices],
        precision=realization.precision,
    )


def functionals_from_mc_realizations(
    realizations: Iterable[DPRealization],
    t_grid: object,
    data: SurvData,
    *,
    L: int = 2_500,
    rng: np.random.Generator | None = None,
    meta: dict[str, Any] | None = None,
) -> Functionals:
    """Evaluate the paper's Monte Carlo functionals from DP realisations.

    This is the literal version of steps (ii)-(iii) of the paper's Section 3
    and the low-level engine behind ``posterior_functionals(method="mc")``.
    For each supplied ``G_b`` it

    1. draws ``L`` atoms ``theta^0_{bl} ~ G_b`` from the realisation's weights
       (one shared set, used for every functional);
    2. evaluates the mixture implied by those atoms,

           ``S_hat_b(t) = (1 / L) sum_l exp(-t**alpha_l / lambda_l)``,
           ``f_hat_b(t) = (1 / L) sum_l alpha_l t**(alpha_l-1) / lambda_l
                          exp(-t**alpha_l / lambda_l)``,

       and reports ``h_hat_b = f_hat_b / S_hat_b`` from the same atoms;
    3. maps the results back to the original time scale with
       :func:`src.dp_gibbs.inverse_map`.

    No pruning happens here: the caller decides whether the tail of ``G_b``
    should be dropped (both production paths prune with ``atom_tol`` first).
    This keeps a single-atom test realisation exact and lets tests compare the
    MC and exact estimators on identical ``G_b`` draws.

    Monte Carlo noise is ``O(1 / sqrt(L))`` in the survival and the density;
    the hazard is a ratio of two averages over the same atoms, hence biased at
    ``O(1 / L)`` but consistent.  The survival and density estimators are
    conditionally unbiased for their finite-sum counterparts on the truncated
    ``G_b`` (plan test 22).

    Parameters
    ----------
    realizations : iterable of DPRealization
        One ``G_b`` per functional row.  Each must have weights that sum to
        one, exactly as for :func:`functionals_from_realizations`; the atom
        law is otherwise not a probability distribution.
    t_grid : array-like
        Original-scale times, 1-D, finite and strictly positive.
    data : SurvData
        The dataset whose ``t_max`` defines the inverse map.
    L : int, optional
        Number of atoms drawn from each realisation (default 2500, the paper's
        value).
    rng : numpy.random.Generator or None, optional
        Random source for the atom draws.  ``None`` uses fresh OS entropy.
    meta : dict, optional
        Provenance to store on the result; ``method``, ``L``, ``n_draws`` and
        ``t_max`` are filled in here.

    Returns
    -------
    Functionals
        Arrays of shape ``(n_draws, len(t_grid))`` on the original scale.
    """
    if not isinstance(data, SurvData):
        raise ValueError("data must be a SurvData instance")
    n_atoms = _as_atom_draw_count(L)
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator or None")
    generator = np.random.default_rng() if rng is None else rng

    def empirical_realizations() -> Iterable[DPRealization]:
        for realization in realizations:
            if not isinstance(realization, DPRealization):
                raise ValueError(
                    "realizations must contain only DPRealization instances, "
                    f"got {type(realization).__name__}"
                )
            total = float(np.sum(realization.weights))
            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
                raise ValueError(
                    "realisation weights must sum to one, got "
                    f"{total!r}; call DPRealization.pruned or use a proper draw"
                )
            yield _draw_atoms_from_realization(realization, n_atoms, generator)

    out_meta: dict[str, Any] = dict(meta) if meta is not None else {}
    out_meta["method"] = "mc"
    out_meta["L"] = n_atoms
    return functionals_from_realizations(
        empirical_realizations(), t_grid, data, meta=out_meta
    )


def _common_truncation(sequence: Sequence[DPChains]) -> tuple[int, float]:
    """The one ``J`` and ``epsilon`` that drive every ``G_b`` of a run.

    ``truncation_number_for_posterior`` is increasing in the precision, so
    the largest ``nu`` gives the conservative level; taking the smallest
    ``epsilon`` is conservative in the same way if a caller passes chains
    whose settings disagree.  Computing this once (never per draw) is the
    plan's common-``J`` rule, test 24.
    """
    max_nu = max(float(np.max(chain.nu)) for chain in sequence)
    epsilon = min(float(chain.settings.epsilon) for chain in sequence)
    j_common = truncation_number_for_posterior(max_nu, sequence[0].n, epsilon)
    return int(j_common), float(epsilon)


def _posterior_realizations(
    sequence: Sequence[DPChains],
    *,
    j_common: int,
    epsilon: float,
    atom_tol: float,
    rng: np.random.Generator,
) -> Iterable[DPRealization]:
    """Draw and prune one ``G_b`` per stored posterior draw, chain by chain.

    Every draw uses the same truncation level ``j_common`` and the same
    random stream, so two calls with the same seed produce the same sequence
    of realisations -- which is what lets the quantile path be checked against
    the CDF of :func:`posterior_functionals` on identical ``G_b`` draws.
    """
    for chain in sequence:
        for b in range(chain.n_draws):
            draw = sample_dp_posterior(
                float(chain.nu[b]),
                float(chain.gamma[b]),
                float(chain.phi[b]),
                chain.theta[b],
                d=chain.prior.d,
                epsilon=epsilon,
                n_terms=j_common,
                rng=rng,
            )
            yield draw.pruned(atom_tol)


def posterior_functionals(
    chains: DPChains | Sequence[DPChains],
    t_grid: object,
    *,
    settings: FunctionalSettings | None = None,
    seed: int | None = None,
) -> Functionals:
    """Posterior functionals from exact atom mixtures (plan Section 4.1).

    For each stored draw ``b`` of each supplied chain this

    1. draws ``G_b ~ [G | theta_b, nu_b, gamma_b, phi_b]`` with
       :func:`src.dp_weibull.sample_dp_posterior`, using the **common**
       truncation level
       ``J_common = truncation_number_for_posterior(max_b nu_b, n, epsilon)``
       computed once for the whole run -- never a per-draw ``J``;
    2. prunes the trailing stick-breaking tail with
       :meth:`src.dp_weibull.DPRealization.pruned(settings.atom_tol)`;
    3. evaluates the exact mixture ``S``, ``F``, ``f`` and ``h`` on
       ``t_grid`` and maps them back to the original time scale.

    ``settings.method="exact"`` (default) is the Rao-Blackwellised estimator
    of the paper's Section 3: the ``L`` atom draws of step (ii) are replaced by
    the truncated weights, which is their ``L -> infinity`` limit and has no
    Monte Carlo noise in ``L``.  ``settings.method="mc"`` draws
    ``settings.L`` atoms from each pruned ``G_b`` instead, reproducing the
    paper's estimator literally; see :func:`functionals_from_mc_realizations`.
    Both methods prune before evaluating kernels and use one shared atom set
    for the hazard.

    Parameters
    ----------
    chains : DPChains or sequence of DPChains
        One chain or several posterior draws of the same dataset.  All chains
        must share ``n`` and ``data.t_max``.  Draws are stacked in the order
        supplied, chain by chain.
    t_grid : array-like
        Original-scale evaluation times, 1-D, finite and strictly positive.
    settings : FunctionalSettings, optional
        Functional-inference knobs.  Defaults to ``FunctionalSettings()``.
    seed : int or None, optional
        Seed for the DP-realisation draws.  ``None`` draws a fresh seed from
        OS entropy, which is recorded in ``meta["seed"]`` so that a run can be
        replayed afterwards.  The same seed and chains reproduce the arrays
        exactly.

    Returns
    -------
    Functionals
        ``(n_draws, len(t_grid))`` functionals for every stored draw, plus
        provenance in ``meta``.

    """
    if settings is None:
        settings = FunctionalSettings()
    if not isinstance(settings, FunctionalSettings):
        raise ValueError("settings must be a FunctionalSettings instance")

    sequence = _as_chain_sequence(chains)
    data = sequence[0].data

    j_common, epsilon = _common_truncation(sequence)

    resolved_seed = _resolve_seed(seed)
    if settings.method == "mc":
        # Two independent child streams: one for the G_b draws, one for the
        # atoms drawn from each G_b.  Reusing a single stream for both would
        # couple the atom indices to the realisation they are drawn from.
        realization_seed, atom_seed = np.random.SeedSequence(resolved_seed).spawn(2)
        rng = np.random.default_rng(realization_seed)
        atom_rng = np.random.default_rng(atom_seed)
    else:
        rng = np.random.default_rng(resolved_seed)
        atom_rng = None

    realizations = _posterior_realizations(
        sequence,
        j_common=j_common,
        epsilon=epsilon,
        atom_tol=settings.atom_tol,
        rng=rng,
    )

    meta: dict[str, Any] = {
        "method": settings.method,
        "J_common": int(j_common),
        "atom_tol": float(settings.atom_tol),
        "seed": int(resolved_seed),
        "epsilon": float(epsilon),
        "n_chains": len(sequence),
        "chain_seeds": tuple(chain.settings.seed for chain in sequence),
    }
    if settings.method == "mc":
        meta["L"] = int(settings.L)
        return functionals_from_mc_realizations(
            realizations, t_grid, data, L=settings.L, rng=atom_rng, meta=meta
        )
    return functionals_from_realizations(realizations, t_grid, data, meta=meta)


def _scale_only_data(t_max: float) -> SurvData:
    """Build a one-row stand-in whose only meaningful field is ``t_max``.

    The inverse map is defined on :class:`SurvData`, but prior functionals
    have no observations: only the time scale matters.  This carries a single
    synthetic observation whose transformed time ``2`` is the original
    ``t = t_max``, which is valid for any positive ``t_max`` and keeps the
    mapping in :func:`src.dp_gibbs.inverse_map` as the single implementation
    of the Jacobian.
    """
    return SurvData(time=np.array([2.0]), event=np.array([True]), t_max=t_max)


def _resolve_time_scale(data: object, t_max: object) -> float:
    """Resolve the original-scale maximum from ``data`` and/or ``t_max``.

    Prior functionals have no data, so the caller supplies the time scale
    either as a :class:`SurvData` (whose ``t_max`` is used) or as a bare
    ``t_max``.  Supplying both is allowed only when they agree; otherwise the
    returned grid would silently mix two units.
    """
    data_t_max: float | None = None
    if data is not None:
        if not isinstance(data, SurvData):
            raise ValueError("data must be a SurvData instance or None")
        data_t_max = float(data.t_max)

    if t_max is None:
        if data_t_max is None:
            raise ValueError(
                "prior functionals need a time scale: pass data=... or "
                "t_max=... so that the original-scale inverse map is defined"
            )
        return data_t_max

    if isinstance(t_max, bool) or not isinstance(
        t_max, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"t_max must be a number or None, got {t_max!r}")
    value = float(t_max)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"t_max must be finite and strictly positive, got {t_max!r}")
    if data_t_max is not None and value != data_t_max:
        raise ValueError(
            f"t_max and data.t_max must agree, got {value!r} and {data_t_max!r}"
        )
    return value


def _as_epsilon(epsilon: object) -> float:
    """Validate the prior truncation tolerance ``epsilon`` in ``(0, 1)``."""
    if isinstance(epsilon, bool) or not isinstance(
        epsilon, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"epsilon must be a number, got {epsilon!r}")
    value = float(epsilon)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"epsilon must lie strictly between 0 and 1, got {epsilon!r}")
    return value


def prior_functionals(
    prior: DPPrior,
    t_grid: object,
    n_draws: int,
    *,
    data: SurvData | None = None,
    t_max: float | None = None,
    settings: FunctionalSettings | None = None,
    epsilon: float = 1e-4,
    seed: int | None = None,
) -> Functionals:
    """Prior functionals ``S``, ``F``, ``f`` and ``h`` for prior bands.

    This is the prior counterpart of :func:`posterior_functionals` (plan
    Section 4.2): for each of ``n_draws`` replications it

    1. draws ``nu_b``, ``gamma_b``, ``phi_b`` from the priors in ``prior``
       with :func:`src.dp_weibull.sample_prior_hyperparameters`;
    2. draws ``G_b ~ DP(nu_b G0(. | phi_b, gamma_b))`` with
       :func:`src.dp_weibull.sample_dp_prior` at the **prior** common
       truncation level ``J_prior = ceil(log(epsilon) / log(max_b nu_b /
       (1 + max_b nu_b)))`` -- the paper's Section 3 rule, evaluated once at
       the largest drawn precision and never per draw;
    3. prunes the trailing stick-breaking tail with ``settings.atom_tol``
       and evaluates the exact mixture ``S``, ``F``, ``f`` and ``h`` on
       ``t_grid``, mapping them back to the original time scale.

    Priors are elicited on the original time scale but applied on
    ``t_tilde = 1 + t / t_max`` (plan Section 5).  The resulting prior bands
    are therefore **not** centred on the elicitation guess: with the Section
    5.1 constants and ``t_max = 4.5518`` the prior mean of ``S(1.1727)`` is
    about ``0.43`` and the prior median of the population median is about
    ``0.61``, not ``1.1727``.  Figure 2 compares the posterior against these
    bands, so the caveat is part of the specification (decision 2 in the
    plan), not a defect to be repaired here.

    Only ``settings.method="exact"`` is supported: a prior band is a summary
    of the distribution of the functional, and the Rao-Blackwellised
    evaluation of a pruned prior realisation has no additional ``L`` noise to
    blur it.

    Parameters
    ----------
    prior : DPPrior
        Fixed prior hyperparameters ``(a_nu, b_nu, a_gamma, b_gamma, a_phi,
        b_phi, d)``.  The priors are not estimated, so this is the same
        container the sampler takes.
    t_grid : array-like
        Original-scale evaluation times, 1-D, finite and strictly positive.
    n_draws : int
        Number of prior replications ``B``.  Each costs one DP realisation,
        not one Markov chain.
    data : SurvData, optional
        Dataset whose ``t_max`` defines the inverse map.  Supply this or
        ``t_max``; supplying both requires them to agree.  The observations
        themselves are not used: a prior draw does not depend on the data.
    t_max : float, optional
        Original-scale maximum, for callers that have no dataset (e.g. a
        band evaluated at the paper's ``t_max = 4.5518``).
    settings : FunctionalSettings, optional
        Functional-inference knobs.  ``atom_tol`` is used for pruning;
        ``method`` must be ``"exact"``.
    epsilon : float, optional
        Target residual DP mass of the prior truncation rule (default
        ``1e-4``, the paper's value).  This is the prior analogue of the
        sampler's ``DPSettings.epsilon``; the posterior path takes it from
        the chains instead.
    seed : int or None, optional
        Seed for the hyperparameter and DP-realisation draws.  ``None`` draws
        a fresh seed from OS entropy, recorded in ``meta["seed"]``.  The
        same seed and arguments reproduce the arrays exactly.

    Returns
    -------
    Functionals
        ``(n_draws, len(t_grid))`` prior functionals on the original scale,
        plus provenance in ``meta`` (``J_prior``, ``max_nu``, ``epsilon``,
        ``atom_tol``, ``seed`` and ``t_max``).

    Raises
    ------
    ValueError
        If ``prior`` is not a :class:`DPPrior`, ``n_draws`` is not a positive
        integer, no time scale is supplied (or two disagreeing ones are),
        the grid, ``epsilon`` or ``seed`` is malformed, or
        ``settings.method`` is not ``"exact"``.
    """
    if not isinstance(prior, DPPrior):
        raise ValueError(
            f"prior must be a DPPrior instance, got {type(prior).__name__}"
        )
    draws = _as_atom_draw_count(n_draws, "n_draws")
    if settings is None:
        settings = FunctionalSettings()
    if not isinstance(settings, FunctionalSettings):
        raise ValueError("settings must be a FunctionalSettings instance")
    if settings.method != "exact":
        raise ValueError(
            "prior_functionals evaluates exact pruned prior realisations; "
            f"settings.method must be 'exact', got {settings.method!r}"
        )
    scale = _resolve_time_scale(data, t_max)
    tolerance = _as_epsilon(epsilon)
    grid = _as_t_grid(t_grid)
    resolved_seed = _resolve_seed(seed)

    rng = np.random.default_rng(resolved_seed)
    hyperparameters = [
        sample_prior_hyperparameters(
            prior.a_nu,
            prior.b_nu,
            prior.a_gamma,
            prior.b_gamma,
            prior.a_phi,
            prior.b_phi,
            rng=rng,
        )
        for _ in range(draws)
    ]

    # The paper's prior truncation rule, evaluated once at the largest drawn
    # precision (the prior analogue of the common-J posterior rule).
    max_nu = max(precision for precision, _, _ in hyperparameters)
    j_prior = truncation_number(max_nu, tolerance)

    realizations = (
        sample_dp_prior(
            precision,
            gamma,
            phi,
            d=prior.d,
            epsilon=tolerance,
            n_terms=j_prior,
            rng=rng,
        ).pruned(settings.atom_tol)
        for precision, gamma, phi in hyperparameters
    )

    meta: dict[str, Any] = {
        "sample": "prior",
        "method": "exact",
        "J_prior": int(j_prior),
        "max_nu": float(max_nu),
        "epsilon": float(tolerance),
        "atom_tol": float(settings.atom_tol),
        "seed": int(resolved_seed),
    }
    return functionals_from_realizations(
        realizations, grid, _scale_only_data(scale), meta=meta
    )


def _cdf_at(realization: DPRealization, t_tilde: float) -> float:
    """Mixture CDF of one realisation at a scalar transformed time."""
    return float(np.asarray(realization.cdf(np.array([t_tilde])))[0])


def _pdf_at(realization: DPRealization, t_tilde: float) -> float:
    """Mixture density of one realisation at a scalar transformed time."""
    return float(np.asarray(realization.pdf(np.array([t_tilde])))[0])


def _invert_cdf(
    realization: DPRealization,
    t_tilde_grid: np.ndarray,
    t_grid: np.ndarray,
    p: float,
) -> float:
    """Invert the mixture CDF of one realisation at probability ``p``.

    The grid CDF gives a piecewise-linear first guess, which on a 300-point
    grid is only accurate to order ``h**2`` in ``t``.  That guess is then
    polished on the *exact* mixture CDF with a Newton step safeguarded by the
    bracket the grid provides: where the density is positive and the step
    stays inside the bracket Newton is used, otherwise the step bisects.  The
    safeguard is what makes the routine terminate on flat regions of the CDF
    (underflowed tail, flat first grid interval) and the Newton refinement is
    what makes a single-atom draw exact to machine precision rather than
    ``O(h**2)``.

    Returns the quantile on the ``t_tilde`` scale; mapping to the original
    scale happens once, in :func:`inverse_map`, so the affine inverse map is
    never applied twice.

    Raises
    ------
    ValueError
        If ``p`` lies outside the range of the grid CDF.  Clamping would
        silently report the grid edge as a quantile; the failure names the
        attained CDF range so a caller knows how far to extend the grid.
    """
    cdf = np.asarray(realization.cdf(t_tilde_grid), dtype=np.float64)
    if not np.all(np.isfinite(cdf)):
        raise ValueError("the realisation CDF is not finite on t_grid")
    if p < cdf[0] or p > cdf[-1]:
        raise ValueError(
            f"p={p!r} is not bracketed by the CDF on t_grid: F ranges over "
            f"[{cdf[0]!r}, {cdf[-1]!r}] on the supplied grid.  Extend t_grid "
            "(a Weibull mixture is already positive at t=0 on the original "
            "scale) so that it spans the requested probability."
        )

    # Bracket from the grid CDF: ``index`` is the first grid point whose CDF
    # reaches p, so the root lies in [t[index-1], t[index]].
    index = int(np.searchsorted(cdf, p, side="left"))
    index = min(max(index, 0), t_grid.size - 1)
    lo = float(t_tilde_grid[max(index - 1, 0)])
    hi = float(t_tilde_grid[index])
    if lo >= hi:
        return lo

    t = float(np.interp(p, cdf, t_tilde_grid))
    t = min(max(t, lo), hi)
    for _ in range(200):
        value = _cdf_at(realization, t) - p
        if value == 0.0:
            return t
        if value < 0.0:
            lo = t
        else:
            hi = t
        density = _pdf_at(realization, t)
        candidate = np.nan
        if np.isfinite(density) and density > 0.0:
            candidate = t - value / density
        if not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)
        if abs(candidate - t) <= 1e-15 * max(1.0, abs(t)):
            return float(candidate)
        if hi - lo <= 1e-15 * max(1.0, hi):
            return float(candidate)
        t = float(candidate)
    return t


def quantiles_from_realizations(
    realizations: Iterable[DPRealization],
    t_grid: object,
    p: object,
    data: SurvData,
) -> np.ndarray:
    """Invert the mixture CDF of already-drawn DP realisations (test 27).

    This is the lower-level engine behind :func:`posterior_quantiles`, shared
    with tests that need a synthetic, known ``G_b``.  For each realisation it
    evaluates ``F_tilde(. | G_b)`` on the transformed grid
    ``t_tilde = 1 + t_grid / t_max``, solves ``F_tilde(eta) = p`` for the
    quantile ``eta`` on that scale (see :func:`_invert_cdf`) and maps every
    row back with :func:`src.dp_gibbs.inverse_map`,
    ``eta_p = t_max * (eta_p_tilde - 1)``.

    The quantile is not a linear functional of ``S``/``F``: inverting the
    cross-sectional mean of the CDF would give the quantile of a different
    distribution, so one inversion is performed per realisation and the
    mapping is applied once, to the quantiles only.

    Parameters
    ----------
    realizations : iterable of DPRealization
        One ``G_b`` per quantile, as in :func:`functionals_from_realizations`.
        Weights must sum to one, otherwise the "CDF" is not a CDF.  A
        generator is consumed once.
    t_grid : array-like
        Original-scale times, 1-D, finite, non-negative and strictly
        increasing.  It must bracket the requested probability: ``F`` is
        already positive at ``t = 0`` because the transformed time there is
        ``1``, so small probabilities may be unreachable.
    p : float
        Target probability, strictly inside ``(0, 1)``.
    data : SurvData
        The dataset whose ``t_max`` defines the inverse map.

    Returns
    -------
    np.ndarray
        ``float64[n_draws]`` quantiles on the original time scale, one per
        realisation.

    Raises
    ------
    ValueError
        If the grid or ``p`` is malformed, a realisation is not a proper
        distribution, or ``p`` is outside the CDF range of a realisation.
    """
    if not isinstance(data, SurvData):
        raise ValueError("data must be a SurvData instance")
    grid = _as_quantile_grid(t_grid)
    probability = _as_probability(p)
    t_tilde = transform_time(data, grid)

    quantiles_tilde: list[float] = []
    for realization in realizations:
        if not isinstance(realization, DPRealization):
            raise ValueError(
                "realizations must contain only DPRealization instances, "
                f"got {type(realization).__name__}"
            )
        total = float(np.sum(realization.weights))
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(
                "realisation weights must sum to one, got "
                f"{total!r}; call DPRealization.pruned or use a proper draw"
            )
        quantiles_tilde.append(_invert_cdf(realization, t_tilde, grid, probability))

    if not quantiles_tilde:
        raise ValueError("realizations must contain at least one realisation")

    mapped = inverse_map(data, grid, quantiles=np.asarray(quantiles_tilde))
    return np.asarray(mapped["quantiles"], dtype=np.float64)


def posterior_quantiles(
    chains: DPChains | Sequence[DPChains],
    t_grid: object,
    p: object,
    *,
    settings: FunctionalSettings | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Posterior quantile functional ``eta_p`` by CDF inversion (test 27).

    For each stored draw ``b`` this

    1. draws ``G_b ~ [G | theta_b, nu_b, gamma_b, phi_b]`` with the common
       truncation level ``J_common`` and prunes it with
       ``settings.atom_tol`` -- exactly the ``G_b`` that
       :func:`posterior_functionals` evaluates, so the same seed gives the
       same realisations;
    2. evaluates the mixture CDF ``F_b`` of ``G_b`` on ``t_grid``;
    3. inverts ``F_b(eta_p^{(b)}) = p`` per draw and maps the result back to
       the original time scale, ``eta_p = t_max * (eta_tilde_p - 1)``.

    Inversion is per draw on purpose.  The paper (Section 3) warns that the
    quantile functional is not a function of linear functionals: inverting an
    averaged survival curve would summarise a different distribution and
    would not commute with the mean.  Percentile bands are therefore taken
    over the ``B`` inverted quantiles, not formed from the CDF band.

    The inversion is always done on the exact truncated mixture CDF, so
    ``settings.method`` must be ``"exact"``; the ``L``-atom Monte Carlo
    estimator of the paper's step (ii) would give a step-function CDF whose
    quantiles are the drawn atoms themselves and are not comparable with the
    Rao-Blackwellised functionals of :func:`posterior_functionals`.

    Parameters
    ----------
    chains : DPChains or sequence of DPChains
        One chain or several posterior draws of the same dataset.  All chains
        must share ``n`` and ``data.t_max``.
    t_grid : array-like
        Original-scale evaluation grid, 1-D, non-negative and strictly
        increasing.  It must bracket ``p``: ``F`` is positive already at
        ``t = 0``, so a tiny ``p`` may be unreachable and raises rather than
        being clamped to the grid edge.  The plan's conventional grid is 300
        points from 0 to ``1.15 * max(t)``.
    p : float
        Target probability, strictly inside ``(0, 1)``.
    settings : FunctionalSettings, optional
        Functional-inference knobs; ``atom_tol`` is used for pruning and
        ``method`` must be ``"exact"``.  Defaults to ``FunctionalSettings()``.
    seed : int or None, optional
        Seed for the DP-realisation draws, with the same meaning and replay
        guarantees as in :func:`posterior_functionals`.  ``None`` draws a
        fresh seed from OS entropy.

    Returns
    -------
    np.ndarray
        ``float64[n_draws]`` quantiles on the original time scale, stacked
        chain by chain.

    Raises
    ------
    ValueError
        If the chains are incompatible, the grid or ``p`` is malformed,
        ``settings.method`` is not ``"exact"``, or ``p`` is not bracketed by
        the CDF of some draw.
    """
    if settings is None:
        settings = FunctionalSettings()
    if not isinstance(settings, FunctionalSettings):
        raise ValueError("settings must be a FunctionalSettings instance")
    if settings.method != "exact":
        raise ValueError(
            "posterior_quantiles inverts the exact mixture CDF per draw; "
            f"settings.method must be 'exact', got {settings.method!r}"
        )

    sequence = _as_chain_sequence(chains)
    data = sequence[0].data
    grid = _as_quantile_grid(t_grid)
    probability = _as_probability(p)

    j_common, epsilon = _common_truncation(sequence)
    rng = np.random.default_rng(_resolve_seed(seed))
    realizations = _posterior_realizations(
        sequence,
        j_common=j_common,
        epsilon=epsilon,
        atom_tol=settings.atom_tol,
        rng=rng,
    )
    return quantiles_from_realizations(realizations, grid, probability, data)


def _as_comparison_fit(value: object, name: str) -> Functionals:
    """Validate one side of a two-population comparison."""
    if not isinstance(value, Functionals):
        raise ValueError(
            f"{name} must be a Functionals instance, got {type(value).__name__}"
        )
    return value


def _comparison_grid(
    fit_a: Functionals, fit_b: Functionals, t_grid: object
) -> np.ndarray:
    """Require one shared, strictly positive grid for both fits.

    ``compare`` never re-interpolates: the stored ``S``/``F`` arrays are
    evaluated at the grid each :class:`Functionals` carries, so a comparison on
    a different grid would have to invent interpolation rules (and the median
    ranking could then depend on those rules).  Requiring the caller's
    ``t_grid`` to be exactly the grid of both fits makes that impossible.
    """
    grid = _as_t_grid(t_grid)
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(
            "t_grid must be strictly increasing for the quantile comparison, "
            "got a duplicate or descending pair"
        )
    for name, fit in (("fit_a", fit_a), ("fit_b", fit_b)):
        if fit.t_grid.shape != grid.shape or not np.allclose(
            fit.t_grid, grid, rtol=1e-12, atol=0.0
        ):
            raise ValueError(
                f"{name}.t_grid does not match the supplied t_grid; a "
                "comparison must use the one grid both fits were evaluated "
                "on, because compare does not re-interpolate"
            )
    return grid


def _comparison_scale(fit_a: Functionals, fit_b: Functionals) -> float | None:
    """Check that both fits describe the same original time scale.

    The original-scale grid is ``t = t_max (t_tilde - 1)``.  Two fits on
    different ``t_max`` are in different units, so a difference of quantiles in
    original time would compare weeks with months.  The check only runs when
    both fits record ``t_max`` in ``meta``; hand-built test functionals need no
    provenance.
    """
    scale_a = fit_a.meta.get("t_max")
    scale_b = fit_b.meta.get("t_max")
    if scale_a is not None and scale_b is not None and float(scale_a) != float(scale_b):
        raise ValueError(
            "the two fits must share t_max to be comparable on the original "
            f"time scale, got {scale_a!r} and {scale_b!r}"
        )
    if scale_a is not None:
        return float(scale_a)
    if scale_b is not None:
        return float(scale_b)
    return None


def _require_non_decreasing_cdf(fit: Functionals, name: str) -> None:
    """Reject a CDF that is not non-decreasing along the grid.

    Quantile inversion searches a monotone CDF; on a non-monotone row the
    search would silently return a point that is not a quantile at all.  A
    tiny negative step is tolerated because ``F = 1 - S`` can round by one ulp
    when ``S`` is built by summing many atoms.
    """
    steps = np.diff(fit.F, axis=1)
    if steps.size and np.any(steps < -1e-12):
        raise ValueError(
            f"{name}.F must be non-decreasing along t_grid to invert the CDF "
            "for the eta comparison"
        )


def _grid_quantiles(
    cdf: np.ndarray, t_grid: np.ndarray, p: float, name: str
) -> np.ndarray:
    """Invert each row of a grid CDF piecewise linearly at probability ``p``.

    ``cdf`` is ``(n_draws, n_grid)`` and non-decreasing along the grid.  The
    quantile of each row is found by locating the first grid point at or above
    ``p`` and linearly interpolating between it and its predecessor.  This is
    the ``O(h**2)`` grid inversion the comparison needs for a per-draw ranking;
    :func:`posterior_quantiles` polishes the same root on the exact mixture CDF
    and is the right tool when the quantile value itself is the deliverable.

    The left edge is **not** a truncation.  A survival CDF is exactly zero at
    ``t = 0``, so a quantile whose level lies below ``cdf[:, 0]`` is bracketed
    by the origin and the first grid point; it is interpolated on the segment
    ``(0, 0) -> (t_grid[0], cdf[:, 0])``.  This is what lets a figure use a
    positive, density-safe grid (the functional path cannot evaluate ``f`` or
    ``h`` at zero) while still ranking per-draw medians, which can fall below
    that grid's first point for a few of the 40,000 draws of a reported run.
    Without it, a single such draw would abort the whole comparison.

    The right edge *is* a truncation: a ``p`` above the last grid value is an
    error rather than a clamp, because the quantile lies beyond the evaluated
    times and returning the grid edge would make the ranking an artefact of
    the grid's length.
    """
    n_draws, n_grid = cdf.shape
    last = cdf[:, -1]
    if np.any(last < p):
        best = float(np.min(last))
        raise ValueError(
            f"p={p!r} is not bracketed by {name}.F on t_grid: the last grid "
            f"point reaches F only up to {best!r}.  Extend t_grid to the right"
        )

    # For a non-decreasing row, the number of entries below ``p`` is the first
    # index at or above it (``searchsorted(..., side="left")``).  ``index = 0``
    # means the quantile is below the first grid point; its lower bracket is the
    # origin ``(0, 0)`` rather than the (nonexistent) previous grid point.
    index = np.clip(np.sum(cdf < p, axis=1), 0, n_grid - 1)
    rows = np.arange(n_draws)
    at_origin = index == 0
    lower = np.maximum(index - 1, 0)
    upper = index
    f_lower = np.where(at_origin, 0.0, cdf[rows, lower])
    f_upper = cdf[rows, upper]
    t_lower = np.where(at_origin, 0.0, t_grid[lower])
    t_upper = t_grid[upper]
    span = f_upper - f_lower
    fraction = np.divide(
        p - f_lower, span, out=np.zeros(n_draws, dtype=np.float64), where=span > 0.0
    )
    return t_lower + fraction * (t_upper - t_lower)


def _hazard_ratio(h_a: np.ndarray, h_b: np.ndarray) -> np.ndarray:
    """Per-draw, per-grid-point ratio ``h_b / h_a`` with explicit limits.

    Both hazards are non-negative and may be ``inf`` where a survival
    underflows.  The ratio is

    * ``h_b / h_a`` where ``h_a`` is finite and positive;
    * ``inf`` where ``h_a == 0`` and ``h_b > 0`` (the correct limit);
    * ``0`` where ``h_a`` is infinite and ``h_b`` is finite;
    * ``nan`` where both are zero or both are infinite -- ``0/0`` and
      ``inf/inf`` are not numbers, and reporting one would silently poison a
      mean band.  Callers summarising the ratio should use nan-aware
      reductions.
    """
    ratio = np.full(h_a.shape, np.nan, dtype=np.float64)
    finite_positive = np.isfinite(h_a) & (h_a > 0.0)
    np.divide(h_b, h_a, out=ratio, where=finite_positive)
    zero_denominator = h_a == 0.0
    ratio[zero_denominator & (h_b > 0.0)] = np.inf
    infinite_denominator = np.isinf(h_a)
    ratio[infinite_denominator & np.isfinite(h_b)] = 0.0
    ratio[infinite_denominator & np.isinf(h_b)] = np.nan
    return ratio


def compare(
    fit_a: Functionals,
    fit_b: Functionals,
    t_grid: object,
    *,
    p: float = 0.5,
) -> dict[str, Any]:
    """Two-population comparison of two independent fits (plan Section 4.4).

    For treatments A and B fitted independently, draw ``b`` of A is paired with
    draw ``b`` of B.  The posterior of two independent datasets factorises, so
    the paired draws are draws from the joint posterior -- but only when the
    fits are genuinely independent **and** have the same number of stored
    draws.  The equal-length condition is checked; independence is the caller's
    obligation.

    On the shared original-scale grid the function returns three comparisons:

    * ``ratio`` -- ``h_B(t) / h_A(t)`` per draw and grid point, the hazard
      ratio of the paper's Figure 5a (the paper writes ``lambda`` for the
      hazard).  It is ``inf`` where ``h_A`` is zero and ``h_B`` positive and
      ``nan`` where the ratio is undefined; see :func:`_hazard_ratio`.
    * ``difference`` -- ``F_B(t) - F_A(t)`` per draw and grid point (Figure
      5b).  A positive value means population B has the larger CDF at ``t``,
      i.e. fails earlier.
    * ``eta_a``, ``eta_b`` and ``eta_difference = eta_a - eta_b`` -- the
      quantile of level ``p`` of each draw, inverted per draw from the stored
      CDF.  ``prob_eta_a_gt_eta_b`` is ``mean(eta_a > eta_b)``, the posterior
      probability ``Pr(eta_A > eta_B | D)`` of Figure 5c (paper: 0.9135).
      Ties count as "not greater", so the reported probability is strictly the
      probability that A's quantile exceeds B's.

    The quantile is obtained by piecewise-linear inversion of each draw's CDF
    on ``t_grid`` (see :func:`_grid_quantiles`).  That is ``O(h**2)`` in ``t``
    and is deliberately not the exact per-draw inversion of
    :func:`posterior_quantiles`: :class:`Functionals` keeps no DP
    realisations, and the comparison only needs the *ranking* of the two
    populations, which is invariant under the strictly increasing time
    transform.  If the quantile *value* is the deliverable, use
    :func:`posterior_quantiles` on the chains instead.

    ``t_grid`` must be exactly the grid both fits were evaluated on; there is
    no re-interpolation.  Both fits must also record the same ``t_max`` in
    ``meta`` when they record one at all.

    Parameters
    ----------
    fit_a : Functionals
        Functionals of population A (e.g. the longer-remission treatment).
    fit_b : Functionals
        Functionals of population B.
    t_grid : array-like
        The shared original-scale grid, positive, finite and 1-D.  It must
        match ``fit_a.t_grid`` and ``fit_b.t_grid``.
    p : float, optional
        Quantile level used for the ``eta`` comparison, in ``(0, 1)``.
        Default ``0.5`` (the population median), which is the Figure 5c
        functional.

    Returns
    -------
    dict
        Keys ``t_grid``, ``ratio`` (``(B, M)``), ``difference`` (``(B, M)``),
        ``eta_a``, ``eta_b``, ``eta_difference`` (``(B,)``),
        ``prob_eta_a_gt_eta_b`` (float), ``p`` and ``meta`` (``p``, the number
        of draws and grid points, both fits' method/seed provenance when
        present, and the shared ``t_max`` when known).

    Raises
    ------
    ValueError
        If either argument is not a :class:`Functionals`, the same object is
        passed twice, the draw counts differ, the grids do not match, the two
        fits record different ``t_max``, ``p`` is not in ``(0, 1)``, a CDF is
        not non-decreasing, or ``p`` is above the last grid point of a fit's
        CDF (a ``p`` below the first grid point is interpolated from the
        origin, where the CDF is exactly zero).
    """
    fit_a = _as_comparison_fit(fit_a, "fit_a")
    fit_b = _as_comparison_fit(fit_b, "fit_b")
    if fit_a is fit_b:
        raise ValueError(
            "fit_a and fit_b must be two different fits; comparing a fit with "
            "itself would always give ratio 1, difference 0 and probability 0.5"
        )
    if fit_a.n_draws != fit_b.n_draws:
        raise ValueError(
            "the two fits must have the same number of stored draws to pair "
            f"draw b with draw b, got {fit_a.n_draws} and {fit_b.n_draws}"
        )
    grid = _comparison_grid(fit_a, fit_b, t_grid)
    scale = _comparison_scale(fit_a, fit_b)
    probability_level = _as_probability(p)
    _require_non_decreasing_cdf(fit_a, "fit_a")
    _require_non_decreasing_cdf(fit_b, "fit_b")

    ratio = _hazard_ratio(fit_a.h, fit_b.h)
    difference = fit_b.F - fit_a.F

    eta_a = _grid_quantiles(fit_a.F, grid, probability_level, "fit_a")
    eta_b = _grid_quantiles(fit_b.F, grid, probability_level, "fit_b")
    eta_difference = eta_a - eta_b
    prob_eta_a_gt_eta_b = float(np.mean(eta_a > eta_b))

    meta: dict[str, Any] = {
        "p": probability_level,
        "n_draws": int(fit_a.n_draws),
        "n_grid": int(grid.size),
        "ratio": "h_B(t) / h_A(t)",
        "difference": "F_B(t) - F_A(t)",
        "pairing": "draw b of A paired with draw b of B (independent fits)",
    }
    for suffix, fit in (("a", fit_a), ("b", fit_b)):
        for key in ("method", "seed"):
            if key in fit.meta:
                meta[f"{key}_{suffix}"] = fit.meta[key]
    if scale is not None:
        meta["t_max"] = scale

    return {
        "t_grid": grid,
        "ratio": ratio,
        "difference": difference,
        "eta_a": eta_a,
        "eta_b": eta_b,
        "eta_difference": eta_difference,
        "prob_eta_a_gt_eta_b": prob_eta_a_gt_eta_b,
        "p": probability_level,
        "meta": meta,
    }
