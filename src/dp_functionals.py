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

Scope
-----
In scope: posterior survival, CDF, density and hazard functionals, by exact
mixture evaluation over a pruned DP realisation (``method="exact"``) or by
the paper's ``L``-atom Monte Carlo estimator (``method="mc"``).

Out of scope (added by later steps or deliberately absent): quantile
functionals, prior functionals, two-population comparisons, plotting, and any
re-sampling of ``theta``.  Nothing here draws from the posterior of the
sampler; it consumes :class:`DPChains` produced by :mod:`src.dp_gibbs`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.dp_gibbs import DPChains, SurvData, inverse_map, transform_time
from src.dp_weibull import (
    DPRealization,
    sample_dp_posterior,
    truncation_number_for_posterior,
)

__all__ = [
    "FunctionalSettings",
    "Functionals",
    "functionals_from_mc_realizations",
    "functionals_from_realizations",
    "posterior_functionals",
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
    usable number.  Quantile inversion (a later step) uses the CDF only and is
    free to start its own grid at zero.
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
    n = sequence[0].n

    # One J for the whole run.  ``truncation_number_for_posterior`` is
    # increasing in the precision, so the largest nu gives the conservative
    # level; taking the smallest epsilon is conservative in the same way if a
    # caller passes chains whose settings disagree.
    max_nu = max(float(np.max(chain.nu)) for chain in sequence)
    epsilon = min(float(chain.settings.epsilon) for chain in sequence)
    j_common = truncation_number_for_posterior(max_nu, n, epsilon)

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

    def realizations() -> Iterable[DPRealization]:
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
                yield draw.pruned(settings.atom_tol)

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
            realizations(), t_grid, data, L=settings.L, rng=atom_rng, meta=meta
        )
    return functionals_from_realizations(realizations(), t_grid, data, meta=meta)
