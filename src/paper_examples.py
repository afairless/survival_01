"""Figure-level recipes and the CLI driver for the Section 5 examples.

Scope
-----
This is the only module that imports ``matplotlib`` (plan Section 11).  It
turns the paper's Sections 5.1 and 5.2 into runnable recipes:

* the presets of plan Section 6.2 -- ``quick`` for tests, ``paper`` for the
  results note and ``paper-lite`` as the documented fallback of decision 5;
* the data and prior glue (``paper_data`` -> ``SurvData`` -> ``DPPrior``);
* one ``Fit`` per (dataset, prior) pair: chains from
  :func:`src.dp_gibbs.sample_chains`, functionals from
  :func:`src.dp_functionals.posterior_functionals`, and on request a prior
  band from :func:`src.dp_functionals.prior_functionals`;
* the five figure recipes of plan Section 6.3, saved as PNG under the
  git-ignored ``docs/research/figures/`` (decision 8: numbers go into
  ``docs/research/dp_gibbs_results.md``, binaries never enter git);
* ``python -m src.paper_examples --example {5.1,5.2} --preset {quick,paper,
  paper-lite} --prior {1,2,3,all} --seed N --out DIR``.

Non-goals
---------
No model code lives here: the sampler, the transform and the functionals are
imported, never reimplemented.  The module is therefore "statistics-free" in
the sense of the plan's module table -- it only schedules, collects and plots
-- and it must not be imported by the sampler modules.

Reproducibility
---------------
Data seeds are fixed (``DATA_SEED_5_1``; the leukemia data are transcribed),
the sampler seeds default to ``SEED_5_1``/``SEED_5_2`` and every random call
takes its seed explicitly.  A preset records its own name in
``DPSettings.preset`` so that a number can be traced back to the settings that
produced it.  The prior is elicited on the original scale but applied on the
``t_tilde`` scale (plan decision 2), so prior bands are compared against
``prior_functionals`` output, never against the elicitation guess.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.dp_functionals import (
    Functionals,
    FunctionalSettings,
    compare,
    posterior_functionals,
    prior_functionals,
)
from src.dp_gibbs import (
    DPChains,
    DPPrior,
    DPSettings,
    SurvData,
    make_surv_data,
    sample_chains,
)
from src.paper_data import (
    A_GAMMA,
    A_PHI,
    B_GAMMA,
    B_PHI,
    NU_PRIORS,
    D,
    TwoLognormalMix,
    leukemia_5_2,
)

__all__ = [
    "DATA_SEED_5_1",
    "DEFAULT_FIGURE_DIR",
    "PRESETS",
    "Preset",
    "Fit",
    "DemoResult",
    "build_parser",
    "data_5_1",
    "data_5_2",
    "example_5_1",
    "example_5_2",
    "figure_1",
    "figure_2",
    "figure_3",
    "figure_4",
    "figure_5",
    "fit_dataset",
    "main",
    "make_prior",
    "prior_band",
    "prior_label",
    "resolve_preset",
    "smooth_hazard",
    "time_grid",
]

#: Location of the regenerated PNGs.  Git-ignored by design (plan decision 8).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = REPO_ROOT / "docs" / "research" / "figures"

#: Seed of the committed Section 5.1 sample (``tests/fixtures/section_5_1.npz``).
DATA_SEED_5_1 = 721383
#: Default sampler seeds.  Separate from the data seeds so that a longer run
#: does not silently reuse the data stream.
SEED_5_1 = 20240601
SEED_5_2 = 20240602


# ---------------------------------------------------------------------------
# Presets (plan Section 6.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    """Bundle of sampler and functional settings for one reproduction mode.

    Attributes
    ----------
    name : str
        ``"quick"``, ``"paper"`` or ``"paper-lite"``; copied into
        ``sampler.preset`` by :func:`fit_dataset`.
    sampler : DPSettings
        Burn-in, saved draws, thinning, grid size, truncation tolerance and
        chain count.
    functionals : FunctionalSettings
        Functional path (``exact`` for every figure), the number of MC atoms
        ``L`` and the evaluation-grid size.
    n_prior_draws : int
        Number of prior replications for a prior band.  The plan fixes the
        sampler settings but not this count; the values below are the driver's
        documented choice.
    """

    name: str
    sampler: DPSettings
    functionals: FunctionalSettings
    n_prior_draws: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("preset name must be a non-empty string")
        if not isinstance(self.sampler, DPSettings):
            raise ValueError("preset sampler must be a DPSettings instance")
        if not isinstance(self.functionals, FunctionalSettings):
            raise ValueError("preset functionals must be a FunctionalSettings instance")
        if self.functionals.method != "exact":
            raise ValueError(
                "the figure recipes need functionals.method='exact'; the MC path "
                "exists only as the test-22 cross-check"
            )
        if isinstance(self.n_prior_draws, bool) or not isinstance(
            self.n_prior_draws, (int, np.integer)
        ):
            raise ValueError(
                f"n_prior_draws must be an integer, got {self.n_prior_draws!r}"
            )
        if int(self.n_prior_draws) < 1:
            raise ValueError("n_prior_draws must be at least 1")
        object.__setattr__(self, "n_prior_draws", int(self.n_prior_draws))

    def with_seed(self, seed: int | None) -> Preset:
        """Return a copy whose sampler uses ``seed``."""
        return replace(self, sampler=replace(self.sampler, seed=seed))


#: Plan Section 6.2.  ``paper`` is the full run (1,505,000 sweeps per chain,
#: 4 chains per prior); ``quick`` is the fast end-to-end mode; ``paper-lite``
#: is the documented fallback of plan decision 5 (B = 4,000, thin 40,
#: burn-in 2,000) and must be named in the results note if it is used.
PRESETS: dict[str, Preset] = {
    "quick": Preset(
        name="quick",
        sampler=DPSettings(
            burn_in=1_000,
            n_save=1_000,
            thin=10,
            n_alpha=256,
            epsilon=1e-4,
            n_chains=2,
            seed=None,
            preset="quick",
        ),
        functionals=FunctionalSettings(
            method="exact", L=200, atom_tol=1e-6, n_t_grid=300
        ),
        n_prior_draws=200,
    ),
    "paper": Preset(
        name="paper",
        sampler=DPSettings(
            burn_in=5_000,
            n_save=10_000,
            thin=150,
            n_alpha=256,
            epsilon=1e-4,
            n_chains=4,
            seed=None,
            preset="paper",
        ),
        functionals=FunctionalSettings(
            method="exact", L=2_500, atom_tol=1e-6, n_t_grid=300
        ),
        n_prior_draws=1_000,
    ),
    "paper-lite": Preset(
        name="paper-lite",
        sampler=DPSettings(
            burn_in=2_000,
            n_save=4_000,
            thin=40,
            n_alpha=256,
            epsilon=1e-4,
            n_chains=4,
            seed=None,
            preset="paper-lite",
        ),
        functionals=FunctionalSettings(
            method="exact", L=1_000, atom_tol=1e-6, n_t_grid=300
        ),
        n_prior_draws=500,
    ),
}


def resolve_preset(value: str | Preset) -> Preset:
    """Return a :class:`Preset` from its name, or pass one through unchanged."""
    if isinstance(value, Preset):
        return value
    if not isinstance(value, str) or value not in PRESETS:
        raise ValueError(f"unknown preset {value!r}; choose one of {sorted(PRESETS)}")
    return PRESETS[value]


# ---------------------------------------------------------------------------
# Priors and data
# ---------------------------------------------------------------------------
def prior_label(index: int) -> str:
    """Readable label of the ``index``-th ``Gamma(shape, rate)`` prior on nu."""
    a_nu, b_nu = NU_PRIORS[index]
    return f"nu ~ Gamma({a_nu:g}, {b_nu:g})"


def make_prior(prior_index: int = 0) -> DPPrior:
    """Build the Section 5 :class:`DPPrior` for one of the three nu priors.

    ``a_gamma``, ``b_gamma``, ``a_phi`` and ``b_phi`` come from
    :mod:`src.paper_data`; only ``(a_nu, b_nu)`` varies between priors.
    """
    if isinstance(prior_index, bool) or not isinstance(prior_index, (int, np.integer)):
        raise ValueError(f"prior_index must be an integer, got {prior_index!r}")
    index = int(prior_index)
    if not 0 <= index < len(NU_PRIORS):
        raise ValueError(
            f"prior_index must lie in [0, {len(NU_PRIORS) - 1}], got {index}"
        )
    a_nu, b_nu = NU_PRIORS[index]
    return DPPrior(
        a_nu=a_nu,
        b_nu=b_nu,
        a_gamma=A_GAMMA,
        b_gamma=B_GAMMA,
        a_phi=A_PHI,
        b_phi=B_PHI,
        d=D,
    )


def _as_prior_indices(indices: Iterable[int] | None) -> tuple[int, ...]:
    """Validate and normalise a selection of nu-prior indices."""
    if indices is None:
        return tuple(range(len(NU_PRIORS)))
    selected = tuple(indices)
    if not selected:
        raise ValueError("at least one prior must be selected")
    out: list[int] = []
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"prior indices must be integers, got {index!r}")
        value = int(index)
        if not 0 <= value < len(NU_PRIORS):
            raise ValueError(
                f"prior index must lie in [0, {len(NU_PRIORS) - 1}], got {value}"
            )
        if value not in out:
            out.append(value)
    return tuple(out)


def data_5_1(seed: int = DATA_SEED_5_1) -> SurvData:
    """Section 5.1 data: one i.i.d. sample from the lognormal mixture."""
    mix = TwoLognormalMix(seed=seed)
    return make_surv_data(mix.times, np.ones(mix.n, dtype=bool))


def data_5_2(arm: str = "A", *, t_max: float | None = None) -> SurvData:
    """Section 5.2 leukemia data for arm ``"A"`` or ``"B"``.

    ``t_max`` overrides the arm's own maximum for the ``t_tilde`` transform.
    The two-population comparison of Figure 5 needs one common original time
    scale (``compare`` refuses two fits that record different ``t_max``), so
    :func:`example_5_2` passes the larger of the two arm maxima for both.
    """
    if arm not in ("A", "B"):
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
    time, event = leukemia_5_2()[arm]
    if t_max is None:
        return make_surv_data(time, event)

    shared = float(t_max)
    if not np.isfinite(shared) or shared <= 0.0:
        raise ValueError("t_max must be a finite, strictly positive number")
    if shared < float(np.max(time)):
        raise ValueError(
            f"t_max={shared!r} is smaller than the largest {arm} time "
            f"{float(np.max(time))!r}"
        )
    return SurvData(
        time=1.0 + np.asarray(time, dtype=np.float64) / shared,
        event=np.asarray(event, dtype=bool),
        t_max=shared,
    )


def time_grid(
    t_max: float,
    n_points: int = 300,
    *,
    extend: float = 1.15,
) -> np.ndarray:
    """Original-scale evaluation grid of ``n_points`` strictly positive times.

    The plan's convention is 300 points from 0 to ``1.15 * max(t)``.  The
    functional path needs strictly positive times (the density and hazard are
    not finite at 0 for every Weibull atom), so the zero endpoint is dropped
    and the grid starts one spacing above the origin.
    """
    t_max = float(t_max)
    if not np.isfinite(t_max) or t_max <= 0.0:
        raise ValueError("t_max must be a finite, strictly positive number")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError(f"n_points must be an integer, got {n_points!r}")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    if not np.isfinite(extend) or extend <= 0.0:
        raise ValueError("extend must be a finite, strictly positive number")
    return np.linspace(0.0, extend * t_max, int(n_points) + 1)[1:]


# ---------------------------------------------------------------------------
# One fit: chains plus functionals
# ---------------------------------------------------------------------------
@dataclass
class Fit:
    """One posterior fit: chains, functionals and (optionally) a prior band.

    Attributes
    ----------
    label : str
        The nu prior in readable form, e.g. ``"nu ~ Gamma(2, 0.9)"``.
    data : SurvData
        The dataset on the transformed scale.
    prior : DPPrior
        The fixed hyperparameters.
    t_grid : np.ndarray
        Original-scale evaluation grid shared with every fit it is compared to.
    chains : list of DPChains
        One entry per chain of the fit.
    functionals : Functionals
        Posterior functionals of the stacked chains on ``t_grid``.
    prior_band : Functionals or None
        Prior functionals for the same prior and grid, when one was drawn.
    """

    label: str
    data: SurvData
    prior: DPPrior
    t_grid: np.ndarray
    chains: list[DPChains]
    functionals: Functionals
    prior_band: Functionals | None = None

    @property
    def n_chains(self) -> int:
        """Number of chains in the fit."""
        return len(self.chains)


def fit_dataset(
    data: SurvData,
    prior: DPPrior,
    *,
    preset: str | Preset = "quick",
    seed: int | None = None,
    t_grid: np.ndarray | None = None,
    max_workers: int | None = None,
) -> Fit:
    """Run ``preset`` chains on ``data`` and evaluate their functionals.

    ``seed`` is the chain-group base seed of plan Section 3.6 (chain ``b`` uses
    ``seed + b``) and is also reused as the functional-path seed, so a fit is
    replayable from one integer.  A common ``t_grid`` must be supplied when the
    result will be passed to :func:`src.dp_functionals.compare`.
    """
    resolved = resolve_preset(preset)
    if not isinstance(data, SurvData):
        raise ValueError("data must be a SurvData instance")
    if not isinstance(prior, DPPrior):
        raise ValueError("prior must be a DPPrior instance")
    settings = replace(resolved.sampler, seed=seed)
    chains = sample_chains(data, prior, [settings], max_workers=max_workers)
    if t_grid is None:
        grid = time_grid(data.t_max, resolved.functionals.n_t_grid)
    else:
        grid = np.asarray(t_grid, dtype=np.float64)
    functionals = posterior_functionals(
        chains, grid, settings=resolved.functionals, seed=seed
    )
    label = f"nu ~ Gamma({prior.a_nu:g}, {prior.b_nu:g})"
    return Fit(
        label=label,
        data=data,
        prior=prior,
        t_grid=grid,
        chains=chains,
        functionals=functionals,
    )


def prior_band(
    fit: Fit,
    *,
    preset: str | Preset = "quick",
    n_draws: int | None = None,
    seed: int | None = None,
) -> Functionals:
    """Prior functionals for ``fit``'s prior and grid (plan Section 4.2).

    The band is elicited on the original scale but drawn on the ``t_tilde``
    scale (plan decision 2).  Compare it against posterior functionals of the
    same fit, never against the elicitation guess.
    """
    resolved = resolve_preset(preset)
    draws = resolved.n_prior_draws if n_draws is None else n_draws
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)):
        raise ValueError(f"n_draws must be an integer, got {draws!r}")
    if int(draws) < 1:
        raise ValueError("n_draws must be at least 1")
    return prior_functionals(
        fit.prior,
        fit.t_grid,
        int(draws),
        t_max=fit.data.t_max,
        settings=resolved.functionals,
        epsilon=resolved.sampler.epsilon,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Summary helpers and figure recipes (plan Section 6.3)
# ---------------------------------------------------------------------------
def _pyplot() -> Any:
    """Import ``matplotlib`` on demand with the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    _pyplot().close(fig)
    return path


def _summary(
    values: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and central ``level`` band of ``values`` along axis 0.

    Non-finite entries (an ``h`` of ``inf`` where ``S`` underflows, or a
    ``nan`` hazard ratio) are treated as missing rather than propagated.
    """
    array = np.asarray(values, dtype=np.float64)
    array = np.where(np.isfinite(array), array, np.nan)
    lo_q = 100.0 * (1.0 - level) / 2.0
    hi_q = 100.0 * (1.0 + level) / 2.0
    mean = np.nanmean(array, axis=0)
    lo = np.nanpercentile(array, lo_q, axis=0)
    hi = np.nanpercentile(array, hi_q, axis=0)
    return mean, lo, hi


def _gaussian_kernel(bandwidth: float) -> np.ndarray:
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be a finite, strictly positive number")
    radius = max(1, int(np.ceil(4.0 * bandwidth)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    return kernel / kernel.sum()


def _convolve_rows(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.size // 2
    pad = [(0, 0)] * (values.ndim - 1) + [(radius, radius)]
    padded = np.pad(values, pad, mode="edge")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), -1, padded
    )


def smooth_hazard(values: np.ndarray, *, bandwidth: float = 5.0) -> np.ndarray:
    """Gaussian smoothing of a hazard trace with ``bandwidth`` grid points.

    The plan's Figure 3 criterion (exactly three sign changes of the first
    difference on ``(0, 5)``) is evaluated on the smoothed mean, because raw
    first differences carry grid noise.  Non-finite values are masked before
    convolution and the result is renormalised by the convolved mask, so one
    ``inf`` cannot poison the curve.  Works on ``(M,)`` and ``(B, M)``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError(f"values must be 1-D or 2-D, got shape {array.shape}")
    kernel = _gaussian_kernel(bandwidth)
    finite = np.isfinite(array)
    numerator = _convolve_rows(np.where(finite, array, 0.0), kernel)
    denominator = _convolve_rows(finite.astype(np.float64), kernel)
    out = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def _plot_band(
    ax: Any,
    t: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: Any = None,
    band_alpha: float = 0.2,
    mean_style: str = "-",
) -> None:
    mean, lo, hi = _summary(values)
    ax.fill_between(t, lo, hi, color=color, alpha=band_alpha, linewidth=0)
    ax.plot(t, mean, color=color, linestyle=mean_style, label=label)


def figure_1(fits: dict[str, Fit], *, path: Path) -> Path:
    """Figure 1: posterior of ``nu`` and ``n*`` under the selected priors."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for label, fit in fits.items():
        nu = np.concatenate([chain.nu for chain in fit.chains])
        n_star = np.concatenate([chain.n_star for chain in fit.chains]).astype(float)
        axes[0].hist(nu, bins=40, density=True, histtype="step", label=label)
        axes[1].hist(n_star, bins="auto", density=True, histtype="step", label=label)
    axes[0].set_xlabel("nu")
    axes[0].set_ylabel("density")
    axes[0].set_title("posterior of nu")
    axes[1].set_xlabel("n*")
    axes[1].set_title("posterior of n*")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, path)


def figure_2(fit: Fit, band: Functionals, *, path: Path) -> Path:
    """Figure 2: posterior and prior of ``S(t)`` and ``f(t)``."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    panels = (
        (axes[0], fit.functionals.S, band.S, "S(t)"),
        (axes[1], fit.functionals.f, band.f, "f(t)"),
    )
    for ax, values, prior_values, title in panels:
        _plot_band(ax, fit.t_grid, values, label="posterior 95%")
        _plot_band(
            ax,
            fit.t_grid,
            prior_values,
            label="prior 95%",
            color="C1",
            band_alpha=0.12,
            mean_style="--",
        )
        ax.set_xlabel("t")
        ax.set_ylabel(title)
        ax.set_title(f"{title} ({fit.label})")
        ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def figure_3(fits: dict[str, Fit], *, path: Path, bandwidth: float = 5.0) -> Path:
    """Figure 3: smoothed posterior hazard ``h(t)`` under each prior."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for label, fit in fits.items():
        smoothed = smooth_hazard(fit.functionals.h, bandwidth=bandwidth)
        _plot_band(ax, fit.t_grid, smoothed, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.set_title(f"posterior hazard (Gaussian smoothing, {bandwidth:g} points)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def figure_4(fits_a: dict[str, Fit], fit_b: Fit, *, path: Path) -> Path:
    """Figure 4: leukemia survival under priors, A vs B, densities, hazards."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    reference_a = next(iter(fits_a.values()))

    ax = axes[0, 0]
    for label, fit in fits_a.items():
        _plot_band(ax, fit.t_grid, fit.functionals.S, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("S_A(t)")
    ax.set_title("S_A under the three priors")
    ax.legend()

    ax = axes[0, 1]
    for label, fit in (("A", reference_a), ("B", fit_b)):
        _plot_band(ax, fit.t_grid, fit.functionals.S, label=f"arm {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("S(t)")
    ax.set_title("S_A vs S_B")
    ax.legend()

    ax = axes[1, 0]
    for label, fit in (("A", reference_a), ("B", fit_b)):
        _plot_band(ax, fit.t_grid, fit.functionals.f, label=f"arm {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.set_title("densities")
    ax.legend()

    ax = axes[1, 1]
    for label, fit in (("A", reference_a), ("B", fit_b)):
        smoothed = smooth_hazard(fit.functionals.h)
        _plot_band(ax, fit.t_grid, smoothed, label=f"arm {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.set_title("hazards (smoothed)")
    ax.legend()

    fig.tight_layout()
    return _save(fig, path)


def figure_5(fit_a: Fit, fit_b: Fit, *, path: Path, p: float = 0.5) -> Path:
    """Figure 5: hazard ratio, CDF difference and ``Pr(eta_A > eta_B)``."""
    plt = _pyplot()
    comparison = compare(fit_a.functionals, fit_b.functionals, fit_a.t_grid, p=p)
    t_grid = comparison["t_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    _plot_band(axes[0], t_grid, comparison["ratio"], label="h_B / h_A")
    axes[0].set_xlabel("t")
    axes[0].set_title("hazard ratio")

    _plot_band(axes[1], t_grid, comparison["difference"], label="F_B - F_A")
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1].set_xlabel("t")
    axes[1].set_title("difference in CDF")

    probability = float(comparison["prob_eta_a_gt_eta_b"])
    # A per-draw median beyond the evaluation grid is censored to +inf by
    # ``compare``; the histogram needs the finite values only (the censored
    # pairs are counted in ``prob_eta_a_gt_eta_b`` and in ``meta``).
    eta_difference = np.asarray(comparison["eta_difference"], dtype=np.float64)
    axes[2].hist(
        eta_difference[np.isfinite(eta_difference)],
        bins=30,
        density=True,
        histtype="step",
    )
    axes[2].axvline(0.0, color="0.4", linewidth=0.8)
    axes[2].set_xlabel(f"eta_A - eta_B (p = {p:g})")
    axes[2].set_title(f"Pr(eta_A > eta_B | D) = {probability:.3f}")

    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Example drivers
# ---------------------------------------------------------------------------
@dataclass
class DemoResult:
    """Outcome of one example: the fits, the written figures and provenance.

    Attributes
    ----------
    label : str
        ``"5.1"`` or ``"5.2"``.
    fits : dict of str to Fit
        Keyed by ``"nu ~ Gamma(shape, rate)"`` for Section 5.1 and by
        ``"A: nu ~ Gamma(...)"``/``"B: ..."`` for Section 5.2.
    figures : tuple of Path
        The PNGs written by the run.
    meta : dict
        Preset name, seeds, chain count and dataset seeds, for the results
        note of plan Section 6.3.
    """

    label: str
    fits: dict[str, Fit]
    figures: tuple[Path, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def example_5_1(
    *,
    preset: str | Preset = "quick",
    prior_indices: Iterable[int] | None = None,
    seed: int | None = SEED_5_1,
    data_seed: int = DATA_SEED_5_1,
    out_dir: Path | str = DEFAULT_FIGURE_DIR,
    max_workers: int | None = None,
) -> DemoResult:
    """Run Section 5.1 end to end and write Figures 1-3.

    Fits the simulated lognormal sample under each selected ``nu`` prior and
    writes ``figure_1.png``, ``figure_2.png`` and ``figure_3.png`` under
    ``out_dir``.  Figure 2 compares the posterior of the first selected prior
    with its prior band -- the elicitation-scale caveat of plan decision 2.
    """
    resolved = resolve_preset(preset)
    indices = _as_prior_indices(prior_indices)
    data = data_5_1(seed=int(data_seed))
    grid = time_grid(data.t_max, resolved.functionals.n_t_grid)

    fits: dict[str, Fit] = {}
    for index in indices:
        prior = make_prior(index)
        fits[prior_label(index)] = fit_dataset(
            data,
            prior,
            preset=resolved,
            seed=seed,
            t_grid=grid,
            max_workers=max_workers,
        )

    reference = next(iter(fits.values()))
    band = prior_band(reference, preset=resolved, seed=seed)
    reference.prior_band = band

    out = Path(out_dir)
    figures = (
        figure_1(fits, path=out / "figure_1.png"),
        figure_2(reference, band, path=out / "figure_2.png"),
        figure_3(fits, path=out / "figure_3.png"),
    )
    return DemoResult(
        label="5.1",
        fits=fits,
        figures=figures,
        meta={
            "preset": resolved.name,
            "seed": seed,
            "data_seed": int(data_seed),
            "prior_indices": indices,
            "n_chains": resolved.sampler.n_chains,
            "burn_in": resolved.sampler.burn_in,
            "n_save": resolved.sampler.n_save,
            "thin": resolved.sampler.thin,
            "t_max": data.t_max,
        },
    )


def example_5_2(
    *,
    preset: str | Preset = "quick",
    prior_indices: Iterable[int] | None = None,
    seed: int | None = SEED_5_2,
    out_dir: Path | str = DEFAULT_FIGURE_DIR,
    max_workers: int | None = None,
) -> DemoResult:
    """Run Section 5.2 end to end and write Figures 4 and 5.

    Arm A is fitted under each selected prior; arm B under ``Gamma(2, 0.9)``
    only, as in the paper (plan Section 6.2).  The reported A-versus-B
    comparison of Figure 5 always uses ``Gamma(2, 0.9)`` for both arms, so that
    prior is fitted even when ``prior_indices`` omits it.  Both arms are
    transformed with the larger of their two maxima: ``compare`` refuses two
    fits that record different ``t_max`` values, and the paper's comparison
    lives on one common week scale.  Figures 4 and 5 are written under
    ``out_dir``.
    """
    resolved = resolve_preset(preset)
    indices = _as_prior_indices(prior_indices)
    if 0 not in indices:
        indices = (0, *indices)
    raw = leukemia_5_2()
    shared_t_max = max(float(np.max(raw[arm][0])) for arm in ("A", "B"))
    data_a = data_5_2("A", t_max=shared_t_max)
    data_b = data_5_2("B", t_max=shared_t_max)
    grid = time_grid(shared_t_max, resolved.functionals.n_t_grid)

    fits: dict[str, Fit] = {}
    for index in indices:
        fit = fit_dataset(
            data_a,
            make_prior(index),
            preset=resolved,
            seed=seed,
            t_grid=grid,
            max_workers=max_workers,
        )
        fits[f"A: {prior_label(index)}"] = fit

    fit_b = fit_dataset(
        data_b,
        make_prior(0),
        preset=resolved,
        seed=seed,
        t_grid=grid,
        max_workers=max_workers,
    )
    fits[f"B: {prior_label(0)}"] = fit_b

    fits_a = {key[3:]: fit for key, fit in fits.items() if key.startswith("A: ")}
    reference_a = fits[f"A: {prior_label(0)}"]

    out = Path(out_dir)
    figures = (
        figure_4(fits_a, fit_b, path=out / "figure_4.png"),
        figure_5(reference_a, fit_b, path=out / "figure_5.png"),
    )
    return DemoResult(
        label="5.2",
        fits=fits,
        figures=figures,
        meta={
            "preset": resolved.name,
            "seed": seed,
            "prior_indices": indices,
            "n_chains": resolved.sampler.n_chains,
            "burn_in": resolved.sampler.burn_in,
            "n_save": resolved.sampler.n_save,
            "thin": resolved.sampler.thin,
            "t_max": shared_t_max,
            "t_max_a": data_a.t_max,
            "t_max_b": data_b.t_max,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
EXAMPLE_CHOICES = ("5.1", "5.2")
PRIOR_CHOICES = ("1", "2", "3", "all")


def build_parser() -> argparse.ArgumentParser:
    """Build the driver's argument parser.

    ``--example`` and ``--preset`` use argparse ``choices`` so that a typo
    exits non-zero with a usage message naming the valid values, rather than
    starting a long run against an unintended configuration.
    """
    parser = argparse.ArgumentParser(
        prog="paper_examples",
        description=(
            "Run a Kottas (2002) Section 5 example and write its figures. "
            "Figures go to the git-ignored docs/research/figures/ directory; "
            "the numbers belong in docs/research/dp_gibbs_results.md."
        ),
    )
    parser.add_argument(
        "--example",
        required=True,
        choices=EXAMPLE_CHOICES,
        help="paper section to reproduce: 5.1 or 5.2",
    )
    parser.add_argument(
        "--preset",
        default="quick",
        choices=tuple(PRESETS),
        help="run length preset (default: quick)",
    )
    parser.add_argument(
        "--prior",
        default="all",
        choices=PRIOR_CHOICES,
        help=(
            "nu prior to use: 1, 2, 3 or all "
            "(indices into paper_data.NU_PRIORS; default: all)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="chain-group base seed (default: the preset's documented seed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"figure directory (default: {DEFAULT_FIGURE_DIR})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="chain pool size (default: min(n_chains, cpu count))",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns 0 on success; argparse exits 2 on bad input."""
    parser = build_parser()
    args = parser.parse_args(argv)

    prior_indices: Iterable[int] | None = None
    if args.prior != "all":
        prior_indices = (int(args.prior) - 1,)
    seed = args.seed
    if args.example == "5.1":
        result = example_5_1(
            preset=args.preset,
            prior_indices=prior_indices,
            seed=SEED_5_1 if seed is None else seed,
            out_dir=args.out,
            max_workers=args.max_workers,
        )
    else:
        result = example_5_2(
            preset=args.preset,
            prior_indices=prior_indices,
            seed=SEED_5_2 if seed is None else seed,
            out_dir=args.out,
            max_workers=args.max_workers,
        )

    print(f"example {result.label}: {len(result.fits)} fit(s)")
    for path in result.figures:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
