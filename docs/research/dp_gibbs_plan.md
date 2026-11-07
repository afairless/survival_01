# Plan: Gibbs Sampler and Functional Inference for the Weibull DP Mixture

Implementation design for the MCMC algorithm in Kottas (2002), *Nonparametric
Bayesian Survival Analysis using Mixtures of Weibull Distributions*, §4.2 and
the Appendix (pp. 22–25), plus everything needed to reproduce §5.1 and §5.2.

This document is a design and test plan. It contains no implementation. It is
written to be consumed by `write-todo-from-plan` → `TODO.md` →
`implement-from-plan`.

## Why this note exists

`src/dp_weibull.py` builds realisations of $G$ and documents itself as
downstream of a Gibbs sampler that does not exist yet:

> **No Gibbs sampler here.** `sample_dp_posterior` is downstream of one.
> — `docs/research/dp_weibull_code_guide.md` §7

`src/kottas_2002_5_1.py:105` therefore fails on an undefined `theta`. The
decision already taken (see §0) is *not* to reach for PyMC, JAGS, Stan or R:
the paper's sampler is a collapsed Po'lya-urn Gibbs sampler with a
non-conjugate base measure, and no existing package implements it. It is
instead to write it, to a standard that lets us claim the implementation is
correct rather than merely plausible.

The hard part of this plan is not the sampler — it is the test suite. A DP
mixture sampler can be wrong in ways that still produce smooth curves that look
like the paper's figures. Section 7 is therefore the core of this document.

## 0. Decisions already taken

| Decision | Choice |
| --- | --- |
| Censoring | Uncensored **and** right-censored (Appendix Eqs. A.2/A.4 and A.3/A.4). Interval/left censoring is out of scope (paper §6 calls it future work). |
| Module layout | `src/dp_gibbs.py` (sampler) + `src/dp_functionals.py` (§3 functional inference). `src/dp_weibull.py` keeps the $G_0$ and DP-realisation primitives. |
| Correctness oracle | Analytic invariants + exact posterior by enumeration on tiny problems + an independently derived reference sampler + simulation-based calibration (SBC). No new runtime dependencies. |
| Paper reproduction | §5.1 (simulated, $n=200$) and §5.2 (leukemia, $2\times20$, censored). §5.3 (liver, $n=622$) is out of scope: the dataset is not in the paper. |
| New dependencies | `pytest`, `ruff`, `pyright` (dev feature only). Nothing at runtime. |

## 1. Scope and non-goals

In scope:

- model (5) of the paper: Weibull kernel $K_W$, $G\sim\mathrm{DP}(\nu G_0)$,
  $G_0(\alpha,\lambda\mid\phi,\gamma)=\mathrm{Uniform}(\alpha\mid0,\phi)\,
  \mathrm{IGamma}(\lambda\mid d,\gamma)$, $d=2$, with $\nu$,
  $\gamma$, $\phi$ random and their priors as in (5);
- the Appendix Gibbs sampler for uncensored and right-censored data;
- functional inference (§3) for the survival function, density, cumulative
  hazard, hazard, quantiles, and differences/ratios of these between two
  populations;
- prior functional inference (§3, last paragraph) for the prior bands in
  Figure 2;
- prior elicitation (§4.3);
- reproduction drivers and figure recipes for §5.1 and §5.2;
- the test suite of §7.

Out of scope (state this in the module docstrings, so the next reader is not
misled):

- covariates / regression (the paper's §6 future work);
- interval, left-censored, or truncated data;
- a *marginal* (non-collapsed) sampler over $G$ (Neal 2000 Algorithms 1–8) as
  an alternative fitting route — the paper's design is the collapsed urn, and
  §3 requires $\theta_{1:n}$ **and** the ability to draw $G\mid\theta$;
- posterior summaries other than means, percentiles and quantile-functionals
  (no Bayes factors, no model comparison);
- any use of `scipy.stats` distributions as *samplers for the whole model*.

Non-goal worth writing down explicitly: **do not re-derive the paper's
formulas in the implementation.** Copy the Appendix equations into the design
table (§2.2), keep the same symbols, and let the tests in §7 check them.
Independent re-derivation belongs in the reference sampler (§7.3), never in the
production path.

## 2. The target algorithm, exactly as the Appendix specifies

### 2.1 State and notation

Data: $n=n_o+n_c$ observations, uncensored times $t_{i_o}$, right-censored
times $z_{i_c}$. Represent them as two arrays of equal length,

```text
time  : float64[n]     t_i  (uncensored) or z_i (censored), all > 1 after §2.3
event : bool[n]        True = uncensored (density), False = right-censored (survival)
```

Latent state of the chain:

| symbol | shape | meaning |
| --- | --- | --- |
| $\alpha_{1:n}$ | `float64[n]` | per-observation Weibull shape |
| $\lambda_{1:n}$ | `float64[n]` | per-observation Weibull scale (paper's $\lambda$) |
| $s_{1:n}$ | `int[n]` | cluster label of observation $i$, $s_i\in\{0,\dots,n^*-1\}$ |
| $\alpha^*_{1:n^*}$ | `float64[n*]` | cluster shape locations |
| $\lambda^*_{1:n^*}$ | `float64[n^*]` | cluster scale locations |
| $n_j$ | `int[n*]` | cluster sizes, $\sum_j n_j = n$ |
| $\nu,\gamma,\phi$ | scalars | DP precision and $G_0$ hyperparameters |
| $u$ | scalar | Escobar–West auxiliary ratio (not carried across sweeps, but see §3.4) |

Invariants maintained at all times (assert these in a debug mode and in tests):

1. `alphas == alphas_star[s]`, `lambdas == lambdas_star[s]`;
2. `n_j == bincount(s)`, no empty clusters, labels are `0..n*-1`;
3. $0 < \alpha_i \le \phi$, $0 < \alpha^*_j \le \phi$;
4. $\lambda > 0$, $\nu > 0$, $\gamma > 0$, $\phi \ge b_\phi$.

### 2.2 Full conditionals

With superscript "$-$" meaning "with observation $i$ removed", and
$q_j^{i} = k_W(t_i\mid\alpha^*_j,\lambda^*_j)$ for uncensored or
$1-K_W(z_i\mid\alpha^*_j,\lambda^*_j)$ for censored.

**(a) $[(\alpha_i,\lambda_i,s_i)\mid\text{rest}]$, for each $i$ in turn** — the
Po'lya urn mixture

$$
q_0^{i}\,h_i(\alpha_i,\lambda_i\mid\gamma,\phi,t_i)
\;+\;\sum_{j=1}^{n^{*-}} n_j^{-} q_j^{i}\,
\delta_{(\alpha^*_j,\lambda^*_j)},
\qquad\text{normalised by } q_0^{i}+\sum_j n_j^{-}q_j^{i}.
$$

The prior-predictive mass (uncensored, Eq. A.2):

$$
q_0^{o}(t) = \frac{d\,\nu\,\gamma^{d}}{\phi}
\int_0^{\phi}\frac{\alpha\,t^{\alpha-1}}{(\gamma+t^{\alpha})^{d+1}}\,d\alpha
= \frac{d\,\nu\,\gamma^{d}}{\phi\,t}
\int_0^{\phi}\frac{\alpha\,t^{\alpha}}{(\gamma+t^{\alpha})^{d+1}}\,d\alpha .
$$

The prior-predictive mass (right-censored, Eq. A.3):

$$
q_0^{c}(z) = \frac{\nu\,\gamma^{d}}{\phi}\int_0^{\phi}
\frac{1}{(\gamma+z^{\alpha})^{d}}\,d\alpha .
$$

The continuous piece factors (both cases):

| | uncensored | right-censored |
| --- | --- | --- |
| $[\alpha_i\mid\cdot]$ | $\propto \alpha\,t^{\alpha}\,\mathbf 1(0<\alpha<\phi)/(\gamma+t^{\alpha})^{d+1}$ | $\propto \mathbf 1(0<\alpha<\phi)/(\gamma+z^{\alpha})^{d}$ |
| $[\lambda_i\mid\alpha_i,\cdot]$ | $\mathrm{IGamma}(d+1,\ \gamma+t^{\alpha})$ | $\mathrm{IGamma}(d,\ \gamma+z^{\alpha})$ |

Note the parenthetical in the paper: the uncensored $[\alpha_i\mid\cdot]$
density *is* the $q_0^o$ integrand. One grid of $\alpha$ values serves both.
This is the single most important implementation fact in the algorithm.

**(b) $[(\alpha^*_j,\lambda^*_j)\mid s,n^*,\gamma,\phi,D]$, for each $j$** — the
Bush–MacEachern location refresh, with $n_j^{o}=|\{i_o:s_{i_o}=j\}|$:

$$
\lambda^*_j \mid \alpha^*_j \sim
\mathrm{IGamma}\!\left(d+n_j^{o},\ \gamma+\!\!\sum_{i_o\in j}\!t_{i_o}^{\alpha^*_j}
+\!\!\sum_{i_c\in j}\!z_{i_c}^{\alpha^*_j}\right),
$$

and $\alpha^*_j$ by the Damien–Wakefield–Walker auxiliary-variable device
(Eqs. A.5–A.6). Per sweep, per cluster: draw

$$
v \sim \mathrm{U}\!\left(0,\ C(\alpha^*_j,D)\right),\qquad
C(\alpha,D)=\prod_{i_o\in j} t_{i_o}^{\alpha},
$$

and one auxiliary per observation in the cluster,

$$
w_{i_o}\sim\mathrm U\!\left(0,\ \exp(-\lambda^{*-1}_j t_{i_o}^{\alpha^*_j})\right),\qquad
w_{i_c}\sim\mathrm U\!\left(0,\ \exp(-\lambda^{*-1}_j z_{i_c}^{\alpha^*_j})\right),
$$

evaluated at the **current** $\alpha^*_j$. Let

$$
\alpha^-_j=\max\left(0,\ \frac{\log v}{\sum_{i_o\in j}\log t_{i_o}}\right),
\qquad
\alpha^+_j=\min\left(\phi,\
\min_{i_o\in j}\frac{\log(-\lambda^*_j\log w_{i_o})}{\log t_{i_o}},\
\min_{i_c\in j}\frac{\log(-\lambda^*_j\log w_{i_c})}{\log z_{i_c}}\right).
$$

Then draw $\alpha^*_j$ on $(\alpha^-_j,\alpha^+_j)$ from a density
**proportional to $(\alpha^*_j)^{n_j^o}$**, not uniformly:

$$
p(\alpha^*_j\mid v,w)=\frac{k\,(\alpha^*_j)^{k-1}}{(\alpha^+_j)^k-(\alpha^-_j)^k},
\qquad k=n_j^o+1,
$$

by inverse CDF, i.e.
$\alpha^*_j=\big[(\alpha^-_j)^k+u\,((\alpha^+_j)^k-(\alpha^-_j)^k)\big]^{1/k}$
with $u\sim\mathrm U(0,1)$. This is the paper's Eq. A.6 statement
"the full conditional for $\alpha^*_j$ is proportional to
$\alpha_j^{*n_j^o}\mathbf 1(\alpha^-_j<\alpha^*_j<\alpha^+_j)$ … drawing from this
distribution is straightforward by using the inverse c.d.f. method". The draw
is **uniform only for an all-censored cluster** ($n_j^o=0$, $k=1$); for
$n_j^o=1$ it is already triangular. An earlier draft of this plan wrote
"$\sim\mathrm U(\alpha^-_j,\alpha^+_j)$" here — that drops the
$\alpha^{*n_j^o}$ factor and biases every uncensored cluster's shape upward
($+8\%$ in $E[\alpha^*]$ at $n_j^o=1$, $+29\%$ at $n_j^o=5$ on $(0.5,1.5)$).
Do not implement the uniform version.

This transition is exact (acceptance probability 1) but it is **not** an
ordinary slice sampler: the auxiliaries are redrawn every sweep from ranges
determined by the current value, then the target is sampled on the resulting
interval by inverse CDF. Implement it literally as written; do not substitute
a different slice sampler and do not "improve" it.

Degenerate cases that must be handled explicitly:

- $n_j^{o}=0$ (cluster of censored observations only): $C=1$, so $v$ is
  $\mathrm U(0,1)$ and there is no constraint from $v$; set $\alpha^-_j=0$
  (the formula divides by an empty sum). Here, and only here, the target is
  uniform ($k=1$).
- $n_j^{c}=0$: the second $\min$ term is absent.
- $n_j=1$: both minima are single terms.
- $n_j^{o}\ge1$ but $n_j^{c}=0$: $k=n_j^o+1$, no censored bound to include;
  this is the common case on uncensored data and the case most likely to be
  coded with the wrong exponent.
- Cluster with $n_j^{o}>0$ but all $t_{i_o}$ equal: fine, only $\log t>0$
  matters.

**(c) Hyperparameters** — Eq. (c) of the Appendix:

| parameter | full conditional |
| --- | --- |
| $\nu$ | Escobar–West: $u\sim\mathrm{Beta}(\nu+1,n)$, then $\nu\sim p\,\mathrm{Gamma}(a_\nu+n^*,b_\nu-\log u)+(1-p)\,\mathrm{Gamma}(a_\nu+n^*-1,b_\nu-\log u)$, $p=\dfrac{a_\nu+n^*-1}{n(b_\nu-\log u)+a_\nu+n^*-1}$ |
| $\phi$ | $\mathrm{Pareto}(a_\phi+n^*,\ \max(b_\phi,\max_j\alpha^*_j))$ |
| $\gamma$ | $\mathrm{Gamma}\!\left(a_\gamma+d\,n^*,\ b_\gamma+\sum_{j=1}^{n^*}1/\lambda^*_j\right)$ |

Conventions (already established in `src/dp_weibull.py`, keep them):
$\mathrm{Gamma}(a,b)$ is shape–**rate**, so SciPy wants `scale=1/b`;
$\mathrm{Pareto}(a,b)$ has density $a b^{a}/\phi^{a+1}$ on $\phi\ge b$, i.e.
`scipy.stats.pareto(b=a, scale=b)`; $\mathrm{IGamma}(a,b)$ is SciPy
`invgamma(a, scale=b)`.

Sweep order is (a) for all $i$ in index order, then (b) for all $j$, then (c).
The order is part of the specification: it fixes the random-number stream and
therefore the golden regression test (§7.5).

### 2.3 The time transformation

The Appendix requires $t_i>1$ (otherwise $\log t_{i_o}$ can vanish and the
$\alpha^-$ formula divides by zero). Use the paper's own transform, applied
once per dataset:

$$
\tilde t_i = 1 + t_i / \max(t),\qquad t_i = T_{\max}\,(\tilde t_i-1),\quad
T_{\max}=\max_i t_i .
$$

Practical rules:

- Transform **before** the sampler sees the data; the sampler's
  `time` array is $\tilde t$.
- $\tilde t\in(1,2]$ for $t>0$, so `log(time) > 0` strictly. Raise
  `ValueError` if any input time is $\le 0$ (a zero time would give
  $\tilde t = 1$ and $\log \tilde t = 0$).
- Every functional maps back with the Jacobian:
  $S(s) = \tilde S(1+s/T_{\max})$,
  $f(s) = \tilde f(1+s/T_{\max})/T_{\max}$,
  $h(s) = \tilde h(1+s/T_{\max})/T_{\max}$,
  $\eta_p = T_{\max}(\tilde\eta_p-1)$.
  Centralise these four lines in one function and test them (§7.4). Getting
  the $1/T_{\max}$ on $f$ and $h$ but not on $S$ wrong is the classic error
  and it is invisible in a plot of $S$.

**Consequence for the prior — accepted and documented (§5, §10).** This map is
an affine shift plus a scale, not a pure scaling, so a Weibull prior elicited
on the *original* time scale (§4.3) is not the same prior after the transform.
Monte Carlo from the §5 prior with $T_{\max}=4.5518$ puts the prior median of
the population median at $\approx0.61$ (the elicitation guess is $1.1727$), and
$\approx44\%$ of prior draws imply a population median at or below $t=0$. This
is the paper's own convention — it elicits $b_\gamma,b_\phi$ in §4.3 and then
fits on $\tilde t$ — so it is kept, not "fixed". What is required is that it
be *stated*: the prior bands of §4.2 are the honest statement of what the prior
says on the original scale, and §6.3's Figure 2 criteria are read against those
bands, never against the elicitation guess. One prior-functional assertion on
the original scale is part of test 21b (§7.4).

### 2.4 The two numerical pieces

**(i) The $\alpha$ quadrature.** Choose a fixed grid in the *unit* interval and
scale it by the current $\phi$:

$$
\alpha_m = \phi\,u_m,\qquad u_m = m/M,\quad m=1,\dots,M,\quad M=256\ \text{(default)}.
$$

Cache, once per chain, `T = exp(outer(log(time), u))`
($T_{im} = t_i^{u_m}\le 2$), so the sweep only needs
$t^{\alpha_m} = T_{im}^{\phi}$ (one `pow` call on an $(n,M)$ array). This keeps
the grid exactly inside $(0,\phi)$ as $\phi$ moves, needs no $\phi$ cap, and
makes the quadrature domain a bounded interval — the computational reason the
paper chose a Uniform prior on $\alpha$ (`docs/research/g0_uniform_inverse_gamma.md` §3.1).

Integrate with the trapezoid rule in $\alpha$ on the *exponentiated* integrand
(weights $\phi/M$, endpoints halved); compute the integrand in log space and
use `logsumexp` with log weights so that a large $t^{\alpha}$ cannot underflow
the sum to zero. Resolution: $\Delta\alpha = \phi/M \approx 0.007$ at
$\phi\approx1.8$, well inside the $-\alpha\approx1.3$ decay scale of the
integrand.

**Accuracy, measured, not assumed.** The grid is $\alpha_m=\phi m/M$,
$m=1..M$, so with halved endpoints the rule is a trapezoid over
$[\phi/M,\phi]$: the dominant error is the omitted first cell, not the
trapezoid error, and it is $O(1/M^2)$. Measured relative error for
$\int_0^\phi \alpha t^{\alpha}/(\gamma+t^{\alpha})^{3}d\alpha$ with
$(\gamma,\phi,t)=(0.6,1.7752,1.5)$: $1.8\!\times\!10^{-3}$ ($M=32$),
$4.6\!\times\!10^{-4}$ (64), $1.1\!\times\!10^{-4}$ (128),
$2.9\!\times\!10^{-5}$ (256), $7.2\!\times\!10^{-6}$ (512); observed order
2.00. A row-relative error of $3\!\times\!10^{-5}$ in $q_0$ is irrelevant to the
urn weights, so $M=256$ stands. Adding $\alpha=0$ as an extra node would give
$4.0\!\times\!10^{-6}$ at $M=256$, but it puts a draw at $\alpha=0$ on the
support (an invalid Weibull shape) and needs masking; do not add it — keep the
plan's grid and assert the measured accuracy (test 3).

Documented alternative, only if profiling demands it: an absolute grid on
$(0,\phi_{\text{cap}}]$ with masking at $\phi$ lets `T` be reused across
sweeps, at the cost of a $\phi_{\text{cap}}$ knob and a "$\phi$ exceeded the
cap, rebuilding the grid" warning path. Do not implement both.

**(ii) Drawing $\alpha$ from the grid.** For each observation the normalised
$q_0$ integrand is the $[\alpha_i\mid\cdot]$ density. Draw by inverse CDF on
the grid: `cumsum` the row, draw $u\sim\mathrm U(0,1)$, `searchsorted`. Vectorise
across all observations whose urn draw selected "new cluster" in one call.

**Known bias, stated once:** drawing $\alpha$ on the grid makes
$\alpha$ discrete with spacing $\Delta\alpha$, so this is an approximation of
order $\Delta\alpha^2$ in the conditional CDF — not of the posterior, which is a
mixture over many such draws. The paper does exactly this. It is measured
(M-sensitivity) and documented, not fixed, because a continuous draw would
require a second independent numerical method (rejection sampling) for no
measurable gain.

## 3. Module design

### 3.1 Files and responsibilities

| file | responsibility | does *not* do |
| --- | --- | --- |
| `src/dp_gibbs.py` | `SurvData` ingestion validation + time transformation, state container, `AlphaGrid`, the three full-conditional steps, one sweep, chain driver, initialisation, diagnostics | plotting, data generation, functionals |
| `src/dp_functionals.py` | §3 steps (i)–(iii) for posterior and prior; survival/density/hazard/quantile functionals; two-population comparisons | sampling $\theta$ |
| `src/paper_data.py` | §5.1 lognormal mixture generator, §5.2 leukemia data, prior-guess constants | MCMC, transformation, figures |
| `src/paper_examples.py` | figure-level recipes + CLI (`--example {5.1,5.2}`, `--preset {quick, paper}`, `--prior`, `--seed`) | statistics |
| `src/dp_weibull.py` | unchanged except the two additions in §3.8 | sampler, transform, data |
| `tests/…` | §7 | — |

`src/kottas_2002_5_1.py` is currently broken scaffolding. Decide in the
implementation step: reduce it to a thin wrapper over
`src/paper_examples.py::example_5_1` (preferred, keeps the history readable) or
delete it. Do not leave it half-working with an undefined `theta`.

### 3.2 Data containers

Three small dataclasses, no inheritance, no methods that sample:

```python
@dataclass(frozen=True)
class SurvData:              # what the sampler consumes, already transformed
    time: np.ndarray         # float64[n], > 1
    event: np.ndarray        # bool[n]
    t_max: float             # max of the ORIGINAL times, for the inverse map

@dataclass(frozen=True)
class DPPrior:               # model (5) hyperparameters, all fixed
    a_nu: float; b_nu: float
    a_gamma: float; b_gamma: float
    a_phi: float; b_phi: float
    d: int = 2

@dataclass
class DPSettings:            # sampler knobs, never model parameters
    burn_in: int = 5_000
    n_save: int = 10_000
    thin: int = 150
    n_alpha: int = 256       # M
    epsilon: float = 1e-4    # J rule
    n_chains: int = 4        # chain b uses seed + b
    seed: int | None = None
    preset: str = "custom"   # "quick" | "paper"; recorded in the results note

@dataclass
class FunctionalSettings:    # consumed by dp_functionals, not by the sampler
    method: str = "exact"    # or "mc"
    L: int = 2_500           # only for method="mc"
    atom_tol: float = 1e-6   # DPRealization.pruned
    n_t_grid: int = 300
```

Plus the chain state (mutable, one instance per chain):

```python
@dataclass
class DPState:
    alpha, lam, s, alpha_star, lam_star, n_j, nu, gamma, phi
```

and the stored output:

```python
@dataclass
class DPChains:              # what dp_functionals consumes
    data: SurvData
    prior: DPPrior
    settings: DPSettings
    theta: np.ndarray        # (B, n, 2)  the paper's theta_b
    nu, gamma, phi: np.ndarray   # (B,)
    n_star: np.ndarray       # (B,)
    log_post: np.ndarray     # (B,), for diagnostics and golden tests
```

Storing $\theta$ as `(B, n, 2)` costs $10^4\cdot200\cdot2\cdot8 = 32$ MB — fine,
and it means `sample_dp_posterior(theta=chain.theta[b])` works with no
reshaping (that function already validates `(n,2)`).

### 3.3 Public API

```python
# src/dp_gibbs.py
@dataclass(frozen=True)
class AlphaGrid:                 # per chain: T does not depend on phi
    u: np.ndarray                # (M,)   u_m = (m+1)/M, m = 0..M-1
    T: np.ndarray                # (n,M)  T_im = time_i ** u_m  (<= 2)
    M: int
def make_grid(data: SurvData, n_alpha: int) -> AlphaGrid
def init_state(data, prior, *, grid, seed) -> DPState
def sweep(state, data, prior, *, grid, rng) -> None              # in place
def sample_step_a(state, data, prior, *, grid, rng) -> None
def sample_step_a_for_obs(state, data, prior, *, i, grid, rng) -> None   # test entry point
def sample_step_b(state, data, prior, *, rng) -> None
def refresh_cluster(state, data, prior, *, j, rng) -> None      # test entry point
def sample_step_c(state, data, prior, *, rng) -> None
def draw_g0_conditional_alpha(time, event, *, gamma, phi, grid, rng) -> tuple[float, float]
def sample_chain(data, prior, settings) -> DPChains
def sample_chains(data, prior, settings_list) -> list[DPChains]
def r_hat(chains, name: str) -> dict          # split R-hat; name in {"nu","n_star","S"}
def ess(chains, name: str) -> dict            # bulk ESS, same `name` vocabulary
def autocorr_time(x: np.ndarray) -> float
```

```python
# src/dp_functionals.py
@dataclass
class Functionals:               # what survival()/density()/hazard() consume
    t_grid: np.ndarray           # (M,)   original time scale
    S, F, f, h: np.ndarray       # (B, M) per-draw functionals, original scale
    meta: dict                   # method, J_common, atom_tol, n_draws, seeds
def posterior_functionals(chains, t_grid, *, settings: FunctionalSettings,
                          seed=None) -> Functionals
def prior_functionals(prior, t_grid, n_draws, *, settings: FunctionalSettings,
                      seed=None) -> Functionals
def posterior_quantiles(chains, t_grid, p, *, settings: FunctionalSettings,
                        seed=None) -> np.ndarray      # (B,)
def survival(fit: Functionals, t_grid=None) -> dict     # {"mean","lo","hi"}
def density(fit: Functionals, t_grid=None) -> dict
def hazard(fit: Functionals, t_grid=None) -> dict
def compare(fit_a: Functionals, fit_b: Functionals, t_grid) -> dict
                                        # ratio, difference, P(eta_a > eta_b)
```

Every public function takes `rng`/`seed` explicitly. No global RNG state. No
`plt` import outside `paper_examples.py`.

### 3.4 One sweep, step by step

Performed by `sweep`, in this fixed order (it *is* the specification):

1. **Precompute** for this sweep: `Tau = T**phi` ($t_i^{\alpha_m}$),
   the $(n,M)$ integrand in log space, the $(n,)$ `q0`
   (uncensored rows use A.2, censored rows A.3), and the $(n,n^*)$ matrix
   $\log q_j^{i}$. All three are vectors/matrices; no Python loop over the grid.
2. **Step (a)**, sequential over $i=0..n-1$:
   - decrement $n_{s_i}$, drop the cluster if it empties (compact labels),
   - form $\log$ weights $\big[\log q_0^{i},\ \log n_j^{-}+\log q_j^{i}\big]$,
   - sample the category by inverse CDF over `logsumexp`-normalised weights,
   - *existing cluster*: copy $\alpha^*_j,\lambda^*_j$ into $\alpha_i,\lambda_i$;
     increment $n_j$,
   - *new cluster*: draw $\alpha_i$ from the grid row for $i$, then
     $\lambda_i\mid\alpha_i$ from the invgamma in §2.2; append a cluster,
   - assert invariants 1–2 every $n/10$ observations in debug mode.
3. **Step (b)**, over $j=0..n^*-1$: draw $\lambda^*_j\mid\alpha^*_j$ from the
   invgamma, then refresh $\alpha^*_j$ by the auxiliary-variable scheme — draw
   $v$ and the $w$'s at the **current** $\alpha^*_j$ and the **new**
   $\lambda^*_j$, form the bounds, then draw
   $\alpha^*_j\propto(\alpha^*_j)^{n_j^o}$ on $(\alpha^-_j,\alpha^+_j)$ (see
   §2.2(b): uniform only when $n_j^o=0$).
   Afterwards set $\alpha_{i}=\alpha^*_{s_i}$, $\lambda_i=\lambda^*_{s_i}$.
4. **Step (c)**: $\nu$ (Escobar–West), $\phi$, $\gamma$ — in that order, because
   the $\phi$ update needs $n^*$ and $\max_j\alpha^*_j$ from step (b).
5. Recompute `log_post` for the stored draw.

Bookkeeping hazards to design against (each gets a test):

- the "remove $i$" operation must be applied *before* forming the weights, and
  the dropped cluster must disappear from the candidate set (otherwise an
  observation can join its own singleton cluster with weight $0\cdot q_j$);
- cluster compaction must be applied after every removal *and* after every
  append, or `n_j` and `s` desynchronise — **and** the columns of the
  precomputed $(n,n^*)$ `log q` matrix must be permuted by the same
  old-label $\to$ new-label map at the same moment. Alternative, also
  acceptable and simpler to test: defer compaction to one relabel pass at the
  end of step (a) and index candidates by the pre-compaction labels meanwhile.
  Pick one; test 6 checks the matrix and `s`/`n_j` agree after mixed
  remove/append sequences;
- step (a)'s "new cluster" branch consumes two uniforms (grid draw + invgamma),
  the "existing" branch consumes one. The RNG consumption differs by branch —
  acceptable, but it means the golden test in §7.5 must use the same code path.

### 3.5 Numerical representation

- Everything in log space until a *sum* is needed; `scipy.special.logsumexp`
  with weights for quadrature sums, `np.cumsum` + `searchsorted` for sampling.
- `1 - exp(-x)` via `-np.expm1(-x)`; censored likelihoods must not cancel.
- `pow` on cached `T` (already in $(0,2]$), never `exp(alpha*log t)` computed
  twice.
- The $\alpha^*_j$ draw is **not** uniform in general: it is
  $\propto(\alpha^*_j)^{n_j^o}$ on $(\alpha^-_j,\alpha^+_j)$, i.e. inverse CDF
  $\alpha=\big[(\alpha^-)^{k}+u\,((\alpha^+)^{k}-(\alpha^-)^{k})\big]^{1/k}$
  with $k=n_j^o+1$, and uniform only for $n_j^o=0$. A silent uniform draw
  biases every uncensored cluster upward (8% at $n_j^o=1$, 29% at $n_j^o=5$ on
  $(0.5,1.5)$).
- Guard the $\alpha^*$ bounds: if $\alpha^+_j\le\alpha^-_j$ (can only happen
  through floating-point equality at a boundary), clamp to the midpoint and
  count the event in a diagnostic counter. A silent `uniform(a, a)` would
  freeze a cluster's shape.
- $\phi$ update: $\max_j\alpha^*_j\le\phi$ is an invariant, so the Pareto's
  lower bound never exceeds the current $\phi$.

### 3.6 Initialisation and multi-chain policy

The paper does not specify initial values; document ours:

- $s_i=i$, $n_j=1$ for all $i$ (each observation its own cluster);
  $\theta_i$ drawn from the continuous piece $h_i$ at
  $\gamma=a_\gamma/b_\gamma$, $\phi=\max(b_\phi,\cdot)$ — i.e. reuse
  `draw_g0_conditional_alpha`;
- $\nu=a_\nu/b_\nu$ (prior mean), $\gamma=a_\gamma/b_\gamma$,
  $\phi=b_\phi$;
- four chains with seeds `seed + 0..3` for anything reported; the paper
  assesses convergence "through multiple chains" (§5).

Initialisation must not be a special case in `sweep`; it only sets the state.

### 3.7 Storage, thinning, diagnostics

- `thin` sweeps between stored draws; `burn_in` discarded first.
  Total sweeps = `burn_in + thin*n_save`.
- Diagnostics implemented in `dp_gibbs.py` with no new dependency: split
  $\hat R$ (Gelman, on 4 chains, cut in half), bulk effective sample size
  (autocorrelation-based), and lag-$k$ autocorrelation of $\nu$, $n^*$, and
  $\tilde S(t)$ at five grid points. Acceptance: $\hat R<1.05$ on all five,
  lag-1 autocorrelation at the stored draws $<0.1$.
- `log_post` is the **unnormalised joint of the collapsed target**, which is
  available in closed form and must be stored exactly as written:
  $$\log p(\theta^*,s,\nu,\gamma,\phi\mid D)=\sum_{j=1}^{n^*}\log g_0(\alpha^*_j,\lambda^*_j\mid\phi,\gamma)
  +\log p(s\mid\nu)+\sum_{i=1}^{n}\log L_i(\theta_{s_i})
  +\log p(\nu)+\log p(\gamma)+\log p(\phi),$$
  with $L_i$ the density or the survival of observation $i$ and the exchangeable
  partition term
  $\log p(s\mid\nu)=n^*\log\nu+\sum_j\log\Gamma(n_j)-\log\Gamma(\nu)+\log\Gamma(\nu+n)$
  (the EPPF, i.e. $\nu^{n^*}\prod_j(n_j-1)!/(\nu)_{(n)}$). Note the
  $\log g_0$ terms are **per cluster**, not per observation. This is a genuine
  log density up to an additive constant, so it can be monitored on a trace and
  used as a scale in the golden test. It is still not normalised: no model
  comparison.

### 3.8 Small changes to `src/dp_weibull.py`

1. `truncation_number_for_posterior(max_nu, n, epsilon=1e-4)` — the paper's
   common-$J$ rule
   $J=\lceil \log\epsilon/\log((n+\max_b\nu_b)/(n+1+\max_b\nu_b))\rceil$, which
   is `truncation_number(n + max_nu)` but deserves a named function and a test
   (the "one $J$ for all draws" rule is easy to get wrong, and per-draw $J$
   silently changes the mixture).
2. `DPRealization.pruned(atom_tol=1e-6)` — drop trailing stick-breaking atoms
   whose cumulative weight is below `atom_tol`, renormalise, and return a new
   `DPRealization`. Needed because $J$ is ~2000 for $n=200$ (§8), while the
   effective support of $G_b$ is ~5–12 atoms.
That is all — two additions. **Nothing else in that module moves**, including
`b_gamma_b_phi_from_weibull`: §5 verified that its $(\sqrt2-1)$ constant and its
$(4\tilde\alpha/3)$ are both correct, and an earlier draft's "fix" would have
broken them. The only change near it is the one-sentence docstring note in §5.

Nothing else in that module should move. `sample_dp_posterior` already accepts
`n_terms` and returns what §3 needs.

## 4. Functional inference (§3 of the paper)

### 4.1 Posterior functionals

For each stored draw $b=1..B$: (i) draw
$G_b\sim[G\mid\theta_b,\nu_b,\gamma_b,\phi_b]$ — call the existing
`sample_dp_posterior(chain.nu[b], chain.gamma[b], chain.phi[b], chain.theta[b],
n_terms=J_common)`; (ii) draw $\theta^0_{bl}\sim G_b$, $l=1..L$; (iii)
$H_b=\frac1L\sum_l H(K_W(\cdot\mid\theta^0_{bl}))$.

Design decisions:

- `J_common = truncation_number_for_posterior(chain.nu.max(), n, epsilon)`.
  Computed **once** for the run, not per draw (§2 of the paper, and §3.6 of the
  code guide: "Per-draw J vs common J").
- `method="exact"` (default) computes
  $H_b=\sum_j w_j H(K_W(\cdot\mid\vartheta_j))$ with the truncated weights —
  Rao-Blackwellised, zero Monte Carlo noise in $L$, and algebraically the
  $L\to\infty$ limit of the paper's estimator; it needs no $L$ draws at all.
  This is what the figures should use. `method="mc"` reproduces the paper
  literally ($L=2500$) and exists to be *compared* with `exact` in §7.4.
- Atom pruning (`pruned(atom_tol)`) applies in both methods before evaluating
  kernels. Justify in the docstring: with precision $\nu+n\approx 200$,
  $J\approx2000$ but $\Pr(w_j>10^{-6})\approx 0$ beyond ~10 atoms; dropping the
  tail changes $S(t)$ by at most `atom_tol`.
- Hazard uses the **same** atom set for numerator and denominator, so the ratio
  is a proper functional of one realisation. Never mix an $f$ from one $G_b$
  with an $S$ from another.
- Store the resulting arrays as `(B, len(t_grid))`, which is 10,000×300×8 =
  24 MB per functional. Compute means/percentiles from that, do not accumulate
  online (the quantile functional needs the full grid).

### 4.2 Prior functionals

Same (ii)–(iii) with $G_b$ drawn from `sample_dp_prior(nu_b, gamma_b, phi_b,
n_terms=J_prior)`, $\nu_b,\gamma_b,\phi_b\sim$ priors, and
$J_{\text{prior}}=\lceil\log\epsilon/\log(\max_b\nu_b/(1+\max_b\nu_b))\rceil$
(paper's §3 last paragraph). Needed for the prior bands in Figure 2 and for
`docs/dp_weibull_explanation.md`'s claim that the specification is
"noninformative". Sample the hyperparameters with
`sample_prior_hyperparameters`, which already exists.

### 4.3 Quantile functionals

The paper (§3) warns the quantile functional is not a function of linear
functionals. Implement by inverting the $M\times B$ matrix of
$F(t_m;G_b)$ draws with interpolation, per draw; then
$\eta_p^{(b)}=\text{interp}(p, F_b, t_{\text{grid}})$ and map back with §2.3.
Grid choice: the paper's figures use the observed support plus a margin;
use 300 points from 0 to $1.15\max(t)$ on the original scale (the paper's
Figure 3 range for §5.1: 0–5).

### 4.4 Two-population comparisons (§5.2)

Independent fits for treatments A and B; because the chains are independent,
pair $b$ with $b$ (valid: both are draws from the product posterior). Then:

- $\lambda_B(t)/\lambda_A(t)$ per draw per $t$ (Fig. 5a);
- $F_B(t)-F_A(t)$ per draw per $t$ (Fig. 5b);
- $\eta_A-\eta_B$ per draw, and $\Pr(\eta_A>\eta_B\mid D)=$ mean of the
  indicator (paper: 0.9135, Fig. 5c).

`compare(fit_a, fit_b, t_grid)` returns all three. Document that pairing is
valid only for independent chains of equal length.

## 5. Prior elicitation (§4.3) — verified, plus an elicitation caveat

The paper fixes $a_\gamma=1$, $a_\phi=2$, then chooses $b_\gamma,b_\phi$ so
that the medians of the induced marginals equal a Weibull guess
$(\tilde\alpha,\tilde\lambda)$ obtained from prior guesses for the population
median and IQR (`weibull_from_median_iqr`, already implemented and correct).

Induced marginals (marginalising $\gamma$ and $\phi$ out of the *parametric*
version):

$$
[\lambda]=\frac{2b_\gamma}{(1+\lambda b_\gamma)^{3}},\qquad
[\alpha]=\frac{2b_\phi^{2}}{3\max(\alpha,b_\phi)^{3}} .
$$

**The existing helper is correct — verified, and an earlier draft of this plan
was wrong about it.** The $\alpha$ marginal is a
$2/3\,\mathrm U(0,b_\phi)+1/3\,\mathrm{Pareto}(b_\phi,2)$ mixture with median
$3b_\phi/4$, so $b_\phi=4\tilde\alpha/3$; that is what the code does. For
$\lambda$, the density $[\lambda]=2b_\gamma/(1+\lambda b_\gamma)^3$ integrates to

$$
F(\lambda)=1-(1+\lambda b_\gamma)^{-2},\qquad
F(\lambda)=\tfrac12 \iff (1+\lambda b_\gamma)^2=2,
$$

so the median is $(\sqrt2-1)/b_\gamma$ and
$b_\gamma=(\sqrt2-1)/\tilde\lambda\approx0.41421/\tilde\lambda$ — exactly
`src/dp_weibull.py:141`. (An earlier draft wrote $F=1-(1+\lambda b)^{-3}$ and
concluded the code was 1.594× too large; that exponent is wrong and the
conclusion was the opposite of the truth. Do **not** apply it: changing the
constant would introduce the bug.)

Numerical check, root-finding on the marginal CDF built by `quad` from the
density:

| $\tilde\lambda$ | 0.5 | 1.7836 | 3.0 |
| --- | --- | --- | --- |
| `b_gamma_b_phi_from_weibull` | 0.828427 | 0.232235 | 0.138071 |
| numerical median of $[\lambda]$ at that $b_\gamma$ | 0.500000 | 1.783600 | 3.000000 |
| median of $[\alpha]$ at $b_\phi$ vs $3b_\phi/4$ | — | 1.331415 vs 1.331415 | — |

For the §5.1 guesses $(\text{median},\text{IQR})=(1.1727,1.3679)$:
$\tilde\alpha=1.3314$, $\tilde\lambda=1.7836$, $b_\gamma=0.23224$,
$b_\phi=1.77522$ — keep all four numbers, and keep
`docs/research/dp_weibull_code_guide.md` §3.4 as it is (the $(\sqrt2-1)$ median
it documents is right).

Required work (its own commit):

1. add test 7: numerical medians of *both* induced marginals equal
   $(\tilde\lambda,\tilde\alpha)$ after the round trip, for three values of each
   input. This test **passes against the current code** and exists to stop the
   constant from being "fixed" later;
2. add one sentence to the `b_gamma_b_phi_from_weibull` docstring: the median
   of $[\lambda]$ is $(\sqrt2-1)/b_\gamma$ because the CDF is
   $1-(1+\lambda b_\gamma)^{-2}$, not $\ldots^{-3}$ (this is a known
   trap, and it already caught one draft of this plan);
3. no constant changes anywhere; delete step 2 of §9 as originally written.

**Caveat — the prior is elicited on one scale and applied on another.** §4.3
elicits $(\tilde\alpha,\tilde\lambda)$ from a population median/IQR on the
*original* time scale, but the sampler fits $\tilde t=1+t/T_{\max}$ (§2.3), and
that map is a shift plus a scale. Measured consequences for the §5.1 prior
($T_{\max}=4.5518$): the prior median of the population median is $\approx0.61$
(intended $1.1727$), $\approx44\%$ of prior draws imply a population median at
or below $t=0$, and the prior mean of $S(1.1727)$ is $0.43$ with a
$2.5$–$97.5\%$ band of $(0.00,0.94)$. This is the paper's own convention and is
kept (decision recorded in §10); it must be stated in the module docstring and
in the results note, and it is why §6.3's Figure 2 criteria are read against
the *prior functionals on the original scale*, not against the elicitation
guess. Test 21b makes that comparison explicit.

Record the provenance in `src/paper_data.py` as module constants. The values
`1.1727, 1.3679` used in `_demo()` and `kottas_2002_5_1.py` are the **exact
population** median and IQR of the §5.1 mixture (computed here:
median $=1.1727$, IQR $=1.3679$, 95th percentile $=3.6828$). The paper says
"values roughly equal to the sample median and interquartile range".

**Decision (recorded in §10): use the population values**, as the existing
scaffolding does. They are reproducible without a sample, and the difference
to a sample-based choice is immaterial to every acceptance criterion in §6.
The realised sample statistics are recorded for reference only — for the §6.1
sample they are median $1.2172$, IQR $1.5336$, max $5.1390$ (§6.1).

## 6. Reproducing §5.1 and §5.2

### 6.1 Data

**§5.1.** $p\,\mathrm{LN}(\mu_1,\sigma_1^2)+(1-p)\mathrm{LN}(\mu_2,\sigma_2^2)$
with $\mu_1=0,\mu_2=1.2,\sigma_1^2=0.25,\sigma_2^2=0.02,p=0.8,n=200$; no
censoring. Note the variance convention: SciPy's `lognorm(s=)` is $\sigma$, so
`s = sqrt(0.25) = 0.5`.

**The existing `TwoLognormalMix` has a real defect and must be fixed.** It
passes the same integer `random_state=self.seed` to both components, so the two
samples share one random stream: the correlation between the driving uniforms
over the overlapping 40 draws is **0.977**. Each component is marginally
correct, but the dataset is not i.i.d. from the mixture (component 2 is a
deterministic function of component 1's first 40 draws), which invalidates
"representative sample" and any repeated-sample study built on it. Fix: build
one `rng = np.random.default_rng(seed)` and pass that object to both
`lognorm.rvs` calls. While moving the class to `src/paper_data.py`, also rename
`sigma_1/sigma_2` to `var_1/var_2` — the fields hold variances and are
`sqrt`-ed inside.

With that fix and seed 721383 the realised sample is median $1.2172$, IQR
$1.5336$ ($q_{.25}=0.8168$, $q_{.75}=2.3505$), max $5.1390$, with 2 observations
above 4. (The shared-seed defect produced the previously recorded
$1.1244/1.2550/4.5518$; those numbers are retired.)

Properties to use as test targets (computed here from the true mixture):

| property | value |
| --- | --- |
| population median | 1.1727 |
| population IQR ($q_{.25},q_{.75}$) | 1.3679 (0.7832, 2.1511) |
| 95th percentile | 3.6828 |
| hazard change points in $(0,5)$ | 1.022, 2.366, 4.454 |
| second density mode | $\approx\exp(1.2-0.02)=3.26$ |
| observations above 4 | 2 (max 5.1390) in our sample — the mechanism behind the widening bands beyond $t=4$; the paper's own sample has 4 above 4, max 4.574 |

**§5.2.** Leukemia remission times in weeks (Lawless 1982, p. 351), transcribed
from the paper:

```
A: 1, 3, 3, 6, 7, 7, 10, 12, 14, 15, 18, 19, 22, 26, 28+, 29, 34, 40, 48+, 49+
B: 1, 1, 2, 2, 3, 4, 5, 8, 8, 9, 11, 12, 14, 16, 18, 21, 27+, 31, 38+, 44
```

Censoring differs by arm: **A has 3** censored (28+, 48+, 49+), **B has 2**
(27+, 38+). Store as (time, event) arrays with a test that the counts are
$n=20$, $n_o=17$, $n_c=3$ for A and $n=20$, $n_o=18$, $n_c=2$ for B, and that the
transcription matches the paper's lists element by element (including the `+`
flags). An earlier draft said "three censored per treatment, counts 20/3 for
each"; that fails on B and must not be used as an assertion.

### 6.2 Settings

| quantity | paper | `--preset paper` | `--preset quick` |
| --- | --- | --- | --- |
| burn-in | ≤5,000 | 5,000 | 1,000 |
| saved draws $B$ | 10,000 | 10,000 | 1,000 |
| thinning | ≤150 | 150 | 10 |
| sweeps | 1,505,000 | 1,505,000 | 11,000 |
| $M$ (grid) | not stated | 256 | 256 |
| $\epsilon$ | $10^{-4}$ | $10^{-4}$ | $10^{-4}$ |
| $L$ (MC variant) | 1,000; 2,500 used | 2,500 | 200 |
| chains per prior | "multiple" | 4 | 2 |

$\nu$ priors: $\mathrm{Gamma}(2,0.9)$, $\mathrm{Gamma}(2,0.1)$,
$\mathrm{Gamma}(3,0.05)$ (§5.1, Fig. 1–3). §5.2 uses
$\mathrm{Gamma}(2,0.9)$ for all reported results and all three for Figure 4(a).
$d=2$, $a_\gamma=1$, $a_\phi=2$, $b_\gamma,b_\phi$ from §5.

Runs required for the full figure set: §5.1 × 3 priors + §5.2 × (A: 3 priors,
B: 1 prior) = **7 fits**, each with `n_chains = 4` chains, i.e. **28 chains**.
Run them through `concurrent.futures.ProcessPoolExecutor` (`sample_chains`) — no
new dependency, and the seeds are independent by construction.

### 6.3 Figure-by-figure acceptance criteria

These are the reproduction targets. They are deliberately *inequalities*: the
paper's figures are posterior summaries with Monte Carlo error and our RNG
differs. "Reproduced" means "same qualitative claim, same order of magnitude,
no contradiction". Two standing caveats apply to every row: our §5.1 sample is
ours, not the paper's (§6.1), and the priors are applied on the $\tilde t$ scale
while being elicited on the original scale (§2.3, §5) — so prior bands are
compared against *our* prior functionals, never against the elicitation guess.

| Fig. | Content | Acceptance |
| --- | --- | --- |
| 1 | posterior of $\nu$ and $n^*$ under the 3 priors | posterior mean of $\nu$ **increases** with the prior mean of $\nu$ (2.22 → 20 → 60), as in the paper; posterior mean of $n^*$ lies in $[3,15]$ under all three priors and increases with the prior mean of $\nu$; $\Pr(3\le n^*\le15\mid D)\ge0.85$ under $\mathrm{Gamma}(2,0.9)$ and $\mathrm{Gamma}(2,0.1)$ and $\ge0.70$ under $\mathrm{Gamma}(3,0.05)$ (see amendment A1); posterior of $\nu$ is narrower than each prior over its bulk (learning), while the right tail under $\mathrm{Gamma}(3,0.05)$ is prior-dominated (paper §5.1) |
| 2 | $S(t)$, $f(t)$: prior and posterior means with 95% bands | posterior mean $S$ at $t=1.1727$ in $[0.42,0.58]$; $\int \hat f\,dt\in[0.95,1.05]$ over the grid; prior 95% band width $\ge$ posterior band width on $(0,4]$, measured from `prior_functionals` on the original scale; density bimodality retained: $\hat f(3.26)>\hat f(2.4)$ |
| 3 | hazard: posterior mean + 95% band; comparison under 3 priors | after Gaussian smoothing of $\hat h$ (bandwidth fixed in the driver, e.g. 5 grid points), exactly 3 sign changes of the first difference on $(0,5)$, the first two within $\pm0.25$ of 1.02 and 2.37 and the third within $\pm0.35$ of 4.45 (see amendment A2; unsmoothed first differences have grid-noise sign changes); band width at $t=4.5$ exceeds that at $t=2.5$ by a factor $\ge1.5$ |
| 4 | leukemia: $S_A$ under 3 priors; $S_A$ vs $S_B$; densities; hazards | 3-prior curves agree within their bands; $\hat S_A>\hat S_B$ on $(0,20)$; hazard of B rises then falls (max of $\hat h_B$ in $[10,20]$) |
| 5 | $h_B/h_A$; $F_B-F_A$; $\eta_A-\eta_B$ | $\Pr(\eta_A>\eta_B\mid D)\in[0.85,0.96]$ (paper 0.9135); the posterior mean of $F_B(t)-F_A(t)$ is $\ge0$ for $t\in(0,20)$ and $\Pr(F_B(t)>F_A(t)\mid D)\ge0.50$ pointwise, with mean $\ge0.75$ over the window (amendment A3); $h_B/h_A$ is not constant: its posterior mean at $t=5$ exceeds that at $t=25$ by $\ge30\%$ and $\Pr(h_B/h_A(5)>h_B/h_A(25)\mid D)\ge0.75$ (amendment A4) |

**Amendments (after the paper-grade probe of step 18, 2026-08-18).** Both were
decided before the reproduction run was accepted; the probe used the paper
preset's burn-in and thinning (5,000 / 150) with 500 saved draws per chain and
four chains at seed 20240601.

- **A1 — row 1, the $n^*$ clause.** The original "$\Pr(3\le n^*\le15)\ge0.85$
  under all three priors" was the plan's own strengthening of the paper's
  "the data support roughly 3 to 15 distinct components" (paper p. 14), whose
  Figure 1 $n^*$ axis runs from 0 to 30 and whose widest-prior histogram has a
  visible tail past 15. The probe gives $\Pr=0.987/0.885/0.730$ with posterior
  means $7.0/9.9/12.8$ for the three priors: the criterion held for the two
  narrower priors and failed only for the deliberately extreme
  $\mathrm{Gamma}(3,0.05)$. Requiring $0.85$ there is not a paper claim, so
  row 1 now asserts the paper's claim instead — mean $n^*\in[3,15]$ for all
  three priors, monotonicity in the prior mean, and $\Pr\ge0.70$ for the
  widest prior.
- **A2 — row 3, the third change point.** 1.022, 2.366 and 4.454 are change
  points of the *population* hazard mixture (§6.1), and the probe recovers
  $4.462/4.245/4.166$ under the three priors. The third point is the one the
  sample's tail controls: our sample has 2 observations above $t=4$
  (max 5.139) against the paper's 4 (max 4.574), a difference §6.1 already
  records, and the paper itself attributes the widening bands beyond $t=4$ to
  exactly those observations. The third tolerance is therefore $\pm0.35$;
  the first two stay at $\pm0.25$.
- **A3 — row 5, the $F_B-F_A$ band clause.** The original "$F_B(t)-F_A(t)\ge0$
  with 95% band mostly above 0" is not what the paper's Figure 5(b) shows: its
  95% interval straddles zero (the lower dashed line is $\approx-0.05$ at
  $t=15$), and only the *80%* interval sits above zero over part of the range.
  The probe agrees with the figure: the posterior mean is positive
  ($\min=0.0101$ on $(0,20)$) and $\Pr(F_B>F_A)$ runs from 0.56 to 0.91
  (mean 0.83) across the window, while the per-draw 2.5% quantile is between
  $-0.168$ and $-0.083$. Demanding a 95% band above zero would require
  $\Pr>0.975$ pointwise, contradicting the row's own
  $\Pr(\eta_A>\eta_B)\approx0.9$. Row 5 now asserts the mean and the pointwise
  probability instead, with the pointwise floor at $0.50$ — the difference is
  more likely positive than not everywhere — so that the probe's minimum of
  $0.56$ keeps Monte Carlo margin.
- **A4 — row 5, the hazard-ratio clause.** "Bands at $t=5$ and $t=25$ do not
  overlap" reproduces a feature of the paper's Figure 5(a) — very tight
  $h_B/h_A$ intervals, $\approx(1.45,2.0)$ near $t=4$ against
  $\approx(0.55,0.9)$ near $t=27$ — that our per-draw ratio does not have:
  the probe gives 95% bands $(0.663,5.455)$ at $t=5$ and $(0.282,3.103)$ at
  $t=25$. The heavy tail comes from the ratio of two small hazards in a
  single draw; the closed-form hazard tests (tests 25, 26) and the
  correctness oracles do not indicate a defect. What *is* reproduced is the
  paper's qualitative claim, non-proportional hazards: the posterior mean
  ratio falls from 2.30 at $t=5$ to 1.18 at $t=25$ with
  $\Pr(\text{ratio}_5>\text{ratio}_{25})=0.81$. Row 5 now asserts that decline
  and probability, and the results note records both sets of band numbers so
  the gap is visible rather than hidden.

Write the achieved numbers into `docs/research/dp_gibbs_results.md` (new, small)
with preset, seeds, chain count, $\hat R$, and the wall-clock time. Figures are
saved as PNG under `docs/research/figures/`, which is `.gitignore`d
(decision recorded in §10): the results note carries the numbers, the figures
are regenerated on demand, and no binaries enter git.

## 7. Test suite design

Layout: `tests/` at repo root; `pytest`; markers `slow` (minutes) and `paper`
(reproduction, opt-in via `-m paper`). Three pieces of configuration are
required, and step 1 of §9 is not done until all three work.

**(1) `pixi.toml` — a dev feature is useless unless it is in an environment.**

```toml
[feature.dev.dependencies]
pytest = ">=8,<9"
ruff = ">=0.6,<1"
pyright = ">=1.1,<2"

[environments]
default = { features = ["dev"], solve-group = "default" }

[tasks]
test = "pytest -m 'not slow and not paper'"
test-all = "pytest"
lint = "ruff check src tests && ruff format --check src tests"
typecheck = "pyright"
```

Then `pixi lock` (the lock file is committed) and verify `pixi run test` and
`pixi run lint` both execute. None of `pytest`, `ruff`, `pyright` exists in the
current environment, so without this the per-step verify gate of the
`incremental-development` skill cannot be satisfied.

**(2) pytest configuration** — new `pyproject.toml` (none exists yet):

```toml
[tool.pytest.ini_options]
markers = ["slow: minutes", "paper: paper reproduction, opt-in"]
addopts = "-q"
```

Unregistered markers produce warnings and make `-m` filters fragile.

**(3) Import path** — every module imports `from src.… import …` (as the
existing scaffolding does), but pytest puts `tests/`, not the repo root, on
`sys.path`. Add a one-line root `conftest.py` (its directory is inserted by
pytest's default *prepend* import mode) or
`pythonpath = ["."]` under the pytest ini options. Without one of them nothing
imports and step 1 is not green. Extend `pyrightconfig.json` at the same time
(`include = ["src", "tests"]`, and align `pythonVersion` with the
`.pixi` environment — it currently says 3.13 while the env is 3.15).

Tiers and budgets:

| tier | what | runtime | runs by default |
| --- | --- | --- | --- |
| T0 | math primitives, grid quadrature, containers | < 5 s | yes |
| T1 | one full conditional at a time | < 30 s | yes |
| T2 | joint correctness: exact posterior, reference sampler, SBC | 1–5 min | SBC only with `slow` |
| T3 | functional inference and the time transformation | < 30 s | yes |
| T4 | golden regression + paper reproduction | model-fit minutes | golden yes, reproduction `paper` |

Every test states *what it would catch* (§7.7). A test that would pass with an
obviously wrong implementation is not worth its runtime.

### 7.1 T0 — primitives

1. **$q_0^o$ against independent 2-D quadrature.** Compare
   $\nu\int_0^\phi\!\!\int_0^\infty k_W(t\mid\alpha,\lambda)
   \mathrm U(\alpha\mid0,\phi)\mathrm{IGamma}(\lambda\mid d,\gamma)\,d\lambda\,d\alpha$
   computed with `scipy.integrate.quad` nested (inner $\lambda$ integration done
   numerically, *not* the closed form) against the closed-form Eq. A.2, for 25
   random $(t,\gamma,\phi,\nu,d)$ with $t\in(1,2]$, $\gamma,\phi>0$,
   $d\in\{1,2,3\}$. Tolerance: relative $10^{-10}$. Catches algebra errors in
   the $\lambda$ marginalisation, the $\gamma^d/\phi$ factors, the $1/t$ or
   $t^{\alpha-1}$ index, and the $d$ versus $d+1$ choice.
2. **$q_0^c$ against the same nested quadrature** using
   $1-K_W = e^{-t^\alpha/\lambda}$. Tolerance relative $10^{-10}$. Catches the
   censored branch specifically, including the $d$ (not $d+1$) shape.
3. **Grid quadrature accuracy.** For fixed $(\gamma,\phi,t,d)$, compare the
   $M$-point trapezoid rule of §2.4 against `quad` for
   $M\in\{32,64,128,256,512\}$; assert the relative error is $\le10^{-4}$ at
   $M=256$, decreases monotonically, and that the observed convergence order is
   $\ge1.8$. Measured ($\gamma,\phi,t,d)=(0.6,1.7752,1.5,2)$: $1.8\!\times\!10^{-3}$,
   $4.6\!\times\!10^{-4}$, $1.1\!\times\!10^{-4}$, $2.9\!\times\!10^{-5}$,
   $7.2\!\times\!10^{-6}$, order 2.00. Do **not** assert $10^{-6}$ at $M=256$: the
   omitted first cell makes the error $O(1/M^2)$ with that constant, so the
   assertion fails on correct code. Catches a wrong weight vector (missing
   $\Delta\alpha$, unhalved endpoints, wrong factor of $\phi$) and a grid that
   silently starts at $\alpha=0$.
4. **$\alpha$-grid draw equals the grid density.** Draw $10^5$ $\alpha$ values
   with the inverse-CDF sampler for a fixed row and compare with the exact
   normalised integrand by total variation distance (split the $(0,\phi)$
   interval at the 20 grid midpoints); assert TV $<0.02$. Catches
   `searchsorted` side/offset bugs and the "forgot to normalise the row" bug.
5. **`SurvData` ingestion and transformation.** Round trip $\tilde t \to t$ for
   random data; assert $\log\tilde t>0$; assert `t_max` is the max of the
   original times. Then the validation matrix, one test per rule, each asserting
   the exception **type and message** (None–One–Many over each input): empty
   arrays; `time`/`event` length mismatch; `event` not boolean or not 0/1;
   time $\le0$; **`NaN` and $\pm$`inf` in `time`** (a `NaN` passes a
   `time <= 0` check silently and then poisons `t_max`, the grid and the whole
   chain — check `np.isfinite` explicitly); `t_max <= 0`. Catches the transform
   being applied to already-transformed data, and an ingestion boundary that
   lets hostile input into the sampler. Add one integration test that
   `np.load` of a fixture is called with `allow_pickle=False`.
6. **Cluster bookkeeping.** Random states → assert invariants §2.1 after
   `compact()` and after simulated removals/appends; assert `n_j.sum()==n`
   after 1,000 random operations.
7. **$b_\gamma$, $b_\phi$ (§5).** For three values each of
   $\tilde\lambda,\tilde\alpha$: build the induced marginals
   $[\lambda]=2b/(1+\lambda b)^2$'s density by `quad` and find its median by
   `brentq`; assert it equals $\tilde\lambda$ to $10^{-6}$ relative, and
   likewise for $[\alpha]$ and $\tilde\alpha$. **This test passes against the
   current code** — it exists to prevent the $(2^{1/3}-1)$ "correction" from
   ever being applied (see §5). Catches a future change to either constant and
   the density/CDF exponent confusion that produced that correction.

### 7.2 T1 — one conditional at a time

Each test fixes the rest of the state, runs one step many times, and compares
with the exact conditional computed by an independent route.

8. **Step (a), urn probabilities.** Fix the state; enumerate
   $q_0^{i}$ and $n_j^{-}q_j^{i}$ exactly (via `quad` and the kernel, not the
   sampler's cached values); apply the sampler's step (a) to observation $i$
   $10^5$ times starting from the same state (`sample_step_a_for_obs`, which
   exists for this test — the public `sample_step_a` sweeps all $i$); compare the empirical
   category frequencies with a chi-square test ($\alpha=10^{-3}$). Run for an
   uncensored and a censored observation, and for a state where $i$ is a
   singleton (so the "remove, cluster disappears" path is exercised).
   Catches: dropped clusters still in the candidate set, wrong $q_j$
   (density vs survival swapped), the missing $n_j^{-}$ factor, off-by-one in
   the inverse CDF, and using $n_j$ instead of $n_j^{-}$.
9. **Step (a), continuous piece.** Given that the draw is a new cluster, the
   $(\alpha_i,\lambda_i)$ has the density of §2.2. Check $\alpha$ by a
   Kolmogorov–Smirnov test against the exact CDF from the grid at $M=4096$
   (high resolution as the reference), and $\lambda$ given $\alpha$ by
   `scipy.stats.invgamma` via `ks_1samp`. Catches: wrong invgamma arguments
   ($d$ vs $d+1$, $\gamma+t^\alpha$), and drawing $\alpha$ from the *censored*
   row for an uncensored observation.
10. **Step (b), $\lambda^*$ refresh.** `ks_1samp` against
    `invgamma(d + n_j_o, scale=gamma + sum_t + sum_z)` for a cluster with
    $n_j^o\in\{0,1,3\}$ and censored members. Catches the empty-uncensored case
    and a missing $\gamma$.
11. **Step (b), $\alpha^*$ refresh.** The strongest single-conditional test.
    The target of the auxiliary-variable step is
    $[\alpha^*_j\mid\lambda^*_j,s,\ldots]\propto\alpha_j^{*n_j^o}\mathbf 1(\alpha^-_j<\alpha^*_j<\alpha^+_j)$
    (A.6). Compute that reference **unconditionally**: integrate $v$ and the
    $w$'s out analytically, i.e. use the unnormalised A.5 density
    $\alpha^{n^o}C(\alpha)\exp(-\lambda^{*-1}(\sum t^\alpha+\sum z^\alpha))$, so
    the reference does not depend on the sampler's bounds code; normalise by
    `quad` on $(0,\phi)$. Compare $10^5$ draws with the reference after
    binning into the 40 equal-mass intervals of the reference (state the rule;
    a TV distance is otherwise undefined). Assert TV $<0.02$ **and** a
    max-CDF deviation $<0.01$ (at $10^5$ draws a KS p-value rejects deviations
    of $\sim10^{-2}$, i.e. it is over-powered; use the effect size). Run for
    $n_j^o\in\{0,1,2,5\}$, $n_j^c=0$, and mixed clusters. **This test catches
    the uniform-instead-of-$\alpha^{n^o}$ error, which is present in every
    cluster with an uncensored member** — including $n_j^o=1$, where the
    correct law is triangular.
12. **Step (c), $\nu$.** Empirical CDF of $10^5$ draws vs the exact two-Gamma
    mixture CDF, KS test. Also assert that $u$ is drawn from
    $\mathrm{Beta}(\nu+1,n)$ with the *current* $\nu$, by checking the
    conditional distribution of $u$ at fixed $\nu$.
13. **Step (c), $\phi$.** KS against `pareto(a_phi + n_star, scale=max(b_phi, max alpha_star))`.
    Assert $\phi\ge\max_j\alpha^*_j$ on every draw.
14. **Step (c), $\gamma$.** KS against
    `gamma(a_gamma + d*n_star, scale=1/(b_gamma + sum(1/lambda_star)))`.
    Catches the $d\,n^*$ vs $n^*$ error, which is invisible in plots.

### 7.3 T2 — joint correctness

15. **Exact posterior by enumeration (the headline test).** Fix
    $\nu,\gamma,\phi$ and $d$; restrict **only** $\alpha$ to a coarse grid
    $\{0.5,0.75,\dots,3\}$. For $n=3$ observations, enumerate every partition
    and every cluster-$\alpha$ assignment, and **integrate $\lambda$ in closed
    form** — for a cluster $j$ the collapsed marginal likelihood is
    $\Gamma(d+n_j^o)\,(\gamma+\sum_{i_o}t^{\alpha}+\sum_{i_c}z^{\alpha})^{-(d+n_j^o)}$
    up to $\alpha$-only factors, so no $\lambda$ discretisation is needed and
    none may be used. Do not put $\lambda$ on a grid: the sampler draws
    $\lambda$ from its continuous conjugate conditional, so a discretised
    reference targets a different stationary law and the test proves nothing.
    Run the sampler in a test-only discrete mode (alpha snapped to the same
    grid, $\lambda$ still continuous; not a production branch), $10^6$ sweeps,
    and compare the marginal posterior over the 5 partitions of $n=3$ and
    $\Pr(s_1=s_2)$ with the exact values (max absolute error $<0.01$, and $<3$
    Monte Carlo standard errors).
    This is the only test that validates the **joint** update (removal,
    urn, relocation, ordering) rather than one conditional at a time. If it
    passes, the algorithm is right; if it fails, no amount of posterior
    smoothness should be trusted.
16. **Independent reference sampler.** `tests/reference_sampler.py`: an
    intentionally naive, independently derived sampler for small $n$ that
    shares *no* code with the production path:
    - $q_0$ by nested `quad` with the inner $\lambda$-integral done numerically
      (does not use the closed form or the grid);
    - the continuous piece by 2-D rejection sampling on
      $(\alpha,\lambda)\in(0,\phi)\times(0,\lambda_{\max})$ instead of
      invgamma + grid;
    - step (b) by rejection sampling from the exact Eq. A.5 density instead of
      the auxiliary-variable scheme;
    - $\nu$ by inverse CDF of the analytic two-Gamma mixture instead of
      Escobar–West augmentation.
    Agreement target: on a synthetic censored dataset with $n=12$, compare
    posterior means and 90% intervals of $\nu$, $n^*$, $S(t)$ at three points,
    and $E[h(t)]$ with $|$difference$|<3$ pooled Monte Carlo standard errors.
    This catches shared-derivation errors (a mistake in A.2 propagated into
    both implementations is impossible here).
17. **Simulation-based calibration (SBC, `slow`).** Generate from the model:
    $\nu,\gamma,\phi\sim$ priors, $G\sim\mathrm{DP}(\nu G_0)$ truncated at
    $\epsilon_{\text{tail}}=10^{-8}$, $\theta_i\sim G$, $t_i\sim K_W$. **Censoring
    must be specified, not hand-waved:** for the censored replicates draw
    $c_i\sim\mathrm U(0,c_{\max})$ independently of $t_i$ (with $c_{\max}$ fixed
    at the §6.1 sample's max, 1.2× that value), set
    $(\text{time},\text{event})=(\min(t_i,c_i),\ t_i\le c_i)$; record the realised
    censoring fraction and hold it in $[0.15,0.25]$ per replicate. Without a
    fixed censoring law the ranks are not uniform even for a correct sampler.
    For $R=100$ replicates with $n=15$, run short chains (`quick` preset, 1
    chain), and test calibration of $\nu,\gamma,\phi$ and of $S(t)$ at
    $t\in\{q_{.25},q_{.5},q_{.75}\}$ of the true generating mixture with:
    (a) rank histograms over 10 bins, each bin count within $\pm3$ binomial SD of
    $R/10$; and (b) one pre-registered effect-size test per statistic
    (the rank distribution's CDF deviation from uniform $<0.15$). Do **not** use
    "KS $p>0.01$ per statistic": at $R=100$ it has low power, and with eight
    statistics it fails spuriously about 8% of the time by construction. State
    the statistic list before running. Catches: conditionals that are
    individually correct but mis-ordered within the sweep, a hyperprior
    inconsistency (rate/scale, prior on $\nu$ used but not in $q_0^{i}$), and
    wrong treatment of censoring. Run with `pytest -m slow`; keep the
    replicates' seeds in a JSON fixture so failures are reproducible.
18. **Multiple chains / mixing.** Four chains from the §3.6 initialisation on
    the §5.1 data at `quick` preset: split $\hat R<1.05$ for $\nu$, $n^*$, and
    $S(t)$ at 5 points; assert $n^*$ visits more than one value in every chain
    (a chain stuck at $n^*=1$ or $n^*=n$ fails).
19. **Data-order invariance.** Permute the observation order, rerun with the
    same seed: posterior means of $S(t)$, $\nu$, and the *distribution* of
    $n^*$ agree within 3 Monte Carlo standard errors. Catches sequential-update
    bugs that make the answer depend on the ordering of the data.
20. **Degenerate-data robustness.** All-uncensored, all-censored, $n=1$,
    $n=2$, duplicated times, all-equal times: the chain runs 100 sweeps
    without producing NaN/negative parameters and maintains invariants.
    (Posterior inference on all-censored data is meaningless but must not
    crash: the $\alpha^-$ empty-sum path is exercised here.)

### 7.4 T3 — functional inference and the transformation

21. **Single-atom exactness.** With a `DPRealization` of one atom
    ($w=1$), every functional must equal the closed-form Weibull value:
    $S(t)=e^{-t^\alpha/\lambda}$, $f(t)$, $h(t)=\alpha t^{\alpha-1}/\lambda$.
    Check against `scipy.stats.weibull_min` with $c=\lambda^{1/\alpha}$.
    Catches the two-$\lambda$ confusion
    (`docs/research/weibull_lambda_definitions.md`).
    **21b — prior-functionals guard (same tier).** Prior functionals must be
    centred where the prior actually is (§5 caveat). With $T_{\max}=4.5518$ and
    the §5 constants, Monte Carlo from `prior_functionals` must reproduce,
    within 3 MC standard errors: prior mean $S(1.1727)=0.43$, its 2.5–97.5%
    band $(0.00,0.94)$, and the prior median of the population median
    $\approx0.61$ — **not** $1.1727$. This makes the elicitation-scale caveat
    of §5 falsifiable: it fails if someone "fixes" the transform or the priors
    so that the two scales coincide. It also guards the Figure 2 comparison,
    which is against these bands, not against the elicitation guess.

22. **`exact` vs `mc`.** For the same $G_b$ draws, the Rao-Blackwellised
    functional and the $L=2500$ Monte Carlo version must agree within
    $1/\sqrt L$ scaled errors; over 200 independent $G_b$, the mean of the MC
    estimator must equal the exact value. This validates that the paper's
    step (iii) estimator is being computed at all. **Scope the unbiasedness
    claim to $S$ and $f$**: the hazard is a ratio of two MC averages over the
    same $L$ atoms, so its MC version is biased at $O(1/L)$ while remaining
    consistent — assert $|\hat h_{\text{mc}}-h_{\text{exact}}|\le3$ MC standard
    errors, not zero bias.
23. **Pruning.** Build a `DPRealization` with a deliberately long tail and
    compare functionals before/after `pruned(1e-6)`: $\sup_t|S-S'|<10^{-6}$ and
    $\sup_t|f-f'|<10^{-6}$. For $h$, assert an **absolute** bound
    $\sup_{t: f(t)>0.05}|h-h'|<10^{-4}$ only where the density is not tiny —
    pruning error is absolute, so the relative error in $h$ can be large in the
    far tail where $f\to0$, and a blanket relative tolerance would fail on
    correct code. Also assert the pruned weights still sum to 1 and that the
    residual-mass atom is retained. Catches a pruning that renormalises wrongly
    or drops the residual atom.
24. **Common $J$.** Assert `truncation_number_for_posterior(max_nu, n)` equals
    the paper's formula for 5 $(n,\nu)$ pairs, and that it is $\ge$ every
    per-draw `truncation_number(nu_b + n)`. Catches the per-draw-$J$
    mistake.
25. **Internal consistency of functionals.** On a grid: $S=1-F$;
    $\int f\approx1$ (trapezoid, tolerance 1e-3); $h=f/S$;
    $-\frac{d}{dt}\log S\approx h$ by finite differences (relative 1e-3).
    Catches swapped $f$/$S$ and dropped normalisations.
26. **Transformation Jacobian.** Synthetic single-atom $G$ on the $\tilde t$
    scale: assert the four maps of §2.3 against the analytic Weibull on the
    original scale, particularly $f$ and $h$ (factor $1/T_{\max}$). Also assert
    that a *median* maps as $\eta=T_{\max}(\tilde\eta-1)$ and that
    $\Pr(\eta_A>\eta_B)$ is invariant to the transform for a synthetic pair.
    Catches the classic missing-Jacobian error, which changes $f$ and $h$ by a
    constant factor and is invisible in $S$.
27. **Quantile inversion.** For a synthetic $G_b$, compare inverted quantiles
    against `weibull_min.ppf`-based mixture quantiles found by `brentq` on the
    exact CDF. Tolerance: 1e-6 **in $t$**, not in $F$: inversion on a 300-point
    grid is piecewise-linear in $F$, so $|F(\hat\eta)-p|$ is $O(h^2)\approx10^{-4}$
    by construction and a 1e-6-in-$F$ assertion fails on correct code. Also
    test the single-atom case, where inversion is exact, at 1e-12 in $t$.

### 7.5 T4 — regression and paper reproduction

28. **Golden regression (`fast`).** Fixed seed, `quick`-like but tiny settings
    ($n=20$ synthetic censored data, 200 sweeps, $M=32$): store $s$, $\nu$,
    $\gamma$, $\phi$, $n^*$ and $\log$ posterior, plus $S(t)$ at 3 points, as
    JSON in `tests/fixtures/golden_dp_gibbs.json`. Assert the discrete and RNG-
    determined quantities ($s$, $n^*$) **exactly**, and the floating summaries
    ($\nu,\gamma,\phi,\log$ posterior, $S$) to 1e-10 relative — not "bitwise",
    which `pow`/`logsumexp` cannot guarantee across CPU and BLAS paths. Guards
    every refactor and pins the RNG consumption order of §3.4. Regenerate only
    deliberately, and make regeneration a separate script invocation (a test
    must never rewrite its own fixture).
29. **Known-answer check of the scaling.** Run `quick` on §5.1 with one prior:
    assert posterior mean of $n^*$ in $[2,25]$ and of $\nu$ in
    $[0.5,15]$ — a loose band that only catches gross breakage (e.g. $\nu$
    sampled from the wrong Gamma convention) without duplicating the paper
    reproduction.
30. **Paper reproduction (`paper` marker).** The §6.3 table, one assertion per
    row, each with the tolerance in the table. Runtime is model-fit minutes
    (see §8); not part of `pixi run test`.
31. **Diagnostics against analytic values.** For simulated AR(1) chains with a
    known stationary variance and lag-1 correlation, assert split $\hat R\to1$
    within 0.05 and that the autocorrelation-based ESS is within 10% of
    $n(1-\rho)/(1+\rho)$. Also assert $\hat R>1.2$ for chains run from
    deliberately different initial values with 20 stored draws. Catches a
    variance-ratio formula applied over the wrong axis (a common silent bug:
    it returns $\hat R\approx1$ for any input) and an ESS computed without the
    autocorrelation sum.
32. **Import and CLI hygiene.** Assert `import src.dp_gibbs` and
    `import src.dp_functionals` succeed in <0.2 s and that `matplotlib` is not
    in `sys.modules` afterwards (this is how the "`plt` only in
    `paper_examples.py`" rule is enforced). Invoke the CLI in-process with
    `--preset` misspelled and `--example 5.3`: assert a non-zero exit and a
    usage message naming the valid choices, with no traceback written to
    stdout.

### 7.6 Fixtures

- `tests/fixtures/section_5_1.npz` — the generated sample (times, seed,
  realised median/IQR/max, one-`default_rng` stream per §6.1).
- `tests/fixtures/section_5_2.npz` — leukemia times and event flags, plus the
  transcribed lists from the paper for the transcription test.
- `tests/fixtures/golden_dp_gibbs.json` — test 28.
- `tests/fixtures/sbc_seeds.json` — test 17.
- The `.npz`/`.json` files are **committed** (`.gitignore` does not exclude
  them). `tests/fixtures/make_fixtures.py` is a regeneration script, run
  manually — it is not imported by any test, and no test writes a fixture.
  If the fixtures were regenerated at test time the "independent of SciPy's RNG
  version" guarantee would be void. Load them with `allow_pickle=False`.
  The regeneration script is tested by comparing its *structural* assertions
  (counts 20/17/3 and 20/18/2, `+`-flag positions, `t_max`) rather than by
  reproducing the committed arrays.

### 7.7 What each test would catch (mutation table)

The suite is accepted only if every row is covered by at least one test that
*fails* under the mutation.

| mutation | caught by |
| --- | --- |
| $q_0$: use $d+1$ instead of $d$ in the censored branch | 1, 2, 8 |
| $q_0$: forget the $\nu$ factor | 8, 16, 29 |
| $q_0$: forget the $1/\phi$ from Uniform$(\alpha\mid0,\phi)$ | 1, 8 |
| urn: use $n_j$ instead of $n_j^-$ | 8, 15 |
| urn: keep the emptied cluster in the candidate set | 8, 15 |
| urn: swap density/survival for censored rows | 8, 17 |
| $\alpha$ draw: use the censored row for uncensored data | 9, 15 |
| $\lambda$: `invgamma(d, gamma + t^alpha)` instead of `d+1` (uncensored) | 9, 16 |
| $\alpha^*$: skip the $w$ auxiliaries (v only) | 11, 16 |
| $\alpha^*$: use $\log t$ denominator with $t$ on the untransformed scale | 11, 15 |
| $\alpha^*$: draw uniform on $(\alpha^-,\alpha^+)$ instead of $\propto\alpha^{n_j^o}$ (A.6) | 11, 16 |
| $\alpha^*$: substitute an ordinary slice sampler | 11, 16 |
| $\alpha^*$: use $n_j$ instead of $n_j^o$ in that exponent | 11, 15 |
| $\nu$: Escobar–West $p$-weight or the $n^*$ vs $n^*-1$ index | 12, 15, 17 |
| $\gamma$: use $n^*$ instead of $d\,n^*$ | 14, 17 |
| $\phi$: forget $n^*$ in the Pareto shape | 13, 17 |
| sweep order: (c) before (b) | 15, 18 |
| functionals: per-draw $J$ instead of common $J$ | 24, 22 |
| functionals: hazard from two different $G_b$ | 22 |
| transform: missing $1/T_{\max}$ Jacobian | 26, 30 |
| transform: applied twice | 5, 20 |
| $b_\gamma$ using $(2^{1/3}-1)/\tilde\lambda$ (i.e. CDF exponent 3) | 7 |
| $b_\gamma$ using the mean $1/\tilde\lambda$ instead of the median | 7 |
| $b_\phi$ from $\tilde\alpha$ instead of $4\tilde\alpha/3$ | 7 |
| diagnostics: $\hat R$ computed across the wrong axis | 31 |
| import: `plt` pulled in by the sampler module | 32 |

## 8. Performance budget

Targets, with the reasoning, because "reproduce the paper" is only practical if
per-sweep cost is known before the 1.5M-sweep run starts.

Cost model per sweep ($n=200$, $M=256$, $n^*\approx10$):

| component | cost | notes |
| --- | --- | --- |
| `Tau = T**phi`, $(200,256)$ `pow` | ~0.10 ms | caches `T` once per chain |
| integrand + `logsumexp` per row | ~0.15 ms | $4\times10^4$ logs |
| `q_j` matrix, $(n,n^*)$ | ~0.02 ms | |
| `log_post`, one vectorised pass | ~0.05 ms | $n$ kernel evaluations plus per-cluster $g_0$ terms; must not loop over the grid |
| step (a) Python loop over $i$ | ~1.0–1.5 ms | ~5–7 µs/observation; the dominant cost |

Budget: **≤ 2.0 ms/sweep** at $n=200$, $M=256$. The components above already
sum to 1.27–1.77 ms, so an earlier figure of 1.2 ms/sweep was not a budget but
a wish; 2.0 ms leaves headroom for the `log_post` pass. Then

- `quick`: 11,000 sweeps ≈ 14–22 s;
- `paper`: 1,505,000 sweeps ≈ 25–50 min per chain;
- full figure set: 28 chains (§6.2) ≈ **1 h wall on 16 cores** (two rounds of
  seven fits), ≈3.5 h on 4 cores. State the core count in the results note;
  an earlier "45–60 min on ≥4 cores" was an arithmetic error.

Implementation gates:

1. Profile after step (a) and step (b) exist; if the per-sweep cost exceeds
   2.0 ms, reduce $M$ to 128 (and re-run test 3 to confirm the $10^{-4}$ relative
   accuracy is retained) before running anything long. Do **not** add
   numba/cython or multiprocessing *inside* a sweep.
2. `sample_chains` parallelises across chains only
   (`ProcessPoolExecutor`, `max_workers=min(n_chains, os.cpu_count())`).
3. Functional inference: with pruning, $B\cdot J_{\text{eff}}\cdot|t_{\text{grid}}|$
   = $10^4\cdot10\cdot300$, tens of seconds; assert < 2 min in the
   reproduction log.

## 9. Implementation sequence

One logical unit per step: code + tests + `pixi run test` green + conventional
commit (per the `incremental-development` skill). Each step is independently
revertible.

**Pre-work, before step 1.** The workspace is not clean: `src/kottas_2002_5_1.py`
is modified and this plan document is untracked, on branch `master`. Commit or
restore those changes, run the existing tests once to establish a green
baseline, and branch (`agent/dp-gibbs`) per the `git-workflow` skill's Pre-Work
Repository Check. Do not start step 1 on a dirty tree.

| # | step | done when |
| --- | --- | --- |
| 1 | dev tooling: `pixi.toml` dev feature **in an `[environments]` entry** (pytest, ruff, pyright), `pixi lock`, `pyproject.toml` pytest markers, root `conftest.py`, `tests/` skeleton, tasks | `pixi run test` runs zero tests, `pixi run lint` and `pixi run typecheck` run |
| 2 | prior-matching guard: test 7 (median round trip, **passing**) + the one-sentence docstring note in `b_gamma_b_phi_from_weibull` (§5) | test 7 passes against the unchanged constant; deleting the median/CDF sentence fails nothing, so also assert the docstring trap in review |
| 3 | `truncation_number_for_posterior`, `DPRealization.pruned` | tests 23, 24 |
| 4 | `SurvData`/`DPPrior`/`DPSettings`/`FunctionalSettings` containers, ingestion validation, time transformation, invariants, `compact()` | tests 5, 6 |
| 5 | `AlphaGrid`, log-space integrands, $q_0^o$, $q_0^c$, grid quadrature | tests 1, 2, 3 |
| 6 | continuous-piece draw ($\alpha$ from grid + $\lambda$ invgamma) | tests 4, 9 |
| 7 | step (a) with full cluster bookkeeping | tests 8, 20 |
| 8 | step (b): $\lambda^*$, then $\alpha^*$ auxiliaries with the $\alpha^{n_j^o}$ draw (§2.2b) | tests 10, 11 (must pass for $n_j^o\in\{0,1,2,5\}$) |
| 9 | step (c): $\nu$, $\phi$, $\gamma$ | tests 12, 13, 14 |
| 10 | `sweep`, `init_state`, `sample_chain`, storage, `log_post` | tests 18, 19, 28 (golden), 20 |
| 11 | diagnostics: split $\hat R$, ESS, autocorrelation | test 31 |
| 12 | `src/dp_functionals.py`: `Functionals`, posterior/prior functionals, `exact` and `mc`, quantiles, mapping back | tests 21, 21b, 22, 25, 26, 27 |
| 13 | `tests/reference_sampler.py` + agreement test | test 16 |
| 14 | SBC harness | test 17 (`-m slow`) |
| 15 | exact-posterior enumeration test (test-only discrete mode) | test 15 |
| 16 | `src/paper_data.py` (one-RNG-stream sample, variance naming) + committed fixtures + transcription test | §6.1 assertions (20/17/3 and 20/18/2) |
| 17 | `src/paper_examples.py` driver, CLI, presets, `compare` | test 32; runs `quick` end-to-end, writes figures |
| 18 | §5.1 reproduction | §6.3 rows 1–3 pass; numbers recorded in results note |
| 19 | §5.2 reproduction | §6.3 rows 4–5 pass |
| 20 | docs: code guide for the two new modules, results note (including the prior-scale caveat and the core count), update `dp_weibull_code_guide.md` cross-references | docs free of stale "no Gibbs sampler here" claims; the prior-scale caveat stated in both the module docstring and the results note |

Steps 15, 13, 14 and 11 are the ones that actually establish correctness; if the
schedule slips, slip the reproduction (18–19), not these.

## 10. Decisions (no open questions)

Every question this plan raised has been decided. Nothing in this section is
pending; changing one of these is a plan change, not an implementation detail.

1. **Prior-guess convention (§5):** **population** values
   $(\text{median},\text{IQR})=(1.1727,1.3679)$ →
   $\tilde\alpha=1.3314$, $\tilde\lambda=1.7836$, $b_\gamma=0.232235$,
   $b_\phi=1.77522$. Recorded as constants in `paper_data.py`; the realised
   sample statistics are recorded in the results note for reference only.
2. **Prior scale (§2.3, §5):** the priors are applied on the $\tilde t$ scale
   even though elicitation is in original-time units. Keep (it is the paper's
   convention), **document it**, and verify it with test 21b and the prior
   bands of §4.2 rather than pretending it does not exist.
3. **$M$ and the $\alpha$ discretisation:** $M=256$ with the grid
   $\alpha_m=\phi m/M$, $m=1..M$, no $\alpha=0$ node. Expected accuracy
   $3\cdot10^{-5}$ relative on $q_0$ (measured); test 3 asserts $\le10^{-4}$ and
   order $\ge1.8$. If a later test fails, raise $M$ — never restructure the
   grid.
4. **`log_post`:** the exact unnormalised joint including the EPPF term (§3.7).
   Diagnostic only; never a model-comparison statistic.
5. **Wall-clock:** keep the full `paper` preset (1,505,000 sweeps × 28 chains),
   ≈1 h on 16 cores. `paper-lite` ($B=4{,}000$, thin 40, burn 2,000) remains
   available as a *documented* fallback only; any use must be named in the
   results note.
6. **All-censored data:** allowed to run, docstring warning, not a
   `ValueError` — the Appendix's formulas remain valid.
7. **Pruning:** `DPRealization.pruned(atom_tol)` stays in the functional path.
   With $n^*\ll J$ most atoms are $G_0$ draws; removing pruning would multiply
   functional-inference cost by ~200×.
8. **Figures:** the plot images go to the git-ignored
   `docs/research/figures/`; the numbers go in
   `docs/research/dp_gibbs_results.md`. No binaries in git.
9. **Naming:** `n` = number of observations, `J` = truncation level, `n_star`
   = cluster count, `n_j` = cluster size, `n_j^o`/`n_j^c` = its uncensored and
   censored parts. Never a bare `n` in new code.
10. **`src/kottas_2002_5_1.py`:** replace with a thin wrapper over
    `paper_examples.example_5_1`; do not keep the half-working body (its `theta`
    is undefined and its guard is `if __name__ == '__main__()':`, so it has
    never run).

## 11. Gotcha checklist (implementation-time)

- $\lambda$ is the paper's *scale-like* parameter in $e^{-t^\alpha/\lambda}$,
  not the standard Weibull scale $c=\lambda^{1/\alpha}$; not the rate either.
- Rate-vs-scale on every Gamma; `scale=` on every SciPy call.
- $d$ appears as $d$ (censored) and $d+1$ (uncensored) in the $\lambda$
  conditionals, and as $d$ in both $q_0$ branches. There is no $d+1$ in $q_0^c$.
- $n_j^-$ in the urn, $n_j$ everywhere else.
- **$\alpha^*_j$ is drawn $\propto(\alpha^*_j)^{n_j^o}$, not uniformly** (A.6).
  Uniform only when $n_j^o=0$; for $n_j^o=1$ the law is triangular. This is the
  single easiest factor to drop and it biases every uncensored cluster.
- The prior is elicited on the original scale but applied on the $\tilde t$
  scale; the prior's implied population median is ≈0.61, not 1.1727. Compare
  bands against `prior_functionals`, never against the elicitation guess.
- One $J$ for the whole run (functional inference), not one per draw.
- The last stick-breaking weight is the residual mass, not a fresh break.
- $\tilde t = 1+t/T_{\max}$; every functional needs the inverse map, and $f$,
  $h$ need the $1/T_{\max}$ Jacobian.
- Residual atom pruning renormalises; never drop the residual-mass atom.
- Hazard numerator and denominator must come from the same $G_b$.
- $\alpha^-$ divides by $\sum_{i_o}\log t$: only defined because $t>1$.
- Step (a) is sequential in $i$; do not vectorise it into a single
  non-sequential update (that changes the transition kernel).
- `plt` only in `paper_examples.py`; the sampler must be importable in a test
  in <0.2 s.

## 12. See also

- `docs/weib_dpm.pdf` — the paper; §3 (functionals), §4.2–4.3, §5, Appendix.
- `docs/dp_weibull_explanation.md` — the model in the paper's notation.
- `docs/research/dp_weibull_code_guide.md` — the existing module; §7 lists the
  same gotchas, and its §3.4 prior-matching section is **correct** (the
  $(\sqrt2-1)$ median) — do not "fix" it.
- `docs/research/g0_uniform_inverse_gamma.md` — why Uniform in $\alpha$ matters
  computationally (§3.1) and the conjugacy that makes every $\lambda$ step an
  inverse gamma (§4.2).
- `docs/research/dirichlet_process_concepts.md` — partition/EPPF concepts used
  by the exact-posterior test (§7.3 test 15).
- `docs/research/weibull_lambda_definitions.md` — the two $\lambda$
  parameterisations, needed by tests 21 and 26.

## 13. References

- Bush, C.A. & MacEachern, S.N. (1996). A semiparametric Bayesian model for
  randomised block designs. *Biometrika* 83, 275–285. — the location refresh
  (step b).
- Damien, P., Wakefield, J. & Walker, S. (1999). Gibbs sampling for Bayesian
  non-conjugate and hierarchical models by using auxiliary variables.
  *JRSS B* 61, 331–344. — the $\alpha^*$ auxiliary-variable scheme.
- Escobar, M.D. & West, M. (1995). Bayesian density estimation and inference
  using mixtures. *JASA* 90, 577–588. — the $\nu$ update and the $n^*$
  approximation $\nu\log(1+n/\nu)$ used in §6.3.
- Ferguson, T.S. (1973). A Bayesian analysis of some nonparametric problems.
  *Ann. Statist.* 1, 209–230. — DP posterior = DP with updated base.
- Gelfand, A.E. & Kottas, A. (2002). A computational approach for full
  nonparametric Bayesian inference under Dirichlet process mixture models.
  *JCGS* 11, 289–305. — §3's scheme and its convergence lemmas.
- Neal, R.M. (2000). Markov chain sampling methods for Dirichlet process
  mixture models. *JCGS* 9, 249–265. — the alternative algorithms that are
  explicitly **not** used here.
- Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
  *Statist. Sinica* 4, 639–650. — the stick-breaking construction.
- West, M., Müller, P. & Escobar, M.D. (1994). Hierarchical priors and mixture
  models, with application in regression and density estimation. — the
  collapsed Gibbs scheme the Appendix follows.
