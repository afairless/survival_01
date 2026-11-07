# Code Guide: `src/dp_gibbs.py` and `src/dp_functionals.py`

## Why this note exists

The implementation plan (`docs/research/dp_gibbs_plan.md`) specifies an
algorithm in the paper's notation and a test suite that decides whether the
implementation is correct. It does not say which function produces which
object, in which order a sweep touches the state, or where the time
transformation is applied. Those are the questions this note answers.

It has two companions:

- `docs/research/dp_weibull_code_guide.md` — the module that owns the base
  measure $G_0$, the truncation rule and the DP realisation. Read it for the
  mathematics of $G$; this note starts where that one stops.
- `docs/research/dp_gibbs_results.md` — the reproduction numbers for the
  paper's §5.1 and §5.2, with the preset, seeds, chain count, $\hat R$ and
  wall clock. Read it before comparing anything here with the paper.

---

## 1. Scope: what the two modules do and do not do

| file | responsibility | does *not* do |
| --- | --- | --- |
| `src/dp_gibbs.py` | ingestion validation and the time transform, state and bookkeeping, the $\alpha$ quadrature grid, the three full conditionals, one sweep, chain storage, initialisation, the collapsed log-posterior, diagnostics | plotting, data generation, functionals |
| `src/dp_functionals.py` | the paper's §3 steps (i)–(iii) for posterior *and* prior, survival/CDF/density/hazard/quantile functionals on the original scale, two-population comparisons | drawing $\theta$ |

Two rules hold everywhere and are enforced by tests, not by convention:

- **No global RNG.** Every function that randomises takes `rng` or `seed`
  explicitly; a chain is replayable from one integer.
- **`matplotlib` only in `src/paper_examples.py`.** The test that imports
  `src.dp_gibbs` and asserts `matplotlib` is absent from `sys.modules` is what
  keeps the numerical modules importable on a headless machine.

The sampler keeps the paper's parameterisation $K_W(t\mid\alpha,\lambda) =
1-\exp(-t^\alpha/\lambda)$: $\lambda$ is *not* the standard Weibull scale and
*not* a rate. See `docs/research/weibull_lambda_definitions.md`.

---

## 2. The data flow

```text
 (time, event)                    prior constants
       |                                 |
       v                                 v
 make_surv_data()  ------------->  DPPrior  (frozen)
       |  t_tilde = 1 + t / t_max        |
       v                                 |
   SurvData (transformed scale)          |
       |                                 |
       +--> time_grid(t_max, 300)  (paper_examples; original scale)
       |                                 |
       v                                 v
   make_grid(data, M=256)  ----->  AlphaGrid  (T = time ** u, cached)
                                         |
                                         v
   init_state(data, prior, grid=...) -> DPState
                                         |
                       sweep() x (burn_in + thin * n_save)
                                         |
                                         v
                                    DPChains  (per chain)
                                         |
                                         v
   posterior_functionals(chains, t_grid) -> Functionals
                                         |        (B, M) arrays per functional)
                                         v
                     compare(fit_a, fit_b, t_grid) / quantiles / bands
                                         |
                                         v
                  paper_examples.py: figure recipes, CLI, presets
```

The transformed scale exists only inside `SurvData`, `DPState` and the two
numerical pieces of §4. `Functionals` and everything downstream are on the
**original** time scale, with the inverse map of
`src.dp_gibbs.inverse_map` applied exactly once.

---

## 3. The sampler (`src/dp_gibbs.py`)

### 3.1 Containers

- `SurvData` — times on the transformed scale plus the event flags. Built by
  `make_surv_data`, which validates lengths, finiteness, positivity and the
  event dtype. All-censored data is accepted by design (plan decision 6); the
  docstring warns that the posterior is then driven by the prior.
- `DPPrior` — frozen hyperparameters $(a_\nu,b_\nu,a_\gamma,b_\gamma,
  a_\phi,b_\phi,d)$. Never changes during a run.
- `DPSettings` — the knobs: `burn_in`, `n_save`, `thin`, `n_alpha`,
  `epsilon`, `n_chains`, `seed`, `preset`. Never model parameters.
- `DPState` — the latent state of one chain: per-observation
  `alpha`, `lam`, `s`, the cluster locations `alpha_star`, `lam_star`, the
  sizes `n_j`, and the scalars `nu`, `gamma`, `phi`. It validates the §2.1
  invariants at construction and on demand; `StateInvariantError` is an
  `AssertionError` subclass so a debug `assert` and a runtime check agree.
- `DPChains` — one chain's stored draws: `theta (B, n, 2)`, `nu`, `gamma`,
  `phi`, `n_star`, `log_post`, plus the `data`, `prior` and `settings` that
  produced them. Cluster labels are deliberately not stored: every functional
  is defined by the observation-level $\theta_i$ mixture, and a caller who
  needs `s` replays the chain from `settings.seed + b`.

Bookkeeping uses the deferred-compaction variant of plan §3.4:
`remove_observation(i)` decrements a cluster count but leaves `i` labelled,
so the urn weights can be formed against the pre-removal partition;
`compact()` performs the single relabel pass that drops empty clusters and
renumbers the rest. Invariants 1 and 4 of §2.1 hold at all times, 2 and 3 once
every removed observation is reassigned and `compact()` has run.

### 3.2 The time transform

The Appendix requires $t>1$ so that every $\log t$ in the $\alpha^-$ bound is
strictly positive. `make_surv_data` applies

$$ \tilde t = 1 + t / t_{\max}, \qquad t = t_{\max}(\tilde t - 1). $$

`transform_time` and `inverse_time` are the public pair. `inverse_map` maps a
whole functional back to the original scale — $S$ and $F$ unchanged, $f$ and
$h$ multiplied by $1/t_{\max}$, $\eta_p$ mapped through $t_{\max}(\eta-1)$.
Dropping the Jacobian is invisible in a plot of $S$ and changes every density
and hazard by a constant, which is why the map lives in one function and test
26 checks it explicitly.

**The prior-scale caveat.** The $b_\gamma,b_\phi$ values are elicited on the
*original* scale (§5 of the paper) but applied to $\tilde t$. The map is a
shift plus a scale, so the prior in force during fitting is not the elicited
prior: for the §5.1 constants the prior median of the population median is
about 0.61, not the elicited 1.1727. This is the paper's convention (plan
decision 2). It is stated in the module docstrings, verified by test 21b, and
repeated in the results note. Never compare a prior band with the elicitation
guess; compare it with `prior_functionals`.

### 3.3 The $\alpha$ grid and the two numerical pieces

`AlphaGrid` holds the unit-interval nodes $u_m=(m+1)/M$, the cache
$T_{im}=t_i^{u_m}$ and the log trapezoid weights. The grid is scaled by the
*current* $\phi$, $\alpha_m = \phi u_m$, so it stays inside $(0,\phi)$ as
$\phi$ moves, needs no cap, and $T$ (bounded by 2 because $\tilde t\in(1,2]$)
does not depend on $\phi$ — a sweep only needs $T^{\phi}$, one `pow` on an
$(n,M)$ array.

- `log_alpha_integrand_uncensored` / `log_alpha_integrand_censored` — Eq. A.2
  and Eq. A.3 in log space, per observation and grid node.
- `log_q0_uncensored` / `log_q0_censored` — the prior-predictive masses
  $q_0^o$ and $q_0^c$, each a log-space integral over the same grid
  (`log_integral_alpha` / `_log_integral_rows`).
- `draw_g0_conditional_alpha` — the continuous-piece draw: an $\alpha$ node
  from the grid by normalised weights, then $\lambda\sim\operatorname{InvGamma}
  (d,\gamma)$.

`M=256` is fixed by plan decision 3: measured relative quadrature error
$3\cdot10^{-5}$, asserted $\le10^{-4}$ by test 3. If accuracy ever fails, raise
$M$; do not restructure the grid.

### 3.4 Step (a): the Polya urn

`sample_step_a` loops over observations in data order and calls
`sample_step_a_for_obs` for each. The public per-observation entry point is
also the test seam for test 8.

For observation $i$ the function

1. removes $i$ from the urn (deferred compaction),
2. forms the log-weights of the remaining clusters and the prior-predictive
   mass `q_0` for that observation (Eq. A.1),
3. draws a category with `_sample_urn_category`,
4. either `assign_existing(i, j)` or `assign_new(i, alpha, lam)` with a fresh
   `draw_g0_conditional_alpha` draw,
5. compacts once at the end of the sweep.

The random-number stream is one uniform per observation plus, for every new
cluster, a grid uniform and an inverse-gamma variate. That order is part of
the specification: the golden regression (test 28) pins it.

### 3.5 Step (b): the cluster locations

`sweep` runs `sample_step_b`, which calls `refresh_cluster` once per cluster
in $j$ order:

- `refresh_cluster` draws $\lambda_j^\ast$ from the inverse-gamma conditional
  ($\gamma$ plus the uncensored cluster members, no quadrature) and then calls
  `refresh_cluster_alpha`;
- `refresh_cluster_alpha` draws $\alpha_j^\ast$ by the auxiliary-variable
  scheme of the Appendix: it introduces $u_j\sim\mathrm{Unif}$, targets
  $\alpha^{n_j^o}$ through the $\alpha^{-}/\alpha^{+}$ bounds and the
  auxiliary draws, and guards the empty-interval case by clamping to the
  midpoint, counting the event in `alpha_bound_guard_count()` rather than
  silently dropping it.

Censored members enter through the survival-only kernel; uncensored members
through the density. A cluster of only censored observations still moves — its
$\alpha^\ast$ draw has a well-defined target — which is why
`n_j^o\in\{0,1,2,5\}$ is an explicit test matrix for test 11.

### 3.6 Step (c): $\nu$, $\phi$, $\gamma$

`sample_step_c` runs, in this order:

1. `refresh_nu` — the Escobar–West two-Gamma mixture conditional: one beta,
   one uniform (to pick the Gamma component) and one gamma variate;
2. `refresh_phi` — the Pareto conditional, with lower bound
   $\max(b_\phi,\max_j\alpha_j^\ast)$ and shape $a_\phi+n^\ast$, read from the
   partition the earlier two steps left behind;
3. `refresh_gamma` — the inverse-gamma conditional given the current
   $\lambda_j^\ast$.

The order is the specification (plan §3.4), not a performance choice: $\phi$
must see the refreshed $\alpha^\ast$, and $\gamma$ must see the refreshed
$\lambda^\ast$.

### 3.7 One sweep and the chain driver

`sweep(state, data, prior, *, grid, rng)` runs (a), (b), (c), then returns
`log_post(state, data, prior)` evaluated on the resulting draw. The return
value is stored in `DPChains.log_post`; a caller that does not need it (a
burn-in sweep) ignores it.

`init_state` starts every observation as its own singleton cluster and draws
its $(\alpha_i,\lambda_i)$ from the continuous piece at the prior means
$\gamma_0=a_\gamma/b_\gamma$, $\phi_0=b_\phi$, with $\nu_0=a_\nu/b_\nu$ and
$\phi=\max(b_\phi,\max_i\alpha_i)$. Initialisation only produces a valid
state; the first sweep is the same call as every other sweep.

`sample_chain` is chain $b=0$ of a group and is seeded with `settings.seed`.
`sample_chains(data, prior, settings_list, max_workers=...)` runs every chain
of every settings entry through a `ProcessPoolExecutor` and returns them in
`(settings index, chain index)` order. Chain $b$ of a group uses
`settings.seed + b`; an unseeded group gets one OS-entropy base seed drawn in
the parent **before** submission and stored in the returned chains, so even an
unseeded run is replayable afterwards.

Parallelism is across chains only. There is no multiprocessing, numba or
cython *inside* a sweep, so the result does not depend on the worker count or
on scheduling order.

### 3.8 The collapsed log-posterior

`log_post` evaluates the exact unnormalised joint of plan §3.7: the
likelihood, the EPPF term, the base-measure terms for the distinct clusters
and the three hyperpriors. It is a diagnostic and the golden regression's
scale. It is **never** a model-comparison statistic: the target is
unnormalised and not comparable across models or even across $d$.

### 3.9 Diagnostics

- `r_hat(chains, name)` — split $\hat R$ (Gelman et al.) for `"nu"`,
  `"n_star"` or `"S"`; the `"S"` trace is the observation-level survival
  mixture at five diagnostic times, which is what catches a chain that has
  frozen its partition.
- `ess(chains, name)` — bulk effective sample size from the autocorrelation
  sum; chains need not have equal length.
- `autocorr_time(x)` — integrated autocorrelation time of one trace.

Test 31 checks all three against simulated AR(1) chains with a known
stationary variance and lag-1 correlation, and asserts that deliberately
disagreeing chains give $\hat R>1.2$ — the failure mode of a variance ratio
applied over the wrong axis is $\hat R\approx1$ for any input.

---

## 4. Functional inference (`src/dp_functionals.py`)

### 4.1 Settings and the storage container

`FunctionalSettings` carries `method` (`"exact"` or `"mc"`), `L` (the MC atom
count, ignored by the exact path), `atom_tol` (the pruning tolerance) and
`n_t_grid` (the conventional 300-point grid; the functions still take the grid
explicitly).

`Functionals` holds `t_grid (M,)` plus `S`, `F`, `f`, `h` as `(B, M)` arrays
on the original scale, and a `meta` dict with provenance. Full arrays are kept
rather than online means because percentile bands and quantile functionals
need the whole cross-section. The container validates shapes and bounds at the
storage boundary.

### 4.2 `exact` versus `mc`

For each stored draw $b$ both paths first draw a realisation
$G_b\sim[G\mid\theta_b,\nu_b,\gamma_b,\phi_b]$ with
`src.dp_weibull.sample_dp_posterior`, prune the stick-breaking tail with
`DPRealization.pruned(atom_tol)`, then:

- `method="exact"` evaluates the truncated weights directly —
  $S_b(t)=\sum_j w_j e^{-t^{\alpha_j}/\lambda_j}$ — the
  Rao–Blackwellised limit of the paper's $L\to\infty$ estimator, with no $L$
  noise. This is the path every figure uses.
- `method="mc"` reproduces the paper's steps (ii)–(iii) literally: draw $L$
  atoms from $G_b$ and average their kernels.

Both paths evaluate $S$, $f$ and $h$ from the **same** atom set, so the
hazard is a ratio of two averages over one atom set, never a ratio of two
independent estimates. Pruning is what keeps
$B\cdot J_{\text{eff}}\cdot|t_{\text{grid}}|$ at $10^4\cdot10\cdot300$ rather
than $200\times$ that (plan decision 7); it changes every functional by at most
`atom_tol`.

Entry points: `functionals_from_realizations` (already-drawn $G_b$, used by
the tests) and `posterior_functionals` (chains in, functionals out).
`functionals_from_mc_realizations` is the MC counterpart.

### 4.3 Posterior and prior functionals

`posterior_functionals(chains, t_grid, *, settings, seed)` stacks every chain
of the fit, uses the **common** truncation level of the group
(`truncation_number_for_posterior`, computed from the largest $\nu+n$ so that
every draw uses the same $J$), and returns the `Functionals` for the fit.

`prior_functionals(prior, t_grid, n_draws, *, data=None, t_max=None,
settings=None, epsilon=1e-4, seed=None)` is the prior counterpart: draw
$(\nu_b,\gamma_b,\phi_b)$ from their priors, draw
$G_b\sim\operatorname{DP}(\nu_bG_0)$, and evaluate the same functionals. Its
truncation level is the paper's Section 3 prior rule at the largest drawn
precision, computed once per run. These are the only correct reference for a
prior band of Figure 2 — see the caveat in §3.2.

### 4.4 Quantiles and two-population comparisons

`posterior_quantiles(chains, t_grid, p, *, settings, seed)` inverts each
draw's CDF exactly on its atom set, then maps back through the inverse time
transform (`quantiles_from_realizations` is the already-drawn variant).

`compare(fit_a, fit_b, t_grid, *, p=0.5)` pairs draw $b$ of A with draw $b$ of
B (valid for independent fits of equal length) and returns

- `ratio` — $h_B/h_A$ per draw and grid point (`inf` where $h_A=0<h_B$,
  `nan` where undefined);
- `difference` — $F_B-F_A$ per draw and grid point;
- `eta_a`, `eta_b`, `eta_difference` and `prob_eta_a_gt_eta_b` — the level-$p$
  quantiles from piecewise-linear grid inversion, and
  $\Pr(\eta_A>\eta_B\mid D)$.

It refuses two fits with different draw counts, different grids or different
`t_max`, a non-monotone CDF, a malformed $p$, and a fit whose *every* draw
leaves $p$ beyond the last grid point. The per-draw quantiles need both edges
of the grid handled explicitly, because a DP posterior has draws whose median
lies outside any finite grid:

- below the first grid point, the CDF is exactly zero at the origin, so the
  quantile is interpolated on `(0, 0) -> (t_grid[0], F[0])`;
- beyond the last grid point, the quantile is censored to `+inf` (it beats
  every finite median and ties with another censored draw, which counts as
  "not greater"); `meta["n_eta_above_grid"]` records how many draws that
  affected, and the Figure 5 histogram drops the non-finite differences.

The per-draw ratio is heavy-tailed where both hazards are small: the
reproduction note records its band widths honestly rather than smoothing them
away.

---

## 5. Assemble a run

A minimal posterior fit and its survival band:

```python
import numpy as np
from src.dp_gibbs import DPSettings, make_surv_data, sample_chains
from src.dp_functionals import FunctionalSettings, posterior_functionals
from src.paper_examples import make_prior

data = make_surv_data(time, event)        # original times and event flags
prior = make_prior(0)                     # Gamma(2, 0.9) on nu, §5 constants
settings = DPSettings(
    burn_in=5_000, n_save=10_000, thin=150,
    n_alpha=256, epsilon=1e-4, n_chains=4,
    seed=20240601, preset="paper",
)
chains = sample_chains(data, prior, [settings])

t_grid = np.linspace(0.02, 1.15 * data.t_max, 300)
fit = posterior_functionals(
    chains, t_grid, settings=FunctionalSettings(method="exact"), seed=20240601
)
mean_S = np.nanmean(fit.S, axis=0)
lo, hi = np.nanpercentile(fit.S, [2.5, 97.5], axis=0)
```

For a paper figure, do not reassemble this by hand: use
`src/paper_examples.py` (`example_5_1`, `example_5_2`, or the CLI
`python -m src.paper_examples --example 5.1 --preset quick`). That module owns
the presets, the prior band, the smoothing and the `plt` import.

---

## 6. Gotcha checklist

- **Two `lambda`s.** $\lambda$ is the paper's scale in
  $e^{-t^\alpha/\lambda}$, not the standard Weibull scale
  $c=\lambda^{1/\alpha}$, and not a rate.
- **`n` means three things.** $n$ observations, $J$ truncation level,
  $n^\ast$ clusters. The module docstrings and plan decision 9 fix the
  vocabulary; a bare `n` in new code is a bug.
- **Two time scales.** `SurvData.time` and everything in the sampler are
  $\tilde t=1+t/t_{\max}$; `Functionals` and the figures are the original
  scale. Mixing them changes densities and hazards by $1/t_{\max}$ and is
  invisible in $S$.
- **The prior-scale caveat is real.** Prior bands are drawn on the $\tilde t$
  scale. Compare them with `prior_functionals`, never with the elicitation
  guess.
- **The last stick-breaking weight is residual.** Weights sum to 1 because of
  that atom, not because of normalisation.
- **Posterior precision is $\nu+n$, not $\nu$.** That changes both the weights
  and the truncation level; use `truncation_number_for_posterior` for a fit.
- **A per-draw hazard ratio is heavy-tailed.** $h_B/h_A$ is a ratio of two
  small numbers; its 95% band is much wider than a ratio-of-bands. This is
  why the §5.2 acceptance test asserts a mean decline and an ordering
  probability rather than non-overlapping bands (plan amendment A4).
- **Deferred compaction.** Inside step (a), `s` may still label an observation
  that has been removed; `compact()` at the end of the sweep is what makes
  the §2.1 invariants hold. Do not read `n_j` mid-step and expect it to match
  `bincount(s)`.
- **`log_post` is not a marginal likelihood.** Never compare it across models
  or reports.
- **`alpha_bound_guard_count()` is a diagnostic.** A non-zero count means a
  cluster shape was clamped for one sweep by the floating-point guard; it is
  not an error, but a reported run should say what it was.

---

## 7. How correctness is established

The test suite of plan §7 is the argument. In the order it is worth reading:

| test | module | what it pins |
| --- | --- | --- |
| 3 | `test_q0_uncensored.py` | quadrature accuracy and order of the common-$J$ rule |
| 7 | `test_prior_matching.py` | the prior-guess constants against induced marginal medians |
| 8 | `test_step_a.py` | the urn step and its bookkeeping |
| 10, 11 | `test_step_b_*.py` | the scale refresh and the $\alpha^\ast$ auxiliaries |
| 12-14 | `test_step_c_*.py` | $\nu$, $\phi$, $\gamma$ conditionals |
| 15 | `test_exact_posterior.py` | exact posterior by enumeration on a tiny problem |
| 16 | `test_reference_sampler.py` | an independently derived sampler |
| 17 | `test_sbc.py` | simulation-based calibration (`-m slow`) |
| 19, 20 | `test_invariance.py`, `test_degenerate.py` | data-order invariance, degenerate data |
| 25, 26 | `test_functionals.py`, `test_quantiles.py` | functionals against closed forms and the inverse map |
| 28 | `test_golden.py` | fixed-seed regression of the whole chain |
| 30 | `test_paper_5_1.py`, `test_paper_5_2.py` | the §6.3 reproduction, opt-in `-m paper` |

Tests 15, 16, 17 are the ones that establish correctness rather than
plausibility: exact enumeration on a small problem, a second implementation,
and calibration of the sampler's own posterior. If one of them fails, do not
"fix" the reproduction tests.

---

## 8. See also

- `docs/research/dp_gibbs_plan.md` — the design and test plan, including the
  §6.3 acceptance table and its amendments A1–A4.
- `docs/research/dp_weibull_code_guide.md` — $G_0$, truncation and
  `DPRealization`.
- `docs/research/dp_gibbs_results.md` — the reproduction numbers.
- `docs/research/dirichlet_process_concepts.md` — partitions, the EPPF and
  what $\nu$ does.
- `docs/dp_weibull_explanation.md` — the model in the paper's notation.
- `docs/research/g0_uniform_inverse_gamma.md` — why Uniform in $\alpha$ and
  inverse-gamma in $\lambda$ make the conditionals what they are.
- `docs/research/weibull_lambda_definitions.md` — the two $\lambda$
  parameterisations.
