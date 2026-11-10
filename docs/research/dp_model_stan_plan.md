# Plan: Kottas (2002) Model Replication in Stan

Status: proposed, not started.
Scope: fit the **model** of Kottas (2002) §5 — the DP mixture of Weibulls with
the paper's priors, on the paper's two datasets — with Stan, and measure how far
the result is from the committed replication record. This is not an attempt to
reproduce the Appendix sampler; that sampler is a Gibbs sweep and has no Stan
expression.

Related documents:

- `docs/research/dp_gibbs_plan.md` — the algorithm of record (§2.2 full
  conditionals, §2.3 the time transform, §6.2 settings, §6.3 acceptance
  criteria, §10 decisions).
- `docs/research/dp_gibbs_replication_report.md` — the numbers this plan
  compares against; `output/replication-2026-11-08T02-26-53Z` is the tree.
- `docs/research/dp_model_numpyro_plan.md` — the sibling plan. Same model, same
  K-ladder, same agreement protocol; the two are meant to be read together and
  their §7 disagree only where the frameworks force it.
- `docs/research/dp_weibull_code_guide.md`, `docs/research/dp_gibbs_code_guide.md`
  — the primitives any port must agree with (`k_W`, `K_W`, `sample_g0`).

This document is a design and test plan. It contains no implementation.

## 0. The claim this plan is allowed to make

A previous assessment concluded that Stan can express the paper's model only
through a finite stick-breaking truncation, and that no Stan program can sample
the Appendix's collapsed Pólya urn. This plan is built around that conclusion,
so it fixes the fidelity ladder **before** any code exists:

| rung | claim | attainable in Stan |
| --- | --- | --- |
| F0 | *same draws* — reproduce the stored chains from the committed seeds | no; different RNG, different algorithm |
| F1 | *same posterior* — target `p(θ, s, ν, γ, φ \| D)` of model (5) | no; the DP has no finite representation |
| F2 | *same posterior under a K-truncated prior*, with the truncation error measured by a K-ladder | **yes — this is the target** |
| F3 | *same acceptance verdicts* on the §6.3 inequality suite | yes, gated on F2 |

F2 is a real claim and a weaker one than F1. Its content is: at the reported
values of `ν`, the truncation moves every reported summary by less than the
Monte Carlo error of the fit, and the K-ladder shows it. The plan's acceptance
gate is therefore **not** "the clauses pass"; it is "the clauses pass *and* the
K-ladder and the agreement protocol of §11 are satisfied". §11 exists because a
truncated fit can pass the clause suite for the wrong reason (§6.3).

## 1. Vocabulary: `K` is not `J`

The existing documents use `J` for a DP truncation and `n*` for the number of
occupied clusters. This plan introduces `K` and must not overload either.

| symbol | meaning | where it lives |
| --- | --- | --- |
| `n` | number of observations | data |
| `K` | number of stick-breaking components in the **prior** mixture | `stan` data block, new |
| `M` | `α`-quadrature nodes of the Gibbs sampler | not used in Stan at all |
| `J` | atoms retained when drawing `G \| θ` for functionals | `dp_weibull.truncation_number`, unchanged |
| `J_prior` | atoms retained when drawing `G` from the prior | `dp_weibull.truncation_number`, unchanged |
| `n*` | occupied clusters of a draw, `≤ K` | generated quantities |
| `n_j` | size of cluster `j` | generated quantities |
| `t̃` | `1 + t/max(t)` | `dp_gibbs.make_surv_data`, unchanged |
| `B` | stored draws per chain | settings |

`K` and `J` are different truncations of different objects and both must be
reported in every table: `K` is a **model** truncation (it changes the prior),
`J` is a **numerical** truncation of a functional estimator (controlled by
`ε = 1e-4` and shared with the Gibbs run, because `posterior_functionals` is
sampler-agnostic).

## 2. The target model, exactly

Data on the transformed scale (`dp_gibbs.make_surv_data`, plan §2.3):

```text
time[n]  > 1          float64, t̃_i
event[n] ∈ {0,1}      1 = uncensored (density), 0 = right-censored (survival)
```

Model (5) of the paper plus the Appendix's `d = 2`:

```text
t_i | α_i, λ_i  ~  K_W(·|α_i, λ_i):  f = α t^(α-1)/λ · exp(−t^α/λ),  S = exp(−t^α/λ)
(α_i, λ_i) | G  ~  G
G | ν, γ, φ     ~  DP(ν · G0)
G0              =  Uniform(α | 0, φ) · InvGamma(λ | d, γ),   d = 2

ν   ~ Gamma(a_ν, b_ν)      a_ν,b_ν ∈ {(2,0.9), (2,0.1), (3,0.05)}
γ   ~ Gamma(1, 0.232235)                      # shape, rate
φ   ~ Pareto(2, 1.77522)                      # density 2b²/φ³ on φ ≥ b
```

### 2.1 The three mapping identities that must be got right

Stan has no Weibull parameterised by a scale-like `λ` in `exp(−t^α/λ)`. The
identities below make the mapping exact; each one is worth a dedicated test
(§10, L1) because every one of them is a silent-error site.

1. **Scale-to-`λ`.** With `σ = λ^(1/α)`, Stan's
   `weibull_lpdf(t | α, σ) = log α + (α−1)log t − log λ − t^α/λ`, which is
   exactly the paper's kernel. Likewise `weibull_lccdf(z | α, σ) = −z^α/λ`.
   Conclusion: declare **`λ*` as the parameter** with its InvGamma prior, then
   pass `σ = λ*^(1/α*)` as a *transformed parameter*. Never declare `σ` as the
   parameter: the paper's prior is on `λ`, and moving it to `σ` needs the
   Jacobian `|dλ/dσ| = α σ^(α−1)`.
2. **InvGamma.** Stan's `inv_gamma(y | a, b)` is `b^a/Γ(a) · y^(−a−1) ·
   exp(−b/y)`, i.e. shape `a`, scale `b`. The paper's `IGamma(λ | d, γ)` is
   shape `d = 2`, scale `γ`. So `lambda_star[k] ~ inv_gamma(d, gamma)`.
3. **Pareto.** Stan's `pareto(y | y_min, α)` is `α y_min^α / y^(α+1)` on
   `y ≥ y_min`. The paper's `Pareto(φ | a_φ, b_φ)` is `a_φ b_φ^{a_φ} /
   φ^{a_φ+1}` on `φ ≥ b_φ`. So `phi ~ pareto(a_phi, b_phi)`.

All three are stated in `src/dp_weibull.py`'s conventions block (plan §2.2c);
reuse that text rather than re-deriving it.

## 3. Five places Stan cannot follow the paper

These are decisions of the framework, not implementation choices, and each one
is named in the report so a reader can see what was traded.

1. **No conditional (Gibbs) updates.** Step (a) of the Appendix is a
   sequential Pólya-urn allocation whose "new cluster" mass is
   `q0(t) = ∫ f(t|α,λ) G0(dα,dλ)`; `q0` is available in closed form up to a
   one-dimensional `α` integral. A Stan program is a single joint density, so
   the urn's *sequential* structure is inexpressible, and `integrate_1d` does
   not help — it would compute `q0` inside a density that still cannot allocate
   sequentially.
2. **No discrete parameters.** `s_{1:n}` must be marginalised. The marginalised
   form is a well-defined continuous-density target, so this is not a defect;
   it does mean the labels, and hence `n*`, are recovered post hoc (§7.6).
3. **No exact DP.** `K` components, residual assigned to the `K`-th atom
   (§7.2), with the truncation budget of §6.
4. **The `α`-grid disappears.** The paper (and our sampler) draws `α` on
   `α_m = φm/M`, `M = 256`, and integrates `q0` on the same grid. Stan draws
   `α` continuously. This is the one place the port is *closer* to the stated
   model than the paper's own implementation is (the measured `q0` error at
   `M = 256` is `2.9e-5`, plan §2.4), and simultaneously the one place where
   the draws can never be numerically comparable.
5. **`log_post` is not comparable.** The Gibbs chains store the collapsed joint
   of a different target. Do not carry it across; store `lp__` and nothing else,
   and never compare the two as a model-comparison statistic (plan decision 4).

## 4. Decisions taken (this plan)

| # | Decision | Consequence |
| --- | --- | --- |
| S1 | Target rung F2, gate on F3 (§0) | The report must carry a K-ladder and an agreement table; a bare "clauses pass" is not a result. |
| S2 | `K` ladder `{40, 80, 160}` for §5.1 and `{30, 60, 120}` for §5.2; `K = 80` is the headline fit | Cheap fits make a full ladder cheaper than one Gibbs fit. The clip in each ladder is chosen from §6, not by taste. |
| S3 | Residual-stick truncation: `β_1..β_{K−1} ~ Beta(1, ν)`, `π_K = 1 − Σ_{k<K} π_k` | This is the standard truncated-SB ("finite DP") prior. It is *not* the DP and must not be called one. |
| S4 | Parameterise `α*_k = φ·u_k`, `u_k ~ U(0,1)`, as default | Simplest to verify: the `1/φ` normalisation of `Uniform(α \| 0, φ)` is produced by the Jacobian and cannot be double-counted. The flat `φ`–`u` ridge is accepted and diagnosed (§7.2, §13). |
| S5 | `upper = φ` bound on `α*` as the *alternative* arm, decided by divergences and ESS | Removes the ridge; carries a genuine double-counting footgun (§7.2) that the L1 test must rule out. |
| S6 | `s` marginalised; `n*` reported both as a Rao-Blackwellised conditional distribution and as a post-hoc draw | The RB version is a *better* estimator of the paper's `n*` than the Gibbs chain's, so disagreements must be read carefully (§11). |
| S7 | New on-disk tree `output/stan-<stamp>/` with its own `MANIFEST.json` and a `sampler` field; **no** use of the `paper_artifacts` schema | The artifact schema encodes Gibbs knobs and a replay guarantee (`SCHEMA_VERSION = 1`, `DPSettings`, `code_sha` replay). A Stan tree must not masquerade as a replay. `paper_artifacts.atomic_savez` is reused as a plain write helper. |
| S8 | Adapter constructs `DPChains`/`Fit`/`DemoResult` **in memory**; `DPSettings` gains `components: int \| None = None` | Everything downstream (`posterior_functionals`, `compare`, `summarize_*`, `evaluate_acceptance`, figure recipes) is reused unchanged. `components=None` keeps the Gibbs report **byte-identical** (asserted). |
| S9 | Optional pixi feature `stan` + environment `stan`; default env and `pixi run test` stay dependency-free | Keeps plan decision 0 ("no new runtime dependencies") true for the default environment; the amendment is recorded in `dp_gibbs_plan.md` §10 (§9 step 10). |
| S10 | Pre-flight gate: compute the prior EPPF tail and residual mass **before** the run; abort if `K` is inadequate | §6. Truncation adequacy is a measurement, never an assumption. |

## 5. Non-goals

- No change to `src/dp_gibbs.py`, `src/dp_functionals.py`, `src/dp_weibull.py`,
  the prior constants, the time transform, the data, the presets or the §6.3
  thresholds. Anything that fails is a finding in the Stan report.
- No SVI, no variational approximation, no Laplace approximation: the
  acceptance criteria are calibrated interval widths (§11.3).
- No covariates, no interval/left censoring, no §5.3 (the liver data are not in
  the paper) — same scope as `dp_gibbs_plan.md` §1.
- No model comparison, no LOO/WAIC, no Bayes factors.
- No figure-level claims of agreement beyond what §11 measures.
- No writes under `docs/research/figures/`; the Stan report's figures go under
  the git-ignored `output/stan-<stamp>/`.

## 6. Choosing `K`: the truncation budget

`K` is not a tuning knob, it is a prior-restriction size. Two measured facts
fix it.

**(a) The prior wants more atoms than the posterior does.** For a Sethuraman
`Beta(1, ν)` stick-breaking, the expected mass beyond `K` atoms is
`(ν/(1+ν))^K`. `E[n* | ν] = ν(ψ(ν+n) − ψ(ν))` for the EPPF at `n = 200`:

| `ν` | 2.22 | 4 | 12 | 20 | 60 |
| --- | --- | --- | --- | --- | --- |
| `E[n* \| ν]` | 10.5 | 16.2 | 34.9 | 48.4 | 88.4 |
| residual mass at `K = 60` | 3e-14 | 1.5e-6 | 8.2e-3 | 5.4e-2 | **3.7e-1** |
| residual mass at `K = 160` | 0 | 0 | 2.0e-11 | 2.8e-6 | 1.7e-2 |
| `K` for residual `≤ 1e-4` | 25 | 42 | 116 | 189 | **558** |

The three `ν` priors have prior means 2.22 / 20 / 60 and prior
`E[n*]` = 10.1 / 44.7 / 82.8 (computed by quadrature over the prior, not from
the chains). The data pull `ν` down hard — posterior means 1.50 / 2.71 / 3.98,
`E[n*]` = 7.02 / 9.88 / 12.78, 99th percentile of `ν` = 4.25 / 8.35 / 11.68 —
so the *posterior* is comfortable at `K = 80` while the *prior* under
`Gamma(3, 0.05)` is not representable below `K ≈ 558`.

**(b) Consequence, stated in the report.** For the two narrower priors the
ladder can demonstrate `1e-4`-level adequacy. For `Gamma(3, 0.05)` it cannot,
and the report says so: the `K = 160` fit leaves ~1.7 % of the prior stick mass
beyond the last atom at `ν = 60`, so Figure 1's *right tail* is where the
truncation lives and no claim of `F1`-fidelity is made there. This is
pre-registered, not discovered afterwards.

**(c) Pre-flight gate (step 6 of §9).** A `scripts/stan_preflight.py` run
prints, for each fit: `E[n*|ν]` and `Pr(n* > K)` under the EPPF at the prior
mean, and the residual mass at the posterior 99th percentile of `ν`. The ladder
is chosen from these numbers; the table is copied into the report. No fitting
starts until the gate prints a stated adequacy verdict per fit.

**(d) `K` and dimension.** At `K = 80` the model has `3K + 2 = 242`
unconstrained parameters for `n = 200` observations. Everything in §7.7 exists
because of that ratio.

## 7. Stan program design

### 7.1 Data block

```text
int<lower=2> n;             int<lower=2> K;
vector<lower=1>[n] time;    array[n] int<lower=0,1> event;
int<lower=1> d;             # always 2
real a_nu; real b_nu; real a_gamma; real b_gamma; real a_phi; real b_phi;
int<lower=0,1> prior_only;  # 1 = skip the likelihood (prior simulation, L0/SBC)
```

`time` is `t̃`; the driver does the transform with `make_surv_data` and the
adapter maps functionals back with `data.t_max`, so the transform is never
reimplemented. `K ≥ 2` because `K = 1` has no stick.

### 7.2 Parameters, and the two `α*` parameterisations

Default (S4):

```text
array[K] beta_raw in (0,1)      # beta_raw[k] ~ beta(1, nu)
array[K] u       in (0,1)       # alpha_star[k] = phi * u[k]
vector<lower=0>[K] lambda_star  # ~ inv_gamma(d, gamma)
real<lower=0> nu; real<lower=0> gamma; real<lower=b_phi> phi;
```

`α*_k = φ u_k` with `u_k ~ U(0,1)` produces the paper's `U(α|0,φ)` exactly:
the change of variable carries the `1/φ`, so the prior density on `u` is 1 and
no explicit `1/φ` may be added. This is the reparameterisation that cannot be
got wrong, at the price of a flat direction: the likelihood depends on `u` only
through the product `φu`, so for each `k` the ray `{φu_k = const}` is
likelihood-flat and is broken only by the Pareto prior and the `u < 1`
constraint. **All reported quantities are constant along that ridge** (they are
functions of `α*`, not of `φ` alone), so the ridge costs ESS, not correctness —
which is why it is diagnosed (§7.7, §13) and not "fixed" by default.

Alternative (S5), for the ridge arm:

```text
array[K] alpha_star with upper = phi    # declares U(0, phi) via the transform
```

Stan's bounded transform supplies the `1/φ` from the Jacobian, so an explicit
`alpha_star ~ uniform(0, phi)` must **not** be written: it would add `−log φ` a
second time. That is the one footgun in this file, and L1 tests it by
comparing the two arms' log densities at fixed parameters (`K = 3`, hand-computed
prior), where the two must agree exactly when both are written correctly.

### 7.3 Transformed parameters

```text
pi[1] = beta_raw[1];  pi[k] = beta_raw[k] * prod_{j<k}(1 - beta_raw[j]);
pi[K] = 1 - sum_{k<K} pi[k];
alpha_star[k] = phi * u[k];            # or the bounded declaration of 7.2
sigma[k] = lambda_star[k]^(1 / alpha_star[k]);
log_pi[k] = log(pi[k]);
```

Assert `fabs(sum(pi) - 1) < 1e-9` and `min(alpha_star) > 0` in a debug path
(`#include`-free: a `reject` in `transformed parameters` at print level 0 in the
test build; the production build relies on the L1 test because a runtime
`reject` would poison NUTS).

### 7.4 Likelihood: marginalised, censored, vectorised

For each observation and component,

```text
ll[i,k] = event[i] ? weibull_lpdf(time[i] | alpha_star[k], sigma[k])
                   : weibull_lccdf(time[i] | alpha_star[k], sigma[k]);
target += log_sum_exp(log_pi[k] + ll[i,k] for k);
```

Notes that belong in the file's docstring, not in the reader's head:

- `log_sum_exp` of a `K`-vector, not `log_mix` (which is two-component). The
  loop is over `k` and vectorised over `i`; that is the shape NUTS needs.
- With `prior_only = 1` the `log_sum_exp` term is skipped entirely.
- `log_pi` must be computed before the likelihood so the compiler shares the
  common subexpression; at `K = 80`, `n = 200` this is a `200×80` matrix per
  gradient evaluation.
- Do not use `weibull_cdf` for the censored term: `1 − F` cancels in the tail.

### 7.5 Priors

```text
beta_raw[k] ~ beta(1, nu);     nu ~ gamma(a_nu, b_nu);
lambda_star[k] ~ inv_gamma(d, gamma);   gamma ~ gamma(a_gamma, b_gamma);
phi ~ pareto(a_phi, b_phi);
# alpha_star: from 7.2 — either the bound alone, or nothing plus phi*u
```

No Jacobian terms are written by hand anywhere: every prior is on a declared
parameter, and the deterministic maps (`α* = φu`, `σ = λ^{1/α}`) are in
`transformed parameters`. If a hand-written `target +=` Jacobian appears in a
review, the file is wrong.

### 7.6 Generated quantities

Small in output, complete in content — the CSV must let the adapter rebuild
`θ` without re-deriving anything:

```text
array[n] int s;                       # one conditional categorical draw per obs
vector[K] pi_out; vector[K] alpha_out; vector[K] lambda_out;
real n_star_draw;                     # #distinct(s)
real n_star_mean;                     # Rao-Blackwellised, see below
simplex[K+1] n_star_pmf;              # Pr(n* = m | params), m = 0..K
real log_dens_obs[n]                  # optional; diagnostics only
```

`n_star_mean` and `n_star_pmf` need no RNG: given `π`, the number of occupied
components is the Poisson-binomial sum of independent `Bernoulli(p_k)` with
`p_k = 1 − (1 − π_k)^n`, so

```text
n_star_pmf[m] = coefficient of x^m in prod_k (1 - p_k + p_k x)   # O(K²)
n_star_mean   = sum_k p_k
Pr(3 <= n* <= 15) = sum_{m=3}^{15} n_star_pmf[m]
```

and the acceptance clause uses the *draw-averaged* `Pr`, which is a
Rao-Blackwellised version of the paper's histogram. `n_star_draw` is kept as a
cross-check between the two (`E[n_star_draw] ≈ E[n_star_mean]` to MC error,
test L2.3). This is the one place the Stan route is *sharper* than the Gibbs
route, and it must be described that way in the report so a smaller `n*`
variance is not mistaken for a truncation artefact.

`θ` is **not** written per observation: the adapter reconstructs
`theta[b,i] = (alpha_out[s_i], lambda_out[s_i])`. At `n = 200`, `K = 80`,
`B = 2000`, that is 400 columns saved per draw and one mapping site to test
(L2.2).

### 7.7 Numerics and sampling control

- `adapt_delta = 0.95` default, 4 chains; a fit with any divergence is **not
  accepted** — raise to 0.99, then re-run, and record the escalation.
- `max_treedepth = 12`; record the treedepth histogram; a fit saturating it is
  reported as a mixing failure, not quietly accepted.
- Init: `u ~ U(0.2, 0.8)`, `lambda_star` at `gamma/d` with `gamma` at its prior
  mean, `beta_raw ~ U(0.2, 0.8)`, `φ = max(b_φ, max α*_init)`, `ν` at its prior
  mean. A single documented init; four chains from it, not four hand-tuned
  inits (the paper's sampler has none either).
- Non-centred sticks (`β = 1 − exp(−e)`, `e ~ Exp(ν)`) are the documented
  fallback if the `β` trace shows a funnel at large `ν`; it is the same prior,
  so switching is not a model change. Record which was used.
- Diagnostics are computed on **label-invariant quantities only**: `ν`, `γ`,
  `φ`, `n*`, `S(t)` at the five monitored times, `η_p`. Per-atom `R̂` is
  meaningless because empty and near-empty atoms are exchangeable
  (§13.3), and reporting one would be a false alarm generator.

## 8. Files and modules

```text
stan/
  kottas_dp_weibull.stan        # model + generated quantities (§7)
  log_density_check.stan        # fixed-parameter target() oracle, tests only
  README.md                     # K/β policy, the two α* parameterisations, build
src/
  sampler_export.py             # SHARED with the NumPyro plan: draws -> DPChains/Fit/DemoResult
  stan_draws.py                 # cmdstanpy CSV -> arrays; output/stan-<stamp> read/write
  stan_examples.py              # example_5_1 / example_5_2 equivalents + CLI
scripts/
  stan_preflight.py             # §6(c) truncation gate
  stan_ladder.sh                # K-ladder over both sections
  stan_verify.sh                # re-render report + agreement table without fitting
tests/
  test_stan_model.py            # L1: log density, priors, mapping identities
  test_sampler_export.py        # L2.2 + DPSettings.components contract
  test_stan_agreement.py        # L3 + §11: z-scores, K-trend, no-free-lunch
```

`src/sampler_export.py` is the only module both plans need. Whichever plan is
implemented first owns it; the second reuses it verbatim. Its public surface:

```python
@dataclass(frozen=True)
class ExportProvenance:      # sampler, components K, warmup, draws, adapt_delta,
    ...                      # max_treedepth, divergences, treedepth saturation,
                             # residual_mass, code_sha, package versions
def build_chains(draws, *, data: SurvData, prior: DPPrior,
                 provenance: ExportProvenance) -> list[DPChains]
def build_fit(chains, *, data, prior, t_grid, band: Functionals | None) -> Fit
def build_result(fits, *, label: str, meta) -> DemoResult
```

with three hard requirements:

1. `theta` has shape `(B, n, 2)` and `alpha/lambda/s/n_j` invariants 1–2 of
   `dp_gibbs_plan.md` §2.1 hold on every exported draw, asserted at the adapter
   boundary (this is the analogue of `DPChains.__post_init__`).
2. `DPSettings.components` is set to `K` and `preset` to
   `"stan-hmc-k<K>"`. When `components is not None`, `paper_report._settings_payload`
   prints `components` and omits `n_alpha`/`thin` rather than printing Gibbs
   knobs that do not exist. Default `components = None` must leave the existing
   report byte-identical — asserted by re-rendering from
   `output/replication-2026-11-08T02-26-53Z` and comparing bytes.
3. The prior band is **loaded from the existing artifacts**
   (`<run_root>/5.1/draws/<fit>/prior-band.npz`, and recomputed with the
   documented seed for the two fits that have none) rather than redrawn. The
   band is sampler-independent; loading it removes one gratuitous source of
   difference from Figure 2's comparison.

## 9. Steps

Every step runs `pixi run test`, `pixi run lint`, `pixi run typecheck`, updates
`docs/research/dp_weibull_code_guide.md` and `dp_gibbs_code_guide.md` where a
module changes, and commits the message in column 2 verbatim. Artifacts under
`output/` are never committed. Steps 1–8 land before any long fit.

| # | Commit message | Logical unit | Key deliverables | Tests | Plan |
| --- | --- | --- | --- | --- | --- |
| 1 | chore: add optional stan environment | Toolchain | `[feature.stan]` with `cmdstanpy`, `[environments] stan = { features = ["stan"], solve-group = "stan" }`, `stan-install` task (CmdStan version pinned, `~/.cmdstan`), lock update; `pytest.importorskip("cmdstanpy")` convention documented | Smoke | S9 |
| 2 | feat: add DPSettings components field | Honest settings record | `DPSettings.components: int \| None = None`; `_settings_payload` emits it and drops `n_alpha`/`thin` when set; docstring naming `K` vs `J` | Unit, byte-identical report regression | S8 |
| 3 | test: add the paper's Weibull primitives as a log-density oracle | Numerical contract | `stan/log_density_check.stan` (fixed params, `target()` only) + `tests/test_stan_model.py` comparing against `dp_weibull.k_W`/`weibull_sf` and a hand-written prior; covers both `α*` parameterisations, `K = 3`, censored and uncensored, all-censored | Unit | §2.1, S5 |
| 4 | feat: add the truncated stick-breaking model | Model file | `stan/kottas_dp_weibull.stan`: data/parameters/transformed/likelihood/priors per §7.1–7.5, `prior_only` switch; `stan/README.md` | Unit (L1), prior-recovery smoke at `n = 20` | §7 |
| 5 | feat: add generated quantities for labels and n-star | Recovery | `s`, `pi_out`, `alpha_out`, `lambda_out`, `n_star_draw`, `n_star_pmf`, `log_dens_obs`; `tests/test_stan_model.py` checks the Poisson-binomial against brute-force enumeration at `K = 4`, and `E[n_star_draw] ≈ n_star_mean` | Unit (L2.3, L2.4) | S6 |
| 6 | feat: add the Stan draws exporter | Adapter | `src/stan_draws.py` (CSV → arrays; `output/stan-<stamp>/` write/read with its own `MANIFEST.json` and `sampler` field; `atomic_savez` reused) and `src/sampler_export.py` (§8 surface); `tests/test_sampler_export.py` incl. the `components` contract and a `quick` round-trip that feeds `r_hat()`/`ess()` | Unit, Integration | S7, S8 |
| 7 | feat: add the Stan fitting driver and CLI | Driver | `src/stan_examples.py`: `example_5_1`/`example_5_2` mirroring `paper_examples` (same `data_5_1`/`data_5_2`, same `time_grid`, same `prior_band` load), CLI `--section/--K/--chains/--warmup/--draws/--adapt-delta/--seed/--save-dir`, per-fit save before the next fit; `tests/test_stan_cli.py` | Unit, Smoke | §8 |
| 8 | test: add the Stan agreement and K-ladder tests | Agreement | `tests/test_stan_agreement.py` (§11: z-score table, K-trend, no-free-lunch guard) and `scripts/stan_preflight.py` (§6c) | Integration (`paper`) | S1, S2, S10 |
| 9 | chore: add the Stan ladder and verification scripts | Run scripts | `scripts/stan_ladder.sh` (preflight → 3 `K` × 7 fits → report → agreement), `scripts/stan_verify.sh` (no fitting: re-render + re-run agreement from saved draws), pixi tasks `stan-5.2`, `stan-5.1`, `stan-ladder`, `stan-verify` | Smoke | §12 |
| 10 | docs: add the Stan replication report and the plan amendment | Report | `docs/research/dp_model_stan_report.md` (generated tables + the §14 wording); one-line pointer in `dp_gibbs_plan.md` §10 as its next free decision number ("a Stan/NumPyro model-replication arm exists; it is not the record") | — | §14 |

## 10. Validation ladder

The Gibbs plan's oracles (§7) are all in-house. This ladder is built so that a
port can be wrong in *one* place and caught.

**L0 — prior simulation (`prior_only = 1`).** Draw `(ν, γ, φ, α*, λ*, π, s)`
from the prior and check: simulated `S(·)` equals
`dp_weibull.sample_dp_prior` functionals at the same parameters; `Pr(n*)`
matches the EPPF-computed distribution at `K` large; `α* ~ U(0,φ)` by a KS
test. Catches the `1/φ` normalisation error (both arms), the stick
normalisation, and the `upper=φ` double count.

**L1 — log density at fixed parameters (the golden test).** With everything
fixed and `K = 3`: (i) the model's `target()` equals a hand-computed
`log p(θ*, π, ν, γ, φ) + Σ log p(t_i | ·)` assembled in NumPy from
`src/dp_weibull.py`; (ii) both `α*` arms agree; (iii) censored terms equal
`log K_W(z)` to 1e-12. `n = 1`, all-censored, and `n_j = 1` are cases, not
afterthoughts. Run in both `prior_only` states.

**L2 — internal consistency of the recovered objects.**
1. `theta[b,i] = (alpha_out[s_i], lambda_out[s_i])` and `#distinct(theta) = n_star_draw`.
2. `Σ_j n_j = n` for the `bincount` of `s`.
3. `E[n_star_draw]` equals `E[n_star_mean]` to MC error over 4 chains.
4. `π` from the CSV satisfies `Σπ = 1`, `π_k > 0`, and matches the
   stick-breaking recomputation from `beta_raw`.
5. Exported `DPChains` pass `dp_gibbs_plan.md` §2.1 invariants 1–2.

**L3 — posterior agreement with the committed Gibbs run (§11).** The exchange
rate is `z`, not "looks the same".

**L4 — SBC (opt-in, `slow`).** `n = 20`, `K = 40`, 100 replications: rank
histograms for `ν`, `γ`, `φ`, `n*`, `S(1.2)`. Uniformity to within the usual
SBC envelope. Reuses `tests/fixtures/sbc_seeds.json` conventions so the
sampling design matches `tests/test_sbc.py`; this is the only *frequentist*
check of the truncated model's calibration.

## 11. Agreement protocol

§0 makes F2 the target, so "the clauses pass" cannot be the gate. Three checks,
all pre-registered.

### 11.1 The z-table

For each of the 12 (§5.1) / 8 (§5.2) label-invariant summary rows, against the
committed run (40 000 draws, `J_common` 328–3157):

```text
se_fw   = sd_fw / sqrt(ESS_fw)          # ess() from dp_gibbs, on exported chains
se_gibbs= sd_gibbs / sqrt(ESS_gibbs)    # ess() on the committed chains
z       = (mean_fw - mean_gibbs) / sqrt(se_fw^2 + se_gibbs^2)
```

`se_gibbs` is small (≥ 10 000 ESS on `ν`/`n*`), so `z` is dominated by the Stan
side. Report `z` for every row and every `K` in the ladder.

### 11.2 The two row classes

| class | rows | expected |
| --- | --- | --- |
| K-insensitive | `S(1.1727)`, `∫f`, the 3 hazard change points, `Pr(S_A>S_B)` pointwise, `Pr(η_A>η_B)`, the mean hazard ratio at 5/25 | `\|z\| ≤ 2.5` at the headline `K`; no trend in `K` |
| K-sensitive | mean and 99th percentile of `ν`, mean and histogram of `n*`, `Pr(3 ≤ n* ≤ 15)`, band widths at `t = 4.5` and `t = 25`, `Pr(F_B>F_A)` mean | monotone `z → 0` as `K` rises; `\|z\| ≤ 2.5` at the largest `K` |

A K-sensitive row that does **not** trend toward zero is a *model or mapping*
bug, not truncation, and blocks the report. A K-insensitive row with `|z| > 2.5`
is a bug at any `K`.

### 11.3 The no-free-lunch guard

The §6.3 suite rewards dispersion-sensitive facts: "prior band ⊋ posterior
band", "band at 4.5 ≥ 1.5 × band at 2.5", `Pr(3≤n*≤15) ≥ 0.70`. Truncation
*shrinks* the posterior, so it makes those clauses **easier**. Therefore every
dispersion clause must additionally satisfy: the Stan width is **within the
MC-error envelope of the Gibbs width** (same `z` criterion), not merely on the
passing side. A fit that passes a width clause while the width moves the wrong
way by more than `2.5` combined SE is reported as a truncation failure even
though `evaluate_acceptance` returns green.

### 11.4 Output

`output/stan-<stamp>/agreement.json` + a markdown table with columns
`row | gibbs | stan(K=40) | stan(K=80) | stan(K=160) | z(80) | trend`. This is
the plan's central table; the §6.3 verdicts are an appendix to it.

## 12. Runbook and effort

```bash
pixi run -e stan stan-install                  # one-off, CmdStan in ~/.cmdstan
pixi run -e stan python scripts/stan_preflight.py --both      # §6c gate
pixi run -e stan scripts/stan_ladder.sh 5.2                   # ~1 h
pixi run -e stan scripts/stan_ladder.sh 5.1                   # ~4-8 h
pixi run -e stan scripts/stan_verify.sh output/stan-<stamp>   # no fitting
```

Planning estimates (to be replaced by measured wall clock in the report, as the
Gibbs run's §2 table does):

| item | estimate | basis |
| --- | --- | --- |
| sampler build + L1/L2 | 2 days | model file + oracle harness |
| one §5.1 fit, `K = 80`, 4 chains × 2 000 + 2 000 | 20–60 min | 242 params, `n·K = 16 000` terms/gradient, geometry-limited |
| one §5.2 fit, `K = 60`, `n = 20` | 3–10 min | 10× fewer likelihood terms |
| full ladder (3 `K` × 7 fits) | 4–12 h | 21 fits, 4 in parallel |
| agreement + report + verification | 1 day | R2-style: no fitting in the verify path |

Compare with the Gibbs record: 13 h 29 m for the 7 headline fits, 1 505 000
sweeps each. The ladder is affordable *because* the fits are short; that is the
trade the plan makes.

## 13. Risks

1. **The `φ`–`u` ridge (S4).** Flat in the likelihood, pinned by the Pareto
   prior and `u < 1`. Costs ESS in `φ` and per-atom `α*`; harmless for
   label-invariant summaries. *Mitigation*: S5 arm, non-centred sticks, and
   §7.7's "no per-atom `R̂`" rule. *Detection*: divergence counts and the `φ`
   ESS in the diagnostics table.
2. **`K = 160` is a 482-parameter model for 200 observations.** If divergences
   persist at `adapt_delta = 0.99`, the ladder's top rung is reported as a
   mixing failure and the headline stays at `K = 80` — the ladder is
   informative even when its top rung fails.
3. **Exchangeable empty atoms.** With `n* ≈ 10` and `K = 80`, seventy atoms are
   prior-only and exchangeable; the posterior has a symmetric ridge, not
   separate modes. NUTS drifts along it. This is why §7.7 restricts
   diagnostics to label-invariant quantities and why the plan must *never*
   report per-atom `R̂` as evidence of failure.
4. **Label switching is not the problem `bayes.md` says it is.** The reported
   quantities are measure-level functionals; `S`, `f`, `h`, `n*` are invariant.
   The plan therefore does not spend effort on relabelling, and would reject a
   "fix" that orders atoms, because ordering would change the prior on `π`.
5. **The wide `ν` prior cannot be represented** (`K ≈ 558` for `1e-4` at
   `ν = 60`). Pre-registered in §6(b); the report says F2 is demonstrated for
   two of the three priors and bounded (not achieved) for the third.
6. **Rao-Blackwellised `n*` looks better than the paper's.** A reader may read
   a narrower `n*` histogram as truncation bias. §7.6 and §11.2 must state
   which estimator produced each row; both are exported.
7. **Toolchain drift.** CmdStan version, `cmdstanpy` version and the C++
   compiler are recorded in `logs/env.txt`; a fit is not reproducible across
   CmdStan versions even at fixed seeds, and the report must not claim replay
   (S7).

## 14. What the report will say

The committed deliverable (`docs/research/dp_model_stan_report.md`) is required
to contain, in this order:

1. The fidelity statement: "Stan reproduces the *Kottas (2002) model* under a
   `K`-truncated stick-breaking prior. It does not reproduce the Appendix
   sampler, and it does not target the untruncated DP posterior."
2. The §6 pre-flight table and the `K`-vs-`E[n*]`/residual-mass table, with the
   `Gamma(3, 0.05)` caveat.
3. The §11.1 z-table and the §11.4 trend column.
4. The §11.3 no-free-lunch table (widths, not just verdicts).
5. The §6.3 clause verdicts, marked "F3 gate, subordinate to 3 and 4".
6. Diagnostics: divergences, treedepth, `R̂`/ESS on label-invariant quantities
   only, `φ` ESS (the ridge), and the escalations of `adapt_delta`.
7. Wall clock and environment.
8. An explicit non-claim: nothing here replaces
   `dp_gibbs_replication_report.md`, and no figure here is byte-comparable with
   the record's.
