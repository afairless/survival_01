# Plan: Kottas (2002) Model Replication in NumPyro

Status: proposed, not started.
Scope: fit the **model** of Kottas (2002) §5 — the DP mixture of Weibulls with
the paper's priors, on the paper's two datasets — with NumPyro, and measure how
far the result is from the committed replication record. This is not an attempt
to reproduce the Appendix sampler, and not an SVI exercise: the deliverable is a
calibrated posterior, because the paper's acceptance criteria are calibrated
interval widths.

Related documents:

- `docs/research/dp_gibbs_plan.md` — the algorithm of record (§2.2 full
  conditionals, §2.3 the time transform, §6.2 settings, §6.3 acceptance
  criteria, §10 decisions).
- `docs/research/dp_gibbs_replication_report.md` — the numbers this plan
  compares against; `output/replication-2026-11-08T02-26-53Z` is the tree.
- `docs/research/dp_model_stan_plan.md` — the sibling plan. Same model, same
  `K`-ladder, same agreement protocol, same shared adapter module. Read the two
  together: §11 below is the same protocol, and only §7 differs, because only
  §7 is about the framework.
- `docs/research/dp_weibull_code_guide.md`, `docs/research/dp_gibbs_code_guide.md`
  — the primitives any port must agree with (`k_W`, `K_W`, `sample_g0`).

This document is a design and test plan. It contains no implementation.

## 0. The claim this plan is allowed to make

Identical ladder to the Stan plan, restated because the claim is what the
document is for:

| rung | claim | attainable in NumPyro |
| --- | --- | --- |
| F0 | *same draws* — reproduce the stored chains from the committed seeds | no; different RNG, different algorithm |
| F1 | *same posterior* — target `p(θ, s, ν, γ, φ \| D)` of model (5) | no; the DP has no finite representation |
| F2 | *same posterior under a K-truncated prior*, with the truncation error measured by a K-ladder | **yes — this is the target** |
| F3 | *same acceptance verdicts* on the §6.3 inequality suite | yes, gated on F2 |

The plan's gate is **not** "the clauses pass". It is "the clauses pass *and* the
`K`-ladder and the agreement protocol of §11 are satisfied", because a truncated
fit can pass the clause suite for the wrong reason (§11.3).

NumPyro has one capability the Stan plan cannot match and this plan therefore
buys: **two algorithmically different samplers of the same truncated model**
(marginalised NUTS and explicit-label `DiscreteHMCGibbs`). Their agreement is an
internal, model-level check that needs no external reference — it catches
mapping and marginalisation errors before the comparison with the Gibbs record
is even attempted.

## 1. Vocabulary: `K` is not `J`

| symbol | meaning | where it lives |
| --- | --- | --- |
| `n` | number of observations | data |
| `K` | number of stick-breaking components in the **prior** mixture | model argument, new |
| `M` | `α`-quadrature nodes of the Gibbs sampler | not used in NumPyro at all |
| `J` | atoms retained when drawing `G \| θ` for functionals | `dp_weibull.truncation_number`, unchanged |
| `J_prior` | atoms retained when drawing `G` from the prior | unchanged |
| `n*` | occupied clusters of a draw, `≤ K` | kernel V1: Poisson-binomial; V2: `#distinct(s)` |
| `t̃` | `1 + t/max(t)` | `dp_gibbs.make_surv_data`, unchanged |
| `B` | stored draws per chain | settings |

`K` is a **model** truncation; `J` is a **numerical** truncation of the
functional estimator, controlled by `ε = 1e-4`, and is shared with the Gibbs run
because `posterior_functionals` is sampler-agnostic. Both belong in every table.

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

### 2.1 The four mapping identities

Each is a silent-error site and each gets its own test (§10, L1). Three are the
same as in the Stan plan; the fourth is NumPyro-specific.

1. **Scale-to-`λ`.** `dist.Weibull(scale=σ, concentration=α)` has
   `log_prob = log α − log σ + (α−1)log(t/σ) − (t/σ)^α` and
   `log_survival` reduced to `−(t/σ)^α` (write it by hand, see 7.4). With
   `σ = λ^(1/α)` this is exactly the paper's kernel with no Jacobian, **provided
   `λ` is the sampled site**. Do not sample `σ` and put the paper's prior on it.
2. **InvGamma — this is the NumPyro trap.** `dist.InverseGamma(concentration,
   rate)` is parameterised by the **rate of the underlying Gamma**, i.e. density
   `∝ λ^(−a−1) exp(−β/λ)` with `β = rate`. The paper's `IGamma(λ | d, γ)` is
   shape `d`, *scale* `γ`, so the call is `dist.InverseGamma(d, rate=gamma)`.
   Writing `scale=gamma` compiles, runs, and biases every `λ`.
3. **Pareto.** `dist.Pareto(scale, alpha)` is `α s^α / x^(α+1)` on `x ≥ s`, so
   `dist.Pareto(b_phi, a_phi)`.
4. **`α*` reparameterisation and its Jacobian.** With `α* = φ·u`, `u ~ U(0,1)`,
   the paper's `U(α|0,φ)` is obtained exactly, because `dα/du = φ` cancels the
   `1/φ`. If instead `α*` is sampled directly on `(0, φ)`, the `1/φ` must appear
   explicitly in the joint (`target += −log(φ)` per atom) and this is where
   NumPyro's hand-written `log_prob` will be wrong if nobody looks.

## 3. Seven places NumPyro cannot follow the paper

1. **No conditional (Gibbs) updates in NUTS.** The Appendix step (a) is a
   sequential Pólya-urn allocation with "new cluster" mass
   `q0(t) = ∫ f(t|α,λ) G0(dα,dλ)`. A NUTS model is one joint density; the urn's
   sequential structure is inexpressible.
2. **NUTS carries no discrete parameters.** `s_{1:n}` must either be
   marginalised (kernel V1) or handled by a Gibbs block (kernel V2). This is the
   single most misstated point in the "DP in NumPyro" folklore: automatic
   enumeration is an **SVI** feature (`TraceEnum_ELBO`), it does not exist under
   `MCMC`/`NUTS`.
3. **No exact DP.** `K` components, residual assigned to the `K`-th atom (§7.2),
   with the truncation budget of §6.
4. **The `α`-grid disappears**, exactly as in the Stan plan: a gain in fidelity
   to the stated model (the measured `q0` error at `M = 256` is `2.9e-5`), a
   loss of numerical comparability with the recorded draws.
5. **`log_post` is not comparable** (different target). Store the model's own
   `log_density` from the MCMC object and nothing else.
6. **No SVI.** Variational inference targets the KL projection `q*` of the
   posterior onto the guide family, not the posterior. The acceptance criteria
   include band widths, quantile probabilities and the number of hazard sign
   changes — all dispersion-sensitive. SVI is admitted only as the *measured
   bias experiment* of §12.1, never as a fit.
7. **No cross-version replay.** JAX/NumPyro results are stable at a fixed key
   within one version and platform, and are not guaranteed across versions or
   CPU vectorisation changes. The report records `jax`/`numpyro` versions and
   does not claim replay.

## 4. Decisions taken (this plan)

| # | Decision | Consequence |
| --- | --- | --- |
| N1 | Target rung F2, gate on F3 (§0) | The report carries a K-ladder and an agreement table; a bare "clauses pass" is not a result. |
| N2 | `K` ladder `{40, 80, 160}` for §5.1, `{30, 60, 120}` for §5.2; `K = 80` headline | Cheap fits make a full ladder cheaper than one Gibbs fit; the ends of each ladder come from §6. |
| N3 | Residual-stick truncation: `β_1..β_{K−1} ~ Beta(1, ν)`, `π_K = 1 − Σ_{k<K} π_k` | The standard truncated-SB ("finite DP") prior. It is *not* the DP and must not be called one. |
| N4 | `jax_enable_x64` **on**, asserted at import in every module and test | JAX defaults to float32. At `n = 200` the log density is O(200) and the tail probabilities of §5.2 are the reported quantities; float32 is not acceptable. |
| N5 | Kernel V1 (marginalised, `logsumexp`) is the **headline**; kernel V2 (`DiscreteHMCGibbs`, explicit `s`) is the internal cross-check | V1 has `3K + 2` parameters and the better geometry; V2 gives labels and `n*` directly. Their agreement is the model-level check of §10 L2.6. |
| N6 | `chain_method="vectorized"`, 4 chains in one `jit`; `"sequential"` is the documented fallback | At `n = 200`, `K = 80`, four chains fit comfortably in 15 GB and vectorising removes the per-chain Python overhead. |
| N7 | New on-disk tree `output/numpyro-<stamp>/` with its own `MANIFEST.json` and a `sampler` field; **no** use of the `paper_artifacts` schema | The artifact schema encodes Gibbs knobs and a replay guarantee. A NumPyro tree must not masquerade as a replay. `paper_artifacts.atomic_savez` is reused as a plain write helper. |
| N8 | Shared adapter `src/sampler_export.py` (owned by whichever plan lands first) constructs `DPChains`/`Fit`/`DemoResult` in memory; `DPSettings` gains `components: int \| None = None` | Everything downstream is reused unchanged. `components=None` keeps the Gibbs report **byte-identical** (asserted). |
| N9 | Optional pixi feature `numpyro` + environment `numpyro`; default env and `pixi run test` stay dependency-free | Keeps plan decision 0 true for the default environment; the amendment is recorded in `dp_gibbs_plan.md` §10 (step 10). |
| N10 | Pre-flight gate: prior EPPF tail and residual mass computed **before** the run | §6. Truncation adequacy is measured, never assumed. |

## 5. Non-goals

- No change to `src/dp_gibbs.py`, `src/dp_functionals.py`, `src/dp_weibull.py`,
  the prior constants, the time transform, the data, the presets or the §6.3
  thresholds.
- No SVI/`TraceEnum_ELBO` in the fitting path (§3.6); the one variational run
  allowed is the optional bias measurement of §12.1, marked non-gate.
- No covariates, no interval/left censoring, no §5.3 — same scope as
  `dp_gibbs_plan.md` §1.
- No model comparison, no LOO/WAIC, no Bayes factors.
- No writes under `docs/research/figures/`; figures go under the git-ignored
  `output/numpyro-<stamp>/`.
- No claim that "automatic enumeration" replaces the hand-written marginalisation
  here: it does not, under MCMC.

## 6. Choosing `K`: the truncation budget

Identical arithmetic to the Stan plan, restated because the number is the plan's
load-bearing quantity. Expected mass beyond `K` atoms under `Beta(1,ν)`
stick-breaking is `(ν/(1+ν))^K`; `E[n*|ν] = ν(ψ(ν+n) − ψ(ν))` at `n = 200`:

| `ν` | 2.22 | 4 | 12 | 20 | 60 |
| --- | --- | --- | --- | --- | --- |
| `E[n* \| ν]` | 10.5 | 16.2 | 34.9 | 48.4 | 88.4 |
| residual mass at `K = 60` | 3e-14 | 1.5e-6 | 8.2e-3 | 5.4e-2 | **3.7e-1** |
| residual mass at `K = 160` | 0 | 0 | 2.0e-11 | 2.8e-6 | 1.7e-2 |
| `K` for residual `≤ 1e-4` | 25 | 42 | 116 | 189 | **558** |

The three `ν` priors have prior means 2.22 / 20 / 60 and prior
`E[n*]` = 10.1 / 44.7 / 82.8. The data pull `ν` down (posterior means
1.50 / 2.71 / 3.98; `E[n*]` = 7.02 / 9.88 / 12.78; 99th percentile of `ν` =
4.25 / 8.35 / 11.68), so the posterior is comfortable at `K = 80` while the
*prior* under `Gamma(3, 0.05)` is not representable below `K ≈ 558`.

**Reported consequence, pre-registered.** `1e-4`-level adequacy is demonstrable
for the two narrower priors and is **not** demonstrable for
`Gamma(3, 0.05)`: at `K = 160` the prior still leaves ~1.7 % of its stick mass
beyond the last atom at `ν = 60`. Figure 1's right tail is exactly where the
truncation lives, and no `F1`-fidelity claim is made there.

**Pre-flight gate (step 6 of §9).** `scripts/numpyro_preflight.py` prints, per
fit, `E[n*|ν]` and `Pr(n* > K)` under the EPPF at the prior mean and the
residual mass at the posterior 99th percentile of `ν`, together with a verdict.
The ladder comes from these numbers; the table is copied into the report.

## 7. Model and kernel design

### 7.1 Model arguments

```python
def model(time, event, *, K, d, a_nu, b_nu, a_gamma, b_gamma, a_phi, b_phi,
          prior_only=False, kernel="marginalised"):   # "marginalised" | "labels"
```

`time` is `t̃`; the driver calls `make_surv_data` and the adapter maps
functionals back with `data.t_max`, so the transform is never reimplemented.
`K ≥ 2`.

### 7.2 Priors and the `α*` parameterisation

```python
nu     = sample("nu",      Gamma(a_nu, b_nu))
gamma  = sample("gamma",   Gamma(a_gamma, b_gamma))
phi    = sample("phi",     Pareto(b_phi, a_phi))
u      = sample("u",       Uniform(0, 1).expand([K]).to_event(1))     # α* = φ u
beta   = sample("beta",    Beta(1, nu).expand([K-1]).to_event(1))    # π_1..π_{K-1}
lam    = sample("lambda_star", InverseGamma(d, rate=gamma).expand([K]).to_event(1))
pi     = stick_breaking(beta)        # π_K = 1 − Σ_{k<K} π_k
alpha  = phi * u
sigma  = lam ** (1 / alpha)          # paper λ -> Weibull scale (§2.1)
```

Notes:

- `u ~ U(0,1)` with `α* = φu` is the reparameterisation that cannot get the
  `1/φ` wrong (§2.1.4). It leaves a likelihood-flat ridge along
  `{φ u_k = const}` for each `k`; the Pareto prior and `u < 1` break it. All
  reported quantities are constant along the ridge, so it costs ESS, not
  correctness (§13.1).
- The alternative — `alpha = sample(..., Uniform(0, phi)...)` — is the ridge
  fix; it requires an explicit `−log φ` per atom, and is a documented arm only
  if §13.1 fires.
- Do not use `dist.TransformedDistribution` to move the prior from `σ` to `λ`:
  sample `λ`, compute `σ`. The Jacobian-free identity of §2.1.1 depends on it.

### 7.3 Likelihood, kernel V1 (marginalised)

```python
ll_u = Weibull(scale=sigma, concentration=alpha).log_prob(time[:, None])   # (n,K)
ll_c = - (time[:, None] ** alpha) / lam                                   # log survival, explicit
ll   = jnp.where(event[:, None], ll_u, ll_c)
if not prior_only:
    numpyro.factor("obs", jnp.sum(logsumexp(log_pi + ll, axis=-1)))
```

A `factor` site, not a `sample` site: no library distribution expresses this
mixture with a per-component censored likelihood, and a `sample` site would
require a custom `Distribution` class whose `log_prob` does the same sum. The
`(n, K)` matrix is `200 × 80`; `logsumexp` is over the component axis and the
sum over observations.

The censored term is written by hand (`−(t^α)/λ`), not as `log1p(−cdf)`, which
cancels in the tail. `dist.RightCensoredDistribution` exists and is used **only**
in the single-component unit test that validates this closed form (L1.3); it is
not used in the mixture, where the per-component vectors are needed anyway.

### 7.4 Likelihood, kernel V2 (explicit labels)

Same model, `s_i ~ Categorical(π)` sampled as a discrete site:

```python
s = sample("s", Categorical(logits=log_pi).expand([n]).to_event(1))
ll = jnp.where(event, Weibull(scale=sigma[s], concentration=alpha[s]).log_prob(time),
                     -(time ** alpha[s]) / lam[s])       # same closed forms as V1
numpyro.factor("obs", jnp.sum(ll))
a_i, l_i = alpha[s], lam[s]                              # per-observation locations
n_star = jnp.unique(s).size
```

`obs_mask` is *not* the censoring switch here (it marks missing/imputed sites);
the censored and uncensored log terms are combined with `jnp.where`, exactly as
in V1, so that L1 can compare the two joints term by term. The per-observation
`(α_i, λ_i)` are gathered from the labelled atoms, which is what
`DPChains.theta` wants and what V1 has to reconstruct from a post-hoc draw.

V2 is *not* a different model: `(θ, s, π)` has the same joint. It differs in
what the sampler does with `s`, and it yields `n*` directly. The `(n, K)`
marginalisation matrix is replaced by an `n`-vector of lookups, so V2's per-step
cost is *lower*; its cost is the discrete update.

### 7.5 Kernel settings

| setting | V1 (headline) | V2 (cross-check) |
| --- | --- | --- |
| kernel | `NUTS(model)` | `DiscreteHMCGibbs(NUTS(model))` |
| `target_accept_prob` | 0.95, escalate to 0.99 on any divergence | same |
| `max_tree_depth` | 12 | 12 |
| `num_warmup` / `num_samples` | 2 000 / 2 000 | 2 000 / 2 000 |
| `num_chains` | 4, `chain_method="vectorized"` | 4, `"sequential"` if the Gibbs block is not vmappable |
| init | `init_to_uniform(radius=0.5)`; `β` and `u` in `(0.2, 0.8)`; `φ = max(b_φ, max α*_init)`; `γ`, `ν` at prior means | same |

`DiscreteHMCGibbs`'s discrete proposal is either a conditional (Gibbs) draw or a
uniform random-walk proposal depending on the installed version's `random_walk`
flag. **The plan does not assume which**: the report records the flag, and L2.6
(below) checks the empirical `s` marginals of V2 against V1's conditional
probabilities at fixed parameters, which is the check that would fail if the
proposal were wrong.

### 7.6 Recovering `s`, `n*`, `θ`

- **V2**: `s` is a state variable; `theta[b,i] = (alpha[s_i], lam[s_i])`
  directly, `n_star = #distinct(s)`.
- **V1**: `s` is recovered post hoc. Two routes, both required:
  1. *Conditional draw*: one `Categorical(logits=log_pi + ll_i)` per
     observation, in a `Predictive`-style pass for the exported `θ`.
  2. *Rao-Blackwellised*: no RNG. Given `π`, `n*` is a Poisson-binomial over
     independent `Bernoulli(p_k)` with `p_k = 1 − (1 − π_k)^n`, so
     `n_star_mean = Σ p_k` and `Pr(3 ≤ n* ≤ 15) = Σ_{m=3..15} pmf[m]` in
     `O(K²)`. The acceptance clause uses the draw-averaged RB `Pr`, which is a
     *sharper* estimator of the same quantity than the Gibbs chain's histogram
     — because the paper's `n*` is exactly `#distinct` of a posterior draw and
     the RB version conditions on the parameters.

Both are exported and the report says which produced each row. `infer_discrete`
is not needed and is not used: it exists to recover discrete latents from a
*guide*, and there is no guide in this plan.

## 8. Files and modules

```text
src/
  jax_config.py                 # enable_x64 + version assertions; imported first everywhere
  sampler_export.py             # SHARED with the Stan plan: draws -> DPChains/Fit/DemoResult
  numpyro_model.py              # kernel="marginalised" | "labels"; log-density helpers
  numpyro_draws.py              # MCMC object -> arrays; output/numpyro-<stamp> read/write
  numpyro_examples.py           # example_5_1 / example_5_2 equivalents + CLI
scripts/
  numpyro_preflight.py          # §6 pre-flight gate
  numpyro_ladder.sh             # K-ladder over both sections, both kernels
  numpyro_verify.sh             # re-render report + agreement table without fitting
tests/
  test_numpyro_model.py         # L1: log density, priors, mapping identities, RightCensored agreement
  test_sampler_export.py        # L2.2 + DPSettings.components contract
  test_numpyro_agreement.py     # L2.6 kernel equivalence, L3 + §11
```

`src/sampler_export.py` is the only module shared with the Stan plan; whichever
plan lands first owns it and the second reuses it verbatim. Its public surface
(also listed in `dp_model_stan_plan.md` §8):

```python
@dataclass(frozen=True)
class ExportProvenance:      # sampler, components K, warmup, draws, target_accept,
    ...                      # divergences, treedepth saturation, kernel flag,
                             # residual_mass, package versions
def build_chains(draws, *, data: SurvData, prior: DPPrior,
                 provenance: ExportProvenance) -> list[DPChains]
def build_fit(chains, *, data, prior, t_grid, band: Functionals | None) -> Fit
def build_result(fits, *, label: str, meta) -> DemoResult
```

with the same three hard requirements: `theta (B, n, 2)` plus the
`dp_gibbs_plan.md` §2.1 invariants asserted at the boundary;
`DPSettings.components = K` with `preset = "numpyro-nuts-k<K>"` and the report's
settings block printing `components` instead of `n_alpha`/`thin`; and the prior
band **loaded** from the existing artifacts rather than redrawn, so Figure 2's
comparison has one fewer source of difference.

## 9. Steps

Every step runs `pixi run test`, `pixi run lint`, `pixi run typecheck`, updates
the code guides where a module changes, and commits the message in column 2
verbatim. Tests that need the framework start with
`pytest.importorskip("numpyro")` / `("jax")` so the default environment collects
and skips them. Artifacts under `output/` are never committed.

| # | Commit message | Logical unit | Key deliverables | Tests | Plan |
| --- | --- | --- | --- | --- | --- |
| 1 | chore: add optional numpyro environment | Toolchain | `[feature.numpyro]` with `numpyro` (+ CPU `jax` resolved by pixi), `[environments] numpyro = { features = ["numpyro"], solve-group = "numpyro" }`, `JAX_ENABLE_X64=1` in the task env, lock update; importorskip convention | Smoke | N9, N4 |
| 2 | feat: add DPSettings components field | Honest settings record | `DPSettings.components: int \| None = None`; `_settings_payload` emits it and drops `n_alpha`/`thin` when set; `K` vs `J` named in the docstring | Unit, byte-identical report regression | N8 |
| 3 | feat: add an x64 and version guard for jax | Numerical base | `src/jax_config.py`: enable x64, assert `jax.config.read("jax_enable_x64")`, record `jax`/`numpyro` versions; raise a clear error if imported after jax arrays exist | Unit | N4 |
| 4 | test: add the paper's Weibull primitives as a log-density oracle | Numerical contract | `tests/test_numpyro_model.py`: kernel log density vs `dp_weibull.k_W`/`weibull_sf` and a hand-written prior; both `α*` arms; `K = 3`; censored, uncensored, all-censored; `RightCensoredDistribution` cross-check on one component; `InverseGamma(rate=)` pinned by a moment test | Unit | §2.1 |
| 5 | feat: add the truncated stick-breaking model and both kernels | Model | `src/numpyro_model.py`: V1 marginalised and V2 labelled, `prior_only`; the plate/factor structure documented in the module docstring; `stick_breaking` shared with the adapter's checks | Unit (L1), prior-recovery smoke at `n = 20` | §7, N3, N5 |
| 6 | feat: add the n-star and label recovery pass | Recovery | V1 conditional `s` draw + Poisson-binomial `n_star_mean`/`pmf`; V2 `#distinct(s)`; brute-force check of the Poisson-binomial at `K = 4` | Unit (L2.3–L2.5) | §7.6 |
| 7 | feat: add the NumPyro draws exporter | Adapter | `src/numpyro_draws.py` (`MCMC` object → arrays; `output/numpyro-<stamp>/` with its own `MANIFEST.json` and `sampler` field; `atomic_savez` reused) + `src/sampler_export.py`; `tests/test_sampler_export.py` incl. the `components` contract and a `quick` round-trip feeding `r_hat()`/`ess()` | Unit, Integration | N7, N8 |
| 8 | feat: add the NumPyro fitting driver and CLI | Driver | `src/numpyro_examples.py`: `example_5_1`/`example_5_2` mirroring `paper_examples` (same `data_5_1`/`data_5_2`, same `time_grid`, band loaded), CLI `--section/--K/--kernel/--chains/--warmup/--draws/--target-accept/--seed/--save-dir`; per-fit save before the next fit; `tests/test_numpyro_cli.py` | Unit, Smoke | §8 |
| 9 | test: add kernel equivalence and the agreement/ladder tests | Agreement | `tests/test_numpyro_agreement.py`: L2.6 (V1 vs V2), §11 z-table, `K`-trend, no-free-lunch guard; `scripts/numpyro_preflight.py` | Integration (`paper`) | N1, N2, N10 |
| 10 | chore: add ladder and verification scripts | Run scripts | `scripts/numpyro_ladder.sh` (preflight → 3 `K` × 7 fits × 2 kernels where affordable → report → agreement), `scripts/numpyro_verify.sh` (no fitting), pixi tasks `numpyro-5.2`, `numpyro-5.1`, `numpyro-ladder`, `numpyro-verify` | Smoke | §12 |
| 11 | docs: add the NumPyro replication report and the plan amendment | Report | `docs/research/dp_model_numpyro_report.md` (§14 wording); one-line pointer in `dp_gibbs_plan.md` §10 as its next free decision number | — | §14 |

## 10. Validation ladder

The Gibbs plan's oracles (§7) are all in-house; this ladder is built so a port
can be wrong in one place and be caught.

**L0 — prior simulation (`prior_only=True`).** Prior draws of
`(ν, γ, φ, α*, λ*, π, s)` must reproduce `dp_weibull.sample_dp_prior`-equivalent
functionals at the same parameters; `α*` uniform on `(0,φ)` by KS; `Pr(n*)`
matches the EPPF at large `K`. Catches the `1/φ` error in both parameterisations,
the stick normalisation, and the `InverseGamma` `rate`/`scale` confusion.

**L1 — log density at fixed parameters.** `K = 3`, everything fixed: the model's
`log_density` equals a NumPy assembly from `src/dp_weibull.py`; V1 and V2 agree
on the *same* joint at the same `(θ, s, π)`; censored terms equal `log K_W` to
1e-12; `n = 1`, all-censored, `n_j = 1` are cases, not afterthoughts. Run with
`log_density` in both `prior_only` states and with `validate_args=True`.

**L2 — internal consistency.**
1. `theta[b,i] = (alpha[s_i], lam[s_i])`; `#distinct(theta) = n_star_draw`.
2. `Σ_j n_j = n` for the `bincount` of `s`.
3. `E[n_star_draw]` equals `E[n_star_mean]` to MC error over 4 chains (V1).
4. `Σπ = 1`, `π_k > 0`, and `π` matches the stick-breaking recomputation.
5. Exported `DPChains` pass `dp_gibbs_plan.md` §2.1 invariants 1–2.
6. **Kernel equivalence (V1 vs V2).** Same data, same `K`, same seeds-free
   comparison: `ν`, `γ`, `φ`, `n*`, `S(t)` at the five monitored times agree
   within `|z| ≤ 2.5` on the §11 formula. This is the plan's strongest
   model-level check and it needs no external reference: the two kernels differ
   in marginalisation and in discrete handling, not in the model.

**L3 — posterior agreement with the committed Gibbs run (§11).**

**L4 — SBC (opt-in, `slow`).** `n = 20`, `K = 40`, 100 replications, rank
histograms for `ν`, `γ`, `φ`, `n*`, `S(1.2)`, reusing the design of
`tests/test_sbc.py` and `tests/fixtures/sbc_seeds.json`. The only *frequentist*
calibration check of the truncated model.

## 11. Agreement protocol

Same three checks as `dp_model_stan_plan.md` §11; repeated here so this document
stands alone.

### 11.1 The z-table

For each of the 12 (§5.1) / 8 (§5.2) label-invariant summary rows, against the
committed run (40 000 draws, `J_common` 328–3157):

```text
se_fw    = sd_fw / sqrt(ESS_fw)
se_gibbs = sd_gibbs / sqrt(ESS_gibbs)
z        = (mean_fw - mean_gibbs) / sqrt(se_fw^2 + se_gibbs^2)
```

### 11.2 The two row classes

| class | rows | expected |
| --- | --- | --- |
| K-insensitive | `S(1.1727)`, `∫f`, the 3 hazard change points, `Pr(S_A>S_B)` pointwise, `Pr(η_A>η_B)`, mean hazard ratio at 5/25 | `\|z\| ≤ 2.5` at the headline `K`; no trend in `K` |
| K-sensitive | mean and 99th percentile of `ν`, mean and histogram of `n*`, `Pr(3 ≤ n* ≤ 15)`, band widths at `t = 4.5` and `t = 25`, `Pr(F_B>F_A)` mean | monotone `z → 0` as `K` rises; `\|z\| ≤ 2.5` at the largest `K` |

A K-sensitive row that does not trend toward zero is a model or mapping bug, not
truncation, and blocks the report. A K-insensitive row with `|z| > 2.5` is a bug
at any `K`.

### 11.3 The no-free-lunch guard

The §6.3 suite rewards dispersion: "prior band ⊋ posterior band", "band at 4.5 ≥
1.5 × band at 2.5", `Pr(3≤n*≤15) ≥ 0.70`. Truncation *shrinks* the posterior and
so makes those clauses **easier**. Every dispersion clause must therefore also
satisfy: the NumPyro width is **within the MC-error envelope of the Gibbs
width**, not merely on the passing side. A fit that passes a width clause while
the width moves the wrong way by more than `2.5` combined SE is reported as a
truncation failure even when `evaluate_acceptance` returns green.

### 11.4 Output

`output/numpyro-<stamp>/agreement.json` plus a markdown table with columns
`row | gibbs | V1(K=40) | V1(K=80) | V1(K=160) | V2(K=80) | z(80) | trend`. The
V2 column is what makes this table different from the Stan plan's; the §6.3
verdicts are an appendix to it.

## 12. Runbook and effort

```bash
pixi run -e numpyro python scripts/numpyro_preflight.py --both    # §6 gate
pixi run -e numpyro scripts/numpyro_ladder.sh 5.2                 # ~1 h
pixi run -e numpyro scripts/numpyro_ladder.sh 5.1                 # ~4-8 h
pixi run -e numpyro scripts/numpyro_verify.sh output/numpyro-<stamp>
```

Planning estimates, to be replaced by measured wall clock in the report:

| item | estimate | basis |
| --- | --- | --- |
| model + L1/L2 + kernel equivalence | 2–3 days | two kernels, one model |
| one §5.1 fit, V1, `K = 80`, 4 × (2 000 + 2 000) vectorised | 15–50 min | 242 params, `n·K = 16 000` terms/gradient, CPU |
| one §5.2 fit, `K = 60`, `n = 20` | 2–8 min | 10× fewer likelihood terms |
| full ladder (3 `K` × 7 fits) + V2 at the headline `K` | 5–14 h | 21 + 7 fits, vectorised chains |
| agreement + report + verification | 1 day | the verify path does not fit |

Compare with the Gibbs record: 13 h 29 m for the 7 headline fits at 1 505 000
sweeps each. The ladder is affordable because each fit is short.

### 12.1 Optional, non-gate: how much would SVI have cost us?

One `TraceEnum_ELBO` run per §5.1 prior (auto-enumeration over `s`, `AutoNormal`
guide on `u, β, λ*, ν, γ, φ`) exists solely to *measure* the variational bias
against the V1 posterior, on the two dispersion-sensitive rows (band width at
`t = 4.5`, `Pr(3≤n*≤15)`). It is reported as a bias measurement with no
acceptance verdict attached, so that the "SVI is not a replication tool" claim
is quantified rather than asserted. `n` is small enough that this costs minutes.

## 13. Risks

1. **The `φ`–`u` ridge (N4/§7.2).** Likelihood-flat, pinned by the Pareto prior
   and `u < 1`. Costs ESS in `φ` and per-atom `α*`; harmless for
   label-invariant summaries. *Mitigation*: the direct-`α*` arm with explicit
   `−log φ`, chosen only if divergences persist. *Detection*: divergence counts
   and the `φ` ESS in the diagnostics table.
2. **float32 creep (N4).** Any code path that builds arrays before
   `jax_config` runs silently downgrades `λ`, `σ` and the `logsumexp`
   accumulation. The L1 tolerance is set in float64 terms, so the guard is a
   test, not a comment.
3. **`DiscreteHMCGibbs` proposal semantics may change across versions.** §7.5
   avoids assuming them; L2.6 is the test that would fail if the discrete
   update were not a valid move (e.g. a random walk that is not
   π-invariant at the boundary `n* = K`).
4. **Exchangeable empty atoms.** With `n* ≈ 10` and `K = 80`, seventy atoms are
   prior-only and exchangeable; the posterior has a symmetric ridge, not
   separate modes. Per-atom `R̂` is meaningless and must not be reported (§7.7
   of the Stan plan; the same rule applies here at the level of `alpha`/`lam`
   traces).
5. **`chain_method="vectorized"` memory.** Four chains × `(n,K)` matrices is
   small here, but the ladder's `K = 160` rung is 4× that; if it OOMs, the
   documented fallback is `"sequential"`, recorded in the fit meta.
6. **The wide `ν` prior cannot be represented** (`K ≈ 558` for `1e-4` at
   `ν = 60`). Pre-registered in §6; the report claims `F2` for two priors and a
   bound for the third.
7. **Rao-Blackwellised `n*` (V1) looks better than the paper's estimator, and
   V2 does not.** V1 and V2 therefore disagree slightly on the `n*` histogram by
   construction. The report must label which estimator produced each row; §11.2
   lists `n*` as K-sensitive, so the K-trend is judged on V1 and V2 separately.
8. **Version drift.** `jax`/`numpyro` versions and `XLA_FLAGS` go into
   `logs/env.txt`; no replay claim (§3.7).

## 14. What the report will say

The committed deliverable (`docs/research/dp_model_numpyro_report.md`) contains,
in this order:

1. The fidelity statement: "NumPyro reproduces the *Kottas (2002) model* under a
   `K`-truncated stick-breaking prior, with two algorithmically different
   samplers of that model. It does not reproduce the Appendix sampler and does
   not target the untruncated DP posterior."
2. The §6 pre-flight table and the `K`-vs-`E[n*]`/residual-mass table with the
   `Gamma(3, 0.05)` caveat.
3. The §11.1 z-table, the V2 column, and the §11.4 trend column.
4. The L2.6 kernel-equivalence table (V1 vs V2) — the internal check with no
   external reference.
5. The §11.3 no-free-lunch table (widths, not just verdicts).
6. The §12.1 variational bias measurement, marked non-gate.
7. The §6.3 clause verdicts, marked "F3 gate, subordinate to 3–5".
8. Diagnostics: divergences, treedepth, `R̂`/ESS on label-invariant quantities
   only, `φ` ESS, `target_accept` escalations, kernel flags.
9. Wall clock and environment.
10. An explicit non-claim: nothing here replaces
    `dp_gibbs_replication_report.md`, no figure is byte-comparable with the
    record's, and the SVI numbers are a bias measurement, not a fit.
