# Code Guide: `src/dp_weibull.py`

## Why this note exists

`src/dp_weibull.py` is short (432 lines) but it implements an infinite-
dimensional object, so the mapping from symbol to function is worth writing
down once. This guide is a companion to three other notes:

- `docs/research/dirichlet_process_concepts.md` — the mathematics: what a DP
  is, why a draw is discrete, what $\nu$ does, and why $J$ is not $n^*$.
- `docs/dp_weibull_explanation.md` — the model, the base measure $G_0$, and the
  Section 5.1 hyperparameter choices, in the paper's notation.
- `docs/research/dp_gibbs_code_guide.md` — the Gibbs sampler and functional
  layer that consume this module's draws.

This note answers the practical questions instead: which function produces
which object, what each argument means, what the demo is doing, and where the
numbers you see on screen come from.

Read the concepts note first if any of $\operatorname{DP}$, $G_0$, $\nu$,
stick-breaking, or truncation is unfamiliar. This note assumes them.

---

## 1. Scope: what the module does and does not do

The module **draws realisations of the mixing distribution** $G$ for the Weibull
DP mixture of Kottas (2002), §4.1. That is steps (i)–(ii) of the paper's
posterior sampling scheme, not the whole inference.

It provides:

- the Weibull kernel in the paper's parameterisation;
- samplers for $G_0$ and for the hyperparameter priors;
- the stick-breaking construction of $G$;
- the truncation rule;
- prior draws $G \sim \operatorname{DP}(\nu G_0)$;
- posterior draws $G \mid \theta_{1:n}$;
- a container that turns a finite $G$ into survival / density / hazard curves.

It does **not** provide:

- the Gibbs sampler over $(\alpha_i, \lambda_i)$ — that is the paper's §7 and
  lives in `src/dp_gibbs.py`; `sample_dp_posterior` expects the sampler's
  output as its `theta` argument;
- a fit or a point estimate for the population survival function — that comes
  from averaging over many posterior draws, in `src/dp_functionals.py`;
- any notion of a cluster count $n^*$ — see §5.

The two downstream modules are documented in
`docs/research/dp_gibbs_code_guide.md`. `src/kottas_2002_5_1.py` is now a thin
wrapper over `src/paper_examples.py::example_5_1`; the §5.1 generator and the
prior constants live in `src/paper_data.py`. Nothing in this module imports
any of them.

---

## 2. The data flow

```text
                    a_nu, b_nu          a_gamma,b_gamma   a_phi,b_phi
                          |                    |                |
                          v                    v                v
              sample_prior_hyperparameters()  (or fixed values)
                          |
                          v
                   nu, gamma, phi
                    /     |      \
                   /      |       \
      truncation_number  |    sample_g0(n)
                   |      |           |
                   v      |           v
          J = n_terms     |     alphas[J], lambdas[J]     <- atom locations ~ G0
                   \      |           /
                    \     v          /
             stick_breaking_weights(nu or nu+n, J)
                          |
                          v
                    weights[J]        <- w_j, sum to 1 (last = residual)
                          |
                          v
                  DPRealization(weights, alphas, lambdas, precision)
                          |
                          v
       .survival(t) .pdf(t) .cdf(t) .hazard(t)
```

For a posterior draw, `theta` (the latent $(\alpha_i,\lambda_i)$ from the Gibbs
sampler) replaces the atoms partially: each of the $J$ atoms is independently
either a fresh $G_0$ draw or one of the observed $\theta_i$ (§6.3).

Geometry to keep in mind: `weights`, `alphas` and `lambdas` are all length-$J$
arrays describing the **same** atom index — atom $j$ is
$\bigl(w_j, \alpha_j, \lambda_j\bigr)$. They are parallel arrays, not a grid.

---

## 3. Function reference

Line numbers refer to `src/dp_weibull.py`.

### 3.1 The Weibull kernel — lines 62–86

| function | returns |
| --- | --- |
| `weibull_cdf(t, alpha, lam)` | $1-\exp(-t^\alpha/\lambda)$ |
| `weibull_sf(t, alpha, lam)` | $\exp(-t^\alpha/\lambda)$ |
| `weibull_pdf(t, alpha, lam)` | $\frac{\alpha t^{\alpha-1}}{\lambda}\exp(-t^\alpha/\lambda)$ |

These are scalar-atom helpers: one $(\alpha,\lambda)$ pair at a time, applied to
an array of $t$. The mixture versions live on `DPRealization`.

**Gotcha.** `lam` here is the paper's scale parameter in
$\exp(-t^\alpha/\lambda)$, *not* the standard Weibull scale $c$ in
$\exp(-(t/c)^\alpha)$. They are related by $c = \lambda^{1/\alpha}$. Passing a
$c$ where a $\lambda$ is expected silently changes the fitted distribution —
this is the whole subject of `docs/research/weibull_lambda_definitions.md`.

**Gotcha.** `weibull_cdf` at $t = 0$ with $\alpha < 1$ computes
$0^{\alpha-1} \cdot 0$ in the density path, not the CDF path — the CDF is safe,
but `weibull_pdf(0, alpha, lam)` diverges for $\alpha < 1$, as it must. The
`_demo` grid starts at `1e-3` for this reason.

### 3.2 The base measure $G_0$ — line 88

```python
sample_g0(n, phi, gamma, *, d=2, rng=None) -> (alphas, lambdas)
```

Draws $n$ i.i.d. atoms from

$$
G_0(\alpha,\lambda\mid\phi,\gamma)
= \operatorname{Uniform}(\alpha\mid0,\phi)\,
  \operatorname{InvGamma}(\lambda\mid d,\gamma),
$$

i.e. $\alpha \sim \operatorname{Uniform}(0,\phi)$ and
$\lambda \sim \operatorname{InvGamma}(d,\gamma)$, independently.

This is the function that answers "which Weibulls are candidates". $\phi$ caps
the shape parameter, so it controls how extreme the allowed hazard shapes can
get; $\gamma$ sets the scale of $\lambda$.

**Gotcha.** SciPy's `invgamma(a, scale=b)` **is** the paper's
$\operatorname{IGamma}(\cdot\mid a,b)$, so `scale=gamma` — not `1/gamma`. The
docstring's NumPy alternative (`1.0 / rng.gamma(shape=d, scale=1.0/gamma)`)
shows why: if $X\sim\operatorname{Gamma}(d, \text{scale}=1/\gamma)$ then
$1/X \sim \operatorname{InvGamma}(d,\gamma)$.

**Gotcha.** `n` here is the number of *atoms*, which is $J$ in the sampling
functions, not the number of observations $n$ in the paper. The name collision
is unfortunate; in `sample_dp_prior` the call is `sample_g0(n_terms, ...)`.

### 3.3 Hyperparameter priors — line 106

```python
sample_prior_hyperparameters(a_nu, b_nu, a_gamma, b_gamma, a_phi, b_phi, *, rng=None)
    -> (nu, gamma, phi)
```

$$
\nu \sim \operatorname{Gamma}(a_\nu, b_\nu),\quad
\gamma \sim \operatorname{Gamma}(a_\gamma, b_\gamma),\quad
\phi \sim \operatorname{Pareto}(a_\phi, b_\phi),
$$

returning three Python floats.

**Gotcha.** The paper writes $\operatorname{Gamma}(\text{shape}, \text{rate})$,
so $\operatorname{Gamma}(2, 0.9)$ has mean $2/0.9 \approx 2.22$. NumPy's
`rng.gamma` takes a **scale**, hence `scale=1.0 / b_nu` in the code. Reading the
paper's second argument as a scale would inflate $\nu$ by $1/0.9^2$ and give
completely different weights.

**Gotcha.** `pareto.rvs(b=a_phi, scale=b_phi)` means the support is
$[b_\phi, \infty)$: $b_\phi$ is a **lower bound** on $\phi$, and $a_\phi$ is the
tail index. $\phi$ is the upper limit of the $\alpha$ prior, so $b_\phi$
effectively sets the smallest admissible upper bound on the Weibull shape.

### 3.4 Prior matching for $b_\gamma, b_\phi$ — lines 136 and 157

These two exist only to choose the hyperparameters of §5.1 sensibly, and they
are easy to mix up because they are inverses of each other in spirit.

```python
weibull_from_median_iqr(median, iqr)      -> (alpha_tilde, lambda_tilde)
b_gamma_b_phi_from_weibull(alpha_tilde, lambda_tilde) -> (b_gamma, b_phi)
```

1. `weibull_from_median_iqr` finds the Weibull whose median and IQR match a
   guess at the population. The IQR/median ratio depends on $\alpha$ only, so
   it solves $2^{1/\alpha} - (4/3)^{1/\alpha} = \text{IQR}/\text{median}$ for
   $\alpha$ by Brent's method, then sets
   $\lambda = m^\alpha/\log 2$ in the paper's parameterisation.
2. `b_gamma_b_phi_from_weibull` inverts the prior medians from §4.3:
   $b_\gamma = (\sqrt2-1)/\tilde\lambda$ and $b_\phi = 4\tilde\alpha/3$.

The reason this is a two-step dance rather than one formula: the prior is on
$(\nu,\gamma,\phi)$, but intuition is available in $(\alpha,\lambda)$ space —
population median and spread are things you can guess from a pilot sample. So
translate guess → Weibull → prior hyperparameters.

For the §5.1 truth $0.8\,\mathrm{LN}(0,0.25) + 0.2\,\mathrm{LN}(1.2,0.02)$ the
population median and IQR are about 1.1727 and 1.3679, giving
$\tilde\alpha \approx 1.3314$, $\tilde\lambda \approx 1.7836$,
$b_\gamma \approx 0.2322$, $b_\phi \approx 1.7752$.

**Gotcha.** `weibull_from_median_iqr` returns the *paper's* $\lambda$, not the
standard scale $c$. The docstring spells this out and is worth reading before
passing the result to SciPy.

### 3.5 The truncation rule — line 188

```python
truncation_number(precision, epsilon=1e-4) -> J
```

Returns the smallest integer $J$ with
$(\text{precision}/(\text{precision}+1))^J \le \varepsilon$, via

$$
J = \left\lceil
\frac{\log\varepsilon}{\log\!\bigl(\text{precision}/(\text{precision}+1)\bigr)}
\right\rceil,
$$

floored at 1. This is the **only** place $J$ is decided, and `n_terms` in the
sampling functions overrides it.

`precision` is `nu` for a prior draw and `nu + n` for a posterior draw. That
difference is not cosmetic: with $n = 200$, $\nu = 24.6$ gives $J = 232$ from
the prior rule but $J = 2074$ from the posterior rule.

**Gotcha.** Using the per-draw rule inside an MCMC loop means a *different* $J$
at every iteration, which makes the sequence of draws awkward to compare and
inconsistent with the paper. The paper picks one common $J$ from
$\max_b \nu_b$ (and $n$) and uses it throughout; that is why the sampling
functions accept `n_terms`. The module docstring says this too.

### 3.6 Stick-breaking weights — line 198

```python
stick_breaking_weights(precision, n_terms, rng) -> weights
```

Implements

$$
z_j \sim \operatorname{Beta}(1, \text{precision}),
\qquad
w_j = z_j \prod_{s<j}(1-z_s) \;(j < J),
\qquad
w_J = \prod_{s<J}(1-z_s),
$$

so the returned length-$J$ vector sums to 1 exactly (up to floating point).
`n_terms == 1` is special-cased to `[1.0]`.

**Gotcha.** The last weight is the **residual stick**, not a
$\operatorname{Beta}(1,\nu)$-scaled break. Do not assume the weights are
exchangeable, or that `weights[-1]` is distributed like the others; it is the
mass of all the infinitely many atoms you chose to drop, folded into one atom.
The sum-to-one property depends entirely on this.

**Gotcha.** `rng` is required here (no default), unlike the sampling functions.
That is deliberate: you want to control the stream.

### 3.7 `DPRealization` — lines 224–286

A dataclass wrapping a truncated $G = \sum_{j<J}w_j\delta_{(\alpha_j,\lambda_j)}$
plus its `precision` (kept only for reference — it is not used in any
computation).

| member | meaning |
| --- | --- |
| `weights`, `alphas`, `lambdas` | length-$J$ parallel arrays |
| `precision` | the $\nu$ or $\nu+n$ used to generate it; informational |
| `n_atoms` (line 249) | `weights.size`, i.e. $J$ |
| `_scaled_time(t)` (252) | the $J \times \lvert t\rvert$ matrix $t^{\alpha_j}/\lambda_j$, via broadcasting |
| `cdf(t)` (259) | $\sum_j w_j\bigl(1-e^{-t^{\alpha_j}/\lambda_j}\bigr)$ |
| `survival(t)` (265) | $\sum_j w_j e^{-t^{\alpha_j}/\lambda_j}$ |
| `pdf(t)` (269) | $\sum_j w_j \frac{\alpha_j t^{\alpha_j-1}}{\lambda_j}e^{-t^{\alpha_j}/\lambda_j}$ |
| `hazard(t)` (282) | `pdf(t) / survival(t)` |

`__post_init__` (237) validates that the three arrays have equal shape and that
weights are non-negative. It does **not** check that they sum to 1 — the class
happily wraps an arbitrary weighted sum, and `cdf`/`survival` would silently
stop being probability statements if the weights did not normalise.

**Note on shapes.** Every summary returns a 1-D `np.ndarray` even for scalar
`t`: `G.survival(1.0)` is `array([...])`, not a float. All of them broadcast
over `t`, so the cost is $O(J \cdot |t|)$ per call — 544 atoms $\times$ 300 grid
points is trivial, but a loop generating thousands of realisations for a paper-
scale $J \approx 2500$ is where the cost sits.

**Note on `hazard`.** It is computed as a ratio of two mixture sums rather than
from a closed form, so it is only as accurate as `pdf` and `survival` at large
$t$. In practice it is stable well past the bulk of the data because the atom
with the smallest $\alpha$ leaves a heavy tail, keeping the survival function
away from underflow.

### 3.8 Prior draws — line 289

```python
sample_dp_prior(nu, gamma, phi, *, d=2, epsilon=1e-4, n_terms=None, rng=None)
    -> DPRealization
```

$$
G \sim \operatorname{DP}\bigl(\nu\,G_0(\cdot\mid\phi,\gamma)\bigr).
$$

Three steps: `n_terms or truncation_number(nu, epsilon)`; stick-breaking with
`precision=nu`; `sample_g0(n_terms, phi, gamma)` for the atoms. The resulting
`precision` attribute is `nu`.

This is the paper's **prior** draw — one realisation of a survival curve before
seeing data. It is what `_demo` plots.

### 3.9 Posterior draws — line 313

```python
sample_dp_posterior(nu, gamma, phi, theta, *, d=2, epsilon=1e-4, n_terms=None, rng=None)
    -> DPRealization
```

$$
G \mid \theta_{1:n}, \nu, \gamma, \phi
\sim
\operatorname{DP}\!\left(
\nu G_0 + \sum_{i=1}^{n}\delta_{\theta_i}
\right),
$$

with `theta` an $(n, 2)$ array of latent $(\alpha_i,\lambda_i)$. The precision is
$\nu + n$; each atom is independently a fresh $G_0$ draw with probability
$\nu/(\nu+n)$, otherwise a uniformly chosen observed $\theta_i$. The
`from_base` boolean mask at line 355 is exactly that Bernoulli decision.

Consequences worth noticing when you read the code:

- the number of atoms drawn from $G_0$ is `n_terms * nu/(nu+n)` in expectation,
  so for $n \gg \nu$ almost all atoms are copies of observed values — the
  posterior is data-dominated;
- the residual last weight still holds, so there are still $J$ atoms regardless
  of how few distinct $\theta_i$ there are;
- `theta` must be a 2-D array with two columns; the validation at line 344
  rejects anything else.

**Gotcha.** This function takes `theta` from a Gibbs sampler that this module
does not implement. Feeding it a single set of point estimates is not a
posterior draw — you call it once per MCMC iteration and average the resulting
curves.

---

## 4. Walkthrough of `_demo()` — line 373

The demo is a single prior-draw illustration, not an inference run. It:

1. builds a Weibull guess from the §5.1 population median and IQR
   (1.1727, 1.3679) and converts it to $(b_\gamma, b_\phi)$;
2. loops over the three $\nu$ priors of §5.1;
3. for each, draws $(\nu,\gamma,\phi)$, then one $G \sim
   \operatorname{DP}(\nu G_0)$;
4. plots `G.survival(t_grid)` and `G.pdf(t_grid)` on a common axis.

Running the equivalent of the demo (same seed, plotting stripped) gives:

| $\nu$ prior | drawn $\nu$ | $\gamma$ | $\phi$ | $J$ = atoms | $\max_j w_j$ | effective components $1/\sum_j w_j^2$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\operatorname{Gamma}(2,0.9)$ | 0.673 | 1.001 | 2.074 | 11 | 0.920 | 1.18 |
| $\operatorname{Gamma}(2,0.1)$ | 10.904 | 8.742 | 3.988 | 105 | 0.149 | 15.65 |
| $\operatorname{Gamma}(3,0.05)$ | 58.554 | 2.722 | 6.167 | 544 | 0.066 | 55.37 |

Read the columns left to right and you have the $\nu$ story of the concepts
note in one table:

- $\nu = 0.67$ gives 11 atoms, one of which carries 92% of the mass. The
  effective number of components is 1.18 — the "mixture" is essentially a single
  Weibull. This is $\nu \to 0$ concentrating on one lump.
- $\nu = 58.6$ gives 544 atoms, the largest is 6.6%, and about 55 matter. This
  is $G$ staying near $G_0$, spread thin.
- $J$ tracks $\nu$ mechanically, because it is `truncation_number(nu)`.

The `1/Σw²` column is the participation ratio: a useful summary of "how many
components are effectively in play", independent of $J$. Comparing it with $J$
is the quickest way to see that the atom count is not the component count.

**Gotcha.** The demo calls `sample_dp_prior(...)` without `n_terms`, so each
prior gets its own $J$. For a paper-reproduction run you would compute a single
common `J` from $\max_b\nu_b$ (and $n$) and pass `n_terms=J` to every draw. The
demo prints `atoms=` so you can see which rule was used.

**Gotcha.** One draw per prior is a *very* noisy summary of the prior. The demo
comment in the source says as much: "In a full analysis you would repeat this
for every posterior (or prior) sample of (nu, gamma, phi)." The plotted curves
would be much smoother if averaged over draws — and the average is the prior
mean functional, which is what the paper's point estimates are based on.

**Gotcha.** `plt.show()` blocks and needs a display. Running
`pixi run python src/dp_weibull.py` on a headless machine will open nothing; the
printed lines are the useful part.

---

## 5. Where "232" comes from, and why it is not the number of clusters

The questions that arise first when reading the demo output:

**Q: Why 232 atoms?**
From `truncation_number(nu, 1e-4)` with a drawn $\nu$. Since
$J$ depends on $\nu$ monotonically, $J = 232$ pins down $\nu \approx 24.6$.
It is a property of that one hyperparameter draw, not of the model: any run whose
$\nu$ lands near 24.6 gets $J = 232$, which is a routine outcome of the
$\operatorname{Gamma}(2,0.1)$ prior (mean 20, sd $\sqrt2/0.1 \approx 14$).
Nothing special is happening at 232 — resample with another seed and you get a
different $J$.

**Q: Is 232 the number of clusters?**
No — and no function in this module computes a cluster count. $J$ is the number
of atoms in the truncated *support* of $G$; every atom has positive weight, but
clusters are defined by latent values tying among observations. The two live in
different places:

| quantity | what it counts | where it lives | random? | typical value |
| --- | --- | --- | --- | --- |
| $J$ (`G.n_atoms`) | atoms in the truncated draw | `truncation_number` | no | 11–544 here; ~2000 with the paper's rule |
| $n^*$ | distinct $\theta_i$ among the $n$ observations | the Gibbs sampler's assignment variables | yes, has a posterior | 3–15 in the paper's §5.1 |

$n^*$ is not in this module because the sampler is not. If you have `theta`
from a sampler, it is one line over the rows the sampler already tracks:

```python
import numpy as np
n_star = np.unique(theta, axis=0).shape[0]   # exact ties only
```

Exact comparison is correct here because the paper defines clusters through the
allocation variables $s_i$: cluster members share their $(\alpha_i,\lambda_i)$
*literally* (the paper's $\alpha^*_j, \lambda^*_j$ are copied to members), rather
than merely being close. Counting near-duplicates instead would give a different
and wrong answer.

**Q: So what should I expect on screen?**
`atoms=` is $J$. It should always be much larger than the number of components
visibly shaping the plotted curve — compare it with the `1/Σw²` column in §4,
not with a cluster count.

---

## 6. Assemble a full run

A single prior draw:

```python
from src.dp_weibull import sample_dp_prior, weibull_from_median_iqr, \
                           b_gamma_b_phi_from_weibull

at, lt = weibull_from_median_iqr(median=1.1727, iqr=1.3679)
b_gamma, b_phi = b_gamma_b_phi_from_weibull(at, lt)

G = sample_dp_prior(nu=2.0, gamma=1.0, phi=2.0, epsilon=1e-4)
G.n_atoms                       # 23   (= truncation_number(2.0))
round(G.weights.sum(), 12)      # 1.0  (to ~1e-16, not exactly)
G.survival([0.5, 1.0, 2.0])     # mixture survival at three times
```

A posterior draw, given latents from your sampler:

```python
import numpy as np
from src.dp_weibull import sample_dp_posterior

theta = np.asarray(theta, dtype=float)          # shape (n, 2) from the Gibbs sampler
G_post = sample_dp_posterior(
    nu=nu, gamma=gamma, phi=phi, theta=theta,
    n_terms=2500,                                # one common J for the whole run
)
```

Averaging functionals over draws, which is what the paper's point estimates
are:

```python
t_grid = np.linspace(1e-3, 6.0, 300)
S_draws = np.array([G.survival(t_grid) for G in posterior_draws])  # (n_draws, |t|)
S_hat = S_draws.mean(axis=0)
S_lo, S_hi = np.percentile(S_draws, [2.5, 97.5], axis=0)
```

---

## 7. Gotcha checklist

- **Two `lambda`s.** `lam` in this module is the paper's scale in
  $\exp(-t^\alpha/\lambda)$, not the standard Weibull scale. See
  `docs/research/weibull_lambda_definitions.md`.
- **Rate vs scale.** Paper's $\operatorname{Gamma}(\text{shape},\text{rate})$;
  NumPy wants `scale=1/rate`. `pareto.rvs(scale=b_phi)` treats $b_\phi$ as a
  lower bound.
- **`n` means two things.** In `sample_g0(n, ...)` it is the atom count $J$; in
  the paper's model it is the number of observations. In
  `sample_dp_posterior`, $n$ = `theta.shape[0]` is the latter.
- **The last weight is residual**, not a fresh break. Weights sum to 1 because
  of that atom, not because of normalisation.
- **Per-draw $J$ vs common $J$.** Pass `n_terms` for anything resembling the
  paper's run.
- **Posterior precision is $\nu + n$**, not $\nu$. This changes both the weights
  and $J$.
- **`DPRealization` does not require normalised weights.**
- **Scalar `t` still returns an array.**
- **`hazard` is a ratio of two sums**, so it inherits their numerical error at
  extreme $t$.
- **`_demo` is one draw per prior**, not an average, and it calls `plt.show()`.
- **No Gibbs sampler here.** `sample_dp_posterior` is downstream of one; the
  sampler is `src/dp_gibbs.py` (see `docs/research/dp_gibbs_code_guide.md`).

---

## 8. See also

- `docs/research/dirichlet_process_concepts.md` — the mathematics behind every
  object this module builds, including the partition definition, the empirical
  check of the Dirichlet marginals, the $J$-versus-$n^*$ distinction, and the
  Gaussian process analogy.
- `docs/dp_weibull_explanation.md` — the model in the paper's notation, the base
  measure, the truncation and posterior-update formulas, and the §5.1
  hyperparameters.
- `docs/research/weibull_lambda_definitions.md` — the two Weibull scale
  parameterisations.
- `docs/research/dp_gibbs_code_guide.md` — the Gibbs sampler and the
  functional layer that consume this module's draws.
- `docs/research/dp_gibbs_results.md` — the §5.1 and §5.2 reproduction numbers.
- `src/paper_data.py` — the §5.1 lognormal mixture (`TwoLognormalMix`), the
  §5.2 leukemia data and the prior-guess constants.
- `src/kottas_2002_5_1.py` — a thin wrapper over the §5.1 driver.
