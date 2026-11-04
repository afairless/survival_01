# From the Beta Distribution to the Dirichlet Process

## Why this note exists

The Dirichlet process is usually introduced with a one-line analogy — "a
Dirichlet distribution generalised to infinitely many dimensions" — and then
immediately used. That analogy is useful but incomplete in a specific way: the
Dirichlet **distribution** lives on a finite simplex, while the Dirichlet
**process** is a distribution over probability *measures*. The step between
them is not "take $N \to \infty$", because the object that survives the limit
is no longer a vector and no longer has a density.

This note builds the chain explicitly:

$$
\text{Beta} \;\longrightarrow\; \text{Dirichlet}
    \;\longrightarrow\; \text{Dirichlet process}
    \;\longrightarrow\; \text{DP mixture},
$$

with the aim of making four things precise:

1. what the DP actually *is* (Ferguson's finite-marginal definition);
2. why stick-breaking is the way to *compute* with it;
3. why a DP draw is discrete even over a continuum, and what that buys you;
4. why the number of atoms $J$ in a computer realisation is **not** the number
   of mixture components $n^*$ that the paper reports.

Throughout, the running example is the Weibull DP mixture of Kottas (2002), as
implemented in `src/dp_weibull.py`. Model notation, the base measure $G_0$ and
the hyperparameter choices are documented in `docs/dp_weibull_explanation.md`;
this note does not repeat them. The implementation view is in
`docs/research/dp_weibull_code_guide.md`.

---

## 1. The finite story: Beta and Dirichlet

### Beta

A Beta distribution is a distribution on the two-point simplex, i.e. on
$(w_1, w_2)$ with $w_1, w_2 \ge 0$ and $w_1 + w_2 = 1$. Writing
$w_1 = z$ and $w_2 = 1 - z$,

$$
z \sim \operatorname{Beta}(\alpha, \beta),
\qquad
p(z) = \frac{z^{\alpha-1}(1-z)^{\beta-1}}{B(\alpha,\beta)},
\qquad
\mathbb{E}[z] = \frac{\alpha}{\alpha+\beta}.
$$

A draw from a Beta is a single probability vector. It is the natural conjugate
prior for the success probability of a coin, i.e. for the weights of a
two-category mixture.

### Dirichlet

A Dirichlet is the same thing on the $(K-1)$-simplex for $K \ge 2$:

$$
(w_1,\dots,w_K) \sim \operatorname{Dir}(\alpha_1,\dots,\alpha_K),
\qquad
p(w) = \frac{1}{B(\alpha)}\prod_{k=1}^{K} w_k^{\alpha_k - 1},
$$

supported on $w_k \ge 0$, $\sum_k w_k = 1$. Its marginal moments are

$$
\mathbb{E}[w_k] = \frac{\alpha_k}{\alpha_0},
\qquad
\operatorname{Var}[w_k] = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)},
\qquad
\alpha_0 = \sum_{k=1}^K \alpha_k .
$$

Three properties matter later.

**(a) Beta is the two-dimensional case.** $\operatorname{Dir}(\alpha_1,\alpha_2)$
is $\operatorname{Beta}(\alpha_1,\alpha_2)$.

**(b) Marginalisation gives Beta again.** For any single component,

$$
w_k \sim \operatorname{Beta}(\alpha_k, \alpha_0 - \alpha_k).
$$

**(c) The aggregation property.** If you merge two components $i$ and $j$,
the resulting vector is still Dirichlet, with the merged parameter being the
*sum*:

$$
(w_1,\dots,w_i + w_j,\dots,w_K)
\sim
\operatorname{Dir}(\dots, \alpha_i + \alpha_j, \dots).
$$

Property (c) is the one that will do the real work. It says the Dirichlet
family is closed under coarsening a partition — merging cells keeps you inside
the family. It is exactly what makes the infinite-dimensional limit exist.

### Why the Dirichlet is the natural prior for mixture weights

A draw from a Dirichlet is *one point in the simplex*: one complete set of
weights. So a Dirichlet prior over mixture weights is a prior over "which
components get how much mass", and it is conjugate to the multinomial, which
is why it appears in every finite mixture model. If you observe counts
$(n_1,\dots,n_K)$ of data allocated to the $K$ components, the posterior is
$\operatorname{Dir}(\alpha_1+n_1,\dots,\alpha_K+n_K)$ — just add the counts.

That conjugate update is the finite-dimensional ancestor of the DP posterior
in §7.

---

## 2. What breaks when you try to take $K \to \infty$

You cannot simply "take $K \to \infty$" in the Dirichlet density, for two
reasons.

1. The normalising constant $B(\alpha)$ and the constraint region are both
   finite-$K$ objects. There is no Lebesgue measure on an infinite simplex, so
   there is no density to write down.
2. More importantly, the target object changes type. A finite mixture's
   weights are a *vector* in $\mathbb{R}^K$. The thing we want in the
   nonparametric setting is not a very long vector but a **probability
   measure** $G$ on the parameter space $\Theta$, with

   $$
   G = \sum_{j=1}^{\infty} w_j\,\delta_{\theta_j},
   \qquad w_j \ge 0,\quad \sum_{j=1}^\infty w_j = 1,\quad \theta_j \in \Theta .
   $$

   That is a function of measurable sets, $A \mapsto G(A)$, not a point in any
   $\mathbb{R}^K$. So the correct question is not "what is the infinite
   Dirichlet density" but "what is a sensible distribution over the set of all
   probability measures on $\Theta$?"

---

## 3. The definition: a Dirichlet on *every* partition

Ferguson (1973) answered that question as follows.

> **Definition.** Let $G_0$ be a probability measure on $\Theta$ and
> $\nu > 0$. Then $G \sim \operatorname{DP}(\nu G_0)$ if for **every** finite
> measurable partition $(A_1,\dots,A_K)$ of $\Theta$,
>
> $$
> \bigl(G(A_1),\dots,G(A_K)\bigr)
> \sim
> \operatorname{Dir}\bigl(\nu G_0(A_1),\dots,\nu G_0(A_K)\bigr).
> $$

Read that carefully, because the phrase *for every partition* is the whole
definition and it is where the intuition "a Dirichlet point with $N$
proportions" goes wrong.

- If you fix one partition with $K$ cells and ask only about those $K$ numbers,
  you get a Dirichlet. Locally, the DP *is* a Dirichlet.
- But the definition quantifies over **all partitions simultaneously**: coarse
  ones, fine ones, ones that cut across each other. A single Dirichlet vector
  cannot satisfy this, because the coarsened vector must also be Dirichlet with
  summed parameters (the aggregation property, §1c). Only an infinite-dimensional
  object can be Dirichlet at every resolution at once.

So the correct one-liner is:

> A Dirichlet process is the unique random probability measure whose restriction
> to any finite partition is Dirichlet.

### Why such an object exists

The existence argument is the standard Kolmogorov-style consistency argument,
and the aggregation property is precisely what supplies the consistency.

1. For each finite partition, declare the law of $(G(A_1),\dots,G(A_K))$ to be
   Dirichlet with parameters $\nu G_0(A_k)$.
2. Check compatibility: if you merge cells, the declared law agrees with the
   Dirichlet law for the coarser partition. Property (c) of §1 says exactly
   this: merging components adds their parameters, which is what the coarser
   partition's Dirichlet would have given anyway.
3. Consistency across all finite partitions implies there is a single random
   measure $G$ inducing all of them.

This is the same mechanism that constructs Gaussian processes (declare
consistent finite-dimensional Gaussian marginals, then invoke Kolmogorov
extension), and it is why the two objects are structurally analogous — see §10.

### Parameters, and what they do

| quantity | role |
| --- | --- |
| $G_0$ | base measure: the *shape* / location of mass. $\mathbb{E}[G(A)] = G_0(A)$ for every $A$. |
| $\nu > 0$ | precision (concentration): how tightly $G$ hugs $G_0$. |

The first two moments of the random number $G(A)$ are

$$
\mathbb{E}[G(A)] = G_0(A),
\qquad
\operatorname{Var}[G(A)] = \frac{G_0(A)\bigl(1 - G_0(A)\bigr)}{\nu + 1}.
$$

Two limits are worth remembering:

- $\nu \to \infty$: the variance vanishes, so $G \to G_0$. A very precise DP is
  nearly its base measure — the nonparametric model collapses to the parametric
  one.
- $\nu \to 0$: the variance goes to $G_0(A)(1-G_0(A))$, the maximum possible for
  a probability, so $G$ becomes extremely lumpy — essentially all mass on a
  single random point drawn from $G_0$.

Note the denominator $\nu + 1$, not $\nu$: the DP is never *less* variable than
a single draw from $G_0$.

### You can check this numerically

Because the definition is about partitions, it can be tested directly with the
stick-breaking sampler. Partition the $\alpha$-axis — which is
$\operatorname{Uniform}(0,\phi)$ under $G_0$ — into $K$ equal-mass cells,
simulate many DP draws, bin the weights by which cell each atom's $\alpha$
falls into, and compare the empirical moments with the Dirichlet formulas of
§1.

```python
import numpy as np
from dp_weibull import stick_breaking_weights

rng = np.random.default_rng(7)
nu, K, reps = 3.0, 4, 40_000
M = np.empty((reps, K))
for r in range(reps):
    w = stick_breaking_weights(nu, 4000, rng)   # truncate far beyond eps
    a = rng.uniform(0.0, 1.0, size=4000)        # atom locations ~ G0 on (0,1)
    idx = np.minimum((a * K).astype(int), K - 1)
    M[r] = np.bincount(idx, weights=w, minlength=K)

print(M.mean(0))          # theory: 1/K each
print(M.var(0))           # theory: (1/K)(1-1/K)/(nu+1)
print(np.cov(M.T)[0, 1])  # theory: -(1/K)^2/(nu+1)
```

Result, from a run of the code in this repository:

| $\nu$ | $K$ | empirical mean | theory $1/K$ | empirical Var | theory $\frac{(1/K)(1-1/K)}{\nu+1}$ | empirical Cov | theory $-\frac{(1/K)^2}{\nu+1}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 4 | 0.2501, 0.2495, 0.2495, 0.2509 | 0.25 | 0.04733, 0.04705, 0.04708, 0.04691 | 0.046875 | $-0.01564$ | $-0.015625$ |
| 0.5 | 3 | 0.3329, 0.3341, 0.3331 | 0.3333 | 0.14781, 0.14804, 0.14753 | 0.148148 | $-0.07416$ | $-0.074074$ |

Means, variances and the negative covariances all match to three decimals. The
negative covariance is the visible signature of the simplex constraint (cells
compete for a fixed unit of mass), and it is exactly what a Dirichlet produces.

This is the concrete sense in which the DP is "a Dirichlet, at every
resolution".

---

## 4. Stick-breaking: from definition to algorithm

The definition of §3 tells you what a DP draw *is* but not how to generate one.
Sethuraman (1994) gave a constructive representation, and it is what the code
actually runs.

Draw independent sequences

$$
z_j \sim \operatorname{Beta}(1, \nu),
\qquad
\theta_j \sim G_0,
\qquad j = 1, 2, \dots
$$

and set

$$
w_1 = z_1,
\qquad
w_j = z_j \prod_{s=1}^{j-1}(1 - z_s),
\qquad
G = \sum_{j=1}^{\infty} w_j\,\delta_{\theta_j}.
$$

### Reading the construction

Think of a unit-length stick. At step $j$:

- you hold a piece of stick of length $\prod_{s<j}(1-z_s)$ — the mass not yet
  allocated;
- you break off a fraction $z_j$ of it and give it to atom $\theta_j$;
- you keep the remainder, $1 - z_j$ of it, for the next atom.

Because $\mathbb{E}[z_j] = 1/(1+\nu)$,

$$
\mathbb{E}[w_j] = \frac{1}{1+\nu}\left(\frac{\nu}{1+\nu}\right)^{j-1},
$$

so the expected weights decay **geometrically** with ratio $\nu/(1+\nu)$.
Small $\nu$ means a fast decay: the first few atoms take most of the stick.
Large $\nu$ means slow decay: mass is spread thinly over very many atoms. This
is the same $\nu$ story as §3, now visible as a mechanism.

The expected remaining stick after $J$ atoms is

$$
\mathbb{E}\!\left[\prod_{s=1}^{J}(1-z_s)\right] = \left(\frac{\nu}{\nu+1}\right)^{J},
$$

which is the formula the truncation rule uses (§6).

### Why $\operatorname{Beta}(1,\nu)$ and not something else

$\operatorname{Beta}(1,\nu)$ is itself the two-dimensional
$\operatorname{Dirichlet}(1,\nu)$ — an uninformative first parameter and a
second parameter equal to the DP precision. Two consequences:

- the construction is a **sequence of two-dimensional Dirichlet draws**, which
  is the cleanest way to see the family resemblance to §1;
- it yields exactly the partition marginals of §3, so the definition and the
  algorithm are the same object; stick-breaking is a representation, not an
  approximation.

### Relation to a finite Dirichlet

There is a clean correspondence between the finite and infinite pictures, which
explains why the DP is described as the $K \to \infty$ limit:

- **Infinite → finite.** Restrict a DP draw to a partition of $K$ cells: those
  $K$ masses are exactly $\operatorname{Dir}(\nu/K,\dots,\nu/K)$ when the cells
  have equal $G_0$-mass. Verified in §3.
- **Finite → infinite.** Conversely, replace the weights by a single
  $K$-dimensional $\operatorname{Dir}(\nu/K,\dots,\nu/K)$ draw. That finite
  approximation converges in distribution to the DP as $K \to \infty$
  (Muliere & Tardella, 1998).

So the ordinary symmetric Dirichlet with parameter $\nu/K$ per component is the
finite shadow of the DP, and this is the precise sense in which "the DP is a
Dirichlet taken to infinitely many dimensions". Note that the *stick-breaking*
weights of §4 are not themselves Dirichlet; it is the restriction of the
*measure* to a finite partition that is (the numerical check in §3), together
with this symmetric finite approximation that converges to it.

> **Remark (further reading).** Sort the DP weights into decreasing order. The
> resulting random sequence follows the Poisson–Dirichlet distribution with
> parameter $\nu$ (the $\alpha = 0$ member of the two-parameter
> Poisson–Dirichlet / Pitman–Yor family). The DP is the $\alpha=0$ case; the
> two-parameter generalisation is what you get if $z_j \sim \operatorname{Beta}(1-\alpha, \nu + j\alpha)$.

---

## 5. Almost surely discrete — the surprising property

Here is the property that makes the DP useful and that is easy to skip over.

Take $G_0$ to be continuous — as in this project, where $G_0$ is
$\operatorname{Uniform}(0,\phi) \times \operatorname{InvGamma}(d,\gamma)$ on
$(\alpha,\lambda)$. A DP draw on that space is

$$
G = \sum_{j=1}^{\infty} w_j \,\delta_{(\alpha_j,\lambda_j)},
$$

a countable weighted sum of points. It is therefore **discrete with probability
one**, even though it is a distribution on a continuous space and even though
$G_0$ is continuous.

Why: the $\theta_j$ are drawn i.i.d. from a continuous $G_0$, so
$\Pr(\theta_i = \theta_j) = 0$ for $i \ne j$. Distinct locations, positive
weights, countably many of them — that is a discrete measure. The discreteness
comes from the DP's *weights* (countably many atoms, §4), not from the base
measure.

Three consequences:

1. **A DP draw is a countable mixture. It is not a density.** You can evaluate
   it at any set, and you can mix kernels against it, but it has no density with
   respect to Lebesgue measure on $\Theta$.
2. **The induced data distribution is a countable mixture** of the kernel,
   $\int K_W(t\mid\theta)\,G(d\theta) = \sum_j w_j K_W(t\mid\theta_j)$ — a
   genuine mixture, not an integral approximation.
3. **Ties become possible, and ties are the clustering mechanism.** Since $G$ is
   discrete, two observations can receive *exactly* the same latent
   $\theta_i$. Under a continuous prior that has probability zero; under a DP
   prior it has positive probability. Count the distinct values
   $\theta_1,\dots,\theta_n$ and you have the number of clusters $n^*$.

Point 3 is the entire reason the DP is a clustering prior, and it is worth
contrasting with the base measure: **$G_0$ is continuous, yet draws from the DP
tie.** The nonparametric prior is strictly "rougher" than the parametric family
it is centred on. The variance statement is the same fact: a parametric model
that fixes $G = G_0$ has $\operatorname{Var}[G(A)] = 0$, while the DP has
$G_0(A)(1-G_0(A))/(\nu+1)$ for every finite $\nu$ — and it is why $\nu \to 0$
gives one big lump.

---

## 6. Truncation: $J$ is a grid size, not a dimension

An infinite sum cannot be computed. The paper (and this code) replaces it with

$$
G_J = \sum_{j=1}^{J} w_j\,\delta_{\theta_j},
\qquad
w_J = 1 - \sum_{j=1}^{J-1} w_j,
$$

i.e. the last atom absorbs the residual stick. $J$ is chosen so that the mass
thrown away has small expectation:

$$
\left(\frac{\text{precision}}{\text{precision}+1}\right)^{J} = \varepsilon,
\qquad
J = \left\lceil
\frac{\log \varepsilon}{\log\!\left(\frac{\text{precision}}{\text{precision}+1}\right)}
\right\rceil,
\qquad \varepsilon = 10^{-4}.
$$

Note what the discarded quantity is: setting $w_J$ to the residual means the
atoms you lose are $J+1, J+2, \dots$, whose total has expectation
$(\nu/(\nu+1))^J = \varepsilon$. So $\varepsilon$ bounds the *tail*, not the
size of $w_J$ (which has expectation $\varepsilon(\nu+1)/\nu$, slightly larger,
since it also contains the mass of atoms $> J$).

This is an **approximation**, in the specific sense that real mass is dropped.
It is not a marginalisation and it is not exact, which is the main difference
from the Gaussian process case in §10.

Amount of approximation, as a function of the precision:

| precision $\nu$ | $J$ with $\varepsilon = 10^{-4}$ |
| --- | --- |
| 0.67 | 11 |
| 2.22 | 25 |
| 3.35 | 36 |
| 10.9 | 105 |
| 24.6 | 232 |
| 58.6 | 544 |

$J$ is not a property of the DP — it is a property of $(\nu, \varepsilon)$ and
of your patience. Change the seed, and the drawn $\nu$ changes, and $J$ with it.
(The value $J = 232$ seen in this project corresponds to $\nu \approx 24.6$; it
is an ordinary Gamma draw from the middle of the
$\operatorname{Gamma}(2,0.1)$ prior.)

### Two different truncation rules in the paper

The paper uses **one common $J$** for an entire MCMC run, computed from the
largest posterior precision it expects to visit:

$$
\left(\frac{n + \max_b \nu_b}{n + 1 + \max_b \nu_b}\right)^{J} = \varepsilon
\qquad \text{(posterior draws)},
$$

and for prior draws of functionals,

$$
\left(\frac{\max_b \nu_b}{1 + \max_b \nu_b}\right)^{J} = \varepsilon .
$$

The `n +` appears because the posterior precision is $\nu + n$ (§7), and the
posterior can place an atom on each of the $n$ observed $\theta_i$. With
$n = 200$ and $\nu$ in the tens, that rule gives $J$ in the thousands:

| $\nu$ | $J$ from $\nu$ alone | $J$ from $n + \nu$, $n=200$ |
| --- | --- | --- |
| 0.67 | 11 | 1853 |
| 24.6 | 232 | 2074 |
| 58.6 | 544 | 2386 |

Both are truncation levels. Neither is a number of clusters.

### $J$ is not $n^*$

This is the distinction that most often causes confusion, so state it plainly:

- $J$ = how many atoms the truncated draw **has** (its support). Fixed in
  advance from $(\nu, \varepsilon, n)$; the same for every draw in a run if you
  use the paper's rule.
- $n^*$ = how many distinct latent values the $n$ observations **occupy**. Random,
  data-dependent, bounded by $n$, and reported as a posterior in the paper.

Support versus occupancy. Every cluster is one of the atoms; almost all atoms
are not clusters. In the paper's §5.1 example, $n^* \approx 3\text{--}15$ while
$J \approx 2000$, and the expected number of clusters given $\nu$ and $n$ is

$$
\mathbb{E}[n^* \mid \nu, n] \approx \nu \log\!\left(1 + \frac{n}{\nu}\right)
\qquad \text{(Escobar \& West, 1995)},
$$

which for $n=200$ gives about 10 at $\nu=2.2$, 54 at $\nu=24.6$, 87 at
$\nu=58.6$ — and must be capped at $n$. That $\nu$ controls $\mathbb{E}[n^*]$ is
how prior information about the number of components enters the model, as the
paper's §4.3 says explicitly.

The hard constraint $J \ge n$ follows immediately: the posterior can allocate an
atom per observation, so a truncation below $n$ would silently clip clusters and
bias $n^*$ downward. Truncating generously is cheap; truncating tightly is a
modelling error, not a computational saving.

---

## 7. Precision and the conjugate update

### Prior draw

$$
G \mid \nu, G_0 \;\sim\; \operatorname{DP}(\nu G_0).
$$

### Posterior draw

After observing latent $\theta_i = (\alpha_i,\lambda_i)$, $i=1,\dots,n$, the
posterior is again a DP (Ferguson 1973; Antoniak 1974) — the DP is conjugate:

$$
G \mid \theta_{1:n}, \nu, G_0
\;\sim\;
\operatorname{DP}\!\left(\nu G_0 + \sum_{i=1}^{n}\delta_{\theta_i}\right),
$$

i.e. precision $\nu + n$ and base measure

$$
\frac{\nu}{\nu+n}G_0 + \frac{1}{\nu+n}\sum_{i=1}^{n}\delta_{\theta_i}.
$$

This is the infinite-dimensional analogue of the Dirichlet–multinomial update
in §1: **add the counts**. Each atom of a posterior DP draw is independently

- a fresh draw from $G_0$, with probability $\nu/(\nu+n)$, or
- one of the observed $\theta_i$, chosen uniformly, with probability $n/(\nu+n)$.

Three things to internalise:

1. **The precision grows with the data.** $\nu \mapsto \nu + n$, so $\varepsilon$
   at fixed $J$ buys less and less as $n$ grows — which is why the paper's $J$
   includes $n$.
2. **The posterior base measure is a mixture of $G_0$ and the empirical
   distribution** of the latent values. For large $n$ it is dominated by the
   data.
3. **$n^*$ is a function of $\theta_{1:n}$, not of $G$.** The clusters live in
   the latent assignment; the DP draw is downstream of them.

---

## 8. DP versus DP mixture

The DP and the model that uses it are different objects, and conflating them is
the second common source of confusion.

The DP is $G$: a random probability measure on the parameter space $\Theta$.
Here $\Theta = (0,\infty)\times(0,\infty)$ is the space of $(\alpha,\lambda)$
pairs. The Weibull kernel $K_W$ then defines the induced data distribution

$$
S(t; G) = \int K_W(t \mid \alpha, \lambda)\,G(d\alpha, d\lambda)
= \sum_{j=1}^{\infty} w_j \exp\!\left(-\frac{t^{\alpha_j}}{\lambda_j}\right).
$$

So:

- $G_0$ determines **which Weibulls are candidates** (their possible shapes and
  scales);
- the DP determines **how much weight each candidate gets and where it sits**;
- the kernel turns the weighted point cloud into a survival curve.

The DP is not "the weights of the Weibull process". It is the random mixing
measure; the Weibull is the kernel mixed by it. This is why the paper calls the
top-level object a *mixing distribution* and the whole model a *DP mixture*.

It is worth noting the alternative, because it clarifies what is special here.
If you placed the DP directly on the time axis — $G$ a DP on $\mathbb{R}$, and
$F(t) = G((-\infty,t])$ — then a draw *is* a random CDF, a DP prior on
distributions. That is a legitimate nonparametric prior for survival curves, and
it is what "a DP prior on distributions" means literally. Kottas (2002) does not
do that: the DP sits on $(\alpha,\lambda)$ and the kernel produces $S(t;G)$. That
choice buys smooth, interpretable, Weibull-shaped survival curves instead of
step functions, at the cost of one extra layer of hierarchy.

---

## 9. The hierarchy, written out

Putting §3–§8 together, the Section 5.1 model is

$$
\begin{aligned}
t_i \mid \alpha_i, \lambda_i
  &\sim \operatorname{Weibull}(\alpha_i, \lambda_i),
  &&\text{kernel (data level)}\\
(\alpha_i, \lambda_i) \mid G
  &\sim G,
  &&\text{latent parameters}\\
G \mid \nu, \gamma, \phi
  &\sim \operatorname{DP}\bigl(\nu\,G_0(\cdot\mid\phi,\gamma)\bigr),
  &&\text{the DP (mixing measure)}\\
\nu \sim \operatorname{Gamma}(a_\nu,b_\nu),\;
\gamma \sim \operatorname{Gamma}(a_\gamma,b_\gamma),\;
\phi \sim \operatorname{Pareto}(a_\phi,b_\phi),
  &&&\text{hyperparameters,}
\end{aligned}
$$

with $G_0(\alpha,\lambda\mid\phi,\gamma) =
\operatorname{Uniform}(\alpha\mid0,\phi)\,
\operatorname{InvGamma}(\lambda\mid d,\gamma)$, $d=2$.

Reading the levels in order is the shortest route to the whole picture: the DP
supplies a random discrete mixing measure; its atoms are candidate Weibull
parameters; the kernel turns each atom into a survival curve; the weighted sum
is $S(t;G)$.

---

## 10. Why this is like Gaussian processes, and where the analogy stops

The structural analogy is exact at the level of *definition*:

| | Gaussian process | Dirichlet process |
| --- | --- | --- |
| prior over | functions $f:\mathcal{T}\to\mathbb{R}$ | probability measures $G$ on $\Theta$ |
| indexed by | points $t \in \mathcal{T}$ | measurable sets $A \subseteq \Theta$ |
| finite marginals | $\bigl(f(t_1),\dots,f(t_N)\bigr)$ is multivariate normal | $\bigl(G(A_1),\dots,G(A_K)\bigr)$ is Dirichlet |
| hyperparameters | mean function $m(t)$, kernel $k(t,t')$ | base measure $G_0$, precision $\nu$ |
| existence | Kolmogorov extension | Kolmogorov extension + Dirichlet aggregation |
| typical draw | smooth function (for smooth kernels) | discrete measure, a.s. |
| computational form | evaluate at finitely many points | truncate the atom sum |

Both are infinite-dimensional conjugate priors that you never represent
directly: you assert consistent finite-dimensional marginals, prove consistency,
and then work with a finite shadow.

But there is one asymmetry worth keeping, because it changes how you reason
about approximation.

- **A GP's finite representation is exact.** Evaluating a GP at $N$ points gives
  an $N$-variate normal that *is* the marginal law of the process at those
  points. Nothing is thrown away; the discretisation is a marginalisation. $N$ is
  chosen by where you observe.
- **The DP's finite representation is approximate.** Truncating at $J$ atoms
  discards mass $(\nu/(\nu+1))^J$ that genuinely belonged to $G$, which is why
  you must pick $\varepsilon$ and why $J$ tracks $\nu$ and $n$. $J$ is chosen by
  an error tolerance.

So: same architecture, different fidelity guarantee. When you read "$G_J$ with
$J = 232$" you are reading a numerical budget, not a property of the process —
whereas "$f$ evaluated at 200 points" is an exact statement about the GP.

A second, looser analogy is often useful: a GP is a prior over functions
parameterised by a mean and covariance; a DP is a prior over *distributions*
parameterised by a base measure and a precision. Both let you say "smooth" or
"concentrated" without committing to a finite basis.

---

## 11. Summary

1. **Beta → Dirichlet**: the Dirichlet is the Beta on the $K$-simplex. A draw is
   one point in the simplex; the family is closed under merging cells
   (aggregation property).
2. **Dirichlet → DP**: the DP is a distribution over probability *measures*,
   defined by requiring $(G(A_1),\dots,G(A_K))$ to be Dirichlet for **every**
   finite partition. That universal quantifier is the definition; a single
   Dirichlet vector cannot satisfy it, which is why the object is necessarily
   infinite-dimensional. Merging consistency is what makes it exist.
3. **Stick-breaking** makes it computable: $z_j \sim \operatorname{Beta}(1,\nu)$,
   $w_j = z_j\prod_{s<j}(1-z_s)$, $\theta_j \sim G_0$. Weights decay
   geometrically with ratio $\nu/(\nu+1)$.
4. **A DP draw is discrete a.s.**, even over a continuous $\Theta$ with
   continuous $G_0$, because there are countably many distinct atoms with
   positive weights. This is precisely what creates ties among latent values,
   hence clusters.
5. **$\nu$ is the concentration knob**: $\nu\to\infty$ gives $G \approx G_0$;
   $\nu\to0$ gives essentially one lump. The posterior precision is $\nu + n$ —
   conjugate "add the counts", just like Dirichlet–multinomial.
6. **$J$ is a computational grid size**, set by $(\nu, \varepsilon)$ and, in the
   paper, also by $n$; it is a truncation budget, not a dimension of the process
   and not the number of components.
7. **$n^*$ is the number of occupied clusters**, a random functional of
   $\theta_{1:n}$ with posterior mass, bounded by $n$; in §5.1 of the paper it is
   about 3–15 while $J$ is about 2000. Support versus occupancy.
8. **The DP is not the Weibull mixture**: the DP is the random mixing measure;
   $K_W$ is the kernel; the survival curve is the weighted sum. Compare the
   alternative where the DP is placed on the time axis and a draw *is* a random
   CDF.
9. **The GP analogy is structural, not numerical**: same Kolmogorov-extension
   architecture, but GP finite realisations are exact marginalisations while DP
   truncations are genuine approximations.

---

## See also

- `docs/dp_weibull_explanation.md` — the model, the base measure $G_0$, the
  stick-breaking and truncation formulas as used in the paper, and the Section
  5.1 hyperparameter choices.
- `docs/research/dp_weibull_code_guide.md` — where each of these objects lives in
  `src/dp_weibull.py`, with a walkthrough of the demo, the $J$-versus-$n^*$
  question in code terms, and the implementation gotchas.
- `docs/research/weibull_lambda_definitions.md` — the two different Weibull scale
  parameters both called `lambda`.

## References

- Kottas, A. (2002). Nonparametric Bayesian Survival Analysis using Mixtures of
  Weibull Distributions.
- Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems.
  *Annals of Statistics*, 1, 209–230.
- Antoniak, C. E. (1974). Mixtures of Dirichlet processes with applications to
  Bayesian nonparametric problems. *Annals of Statistics*, 2, 1152–1174.
- Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
  *Statistica Sinica*, 4, 639–650.
- Escobar, M. D. and West, M. (1995). Bayesian density estimation and inference
  using mixtures. *JASA*, 90, 577–588.
- Kingman, J. F. C. (1975). Random discrete distributions. *JRSS B*, 37, 1–22.
- Muliere, P. and Tardella, L. (1998). Approximating distributions of random
  functionals of Ferguson–Dirichlet priors. *The Canadian Journal of
  Statistics*, 26, 283–297.
- Pitman, J. and Yor, M. (1997). The two-parameter Poisson–Dirichlet distribution
  derived from a stable subordinator. *Annals of Probability*, 25, 855–900.
