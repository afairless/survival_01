# Why `G_0` Is Uniform in the Shape and Inverse Gamma in the Scale

## Why this note exists

`docs/dp_weibull_explanation.md` §2 states the base measure of the Kottas (2002)
Weibull Dirichlet process mixture:

$$
G_0(\alpha,\lambda\mid\phi,\gamma)
=
\operatorname{Uniform}(\alpha\mid 0,\phi)\,
\operatorname{InvGamma}(\lambda\mid d,\gamma),
\qquad d=2,
$$

and §2 of that note is enough to *implement* it — draw `alpha` from `rng.uniform`,
draw `lambda` from `scipy.stats.invgamma`. It does not say **why** those two
families were picked, and the paper states the reasons compactly, spread over
§4.1 (model), §4.2 (inference) and §4.3 (prior specification), with the
consequences buried in Appendix equations (A.2)–(A.6).

This note collects that reasoning in one place:

1. `G_0` is a prior on **Weibull kernel parameters**, not a prior on the survival
   curve, so "why these distributions" is a question about the *kernel*.
2. The uniform on $\alpha$ is a **computational** device (bounded quadrature
   domain) that also carries the *only* substantive shape information in the
   prior.
3. The inverse gamma on $\lambda$ is a **conjugacy** device: it is the
   Laplace-transform conjugate prior to the Weibull's $\exp(-t^\alpha/\lambda)$
   term, and it is what makes the two-dimensional integral collapse to one
   dimension.
4. The two choices are made *as a pair*; neither works alone.

The note closes with the three distinct ways `G_0` is related to the Weibull
distribution, which is the part most easily misread.

## 1. `G_0` is about the kernel, not about $S(t)$

The hierarchy is

$$
t_i\mid\alpha_i,\lambda_i
\sim K_W(\cdot\mid\alpha_i,\lambda_i),
\qquad
(\alpha_i,\lambda_i)\mid G \sim G,
\qquad
G\mid\nu,\gamma,\phi \sim \operatorname{DP}\bigl(\nu G_0(\cdot\mid\phi,\gamma)\bigr).
$$

`G_0` is a probability measure on the *parameter space* of the Weibull kernel. It
satisfies $\mathbb{E}[G(A)] = G_0(A)$ for every measurable $A$, so it sets both
the location and the spread of the mixing distribution; $\nu$ then decides how
tightly a realisation hugs it ($\nu\to\infty$ gives $G = G_0$).

Two consequences are worth stating up front:

- Asking "why uniform and inverse gamma" is asking about the **support and
  integrability properties of the kernel**, not about the shape of a survival
  curve. A Weibull density would be a category error here: `G_0` is a
  distribution over $(\alpha,\lambda)$, whose two coordinates live on different
  scales and get different families.
- `G_0` is *not* a convenience prior that drops out. It is the probability of
  opening a new cluster in the Gibbs sampler: the Polya-urn weight is
  $\nu\,q_0$ with

  $$
  q_0 = \int k_W(t\mid\alpha,\lambda)\,G_0(d\alpha,d\lambda),
  $$

  so every numerical property of `G_0` is a property of the sampler.

## 2. The parameterisation of the kernel drives everything

The paper writes

$$
K_W(t\mid\alpha,\lambda)
=
1-\exp\!\left(-\frac{t^\alpha}{\lambda}\right),
\qquad
k_W(t\mid\alpha,\lambda)
=
\lambda^{-1}\alpha t^{\alpha-1}\exp\!\left(-\frac{t^\alpha}{\lambda}\right),
$$

that is, $\lambda$ appears **inside the exponent as a divisor** — a rate-like
scale, not the standard Weibull scale $c$ (the mapping is $c=\lambda^{1/\alpha}$;
see `docs/research/weibull_lambda_definitions.md`).

Everything below follows from that placement. In the kernel, $\lambda$ occurs
only as $\lambda^{-1}$ and $\exp(-t^\alpha/\lambda)$, so any prior on $\lambda$
whose density is a power of $\lambda$ times an exponential in $1/\lambda$ will
multiply into a gamma-type integral. `InvGamma` is exactly such a family.

## 3. Why $\operatorname{Uniform}(\alpha\mid 0,\phi)$

### 3.1 The computational reason: a bounded quadrature domain

The paper is explicit that no closed form exists for the integral over both
parameters:

> Although a base distribution $G_0$ yielding a closed form expression for the
> integral $\int k_W(\cdot\mid\alpha,\lambda)\,G_0(d\alpha,d\lambda)$ is not
> available, the choice (4) achieves essentially all of the aforementioned
> goals. Under this choice, the integral above reduces to that of a smooth
> function over a bounded interval and is easy to compute using numerical
> integration.

Truncating $\alpha$ to $(0,\phi)$ is what makes "bounded interval" true.
Combined with the analytic $\lambda$ integral of §4, the two-dimensional
integral becomes the one-dimensional integral of (A.2):

$$
q_0^{o}
=
\nu\,\frac{d\,\gamma^{d}}{\phi\,t}\int_0^{\phi}
\frac{\alpha\,t^{\alpha}}{(\gamma+t^{\alpha})^{d+1}}\,d\alpha ,
$$

and for a censored time $z$, (A.3):

$$
q_0^{c}
=
\nu\,\frac{\gamma^{d}}{\phi}\int_0^{\phi}
\frac{1}{(\gamma+z^{\alpha})^{d}}\,d\alpha .
$$

Both are smooth integrands on a finite interval, so a trapezoid or Simpson grid
in $\alpha$ is enough. That is what lets the paper use the standard Gibbs sampler
of West et al. (1994) and Bush & MacEachern (1996) — "generally more efficient
and easier to implement" — instead of a Metropolis-within-Gibbs scheme.

The same grid pays for itself twice: the continuous piece of the full
conditional is

$$
h_o(\alpha,\lambda\mid\gamma,\phi,t)\ \propto\ k_W(t\mid\alpha,\lambda)\,
g_0(\alpha,\lambda\mid\phi,\gamma),
$$

which factorises into $[\alpha\mid\gamma,\phi,t]$ times an inverse gamma for
$\lambda$ given $\alpha$. The $\alpha$ draw is obtained by discretising
$[\alpha\mid\gamma,\phi,t]$ — and the paper notes that "we already have the
required values of the function from the evaluation of integral (A.2)". One grid,
used for the Polya-urn weight *and* for the draw.

### 3.2 The statistical reason: flat, but only over shapes you believe in

Uniform per unit shape is the honest "I have no preference among shapes" prior
inside the range $0<\alpha<\phi$, and $\alpha$ is precisely the parameter that
governs hazard *shape*: $\alpha<1$ gives a decreasing hazard, $\alpha=1$ a
constant one, $\alpha>1$ an increasing one. Flatness in $\alpha$ is therefore
flatness in a *qualitative* feature, not in a nuisance parameter.

Two remarks:

- The support also has a meaning. To *force* a decreasing hazard the paper
  restricts $\alpha\in(0,1)$ "with the corresponding adjustment for $G_0$ in (4)".
  Bounding $\alpha$ is how the model expresses that kind of structural prior.
- On the log-time scale the Weibull is a Gumbel location-scale law with scale
  $1/\alpha$, so a uniform bound $\alpha\le\phi$ is a bound $\text{scale}\ge
  1/\phi$ on the Gumbel axis: a statement about tail steepness, not about
  typical survival times.

### 3.3 Why $\phi$ is random

> Of course, fixing $\phi$ would be restrictive and choosing its value awkward.
> Assuming $\phi$ random overcomes these difficulties.

The truncation point is exactly the kind of value an analyst cannot state
credibly, so it is given a hyperprior — $\operatorname{Pareto}(a_\phi,b_\phi)$,
with density $a_\phi b_\phi^{a_\phi}/\phi^{a_\phi+1}$ on $\phi\ge b_\phi$. A
Pareto tail is used rather than, say, an exponential because it is the natural
heavy-tailed prior on a *scale-like truncation point*, its own truncation
$\phi\ge b_\phi$ is harmless, and the resulting conditional stays Pareto, so the
Gibbs step is a draw from a standard family:

$$
[\phi\mid\cdots] = \operatorname{Pareto}\Bigl(a_\phi+n^{*},\ \max\bigl\{b_\phi,
\max_{j\le n^{*}}\alpha^{*}_{j}\bigr\}\Bigr).
$$

The constraint is structural: $G_0$ puts no mass above $\phi$, so $\phi$ must
exceed every atom's shape, and the conditional says exactly that.

## 4. Why $\operatorname{InvGamma}(d,\gamma)$ on $\lambda$

### 4.1 Conjugacy in $1/\lambda$: the closure that makes it work

With $\operatorname{InvGamma}(d,\gamma)$ density
$\gamma^{d}\lambda^{-(d+1)}e^{-\gamma/\lambda}/\Gamma(d)$, the products
$k_W\,g_0$ and $(1-K_W)\,g_0$ are both of the form
$\lambda^{-(d+1+m)}e^{-A/\lambda}$ with $A=\gamma+t^\alpha$, and

$$
\int_0^\infty \lambda^{-(d+2)}e^{-A/\lambda}\,d\lambda
=
\frac{\Gamma(d+1)}{A^{d+1}},
$$

so the $\lambda$ integration is exact:

$$
\int k_W(t\mid\alpha,\lambda)\,G_0(d\lambda)
=
\frac{d\,\gamma^{d}\,\alpha\,t^{\alpha-1}}{(\gamma+t^{\alpha})^{d+1}},
\qquad
\int \bigl(1-K_W(t\mid\alpha,\lambda)\bigr)\,G_0(d\lambda)
=
\left(\frac{\gamma}{\gamma+t^{\alpha}}\right)^{d}.
$$

These are the integrands of (A.2) and (A.3) respectively. **The inverse gamma is
chosen so that the $\lambda$ half of the integral has a closed form**, leaving
only the bounded $\alpha$ integral of §3.1. That is the trade the paper is
describing when it calls (4) a choice that "achieves essentially all of the
aforementioned goals" despite the absence of a closed form for the whole thing.

Equivalently and maybe more memorably: $\lambda\sim\operatorname{InvGamma}(d,\gamma)$
is the same statement as $1/\lambda\sim\operatorname{Gamma}(d,\text{rate}=\gamma)$,
i.e. a gamma prior on the rate-like quantity that the kernel actually uses, and
the Weibull survival function $\exp(-t^\alpha/\lambda)$ is exactly the Laplace
transform of that gamma at $t^\alpha$. Gamma mixing is what buys the closed form —
the same mechanism as the classical gamma-frailty models.

### 4.2 Conjugacy in the sampler: every $\lambda$ step stays inverse gamma

Because the prior is inverse gamma and the kernel contributes only
$e^{-\text{const}/\lambda}$ and a power of $\lambda$, each conditional is inverse
gamma with adjusted parameters, with no tuning:

| full conditional | distribution |
| --- | --- |
| $[\lambda_i\mid\alpha_i,\cdot]$, $t_i$ uncensored | $\operatorname{InvGamma}(d+1,\ \gamma+t_i^{\alpha_i})$ |
| $[\lambda_i\mid\alpha_i,\cdot]$, $z_i$ censored | $\operatorname{InvGamma}(d,\ \gamma+z_i^{\alpha_i})$ |
| $[\lambda^{*}_j\mid s,\cdot]$, cluster location | $\operatorname{InvGamma}(d+n_j^{o}+1,\ \gamma+\sum t_{i}^{\alpha^{*}_j}+\sum z_{i}^{\alpha^{*}_j})$ |

The uncensored case shifts the shape by one because $k_W$ contributes
$\lambda^{-1}$; the censored case does not because $1-K_W$ is free of
$\lambda$ given $\alpha$ — censoring enters only through the rate's
$\sum z^{\alpha^{*}_j}$. The same shift appears in the Polya-urn weights: the
exponent $d+1$ in (A.2) versus $d$ in (A.3).

The shape draws are the mirror image: indestructible one-dimensional problems
handled by inverse CDF on the existing grid,

$$
[\alpha^{*}_j\mid\cdot]\ \propto\ (\alpha^{*}_j)^{n_j^{o}}
\mathbf{1}\bigl(\underline{\alpha}^{*}_j<\alpha^{*}_j<\overline{\alpha}^{*}_j\bigr),
\qquad
\underline{\alpha}^{*}_j,\overline{\alpha}^{*}_j \text{ as in (A.6)} .
$$

A truncated uniform shape prior plus an inverse gamma scale prior therefore
yield a Gibbs sampler in which **every** step is either inverse CDF on a
pre-computed grid or a draw from a standard family. That is the design goal the
paper set for `G_0`.

### 4.3 Why $d=2$

> We set $d=2$ yielding an inverse Gamma distribution with infinite variance.

For $\operatorname{InvGamma}(a,b)$ the mean is $b/(a-1)$ and the variance is
$b^{2}/\bigl((a-1)^{2}(a-2)\bigr)$. Hence

- $d>1$ is needed for a finite mean; with $d=2$ the mean is $\gamma$, i.e. the
  scale prior is *centred* on $\gamma$;
- $d\le 2$ gives infinite variance, so $d=2$ is the smallest value that keeps the
  prior centred at a finite value while leaving the spread effectively
  unbounded — "the least informative centred choice".

$d$ is not a stray constant: in the conjugate sums of §4.2 it behaves like a
prior sample size (two pseudo-observations' worth of $\lambda$ information), and
in §5 below it also fixes the tail of the induced parametric family.

## 5. How this is related to the Weibull distribution — three different links

These get conflated easily, so keep them separate.

### 5.1 The atoms are Weibull parameters

Conditionally on $(\phi,\gamma)$, a draw $(\alpha,\lambda)\sim G_0(\cdot\mid\phi,\gamma)$
is a *Weibull parameter pair*. The first stage of the model is then a Weibull, and
the $G\equiv G_0$ special case is a parametric Weibull model. `G_0` sets the
candidate Weibulls; $\nu$ sets how much the mixture departs from that single
Weibull.

Note that the prior spreads over *every scale and shape in the considered range*,
not over "a Weibull plus noise": the induced prior on, say, the population median
$(\lambda\ln 2)^{1/\alpha}$ is extremely diffuse, because small $\alpha$ with
moderate $\lambda$ already moves the median by orders of magnitude. That
diffuseness is intentional — it is where the DP mixture's flexibility comes from.

### 5.2 Integrating $\lambda$ out gives a Burr XII, not a Weibull

Using the identity in §4.1, the survival of a *kernel composed with the
$\lambda$-prior* is

$$
S(t\mid\alpha,\gamma,d)
=
\int_0^\infty \exp\!\left(-\frac{t^{\alpha}}{\lambda}\right)
\operatorname{InvGamma}(\lambda\mid d,\gamma)\,d\lambda
=
\left(1+\frac{t^{\alpha}}{\gamma}\right)^{-d}
=
\left(\frac{\gamma}{\gamma+t^{\alpha}}\right)^{d}.
$$

That is a **Burr type XII** (generalised log-logistic / Lomax-type) survival
function whose tail decays like $t^{-\alpha d}$. So:

- the inner layer of `G_0` is a scale mixture that turns a Weibull kernel into a
  Burr XII component;
- the "parametric version" of the model that §4.3 uses for elicitation is a
  mixture of Burr XII laws over $\alpha$, with Weibull kernels underneath;
- the prior mean survival of the full model is the average of these over the
  shape prior, since $\mathbb{E}[S(t;G)] = \int K_W(t\mid\alpha,\lambda)G_0(d\alpha,d\lambda)
  = \frac{1}{\phi}\int_0^{\phi}\bigl(\gamma/(\gamma+t^{\alpha})\bigr)^{d}d\alpha$.

The Weibull appears as the *kernel* and (after marginalising $\lambda$) as the
*scale-mixture component*; it never appears as the shape of `G_0`.

### 5.3 The hyperparameters are elicited from a Weibull fit

This is the most direct link, from §4.3. Setting $a_\gamma=1$ (exponential) and
$a_\phi=2$ (infinite prior variance) and marginalising $\gamma$ and $\phi$ out of
the parametric version gives the induced marginals

$$
[\lambda] = \frac{2b_\gamma}{(1+\lambda b_\gamma)^{3}},
\qquad
[\alpha] = \frac{2b_\phi^{2}}{3\bigl(\max\{\alpha,b_\phi\}\bigr)^{3}} .
$$

The elicitation then runs *in Weibull terms*:

1. Elicit a prior median and IQR for the survival population.
2. Solve for the Weibull $(\tilde\alpha,\tilde\lambda)$ matching them — the
   IQR/median ratio identifies $\tilde\alpha$ alone, then
   $\tilde\lambda = m^{\tilde\alpha}/\ln 2$. Implemented as
   `weibull_from_median_iqr` in `src/dp_weibull.py`.
3. Equate the medians of the induced marginals to those targets. The medians are
   $(\sqrt2-1)/b_\gamma$ for $[\lambda]$ and $3b_\phi/4$ for $[\alpha]$, so

   $$
   b_\gamma = \frac{\sqrt2-1}{\tilde\lambda},
   \qquad
   b_\phi = \frac{4\tilde\alpha}{3},
   $$

   implemented as `b_gamma_b_phi_from_weibull`.

So `G_0` is *centred on the Weibull that encodes the analyst's prior beliefs*:
the shape prior's truncation point is tied to $\tilde\alpha$, and the scale
prior's centre is tied to $\tilde\lambda$. That is the sense in which this base
measure is "the Weibull the prior believes in" even though it is not itself a
Weibull density.

## 6. Misconceptions worth guarding against

1. **"`G_0` is a Weibull prior."** No: it is a product of a truncated uniform
   (shape) and an inverse gamma (scale), on the kernel's parameter space.
2. **"$\lambda$ in `G_0` is the Weibull scale."** No: it is the paper's
   $\lambda_p = c^{\alpha}$; passing it to `scipy.stats.weibull_min(scale=...)`
   without converting is a bug.
3. **"Marginalising $\lambda$ gives a Weibull."** It gives a Burr XII
   $(\gamma/(\gamma+t^\alpha))^{d}$; the Weibull is only recovered by fixing
   $\lambda$.
4. **"Uniform in $\alpha$ means a vague prior on survival."** It is vague, but
   for the reason in §5.1, not because the density is flat in time. And the flat
   region is $(0,\phi)$, with $\phi$ data-informed through a Pareto hyperprior.
5. **"$d$ is an arbitrary constant."** $d=2$ is the smallest value giving a
   finite-mean, infinite-variance inverse gamma; it also sets the Burr tail
   exponent $\alpha d$ and acts as a conjugate pseudo-count in the $\lambda$
   conditionals.
6. **"The base measure only matters a priori."** It supplies the new-cluster
   Polya-urn weight $\nu q_0$ — the probabilities in (A.2)/(A.3) — so its
   tractability *is* the sampler's tractability.

## 7. Where this lives in the code

- `sample_g0` — draws $(\alpha,\lambda)\sim G_0$: uniform shape, `invgamma` scale
  with `d=2`.
- `sample_prior_hyperparameters` — draws $(\nu,\gamma,\phi)$ from
  Gamma/Gamma/Pareto.
- `b_gamma_b_phi_from_weibull` — §5.3, medians of the induced marginals.
- `weibull_from_median_iqr` — §5.3 step 2, the Weibull matching a median and IQR.
- `docs/research/dp_weibull_code_guide.md` — the surrounding function-by-function
  walkthrough.

## See also

- `docs/dp_weibull_explanation.md` — the model, the base measure $G_0$ as it is
  implemented, the stick-breaking and truncation formulas, and the Section 5.1
  hyperparameter choices.
- `docs/research/weibull_lambda_definitions.md` — the two different Weibull scale
  parameters both called `lambda`, and why $c=\lambda^{1/\alpha}$.
- `docs/research/dirichlet_process_concepts.md` — what a DP is, why $G_0$ is the
  centre of the process, and the conjugate update that makes $\nu$ interpretable.
- `docs/research/dp_weibull_code_guide.md` — function-by-function guide to
  `src/dp_weibull.py`.

## References

- Kottas, A. (2002). Nonparametric Bayesian Survival Analysis using Mixtures of
  Weibull Distributions. §4.1 (model and $G_0$), §4.2–§4.3 (inference and prior
  specification), Appendix A (equations (A.2)–(A.6)).
- Kottas, A. (2006). Nonparametric Bayesian survival analysis using mixtures of
  Weibull distributions. *Journal of Statistical Planning and Inference*, 136,
  578–596.
- Escobar, M. D. and West, M. (1995). Bayesian density estimation and inference
  using mixtures. *JASA*, 90, 577–588.
- West, M., Müller, P. and Escobar, M. D. (1994). Hierarchical priors and mixture
  models, with application in regression and density estimation. In *Aspects of
  Uncertainty*, 363–386.
- Bush, C. A. and MacEachern, S. N. (1996). A semiparametric Bayesian model for
  randomised block designs. *Biometrika*, 83, 275–285.
- Burr, I. W. (1942). Cumulative frequency functions. *Annals of Mathematical
  Statistics*, 13, 215–232.
