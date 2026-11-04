# The Two Weibull Scale Parameters Called `lambda`

## Why this note exists

In `Kottas (2002)`, Section 4.1, the Weibull kernel is written with a parameter
called `lambda`:

$$
K_W(t\mid \alpha,\lambda)
=
1-\exp\!\left(-\frac{t^\alpha}{\lambda}\right).
$$

Wikipedia and many statistics packages use a different scale parameter, also
often called `lambda`:

$$
F(t\mid k,\lambda)
=
1-\exp\!\left[-\left(\frac{t}{\lambda}\right)^k\right].
$$

These are the same Weibull family, but the two `lambda` values are **not
numerically equal**. Confusing them changes the median formula and gives the
wrong scale when passing values to SciPy or NumPy.

This note fixes notation so the two conventions cannot be mixed up:

- write \(c\) for the standard Weibull scale parameter;
- write \(\lambda_p\) for the paper's scale parameter.

The code in `src/dp_weibull.py` uses \(\lambda_p\).

## Standard parameterisation

The standard Weibull distribution function is usually written as

$$
F(t\mid k,c)
=
1-\exp\!\left[-\left(\frac{t}{c}\right)^k\right],
\qquad t\ge 0,
$$

where \(k>0\) is the shape and \(c>0\) is the scale. Wikipedia uses the symbol
\(\lambda\) for \(c\):

$$
F(t\mid k,\lambda)
=
1-\exp\!\left[-\left(\frac{t}{\lambda}\right)^k\right].
$$

In this note that Wikipedia scale is called \(c\).

The median follows from setting \(F(m)=1/2\):

$$
1-\exp\!\left[-\left(\frac{m}{c}\right)^k\right]
=
\frac12.
$$

Therefore

$$
\exp\!\left[-\left(\frac{m}{c}\right)^k\right]
=
\frac12,
$$

so

$$
\left(\frac{m}{c}\right)^k=\ln 2,
$$

and hence

$$
m=c(\ln 2)^{1/k}.
$$

Solving for the standard scale gives

$$
c=\frac{m}{(\ln 2)^{1/k}}.
$$

The standard scale \(c\) has units of time.

## Kottas parameterisation

The paper uses

$$
K_W(t\mid \alpha,\lambda_p)
=
1-\exp\!\left(-\frac{t^\alpha}{\lambda_p}\right).
$$

This is the same family with

$$
k=\alpha.
$$

To see the relationship, compare the exponential terms:

$$
\left(\frac{t}{c}\right)^\alpha
=
\frac{t^\alpha}{c^\alpha}
=
\frac{t^\alpha}{\lambda_p}.
$$

Therefore

$$
c^\alpha=\lambda_p
$$

and

$$
c=\lambda_p^{1/\alpha}.
$$

The paper's \(\lambda_p\) is the standard scale raised to the shape power:

$$
\lambda_p=c^\alpha.
$$

Unlike \(c\), \(\lambda_p\) has units of \(\text{time}^\alpha\), not time.

The median under the paper's parameterisation is

$$
m=\left(\lambda_p\ln 2\right)^{1/\alpha}.
$$

Solving for \(\lambda_p\):

$$
m^\alpha=\lambda_p\ln 2,
$$

so

$$
\lambda_p=\frac{m^\alpha}{\ln 2}.
$$

That is the formula used in `weibull_from_median_iqr`:

```python
lam_p = median**alpha / np.log(2.0)
```

## Side-by-side comparison

| Quantity | Standard form | Kottas form |
| --- | --- | --- |
| CDF | \(1-\exp[-(t/c)^\alpha]\) | \(1-\exp(-t^\alpha/\lambda_p)\) |
| Scale symbol here | \(c\) | \(\lambda_p\) |
| Common other symbol | \(\lambda\) (Wikipedia) | \(\lambda\) (paper; code `lam`) |
| Relationship | \(c=\lambda_p^{1/\alpha}\) | \(\lambda_p=c^\alpha\) |
| Median | \(m=c(\ln 2)^{1/\alpha}\) | \(m=(\lambda_p\ln 2)^{1/\alpha}\) |
| Scale from median | \(c=m/(\ln 2)^{1/\alpha}\) | \(\lambda_p=m^\alpha/\ln 2\) |
| Units of scale | time | \(\text{time}^\alpha\) |
| SciPy `weibull_min` | `weibull_min(c=alpha, scale=c)` | convert first: `scale=lam_p**(1/alpha)` |

## Density functions

The standard density is

$$
f(t\mid \alpha,c)
=
\frac{\alpha}{c}
\left(\frac{t}{c}\right)^{\alpha-1}
\exp\!\left[-\left(\frac{t}{c}\right)^\alpha\right].
$$

The paper's density is

$$
k_W(t\mid \alpha,\lambda_p)
=
\frac{\alpha t^{\alpha-1}}{\lambda_p}
\exp\!\left(-\frac{t^\alpha}{\lambda_p}\right).
$$

Substituting \(c=\lambda_p^{1/\alpha}\) into the standard density recovers the
paper's density, so the two are equivalent.

## Worked example

Take

$$
\alpha\approx 1.3314,
\qquad
m\approx 1.1727.
$$

Using the standard parameterisation:

$$
c
=
\frac{1.1727}{(\ln 2)^{1/1.3314}}
\approx 1.5443.
$$

Using the Kottas parameterisation:

$$
\lambda_p
=
\frac{1.1727^{1.3314}}{\ln 2}
\approx 1.7836.
$$

The two are related by

$$
c
=
\lambda_p^{1/\alpha}
=
1.7836^{1/1.3314}
\approx 1.5443.
$$

Both produce the same median:

$$
c(\ln 2)^{1/\alpha}
=
\lambda_p^{1/\alpha}(\ln 2)^{1/\alpha}
=
(\lambda_p\ln 2)^{1/\alpha}
\approx 1.1727.
$$

## Code checklist

Before using a Weibull `lambda`, check which CDF form you have:

1. If the CDF is \(1-\exp[-(t/c)^\alpha]\), the scale is the standard
   parameter \(c\).
2. If the CDF is \(1-\exp(-t^\alpha/\lambda)\), the scale is the paper's
   parameter \(\lambda_p\).
3. The median formulas are different:

   ```python
   # Standard Weibull scale c
   c = median / np.log(2.0) ** (1.0 / alpha)

   # Kottas paper scale lambda_p
   lam_p = median**alpha / np.log(2.0)
   ```

4. They are related by:

   ```python
   assert np.isclose(c, lam_p ** (1.0 / alpha))
   assert np.isclose(lam_p, c**alpha)
   ```

5. Do **not** pass `lam_p` directly as SciPy's `scale` unless you convert it:

   ```python
   from scipy.stats import weibull_min

   # Kottas distribution K_W(t | alpha, lam_p)
   dist = weibull_min(c=alpha, scale=lam_p ** (1.0 / alpha))
   ```

## Where this matters in the repository

The following functions in `src/dp_weibull.py` all use the Kottas
parameterisation:

- `weibull_cdf`
- `weibull_sf`
- `weibull_pdf`
- `DPRealization.cdf`
- `DPRealization.survival`
- `DPRealization.pdf`
- `DPRealization.hazard`
- `weibull_from_median_iqr`

The G0 sampler also uses the paper's inverse-gamma prior on \(\lambda_p\):

$$
\lambda_p\mid d,\gamma
\sim
\operatorname{InvGamma}(\text{shape}=d,\ \text{scale}=\gamma),
\qquad d=2.
$$

## References

- Kottas, A. (2002). Nonparametric Bayesian Survival Analysis using Mixtures
  of Weibull Distributions. Section 4.1.
- Wikipedia, “Weibull distribution”:
  <https://en.wikipedia.org/wiki/Weibull_distribution>
- SciPy, `scipy.stats.weibull_min`:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html>
