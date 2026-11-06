"""Plan Section 3.7: the collapsed log-posterior ``log_post``.

The collapsed target is the unnormalised joint of the sampler in closed form:

    ``log p(theta*, s, nu, gamma, phi | D)
        = sum_j log g0(alpha*_j, lambda*_j | phi, gamma)
        + log p(s | nu)
        + sum_i log L_i(theta_{s_i})
        + log p(nu) + log p(gamma) + log p(phi)``,

with ``g0 = Uniform(alpha | 0, phi) * InvGamma(lambda | d, gamma)``, the EPPF
``p(s | nu) = nu**n_star prod_j (n_j - 1)! / (nu)_(n)`` and ``L_i`` the
Weibull density (uncensored) or survival (right-censored).

The tests compare the production value with a deliberately independent
reference built from ``scipy.stats`` (``uniform``, ``invgamma``, ``gamma``,
``pareto``, ``weibull_min``) and from the rising-factorial form of the EPPF,
so no code path of the module is reused for the expected value.  Four
mistakes that are silent on a trace are singled out:

* the ``g0`` term counted once per **observation** instead of once per
  cluster -- the merge test's delta contains exactly one ``g0`` term;
* the EPPF's two Gamma terms with the plan's inverted inline display (the
  term would grow with ``n`` instead of being a partition probability) -- the
  merge test uses the rising factorial
  ``(nu)_(n) = prod_{k=0}^{n-1} (nu + k)`` and the reference uses
  ``gammaln`` with the signs of the EPPF definition;
* the density used for a censored observation -- the event-flip test isolates
  one observation's contribution;
* a missing prior term -- the full reference includes all three hyperpriors.

The score must also be invariant to cluster numbering and to the order of the
observations, and it must refuse a state that is not a valid point of the
target (non-compacted, or ``phi < b_phi``).
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgamma, pareto, uniform, weibull_min

from src.dp_gibbs import (
    DPPrior,
    DPState,
    StateInvariantError,
    SurvData,
    log_post,
    make_surv_data,
)

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
    d=2,
)


def _data() -> SurvData:
    """A small mixed-censoring dataset on the transformed scale."""
    return make_surv_data(
        time=np.array([0.4, 0.8, 1.0, 1.6, 2.4, 3.0, 4.5, 5.2]),
        event=np.array([True, False, True, True, False, True, True, False]),
    )


def _state(
    labels: object,
    alpha_star: object,
    lam_star: object,
    *,
    nu: float = 1.5,
    gamma: float = 0.8,
    phi: float = 2.5,
) -> DPState:
    """Build a valid compacted state with the given partition and locations."""
    labels = np.asarray(labels, dtype=np.intp)
    alpha_star = np.asarray(alpha_star, dtype=np.float64)
    lam_star = np.asarray(lam_star, dtype=np.float64)
    n_j = np.bincount(labels, minlength=alpha_star.size).astype(np.intp)
    if np.any(n_j == 0):
        raise ValueError("the test state must not contain an empty cluster")
    return DPState(
        alpha=alpha_star[labels].copy(),
        lam=lam_star[labels].copy(),
        s=labels.copy(),
        alpha_star=alpha_star.copy(),
        lam_star=lam_star.copy(),
        n_j=n_j,
        nu=nu,
        gamma=gamma,
        phi=phi,
    )


def _reference_log_post(state: DPState, data: SurvData, prior: DPPrior) -> float:
    """Independent scalar reference for the collapsed log-posterior.

    ``g0``, the likelihood and the three hyperpriors come from ``scipy.stats``
    distributions, never from the module's arithmetic.  The EPPF is evaluated
    in its rising-factorial form,
    ``n_star log nu + sum_j gammaln(n_j) - sum_{k=0}^{n-1} log(nu + k)``, which
    shares only ``numpy`` with the production ``lgamma`` expression.  The
    Weibull is ``weibull_min`` with ``scale = lambda**(1 / alpha)``, so its
    density and survival are an arithmetic path independent of the module's
    ``t**alpha / lambda``.
    """
    total = 0.0
    for j in range(state.n_star):
        total += float(uniform.logpdf(state.alpha_star[j], loc=0.0, scale=state.phi))
        total += float(invgamma.logpdf(state.lam_star[j], a=prior.d, scale=state.gamma))

    total += state.n_star * math.log(state.nu)
    total += float(np.sum(gammaln(np.asarray(state.n_j, dtype=np.float64))))
    total -= float(np.sum(np.log(state.nu + np.arange(state.n))))

    for i in range(state.n):
        scale = float(state.lam[i]) ** (1.0 / float(state.alpha[i]))
        if data.event[i]:
            total += float(
                weibull_min.logpdf(data.time[i], c=state.alpha[i], scale=scale)
            )
        else:
            total += float(
                weibull_min.logsf(data.time[i], c=state.alpha[i], scale=scale)
            )

    total += float(gamma_dist.logpdf(state.nu, a=prior.a_nu, scale=1.0 / prior.b_nu))
    total += float(
        gamma_dist.logpdf(state.gamma, a=prior.a_gamma, scale=1.0 / prior.b_gamma)
    )
    total += float(pareto.logpdf(state.phi, b=prior.a_phi, scale=prior.b_phi))
    return total


CASES = [
    pytest.param(
        _state(
            labels=[0, 0, 1, 1, 1, 2, 2, 2],
            alpha_star=[0.6, 0.9, 1.4],
            lam_star=[0.7, 1.5, 2.2],
        ),
        id="three-clusters-mixed-censoring",
    ),
    pytest.param(
        _state(
            labels=[0, 0, 0, 0, 0, 0, 0, 0],
            alpha_star=[1.1],
            lam_star=[1.9],
        ),
        id="one-cluster",
    ),
    pytest.param(
        _state(
            labels=[0, 1, 0, 2, 2, 1, 3, 3],
            alpha_star=[0.4, 0.8, 1.0, 2.0],
            lam_star=[0.5, 5.0, 1.2, 0.3],
            nu=0.7,
            gamma=1.4,
            phi=2.2,
        ),
        id="four-clusters-distinct-scales",
    ),
]


@pytest.mark.parametrize("state", CASES)
def test_matches_independent_reference(state: DPState) -> None:
    """``log_post`` equals the ``scipy``/rising-factorial reference."""
    data = _data()
    expected = _reference_log_post(state, data, PRIOR)
    got = log_post(state, data, PRIOR)
    assert isinstance(got, float)
    assert np.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_merging_identical_clusters_changes_only_g0_and_eppf() -> None:
    """The delta of merging identical locations is one ``g0`` term plus EPPF.

    The two clusters of ``state_split`` have exactly the same location, so
    every observation keeps its ``(alpha_i, lambda_i)`` and the likelihood is
    unchanged.  Merging them must therefore remove exactly **one** ``g0``
    factor and replace the EPPF term of the split partition by the merged
    one:

        ``Delta = log g0(alpha*, lambda*) + log nu
            + lgamma(n_0) + lgamma(n_1) - lgamma(n_0 + n_1)``.

    A ``g0`` term counted per observation would remove ``n_1`` factors instead
    of one; a missing EPPF would remove the partition part of the delta.
    """
    data = _data()
    alpha_star = 0.9
    lam_star = 1.4
    labels_split = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.intp)
    split = _state(labels_split, [alpha_star, alpha_star], [lam_star, lam_star])
    merged = _state(np.zeros(8, dtype=np.intp), [alpha_star], [lam_star])

    n_0 = int(np.count_nonzero(labels_split == 0))
    n_1 = int(np.count_nonzero(labels_split == 1))
    one_g0_term = float(uniform.logpdf(alpha_star, loc=0.0, scale=split.phi)) + float(
        invgamma.logpdf(lam_star, a=PRIOR.d, scale=split.gamma)
    )
    expected_delta = (
        one_g0_term
        + math.log(split.nu)
        + float(gammaln(n_0))
        + float(gammaln(n_1))
        - float(gammaln(n_0 + n_1))
    )

    got_delta = log_post(split, data, PRIOR) - log_post(merged, data, PRIOR)
    assert got_delta == pytest.approx(expected_delta, rel=1e-12, abs=1e-12)


def test_label_and_observation_order_invariance() -> None:
    """Renumbering clusters and permuting observations leaves the score intact."""
    data = _data()
    state = _state(
        labels=[0, 0, 1, 1, 1, 2, 2, 2],
        alpha_star=[0.6, 0.9, 1.4],
        lam_star=[0.7, 1.5, 2.2],
    )
    permutation = np.array([7, 0, 5, 2, 6, 1, 4, 3], dtype=np.intp)
    # Reverse the cluster numbering: 0 -> 2, 1 -> 0, 2 -> 1.  ``relabel``
    # maps old to new, so the new location arrays are indexed by its inverse.
    relabel = np.array([2, 0, 1], dtype=np.intp)
    old_of_new = np.argsort(relabel)

    permuted = _state(
        labels=relabel[state.s[permutation]],
        alpha_star=state.alpha_star[old_of_new],
        lam_star=state.lam_star[old_of_new],
        nu=state.nu,
        gamma=state.gamma,
        phi=state.phi,
    )
    permuted_data = SurvData(
        time=data.time[permutation],
        event=data.event[permutation],
        t_max=data.t_max,
    )

    assert log_post(permuted, permuted_data, PRIOR) == pytest.approx(
        log_post(state, data, PRIOR), rel=1e-12, abs=1e-12
    )


def test_censored_observation_contributes_log_survival() -> None:
    """Flipping one observation's indicator changes its term by logsf - logpdf."""
    data = _data()
    state = _state(
        labels=[0, 0, 1, 1, 1, 2, 2, 2],
        alpha_star=[0.6, 0.9, 1.4],
        lam_star=[0.7, 1.5, 2.2],
    )
    i = 1  # censored in ``data``: the survival branch is in force
    flipped_data = SurvData(
        time=data.time,
        event=np.where(np.arange(data.time.size) == i, True, data.event),
        t_max=data.t_max,
    )
    scale = float(state.lam[i]) ** (1.0 / float(state.alpha[i]))
    expected_delta = float(
        weibull_min.logpdf(data.time[i], c=state.alpha[i], scale=scale)
    ) - float(weibull_min.logsf(data.time[i], c=state.alpha[i], scale=scale))

    got_delta = log_post(state, flipped_data, PRIOR) - log_post(state, data, PRIOR)
    assert got_delta == pytest.approx(expected_delta, rel=1e-12, abs=1e-12)


def test_rejects_non_compacted_state() -> None:
    """A state left in the deferred-removal form is not a point of the target."""
    data = _data()
    state = _state(
        labels=[0, 0, 1, 1, 1, 2, 2, 2],
        alpha_star=[0.6, 0.9, 1.4],
        lam_star=[0.7, 1.5, 2.2],
    )
    state.remove_observation(0)
    with pytest.raises(StateInvariantError):
        log_post(state, data, PRIOR)


def test_rejects_phi_below_prior_lower_bound() -> None:
    """Outside the Pareto support there is no finite target density."""
    data = _data()
    state = _state(
        labels=np.zeros(8, dtype=np.intp),
        alpha_star=[0.4],
        lam_star=[1.0],
        phi=0.6,
    )
    with pytest.raises(StateInvariantError):
        log_post(state, data, PRIOR)


def test_rejects_data_state_mismatch() -> None:
    """A state scored against another dataset must fail at the boundary."""
    state = _state(
        labels=[0, 0, 1],
        alpha_star=[0.6, 0.9],
        lam_star=[0.7, 1.5],
    )
    other = make_surv_data(
        time=np.array([1.0, 2.0, 3.0, 4.0]), event=np.ones(4, dtype=bool)
    )
    with pytest.raises(ValueError, match="same number of observations"):
        log_post(state, other, PRIOR)


@pytest.mark.parametrize(
    "prior",
    [
        pytest.param(dataclasses.replace(PRIOR, a_nu=0.0), id="a-nu-zero"),
        pytest.param(dataclasses.replace(PRIOR, b_gamma=-1.0), id="b-gamma-negative"),
        pytest.param(dataclasses.replace(PRIOR, a_phi=float("nan")), id="a-phi-nan"),
        pytest.param(dataclasses.replace(PRIOR, d=0), id="d-zero"),
    ],
)
def test_rejects_invalid_prior_hyperparameters(prior: DPPrior) -> None:
    """Every density in the target must have finite, positive parameters."""
    data = _data()
    state = _state(
        labels=np.zeros(8, dtype=np.intp),
        alpha_star=[0.9],
        lam_star=[1.4],
    )
    with pytest.raises(ValueError):
        log_post(state, data, prior)
