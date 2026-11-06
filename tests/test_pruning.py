"""Test 23 (§7.1): pruning a negligible tail leaves the functionals unchanged.

A posterior realisation carries ``J`` atoms because the truncation rule is
fixed by the largest precision in the run, but almost all of those atoms have
weights far below any useful tolerance.  ``DPRealization.pruned`` drops a
trailing block whose total mass is at most ``atom_tol`` and renormalises the
rest.

The functionals here are recomputed from the atom locations and weights with
explicit closed-form Weibull formulas inside the test, so an error shared with
the implementation cannot cancel.  The survival and density errors are
bounded by the dropped mass; the hazard is only checked where the density is
not tiny, because the pruning error there is absolute while the hazard itself
is a ratio.
"""

import numpy as np
import pytest

from src.dp_weibull import DPRealization

N_HEAD = 5
N_TAIL = 494
N_ATOMS = N_HEAD + N_TAIL + 1
HEAD_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.0995])
TAIL_WEIGHT = 1e-9
RESIDUAL_WEIGHT = 1e-12


def make_long_tail_realization() -> DPRealization:
    """A realisation with five real atoms, a 494-atom tail and a tiny residual."""
    weights = np.concatenate(
        [
            HEAD_WEIGHTS,
            np.full(N_TAIL, TAIL_WEIGHT),
            np.array([RESIDUAL_WEIGHT]),
        ]
    )
    weights = weights / weights.sum()

    alphas = np.linspace(1.3, 1.9, N_ATOMS)
    lambdas = np.linspace(2.0, 4.0, N_ATOMS)
    return DPRealization(weights, alphas, lambdas, precision=224.6)


def closed_form_functionals(
    realization: DPRealization, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate S, f and h from the Weibull kernels without using the methods."""
    t = np.asarray(t, dtype=float)[None, :]
    alpha = realization.alphas[:, None]
    lam = realization.lambdas[:, None]
    z = np.power(t, alpha) / lam
    exp_neg_z = np.exp(-z)

    survival = np.sum(realization.weights[:, None] * exp_neg_z, axis=0)
    density = np.sum(
        realization.weights[:, None]
        * alpha
        * np.power(t, alpha - 1.0)
        / lam
        * exp_neg_z,
        axis=0,
    )
    # Use a guard so that the ratio is finite where S underflows to zero.
    hazard = np.divide(
        density, survival, out=np.zeros_like(density), where=survival > 0
    )
    return survival, density, hazard


def test_pruned_bounds_and_residual_atom() -> None:
    """Pruning a sub-tolerance tail changes S and f by less than the tolerance."""
    realization = make_long_tail_realization()
    atom_tol = 1e-6
    pruned = realization.pruned(atom_tol)

    t = np.linspace(0.05, 8.0, 400)
    survival, density, hazard = closed_form_functionals(realization, t)
    survival_p, density_p, hazard_p = closed_form_functionals(pruned, t)

    dropped_mass = float(realization.weights.sum() - realization.weights[:N_HEAD].sum())
    dropped_mass -= float(realization.weights[-1])
    assert 0.0 < dropped_mass <= atom_tol

    assert np.max(np.abs(survival - survival_p)) < atom_tol
    assert np.max(np.abs(density - density_p)) < atom_tol

    # The hazard ratio amplifies absolute errors wherever f is tiny; restrict
    # the bound to the region where the density is not negligible.
    informative = density > 0.05
    assert np.count_nonzero(informative) > 0
    assert np.max(np.abs(hazard[informative] - hazard_p[informative])) < 1e-4

    assert pruned.n_atoms == N_HEAD + 1
    assert pruned.n_atoms < realization.n_atoms
    assert np.isclose(np.sum(pruned.weights), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(pruned.weights >= 0.0)

    # The residual stick survives, at the same location and with positive mass.
    assert pruned.alphas[-1] == realization.alphas[-1]
    assert pruned.lambdas[-1] == realization.lambdas[-1]
    assert pruned.weights[-1] > 0.0
    assert pruned.precision == realization.precision


def test_closed_form_matches_realization_methods() -> None:
    """The independent formulas agree with the container's own evaluation."""
    realization = make_long_tail_realization()
    pruned = realization.pruned(1e-6)
    t = np.linspace(0.05, 8.0, 50)

    for item in (realization, pruned):
        survival, density, hazard = closed_form_functionals(item, t)
        assert np.allclose(item.survival(t), survival, rtol=0.0, atol=1e-14)
        assert np.allclose(item.pdf(t), density, rtol=0.0, atol=1e-14)
        assert np.allclose(item.hazard(t), hazard, rtol=0.0, atol=1e-14)


def test_tolerance_limits_the_dropped_block() -> None:
    """Atoms are dropped one at a time until the next would exceed the tolerance."""
    weights = np.array([0.5, 0.2, 3e-7, 3e-7, 3e-7, 0.3])
    alphas = np.linspace(1.3, 1.7, weights.size)
    lambdas = np.linspace(2.0, 3.0, weights.size)
    realization = DPRealization(weights, alphas, lambdas, precision=10.0)

    # Walking backwards, one 3e-7 atom fits under 5e-7 but two do not.
    pruned = realization.pruned(5e-7)

    assert pruned.n_atoms == 5
    assert pruned.alphas[-1] == alphas[-1]
    assert np.allclose(pruned.alphas, alphas[[0, 1, 2, 3, 5]])
    assert np.isclose(np.sum(pruned.weights), 1.0, rtol=0.0, atol=1e-15)


def test_no_pruning_when_the_tail_is_above_tolerance() -> None:
    """A short realisation with appreciable tail mass is only renormalised."""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    alphas = np.array([1.2, 1.4, 1.6, 1.8])
    lambdas = np.array([2.0, 2.5, 3.0, 3.5])
    realization = DPRealization(weights, alphas, lambdas, precision=5.0)

    pruned = realization.pruned(1e-6)

    assert pruned.n_atoms == 4
    assert np.allclose(pruned.weights, weights)
    assert np.allclose(pruned.alphas, alphas)
    assert np.allclose(pruned.lambdas, lambdas)


def test_input_weights_need_not_sum_to_one() -> None:
    """Pruning renormalises, including a realisation built on an unnormalised scale."""
    realization = DPRealization(
        np.array([0.8, 0.4, 0.8]),
        np.array([1.2, 1.5, 1.8]),
        np.array([2.0, 3.0, 4.0]),
        7.0,
    )

    pruned = realization.pruned(1e-6)

    assert np.isclose(np.sum(pruned.weights), 1.0, rtol=0.0, atol=1e-15)
    assert np.allclose(pruned.weights, np.array([0.8, 0.4, 0.8]) / 2.0)


def test_single_residual_atom_is_kept() -> None:
    """A one-atom realisation has only the residual stick, which stays."""
    realization = DPRealization(np.array([0.25]), np.array([1.5]), np.array([3.0]), 4.0)

    pruned = realization.pruned(1e-6)

    assert pruned.n_atoms == 1
    assert pruned.weights[0] == pytest.approx(1.0)
    assert pruned.alphas[0] == 1.5
    assert pruned.lambdas[0] == 3.0


@pytest.mark.parametrize("atom_tol", [0.0, -1e-6, 1.0, 1.5, float("nan")])
def test_rejects_invalid_tolerance(atom_tol: float) -> None:
    """The tolerance is a mass in (0, 1); anything else is a programming error."""
    realization = DPRealization(
        np.array([0.5, 0.5]), np.array([1.2, 1.8]), np.array([2.0, 4.0]), 3.0
    )

    with pytest.raises(ValueError):
        realization.pruned(atom_tol)


def test_rejects_realisation_without_mass() -> None:
    """Renormalisation is undefined when every weight is zero."""
    realization = DPRealization(
        np.zeros(2), np.array([1.2, 1.8]), np.array([2.0, 4.0]), 3.0
    )

    with pytest.raises(ValueError):
        realization.pruned(1e-6)
