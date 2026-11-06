"""Test 6 (plan Sections 2.1 and 7.1): cluster bookkeeping and invariants.

The collapsed urn of the Appendix mutates the partition on every observation:
step (a) decrements the old cluster, may drop it, and then either reuses an
existing location or appends a new one.  A count that drifts from ``s``, or a
label that survives a compaction it should not, corrupts the urn weights and
then every later conditional -- invisibly, because the sampler still runs.

This module pins the state contract:

* random valid states satisfy the four Section 2.1 invariants;
* the deferred-compaction sequence ``remove -> assign -> compact`` keeps
  ``n_j``, ``s`` and the cluster locations in sync, and returns the
  old-label -> new-label map needed to permute cached ``log q`` columns;
* compaction drops empty clusters, renumbers the rest and preserves the
  partition and every per-observation location;
* 1,000 random remove/assign/append operations leave ``sum(n_j) == n`` and,
  after a final compaction, all invariants;
* each invariant detector fires on the specific corruption it is meant to
  catch.

The property-based tests use seeded NumPy generators (the repository has no
Hypothesis dependency); the operation sequence is random, the properties are
deterministic.
"""

import dataclasses

import numpy as np
import pytest

from src.dp_gibbs import DPPrior, DPState, StateInvariantError

PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.77522,
)


def dp_state(
    *,
    alpha: object,
    lam: object,
    s: object,
    alpha_star: object,
    lam_star: object,
    n_j: object,
    nu: float,
    gamma: float,
    phi: float,
) -> DPState:
    """Construct from array-like fields, keeping invalid dtypes visible.

    The production container accepts arrays; tests are clearer with Python
    lists, so this wrapper converts the numeric fields but leaves ``s`` and
    ``n_j`` as given (a float label must still reach the validator).
    """
    return DPState(
        alpha=np.asarray(alpha, dtype=np.float64),
        lam=np.asarray(lam, dtype=np.float64),
        s=np.asarray(s),
        alpha_star=np.asarray(alpha_star, dtype=np.float64),
        lam_star=np.asarray(lam_star, dtype=np.float64),
        n_j=np.asarray(n_j),
        nu=nu,
        gamma=gamma,
        phi=phi,
    )


def make_state(
    s: object,
    alpha_star: object,
    lam_star: object,
    *,
    phi: float = 2.5,
    nu: float = 2.5,
    gamma: float = 0.5,
) -> DPState:
    """Build a state from a partition, deriving every dependent field."""
    s = np.asarray(s, dtype=np.intp)
    alpha_star = np.asarray(alpha_star, dtype=np.float64)
    lam_star = np.asarray(lam_star, dtype=np.float64)
    n_j = np.bincount(s, minlength=alpha_star.size).astype(np.intp)
    return dp_state(
        alpha=alpha_star[s],
        lam=lam_star[s],
        s=s,
        alpha_star=alpha_star,
        lam_star=lam_star,
        n_j=n_j,
        nu=nu,
        gamma=gamma,
        phi=phi,
    )


def random_state(rng: np.random.Generator) -> DPState:
    """Draw a valid state with a random partition and random locations."""
    n = int(rng.integers(1, 25))
    n_star = int(rng.integers(1, n + 1))
    # Start with one observation in every cluster so that no cluster is empty,
    # then scatter the remaining observations.
    s = np.concatenate(
        [
            np.arange(n_star, dtype=np.intp),
            rng.integers(0, n_star, size=n - n_star, dtype=np.intp),
        ]
    )
    rng.shuffle(s)
    phi = float(rng.uniform(PRIOR.b_phi, 2.0 * PRIOR.b_phi))
    alpha_star = rng.uniform(1e-3, phi, size=n_star)
    lam_star = np.exp(rng.uniform(-2.0, 2.0, size=n_star))
    return make_state(
        s,
        alpha_star,
        lam_star,
        phi=phi,
        nu=float(rng.uniform(1e-2, 10.0)),
        gamma=float(rng.uniform(1e-2, 5.0)),
    )


def assert_counts_and_locations(state: DPState) -> None:
    """Assert the subset of invariants that must hold between remove/assign."""
    np.testing.assert_array_equal(state.alpha, state.alpha_star[state.s])
    np.testing.assert_array_equal(state.lam, state.lam_star[state.s])
    assert int(state.n_j.sum()) == state.n


class TestConstruction:
    """A hand-built state must already satisfy Section 2.1."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_states_satisfy_invariants(self, seed):
        """Every generated state passes the full check, including ``b_phi``."""
        state = random_state(np.random.default_rng(seed))

        state.check_invariants(PRIOR)

    def test_n_and_n_star(self):
        """The two sizes are the observation count and the cluster count."""
        state = make_state([0, 0, 1], [1.0, 1.5], [2.0, 3.0])

        assert state.n == 3
        assert state.n_star == 2

    def test_arrays_are_owned_and_normalised(self):
        """Construction copies and dtypes the arrays so aliases cannot drift."""
        alpha_star = np.array([1.0, 1.5])
        lam_star = np.array([2.0, 3.0])
        state = make_state([0, 0, 1], alpha_star, lam_star)

        assert state.alpha.dtype == np.float64
        assert state.lam.dtype == np.float64
        assert state.s.dtype == np.intp
        assert state.n_j.dtype == np.intp
        alpha_star[0] = 0.25
        assert state.alpha_star[0] == 1.0

    def test_integer_lists_are_accepted(self):
        """Plain Python lists are a supported way to build a test state."""
        state = dp_state(
            alpha=[1.0, 1.5],
            lam=[2.0, 3.0],
            s=[0, 1],
            alpha_star=[1.0, 1.5],
            lam_star=[2.0, 3.0],
            n_j=[1, 1],
            nu=2.0,
            gamma=0.5,
            phi=2.5,
        )

        state.check_invariants(PRIOR)

    def test_rejects_nonpositive_phi(self):
        """Invariant 4: ``phi`` is the upper bound of the uniform alpha prior."""
        with pytest.raises(ValueError, match="phi must be a finite"):
            dp_state(
                alpha=[1.0],
                lam=[2.0],
                s=[0],
                alpha_star=[1.0],
                lam_star=[2.0],
                n_j=[1],
                nu=2.0,
                gamma=0.5,
                phi=0.0,
            )

    def test_rejects_nan_location(self):
        """A NaN location would poison every later ``q_j`` evaluation."""
        with pytest.raises(ValueError, match="alpha_star must contain only finite"):
            dp_state(
                alpha=[1.0],
                lam=[2.0],
                s=[0],
                alpha_star=[np.nan],
                lam_star=[2.0],
                n_j=[1],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_float_labels(self):
        """A non-integral label would silently truncate on indexing."""
        with pytest.raises(ValueError, match="s must contain integers"):
            dp_state(
                alpha=[1.0],
                lam=[2.0],
                s=[0.5],
                alpha_star=[1.0],
                lam_star=[2.0],
                n_j=[1],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_length_mismatch(self):
        """``alpha`` and ``s`` describe the same observations."""
        with pytest.raises(ValueError, match="alpha and s must have the same length"):
            dp_state(
                alpha=[1.0, 1.0],
                lam=[2.0],
                s=[0],
                alpha_star=[1.0],
                lam_star=[2.0],
                n_j=[1],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_empty_state(self):
        """An empty chain has no urn to update."""
        with pytest.raises(ValueError, match="alpha must not be empty"):
            dp_state(
                alpha=[],
                lam=[],
                s=[],
                alpha_star=[],
                lam_star=[],
                n_j=[],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_count_mismatch(self):
        """``n_j`` is authoritative for drop decisions; it must start correct."""
        with pytest.raises(StateInvariantError, match="n_j is not bincount"):
            dp_state(
                alpha=[1.0],
                lam=[2.0],
                s=[0],
                alpha_star=[1.0],
                lam_star=[2.0],
                n_j=[2],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_empty_cluster(self):
        """A cluster with count zero is not a valid post-compaction state."""
        with pytest.raises(StateInvariantError, match="invariant 2.*non-empty"):
            dp_state(
                alpha=[1.0],
                lam=[2.0],
                s=[0],
                alpha_star=[1.0, 1.5],
                lam_star=[2.0, 3.0],
                n_j=[1, 0],
                nu=2.0,
                gamma=0.5,
                phi=2.5,
            )

    def test_rejects_alpha_above_phi(self):
        """Invariant 3: the uniform support is ``0 < alpha <= phi``."""
        with pytest.raises(StateInvariantError, match="must not exceed phi"):
            make_state([0], [3.0], [2.0], phi=2.5)


class TestCompaction:
    """One relabel pass drops empty clusters and renumbers the partition."""

    def test_canonical_state_maps_identically(self):
        """With no empty cluster, compaction must be a no-op."""
        state = make_state([0, 0, 1, 2], [1.0, 1.5, 2.0], [2.0, 3.0, 4.0])

        old_to_new = state.compact()

        np.testing.assert_array_equal(old_to_new, np.arange(3))
        np.testing.assert_array_equal(state.s, [0, 0, 1, 2])
        state.check_invariants(PRIOR)

    def test_drops_cluster_emptied_by_a_move(self):
        """A moved singleton's old cluster disappears and labels renumber."""
        state = make_state(
            [0, 1, 2, 3],
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            phi=2.5,
        )
        state.remove_observation(1)
        assert state.n_j.tolist() == [1, 0, 1, 1]
        state.assign_existing(1, 2)
        assert state.n_j.tolist() == [1, 0, 2, 1]
        old_to_new = state.compact()

        np.testing.assert_array_equal(old_to_new, [0, -1, 1, 2])
        np.testing.assert_array_equal(state.s, [0, 1, 1, 2])
        np.testing.assert_array_equal(state.n_j, [1, 2, 1])
        np.testing.assert_array_equal(state.alpha_star, [0.5, 1.5, 2.0])
        np.testing.assert_array_equal(state.lam_star, [1.0, 3.0, 4.0])
        # The moved observation takes its new cluster's location; every other
        # observation keeps the value it had before the relabel pass.
        np.testing.assert_array_equal(state.alpha, [0.5, 1.5, 1.5, 2.0])
        np.testing.assert_array_equal(state.lam, [1.0, 3.0, 3.0, 4.0])
        state.check_invariants(PRIOR)

    def test_drops_a_cluster_appended_but_left_empty(self):
        """The step-(a) new-cluster branch can be abandoned harmlessly."""
        state = make_state([0, 0, 1], [1.0, 1.5], [2.0, 3.0])

        j = state.append_cluster(2.0, 4.0)
        assert j == 2
        old_to_new = state.compact()

        np.testing.assert_array_equal(old_to_new, [0, 1, -1])
        assert state.n_star == 2
        state.check_invariants(PRIOR)

    def test_refuses_to_compact_with_a_pending_removal(self):
        """Compacting between remove and assign would drop a live observation."""
        state = make_state([0, 1], [1.0, 1.5], [2.0, 3.0])
        state.remove_observation(1)

        with pytest.raises(StateInvariantError, match="still labels an empty"):
            state.compact()

    @pytest.mark.parametrize("seed", range(20))
    def test_random_compaction_preserves_partition_and_locations(self, seed):
        """Membership and per-observation values survive the relabel pass."""
        rng = np.random.default_rng(500 + seed)
        state = random_state(rng)
        # Move a random set of observations, half to existing clusters and half
        # to fresh ones, so the old partition has a mix of empty and occupied
        # clusters.  Records the labels immediately before compaction.
        for i in rng.choice(
            state.n, size=int(rng.integers(1, state.n + 1)), replace=False
        ):
            i = int(i)
            state.remove_observation(i)
            if rng.random() < 0.5:
                state.assign_existing(i, int(rng.integers(0, state.n_star)))
            else:
                state.assign_new(
                    i,
                    float(rng.uniform(1e-3, state.phi)),
                    float(rng.uniform(0.1, 5.0)),
                )
        old_n_star = state.n_star
        old_s = state.s.copy()
        old_n_j = state.n_j.copy()

        old_to_new = state.compact()
        state.check_invariants(PRIOR)

        # The map is a relabelling: dropped labels go to -1, kept labels are
        # exactly 0..m-1 in increasing <-> increasing order.
        assert old_to_new.size == old_n_star
        keep_mask = old_to_new >= 0
        kept = old_to_new[keep_mask]
        np.testing.assert_array_equal(kept, np.arange(kept.size))
        assert state.n_star == kept.size
        assert np.all(old_n_j[~keep_mask] == 0)
        # Labels and counts are exactly the old ones pushed through the map.
        np.testing.assert_array_equal(state.s, old_to_new[old_s])
        np.testing.assert_array_equal(state.n_j, old_n_j[keep_mask])
        np.testing.assert_array_equal(state.alpha, state.alpha_star[state.s])
        np.testing.assert_array_equal(state.lam, state.lam_star[state.s])

    @pytest.mark.parametrize("seed", range(10))
    def test_compaction_is_idempotent(self, seed):
        """A second pass finds the canonical labels already in place."""
        state = random_state(np.random.default_rng(700 + seed))
        state.compact()

        first = state.s.copy()
        second = state.compact()

        np.testing.assert_array_equal(state.s, first)
        np.testing.assert_array_equal(second, np.arange(state.n_star))


class TestRemoveAndAssign:
    """The primitive mutations used by step (a)."""

    def test_remove_decrements_without_relabelling(self):
        """Deferred compaction keeps the old label available for candidates."""
        state = make_state([0, 0, 1], [1.0, 1.5], [2.0, 3.0])

        removed = state.remove_observation(0)

        assert removed == 0
        assert state.s[0] == 0
        assert state.n_j.tolist() == [1, 1]
        assert int(state.n_j.sum()) == 2
        np.testing.assert_array_equal(state.alpha, state.alpha_star[state.s])

    def test_assign_existing_restores_the_count_and_copies_the_location(self):
        """Reusing a cluster copies its location, so invariant 1 holds at once."""
        state = make_state([0, 1], [1.0, 1.5], [2.0, 3.0])
        state.remove_observation(1)
        # Refresh cluster 0's location (step (b) does this to all members).
        state.alpha_star[0] = 1.25
        state.lam_star[0] = 2.25
        state.alpha[0] = 1.25
        state.lam[0] = 2.25

        state.assign_existing(1, 0)

        assert state.s.tolist() == [0, 0]
        assert state.n_j.tolist() == [2, 0]
        assert state.alpha[1] == 1.25
        assert state.lam[1] == 2.25
        assert_counts_and_locations(state)

        old_to_new = state.compact()
        assert old_to_new.tolist() == [0, -1]
        assert state.n_j.tolist() == [2]
        state.check_invariants(PRIOR)

    def test_assign_new_appends_a_singleton_cluster(self):
        """The new-cluster branch creates a label and assigns in one call."""
        state = make_state([0, 0], [1.0], [2.0])
        state.remove_observation(1)

        j = state.assign_new(1, 1.5, 3.0)

        assert j == 1
        assert state.s.tolist() == [0, 1]
        assert state.n_j.tolist() == [1, 1]
        assert state.alpha_star.tolist() == [1.0, 1.5]
        assert state.lam_star.tolist() == [2.0, 3.0]
        state.check_invariants(PRIOR)

    def test_assign_new_drops_the_old_singleton_on_compaction(self):
        """Moving the only member of an old cluster leaves it to be dropped."""
        state = make_state([0, 1], [1.0, 1.5], [2.0, 3.0])
        state.remove_observation(1)

        j = state.assign_new(1, 2.0, 4.0)

        assert j == 2
        assert state.n_j.tolist() == [1, 0, 1]
        old_to_new = state.compact()
        assert old_to_new.tolist() == [0, -1, 1]
        assert state.s.tolist() == [0, 1]
        assert state.n_j.tolist() == [1, 1]
        assert state.alpha_star.tolist() == [1.0, 2.0]
        state.check_invariants(PRIOR)

    def test_double_remove_is_rejected(self):
        """Removing an observation twice would under-count its cluster."""
        state = make_state([0, 1], [1.0, 1.5], [2.0, 3.0])
        state.remove_observation(1)

        with pytest.raises(StateInvariantError, match="already removed"):
            state.remove_observation(1)

    def test_assign_before_remove_is_rejected(self):
        """Assigning an already-counted observation would over-count it."""
        state = make_state([0, 1], [1.0, 1.5], [2.0, 3.0])

        with pytest.raises(StateInvariantError, match="before remove_observation"):
            state.assign_existing(0, 1)

    def test_append_cluster_rejects_alpha_above_phi(self):
        """Invariant 3 applies to the new location as well."""
        state = make_state([0], [1.0], [2.0], phi=2.5)

        with pytest.raises(StateInvariantError, match="must not exceed phi"):
            state.append_cluster(2.6, 1.0)

    def test_append_cluster_rejects_nonpositive_scale(self):
        """Invariant 4 applies to the new location as well."""
        state = make_state([0], [1.0], [2.0])

        with pytest.raises(ValueError, match="lam must be a finite"):
            state.append_cluster(1.0, 0.0)

    def test_out_of_range_observation(self):
        """A stale index after a data change must fail loudly."""
        state = make_state([0], [1.0], [2.0])

        with pytest.raises(IndexError, match="observation index 1"):
            state.remove_observation(1)

    def test_out_of_range_cluster(self):
        """A stale label after compaction must fail loudly."""
        state = make_state([0], [1.0], [2.0])

        with pytest.raises(IndexError, match="cluster index 1"):
            state.assign_existing(0, 1)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_operation_sequence_keeps_counts(self, seed):
        """1,000 mixed remove/assign/append operations preserve ``sum(n_j)``."""
        rng = np.random.default_rng(900 + seed)
        state = random_state(rng)

        for step in range(1_000):
            i = int(rng.integers(0, state.n))
            state.remove_observation(i)
            assert int(state.n_j.sum()) == state.n - 1
            np.testing.assert_array_equal(state.alpha, state.alpha_star[state.s])
            np.testing.assert_array_equal(state.lam, state.lam_star[state.s])

            if rng.random() < 0.7:
                state.assign_existing(i, int(rng.integers(0, state.n_star)))
            else:
                state.assign_new(
                    i,
                    float(rng.uniform(1e-3, state.phi)),
                    float(rng.uniform(1e-3, 10.0)),
                )

            assert int(state.n_j.sum()) == state.n
            assert_counts_and_locations(state)

            # Compact periodically so empty clusters are dropped rather than
            # accumulating for the whole run.
            if step % 25 == 24:
                state.compact()

        state.compact()
        state.check_invariants(PRIOR)


class TestInvariantDetection:
    """Each Section 2.1 invariant has a detector that names the violation."""

    def test_alpha_not_matching_star(self):
        state = random_state(np.random.default_rng(1))
        state.alpha[0] = state.alpha[0] / 2.0

        with pytest.raises(StateInvariantError, match="invariant 1.*alpha"):
            state.check_invariants(PRIOR)

    def test_lam_not_matching_star(self):
        state = random_state(np.random.default_rng(2))
        state.lam[0] = state.lam[0] + 1.0

        with pytest.raises(StateInvariantError, match="invariant 1.*lam"):
            state.check_invariants(PRIOR)

    def test_count_drift(self):
        state = random_state(np.random.default_rng(3))
        state.n_j[0] += 1

        with pytest.raises(StateInvariantError, match="invariant 2.*n_j"):
            state.check_invariants(PRIOR)

    def test_label_out_of_range(self):
        state = random_state(np.random.default_rng(4))
        state.s[0] = state.n_star

        with pytest.raises(StateInvariantError, match="labels must lie"):
            state.check_invariants(PRIOR)

    def test_empty_cluster(self):
        state = random_state(np.random.default_rng(5))
        if state.n_star == 1:
            state.append_cluster(0.5, 1.0)
        state.n_j[-1] = 0

        with pytest.raises(StateInvariantError, match="invariant 2"):
            state.check_invariants(PRIOR)

    def test_alpha_above_phi(self):
        state = random_state(np.random.default_rng(6))
        state.alpha_star[0] = state.phi + 1.0
        state.alpha[state.s == 0] = state.phi + 1.0

        with pytest.raises(StateInvariantError, match="invariant 3.*exceed phi"):
            state.check_invariants(PRIOR)

    def test_nonpositive_lambda(self):
        state = random_state(np.random.default_rng(7))
        state.lam_star[0] = 0.0
        state.lam[state.s == 0] = 0.0

        with pytest.raises(StateInvariantError, match="invariant 4.*positive"):
            state.check_invariants(PRIOR)

    def test_nonpositive_precision(self):
        state = random_state(np.random.default_rng(8))
        state.nu = 0.0

        with pytest.raises(StateInvariantError, match="invariant 4.*positive"):
            state.check_invariants(PRIOR)

    def test_phi_below_pareto_bound_only_checked_with_a_prior(self):
        """Invariant 4's ``phi >= b_phi`` needs the prior's Pareto scale."""
        phi = PRIOR.b_phi / 2.0
        state = make_state([0], [phi / 2.0], [1.0], phi=phi)

        state.check_invariants()
        with pytest.raises(StateInvariantError, match="at least prior.b_phi"):
            state.check_invariants(PRIOR)

    def test_dataclass_fields_match_the_plan(self):
        """The state vocabulary is fixed by Section 2.1."""
        names = {field.name for field in dataclasses.fields(DPState)}

        assert names == {
            "alpha",
            "lam",
            "s",
            "alpha_star",
            "lam_star",
            "n_j",
            "nu",
            "gamma",
            "phi",
        }
