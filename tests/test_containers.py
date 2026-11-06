"""Test 5 (containers, plan Section 3.2): ``DPPrior`` and ``DPSettings``.

The two configuration containers are pure data, so the tests here pin the
contract the rest of the sampler relies on:

* ``DPPrior`` carries exactly the model-(5) hyperparameters, is frozen (a run
  must not mutate the prior) and supports value equality, so a golden test can
  compare whole configurations;
* ``DPSettings`` carries only sampler knobs, is mutable (a driver adjusts
  seeds and presets), and its defaults are numerically the ``paper`` preset of
  plan Section 6.2.

What a wrong container would break: a default that disagrees with Section 6.2
silently changes every reproduction number, and an accidentally frozen or
shared setting turns a driver's per-chain override into a no-op or a leak
between chains.
"""

import dataclasses

import pytest

from src.dp_gibbs import DPPrior, DPSettings


def section_5_1_prior() -> DPPrior:
    """The Section 5.1 prior guess: ``Gamma(2, 0.9)`` with the §5 constants."""
    return DPPrior(
        a_nu=2.0,
        b_nu=0.9,
        a_gamma=1.0,
        b_gamma=0.232235,
        a_phi=2.0,
        b_phi=1.77522,
    )


class TestDPPrior:
    """Model hyperparameters: frozen, value-equal, ``d`` defaulting to 2."""

    def test_default_d_is_two(self):
        """``d = 2`` is the paper's inverse-gamma shape (Section 2.1)."""
        assert section_5_1_prior().d == 2

    def test_d_can_be_overridden(self):
        """``d`` is a keyword with a default, not a required field."""
        prior = dataclasses.replace(section_5_1_prior(), d=3)

        assert prior.d == 3
        assert prior.a_nu == 2.0

    def test_field_names_match_the_model(self):
        """The field set is exactly ``a_*``/``b_*`` for nu, gamma, phi plus d."""
        names = {f.name for f in dataclasses.fields(DPPrior)}

        assert names == {
            "a_nu",
            "b_nu",
            "a_gamma",
            "b_gamma",
            "a_phi",
            "b_phi",
            "d",
        }

    def test_frozen(self):
        """Mutating the prior mid-run would silently change the target."""
        prior = section_5_1_prior()

        name = "a_nu"
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(prior, name, 3.0)

    def test_value_equality_and_hash(self):
        """Two runs built from the same constants are the same prior."""
        first = section_5_1_prior()
        second = section_5_1_prior()

        assert first == second
        assert hash(first) == hash(second)
        assert first != dataclasses.replace(first, b_nu=0.1)

    def test_holds_arbitrary_positive_constants(self):
        """Nothing is validated or coerced: the container is a plain record."""
        prior = DPPrior(
            a_nu=3,
            b_nu=0.05,
            a_gamma=1,
            b_gamma=0.828427,
            a_phi=2,
            b_phi=2.5,
            d=2,
        )

        assert prior.a_nu == 3
        assert prior.b_gamma == 0.828427


class TestDPSettings:
    """Sampler knobs: mutable, per-instance, defaults equal the paper preset."""

    def test_defaults_are_the_paper_preset(self):
        """Plan Section 6.2: burn 5,000, B 10,000, thin 150, M 256, 4 chains."""
        settings = DPSettings()

        assert settings.burn_in == 5_000
        assert settings.n_save == 10_000
        assert settings.thin == 150
        assert settings.n_alpha == 256
        assert settings.epsilon == 1e-4
        assert settings.n_chains == 4

    def test_seed_defaults_to_none(self):
        """``seed=None`` means "draw one from the OS entropy pool", not seed 0."""
        assert DPSettings().seed is None

    def test_preset_default_is_custom(self):
        """Defaults are not a named preset: the results note must say so."""
        assert DPSettings().preset == "custom"

    def test_mutable_knobs(self):
        """Drivers override knobs in place; a frozen container would break them."""
        settings = DPSettings()
        settings.seed = 20_240_101
        settings.preset = "quick"

        assert settings.seed == 20_240_101
        assert settings.preset == "quick"

    def test_instances_are_independent(self):
        """Two chains must not share one settings record."""
        first = DPSettings()
        second = DPSettings()
        first.seed = 7

        assert second.seed is None

    def test_value_equality(self):
        """Equal knobs compare equal, so presets can be compared directly."""
        quick = DPSettings(
            burn_in=1_000,
            n_save=1_000,
            thin=10,
            n_alpha=256,
            epsilon=1e-4,
            n_chains=2,
            preset="quick",
        )

        assert quick == dataclasses.replace(quick)
        assert quick != DPSettings()

    def test_field_names(self):
        """The knob vocabulary is fixed; a rename breaks stored configurations."""
        names = {f.name for f in dataclasses.fields(DPSettings)}

        assert names == {
            "burn_in",
            "n_save",
            "thin",
            "n_alpha",
            "epsilon",
            "n_chains",
            "seed",
            "preset",
        }
