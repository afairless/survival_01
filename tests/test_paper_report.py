"""Unit tests for the artifact report payloads (plan §6.3, row 8).

The report module turns a loaded ``DemoResult`` into the numbers the committed
report quotes.  These tests pin four contracts:

* the two summaries expose every Figure 1-5 quantity, with one row per fitted
  prior and JSON-safe values (``tests/test_paper_report.py`` is the drift guard
  for the acceptance target table mirroring the paper tests);
* the first-500-draw slice is built with ``dataclasses.replace`` and keeps the
  full trace's ``J_common``/``J_prior`` -- slicing the stored functional arrays
  must not silently change the truncation level;
* the diagnostics on the slice reproduce ``r_hat``/``ess`` computed directly
  from the sliced chains, and leave the inputs untouched;
* the baseline delta names the rows that cannot match by construction and uses
  the expected sign for every comparable row.

The fixtures are deliberately synthetic: a report payload is a pure function
of chains and functionals, so a tiny deterministic ``DemoResult`` exercises it
without a second of MCMC.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import paper_report
from src.dp_functionals import Functionals
from src.dp_gibbs import DPChains, DPPrior, DPSettings, SurvData, ess, r_hat
from src.paper_examples import REPO_ROOT, DemoResult, Fit, prior_label
from tests import test_paper_5_1, test_paper_5_2

PRIOR_TABLE = ((2.0, 0.9), (2.0, 0.1), (3.0, 0.05))


# ---------------------------------------------------------------------------
# Synthetic containers
# ---------------------------------------------------------------------------
def _prior(index: int) -> DPPrior:
    """The ``index``-th Section 5 ``nu`` prior with the paper's constants."""
    a_nu, b_nu = PRIOR_TABLE[index]
    return DPPrior(
        a_nu=a_nu,
        b_nu=b_nu,
        a_gamma=1.0,
        b_gamma=0.232235,
        a_phi=2.0,
        b_phi=1.0,
        d=2,
    )


def _chain(
    *,
    n_draws: int = 8,
    n: int = 6,
    seed: int = 1,
    prior: DPPrior | None = None,
    t_max: float = 5.0,
) -> DPChains:
    """A small chain that satisfies every ``DPChains`` invariant."""
    rng = np.random.default_rng(seed)
    data = SurvData(
        time=np.linspace(1.05, 1.9, n),
        event=np.ones(n, dtype=bool),
        t_max=t_max,
    )
    alpha = rng.uniform(0.4, 0.9, size=(n_draws, n))
    phi = rng.uniform(1.2, 2.0, size=n_draws)
    lam = rng.uniform(0.5, 3.0, size=(n_draws, n))
    return DPChains(
        data=data,
        prior=_prior(0) if prior is None else prior,
        settings=DPSettings(
            burn_in=5,
            n_save=n_draws,
            thin=1,
            n_alpha=16,
            epsilon=1e-4,
            n_chains=2,
            seed=seed,
            preset="quick",
        ),
        theta=np.stack([alpha, lam], axis=2),
        nu=rng.uniform(0.5, 3.0, size=n_draws),
        gamma=rng.uniform(0.2, 1.5, size=n_draws),
        phi=phi,
        n_star=rng.integers(1, n + 1, size=n_draws).astype(np.intp),
        log_post=-rng.uniform(1.0, 10.0, size=n_draws),
    )


def _functionals(
    t_grid: np.ndarray,
    n_draws: int,
    *,
    seed: int = 1,
    low: float = 0.15,
    high: float = 0.25,
    bump: bool = False,
    j_common: int = 40,
    j_prior: int = 11,
) -> Functionals:
    """Synthetic survival/density/hazard arrays with a monotone CDF.

    ``S_b(t) = exp(-rate_b t)`` makes ``F = 1 - S`` non-decreasing, which
    ``compare`` requires.  The hazard carries a small deterministic wiggle so
    that the smoothed band has a non-zero width; ``bump`` gives arm B the late
    peak the Section 5.2 criteria look for.
    """
    rng = np.random.default_rng(seed)
    rates = rng.uniform(low, high, size=n_draws)
    shape = 1.0 + 0.1 * np.sin(t_grid)[None, :]
    hazard = rates[:, None] * shape
    if bump:
        hazard = hazard + 0.8 * np.exp(-0.5 * ((t_grid[None, :] - 12.0) / 2.0) ** 2)
    survival = np.exp(-rates[:, None] * t_grid[None, :])
    return Functionals(
        t_grid=t_grid,
        S=survival,
        F=1.0 - survival,
        f=hazard * survival,
        h=hazard,
        meta={
            "method": "exact",
            "J_common": j_common,
            "J_prior": j_prior,
            "seed": seed,
        },
    )


def _fit(
    label: str,
    *,
    t_grid: np.ndarray,
    n_draws: int = 8,
    n_chains: int = 2,
    seed: int = 1,
    prior: DPPrior | None = None,
    band: bool = False,
    low: float = 0.15,
    high: float = 0.25,
    bump: bool = False,
) -> Fit:
    """A consistent ``Fit`` built from :func:`_chain` and :func:`_functionals`."""
    chosen = _prior(0) if prior is None else prior
    chains = [
        _chain(
            n_draws=n_draws,
            seed=seed + index,
            prior=chosen,
            t_max=float(t_grid[-1]),
        )
        for index in range(n_chains)
    ]
    prior_band = (
        _functionals(
            t_grid,
            5,
            seed=seed + 200,
            low=0.02,
            high=0.8,
            j_common=12,
            j_prior=7,
        )
        if band
        else None
    )
    return Fit(
        label=label,
        data=chains[0].data,
        prior=chosen,
        t_grid=t_grid,
        chains=chains,
        functionals=_functionals(
            t_grid,
            n_draws * n_chains,
            seed=seed + 100,
            low=low,
            high=high,
            bump=bump,
        ),
        prior_band=prior_band,
    )


@pytest.fixture(scope="module")
def result_5_1() -> DemoResult:
    """A three-prior Section 5.1 result with a reference prior band."""
    t_grid = np.linspace(0.1, 5.75, 60)
    fits = {}
    for index in range(len(PRIOR_TABLE)):
        label = prior_label(index)
        fits[label] = _fit(
            label,
            t_grid=t_grid,
            seed=100 * (index + 1),
            prior=_prior(index),
            band=index == 0,
        )
    return DemoResult(
        label="5.1",
        fits=fits,
        meta={
            "preset": "quick",
            "seed": 20240601,
            "data_seed": 721383,
            "prior_indices": (0, 1, 2),
            "n_chains": 2,
            "n_save": 8,
            "t_max": float(t_grid[-1]),
        },
    )


@pytest.fixture(scope="module")
def result_5_2() -> DemoResult:
    """A three-prior arm-A result plus arm B on one shared grid."""
    t_grid = np.linspace(0.1, 50.0, 80)
    fits = {}
    for index in range(len(PRIOR_TABLE)):
        label = prior_label(index)
        fits[f"A: {label}"] = _fit(
            f"A: {label}",
            t_grid=t_grid,
            seed=200 * (index + 1),
            prior=_prior(index),
            low=0.1,
            high=0.2,
        )
    label_b = prior_label(0)
    fits[f"B: {label_b}"] = _fit(
        f"B: {label_b}",
        t_grid=t_grid,
        seed=900,
        prior=_prior(0),
        low=0.12,
        high=0.22,
        bump=True,
    )
    return DemoResult(
        label="5.2",
        fits=fits,
        meta={
            "preset": "quick",
            "seed": 20240602,
            "prior_indices": (0, 1, 2),
            "n_chains": 2,
            "n_save": 8,
            "t_max": float(t_grid[-1]),
        },
    )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------
def test_report_module_does_not_pull_in_matplotlib() -> None:
    """The payload unit must stay importable on a headless machine."""
    code = "\n".join(
        [
            "import sys",
            "import src.paper_report",
            "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'",
            "print('ok')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_acceptance_targets_mirror_the_paper_tests() -> None:
    """Every target constant is the one the committed acceptance tests use."""
    modules = {"5.1": test_paper_5_1, "5.2": test_paper_5_2}
    for section, module in modules.items():
        targets = paper_report.ACCEPTANCE_TARGETS[section]
        assert targets, f"no acceptance targets for {section}"
        for name, value in targets.items():
            assert hasattr(module, name), (
                f"{section} target {name} is not defined in {module.__name__}"
            )
            assert value == getattr(module, name), (
                f"{section} target {name}={value!r} disagrees with "
                f"{module.__name__}.{name}={getattr(module, name)!r}"
            )


def test_slice_draws_is_the_notes_probe_length() -> None:
    """The old note's probe is the first 500 stored draws of each chain."""
    assert paper_report.SLICE_DRAWS == 500


# ---------------------------------------------------------------------------
# Section 5.1 payload
# ---------------------------------------------------------------------------
def test_summarize_5_1_exposes_every_figure_key(result_5_1: DemoResult) -> None:
    """Figures 1-3 are present with one row per fitted prior."""
    summary = paper_report.summarize_5_1(result_5_1)

    assert summary["label"] == "5.1"
    assert summary["preset"] == "quick"
    assert len(summary["prior_labels"]) == len(PRIOR_TABLE)

    figure_1 = summary["figure_1"]
    for key in (
        "nu_prior_mean",
        "nu_posterior_mean",
        "nu_posterior_95",
        "nu_prior_95_width",
        "nu_posterior_width",
        "nu_posterior_99",
        "n_star_posterior_mean",
        "n_star_probability",
        "n_star_observed_range",
        "learning",
    ):
        assert len(figure_1[key]) == len(PRIOR_TABLE), key
    assert np.asarray(figure_1["nu_posterior_95"]).shape == (3, 2)
    assert np.asarray(figure_1["n_star_observed_range"]).shape == (3, 2)

    figure_2 = summary["figure_2"]
    for key in (
        "survival_at_median",
        "density_mass",
        "second_mode_density",
        "first_mode_density",
    ):
        assert len(figure_2[key]) == len(PRIOR_TABLE), key
    assert figure_2["prior_band_width"] is not None
    assert figure_2["posterior_band_width"] is not None
    assert isinstance(figure_2["prior_band_is_wider"], bool)
    assert figure_2["band_prior_label"] == summary["prior_labels"][0]

    figure_3 = summary["figure_3"]
    assert len(figure_3["change_points"]) == len(PRIOR_TABLE)
    assert len(figure_3["band_width_near"]) == len(PRIOR_TABLE)
    assert len(figure_3["band_width_far"]) == len(PRIOR_TABLE)
    assert len(figure_3["band_ratio"]) == len(PRIOR_TABLE)
    assert figure_3["smoothing_bandwidth"] == 5.0

    json.dumps(summary)


def test_summarize_5_1_without_a_prior_band_reports_null(
    result_5_1: DemoResult,
) -> None:
    """A fit without a prior band is summarised, not rejected."""
    key = next(iter(result_5_1.fits))
    bare = DemoResult(
        label="5.1",
        fits={key: replace(result_5_1.fits[key], prior_band=None)},
        meta=dict(result_5_1.meta),
    )
    summary = paper_report.summarize_5_1(bare)

    assert summary["figure_2"]["prior_band_width"] is None
    assert summary["figure_2"]["posterior_band_width"] is None
    assert summary["figure_2"]["prior_band_is_wider"] is None


def test_summarize_5_1_rejects_another_section(result_5_2: DemoResult) -> None:
    """Summarising the wrong section is a bug, not an empty table."""
    with pytest.raises(ValueError, match="5.1"):
        paper_report.summarize_5_1(result_5_2)


def test_summarize_5_1_needs_at_least_one_fit() -> None:
    with pytest.raises(ValueError, match="no fits"):
        paper_report.summarize_5_1(DemoResult(label="5.1", fits={}))


# ---------------------------------------------------------------------------
# Section 5.2 payload
# ---------------------------------------------------------------------------
def test_summarize_5_2_exposes_every_figure_key(result_5_2: DemoResult) -> None:
    """Figures 4-5 are present, including the not-comparable count."""
    summary = paper_report.summarize_5_2(result_5_2)

    assert summary["label"] == "5.2"
    assert len(summary["prior_labels"]) == len(PRIOR_TABLE)

    figure_4 = summary["figure_4"]
    assert np.asarray(figure_4["prior_agreement"]).shape == (3, 3)
    assert figure_4["prior_agreement_min"] <= 1.0
    assert np.isfinite(figure_4["survival_difference_min"])
    assert figure_4["arm_b_peak_time"] >= 10.0
    assert figure_4["arm_b_peak_value"] > 0.0
    assert figure_4["arm_b_above_peak_max"] is not None
    assert figure_4["arm_b_above_peak_max"] <= figure_4["arm_b_peak_value"]

    figure_5 = summary["figure_5"]
    assert 0.0 <= figure_5["prob_eta_a_gt_eta_b"] <= 1.0
    assert figure_5["difference_min"] >= 0.0
    low, high = figure_5["difference_probability_range"]
    assert 0.0 <= low <= high <= 1.0
    assert 0.0 <= figure_5["difference_probability_mean"] <= 1.0
    assert len(figure_5["hazard_ratio_mean"]) == 2
    assert figure_5["hazard_ratio_decline"] is not None
    assert figure_5["hazard_ratio_decline"] > 0.0
    assert 0.0 <= figure_5["hazard_ratio_order_probability"] <= 1.0
    assert np.asarray(figure_5["hazard_ratio_band"]).shape == (2, 2)
    assert set(figure_5["n_eta_above_grid"]) == {"fit_a", "fit_b"}
    assert figure_5["ratio_times"] == [5.0, 25.0]

    json.dumps(summary)


def test_summarize_5_2_rejects_a_result_without_arm_b() -> None:
    t_grid = np.linspace(0.1, 50.0, 40)
    fit = _fit("A: " + prior_label(0), t_grid=t_grid)
    with pytest.raises(ValueError, match="arm-B"):
        paper_report.summarize_5_2(
            DemoResult(label="5.2", fits={"A: " + prior_label(0): fit})
        )


def test_summarize_5_2_rejects_another_section(result_5_1: DemoResult) -> None:
    with pytest.raises(ValueError, match="5.2"):
        paper_report.summarize_5_2(result_5_1)


# ---------------------------------------------------------------------------
# Slicing and the full-trace truncation level
# ---------------------------------------------------------------------------
def test_slice_result_keeps_the_full_trace_j(result_5_1: DemoResult) -> None:
    """The first-500 summary is the same estimator, just fewer stored draws."""
    sliced = paper_report.slice_result(result_5_1, 4)

    assert sliced.meta["n_save"] == 4
    for key, full in result_5_1.fits.items():
        short = sliced.fits[key]
        assert len(short.chains) == len(full.chains)
        for chain in short.chains:
            assert chain.n_draws == 4
            assert chain.settings.n_save == 4
        assert short.functionals.n_draws == 4 * len(full.chains)
        assert short.functionals.meta["J_common"] == full.functionals.meta["J_common"]
        if full.prior_band is None:
            assert short.prior_band is None
        else:
            assert short.prior_band is not None
            assert short.prior_band.n_draws == 4
            assert short.prior_band.meta["J_prior"] == full.prior_band.meta["J_prior"]

    # The input is untouched.
    assert all(
        chain.n_draws == 8 for fit in result_5_1.fits.values() for chain in fit.chains
    )
    summary = paper_report.summarize_5_1(sliced)
    assert len(summary["figure_1"]["nu_posterior_mean"]) == len(PRIOR_TABLE)


def test_slice_result_caps_at_the_available_draws(result_5_2: DemoResult) -> None:
    """A 1 000-draw request on an 8-draw fit is the whole fit, not an error."""
    sliced = paper_report.slice_result(result_5_2, 1_000)
    for fit in sliced.fits.values():
        assert all(chain.n_draws == 8 for chain in fit.chains)


def test_slice_result_rejects_an_invalid_count(result_5_1: DemoResult) -> None:
    with pytest.raises(ValueError, match="n_draws"):
        paper_report.slice_result(result_5_1, 0)
    with pytest.raises(ValueError, match="n_draws"):
        paper_report.slice_result(result_5_1, True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_slices_with_replace_and_reproduces_r_hat_ess() -> None:
    """The 500-draw block is exactly ``replace(settings, n_save=500)``."""
    t_grid = np.linspace(0.1, 50.0, 40)
    chains = [_chain(n_draws=600, seed=11 + index) for index in range(2)]
    fit = Fit(
        label="nu ~ Gamma(2, 0.9)",
        data=chains[0].data,
        prior=_prior(0),
        t_grid=t_grid,
        chains=chains,
        functionals=_functionals(t_grid, 1_200, seed=12, j_common=42, j_prior=9),
        prior_band=_functionals(t_grid, 10, seed=13, j_common=12, j_prior=7),
    )
    result = DemoResult(
        label="5.1",
        fits={"nu ~ Gamma(2, 0.9)": fit},
        meta={"preset": "quick", "n_save": 600},
    )

    payload = paper_report.diagnostics(result)
    block = payload["fits"]["nu ~ Gamma(2, 0.9)"]

    assert payload["slice_draws"] == 500
    assert block["full"]["draws_per_chain"] == 600
    assert block["first_500"]["draws_per_chain"] == 500
    assert block["j"] == {"J_common": 42, "J_prior": 7}
    assert len(block["full"]["S"]["r_hat"]) == 5
    assert len(block["full"]["S"]["ess"]) == 5
    assert len(block["full"]["S"]["t"]) == 5

    sliced = [
        replace(
            chain,
            settings=replace(chain.settings, n_save=500),
            theta=chain.theta[:500],
            nu=chain.nu[:500],
            gamma=chain.gamma[:500],
            phi=chain.phi[:500],
            n_star=chain.n_star[:500],
            log_post=chain.log_post[:500],
        )
        for chain in chains
    ]
    for name in ("nu", "n_star"):
        expected_r = r_hat(sliced, name)[name]
        expected_e = ess(sliced, name)[name]
        np.testing.assert_allclose(
            block["first_500"][name]["r_hat"], expected_r, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            block["first_500"][name]["ess"], expected_e, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            block["full"][name]["r_hat"], r_hat(chains, name)[name]
        )
        np.testing.assert_allclose(block["full"][name]["ess"], ess(chains, name)[name])

    np.testing.assert_allclose(
        block["first_500"]["S"]["r_hat"], r_hat(sliced, "S")["S"]
    )
    np.testing.assert_allclose(block["full"]["S"]["r_hat"], r_hat(chains, "S")["S"])

    # The full chains and their stored arrays are not modified.
    assert all(chain.n_draws == 600 for chain in fit.chains)


def test_diagnostics_uses_the_available_draws_when_short(
    result_5_1: DemoResult,
) -> None:
    payload = paper_report.diagnostics(result_5_1)
    for block in payload["fits"].values():
        assert block["full"]["draws_per_chain"] == 8
        assert block["first_500"]["draws_per_chain"] == 8
        assert block["j"]["J_common"] == 40
    first = payload["fits"][next(iter(payload["fits"]))]
    assert first["j"]["J_prior"] == 7
    for block in list(payload["fits"].values())[1:]:
        assert block["j"]["J_prior"] is None


def test_diagnostics_requires_a_fit(result_5_1: DemoResult) -> None:
    with pytest.raises(ValueError, match="at least one fit"):
        paper_report.diagnostics(DemoResult(label="5.1", fits={}))


# ---------------------------------------------------------------------------
# Baseline and deltas
# ---------------------------------------------------------------------------
def test_committed_baseline_is_loadable() -> None:
    baseline = paper_report.load_baseline()

    assert baseline["schema_version"] == paper_report.BASELINE_SCHEMA_VERSION
    assert baseline["source"] == "docs/research/dp_gibbs_results.md"
    assert set(baseline["sections"]) == {"5.1", "5.2"}
    assert baseline["sections"]["5.1"]["not_comparable"] == [
        "figure_2.prior_band_width"
    ]
    assert baseline["sections"]["5.2"]["not_comparable"] == [
        "figure_5.n_eta_above_grid"
    ]


def test_committed_baseline_keeps_the_notes_arithmetic() -> None:
    """The note's ``t=4.5 / t=2.5`` columns must not be swapped.

    ``band_width_near`` is the width at ``HAZARD_BAND_TIMES[0]`` (2.5) and
    ``band_width_far`` the width at ``HAZARD_BAND_TIMES[1]`` (4.5); the note
    reports them as ``4.65 / 0.386 = 12.1``.  A transcription that swaps the
    two columns leaves every delta row silently wrong, so the arithmetic of
    the note (``ratio == far / near``) is asserted here.
    """
    values = paper_report.load_baseline()["sections"]["5.1"]["values"]
    near = np.asarray(values["figure_3.band_width_near"], dtype=np.float64)
    far = np.asarray(values["figure_3.band_width_far"], dtype=np.float64)
    ratio = np.asarray(values["figure_3.band_ratio"], dtype=np.float64)

    assert np.all(near < far), "the 2.5 column is the narrow one"
    np.testing.assert_allclose(far / near, ratio, rtol=0.02)


def test_committed_baseline_keeps_the_hazard_ratio_decline() -> None:
    """The note's ``2.298 vs 1.176 = 1.95x`` must survive transcription."""
    values = paper_report.load_baseline()["sections"]["5.2"]["values"]
    near, far = values["figure_5.hazard_ratio_mean"]

    assert values["figure_5.hazard_ratio_decline"] == pytest.approx(
        near / far, rel=0.02
    )


def test_load_baseline_rejects_the_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"schema_version": 99, "sections": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        paper_report.load_baseline(path)


def test_delta_vs_baseline_5_1_signs_and_not_comparable(
    result_5_1: DemoResult,
) -> None:
    summary = paper_report.summarize_5_1(result_5_1)
    delta = paper_report.delta_vs_baseline(summary)

    rows = delta["rows"]
    assert delta["label"] == "5.1"
    assert delta["not_comparable"] == ["figure_2.prior_band_width"]

    prior_mean = rows["figure_1.nu_prior_mean"]
    assert prior_mean["observed"] == summary["figure_1"]["nu_prior_mean"]
    assert prior_mean["baseline"] == [2.22, 20.0, 60.0]
    assert prior_mean["delta"][0] == pytest.approx(
        summary["figure_1"]["nu_prior_mean"][0] - 2.22
    )
    assert prior_mean["delta"][1] == pytest.approx(0.0)
    assert prior_mean["not_comparable"] is False

    band = rows["figure_2.prior_band_width"]
    assert band["not_comparable"] is True
    assert band["delta"] == pytest.approx(
        summary["figure_2"]["prior_band_width"] - 0.833
    )

    assert "figure_3.change_points" in rows


def test_delta_vs_baseline_5_2_marks_n_eta(result_5_2: DemoResult) -> None:
    summary = paper_report.summarize_5_2(result_5_2)
    delta = paper_report.delta_vs_baseline(summary)

    rows = delta["rows"]
    assert delta["not_comparable"] == ["figure_5.n_eta_above_grid"]

    probability = rows["figure_5.prob_eta_a_gt_eta_b"]
    assert probability["delta"] == pytest.approx(
        summary["figure_5"]["prob_eta_a_gt_eta_b"] - 0.89
    )

    for arm in ("fit_a", "fit_b"):
        row = rows[f"figure_5.n_eta_above_grid.{arm}"]
        assert row["not_comparable"] is True
        assert row["baseline"] is None
        assert row["delta"] is None
        assert isinstance(row["observed"], int)

    assert "figure_4.prior_agreement" in rows


def test_delta_vs_baseline_rejects_a_missing_row(
    result_5_2: DemoResult,
) -> None:
    summary = paper_report.summarize_5_2(result_5_2)
    summary["figure_4"].pop("arm_b_peak_time")
    with pytest.raises(ValueError, match="arm_b_peak_time"):
        paper_report.delta_vs_baseline(summary)


def test_delta_vs_baseline_rejects_an_unknown_section(
    result_5_1: DemoResult,
) -> None:
    summary = paper_report.summarize_5_1(result_5_1)
    summary["label"] = "9.9"
    with pytest.raises(ValueError, match="9.9"):
        paper_report.delta_vs_baseline(summary)


def test_delta_vs_baseline_returns_none_for_a_shape_change(
    result_5_1: DemoResult,
) -> None:
    summary = paper_report.summarize_5_1(result_5_1)
    summary["figure_1"]["nu_prior_mean"] = [2.22, 20.0]
    delta = paper_report.delta_vs_baseline(summary)
    assert delta["rows"]["figure_1.nu_prior_mean"]["delta"] is None
