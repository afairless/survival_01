"""CLI hygiene and the figure driver (plan Section 7.5, test 32).

Two separate contracts are checked here.

**Import hygiene.**  ``src.dp_gibbs`` and ``src.dp_functionals`` must not pull
``matplotlib`` into ``sys.modules``; that is how the plan's "``plt`` only in
``paper_examples.py``" rule is enforced.  The 0.2 s budget of test 32 refers to
the modules' own import cost, so the timer starts after the unavoidable
third-party imports (NumPy and SciPy are hard dependencies of these modules);
the fresh-process check for ``matplotlib`` is separate.

**CLI hygiene.**  A misspelled ``--preset`` or an unknown ``--example`` must
exit non-zero with a usage message naming the valid choices, and no traceback
on stdout.  ``argparse``'s ``choices`` provides exactly that, and the test
invokes :func:`src.paper_examples.main` in process so that the registered
choices -- not the help text -- are the thing under test.

The end-to-end smoke tests use a :class:`src.paper_examples.Preset` with tiny
sweeps so that the driver, the figure recipes and the PNG output are exercised
without a minutes-long MCMC run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import paper_examples
from src.dp_functionals import FunctionalSettings
from src.dp_gibbs import DPSettings
from src.paper_data import NU_PRIORS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A deliberately tiny preset: enough sweeps to produce stored draws, not
#: enough to say anything about the posterior.  Used only by the smoke tests.
TINY_PRESET = paper_examples.Preset(
    name="tiny",
    sampler=DPSettings(
        burn_in=50,
        n_save=10,
        thin=5,
        n_alpha=32,
        epsilon=1e-4,
        n_chains=1,
        seed=None,
        preset="tiny",
    ),
    functionals=FunctionalSettings(method="exact", L=8, atom_tol=1e-6, n_t_grid=300),
    n_prior_draws=4,
)


# ---------------------------------------------------------------------------
# Test 32: import hygiene
# ---------------------------------------------------------------------------
def test_sampler_imports_do_not_pull_in_matplotlib() -> None:
    """Test 32: the sampler modules stay clear of ``matplotlib``."""
    code = "\n".join(
        [
            "import sys",
            "import numpy, scipy.special, scipy.stats",
            "import time",
            "start = time.perf_counter()",
            "import src.dp_gibbs, src.dp_functionals",
            "elapsed = time.perf_counter() - start",
            "assert elapsed < 0.2, f'project imports took {elapsed:.3f}s'",
            "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'",
            "print('ok')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# Test 32: CLI choices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "bad_token", "valid_token"),
    [
        (["--example", "5.1", "--preset", "quickk"], "quickk", "quick"),
        (["--example", "5.3", "--preset", "quick"], "5.3", "5.2"),
    ],
)
def test_cli_rejects_invalid_choices(
    argv: list[str],
    bad_token: str,
    valid_token: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo exits non-zero with a usage message and no stdout traceback."""
    with pytest.raises(SystemExit) as excinfo:
        paper_examples.main(argv)
    assert excinfo.value.code != 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "usage:" in captured.err
    assert bad_token in captured.err
    assert valid_token in captured.err


def test_cli_help_names_every_choice(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        paper_examples.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for token in ("5.1", "5.2", "quick", "paper", "paper-lite", "all"):
        assert token in out


# ---------------------------------------------------------------------------
# Presets and glue (unit)
# ---------------------------------------------------------------------------
def test_presets_match_plan_section_6_2() -> None:
    quick = paper_examples.PRESETS["quick"]
    paper = paper_examples.PRESETS["paper"]

    assert paper.sampler.n_alpha == quick.sampler.n_alpha == 256
    assert paper.sampler.epsilon == quick.sampler.epsilon == 1e-4
    assert paper.functionals.method == quick.functionals.method == "exact"

    assert (paper.sampler.burn_in, paper.sampler.n_save, paper.sampler.thin) == (
        5_000,
        10_000,
        150,
    )
    assert (quick.sampler.burn_in, quick.sampler.n_save, quick.sampler.thin) == (
        1_000,
        1_000,
        10,
    )
    assert paper.sampler.n_chains == 4
    assert quick.sampler.n_chains == 2
    assert paper.functionals.L == 2_500
    assert quick.functionals.L == 200

    # The total sweep count of the plan's settings table.
    assert paper.sampler.burn_in + paper.sampler.n_save * paper.sampler.thin == (
        1_505_000
    )
    assert quick.sampler.burn_in + quick.sampler.n_save * quick.sampler.thin == 11_000


def test_paper_lite_is_the_documented_fallback() -> None:
    lite = paper_examples.PRESETS["paper-lite"]
    assert (lite.sampler.burn_in, lite.sampler.n_save, lite.sampler.thin) == (
        2_000,
        4_000,
        40,
    )
    assert lite.sampler.preset == "paper-lite"


def test_resolve_preset_accepts_names_and_objects() -> None:
    assert paper_examples.resolve_preset("quick") is paper_examples.PRESETS["quick"]
    assert paper_examples.resolve_preset(TINY_PRESET) is TINY_PRESET
    with pytest.raises(ValueError, match="unknown preset"):
        paper_examples.resolve_preset("paper-ish")


def test_make_prior_uses_paper_constants() -> None:
    for index, (a_nu, b_nu) in enumerate(NU_PRIORS):
        prior = paper_examples.make_prior(index)
        assert (prior.a_nu, prior.b_nu) == (a_nu, b_nu)
        assert prior.d == 2
        assert paper_examples.prior_label(index).startswith("nu ~ Gamma(")
    with pytest.raises(ValueError, match="prior_index"):
        paper_examples.make_prior(len(NU_PRIORS))


def test_time_grid_is_positive_and_covers_the_data() -> None:
    grid = paper_examples.time_grid(4.0, 300)
    assert grid.shape == (300,)
    assert np.all(grid > 0.0)
    assert np.all(np.diff(grid) > 0.0)
    assert grid[0] < 0.1
    assert np.isclose(grid[-1], 1.15 * 4.0)
    with pytest.raises(ValueError):
        paper_examples.time_grid(0.0)


def test_smooth_hazard_preserves_a_constant_and_ignores_inf() -> None:
    constant = np.full((3, 40), 2.5)
    np.testing.assert_allclose(paper_examples.smooth_hazard(constant), constant)

    with_inf = np.full((2, 20), 1.0)
    with_inf[0, 0] = np.inf
    smoothed = paper_examples.smooth_hazard(with_inf)
    assert np.all(np.isfinite(smoothed))
    assert np.allclose(smoothed, 1.0)


def test_figure_directory_is_git_ignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docs/research/figures/" in ignored


# ---------------------------------------------------------------------------
# Smoke: the examples run end to end and write their figures
# ---------------------------------------------------------------------------
def test_example_5_1_writes_figures(tmp_path: Path) -> None:
    result = paper_examples.example_5_1(
        preset=TINY_PRESET,
        prior_indices=(0, 1),
        seed=11,
        out_dir=tmp_path,
    )

    assert result.label == "5.1"
    assert len(result.fits) == 2
    assert {path.name for path in result.figures} == {
        "figure_1.png",
        "figure_2.png",
        "figure_3.png",
    }
    for path in result.figures:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert result.meta["preset"] == "tiny"
    assert result.meta["n_chains"] == 1
    for fit in result.fits.values():
        assert fit.n_chains == 1
        assert fit.functionals.n_draws == 10


def test_example_5_2_writes_figures(tmp_path: Path) -> None:
    result = paper_examples.example_5_2(
        preset=TINY_PRESET,
        prior_indices=(0, 1),
        seed=12,
        out_dir=tmp_path,
    )

    assert result.label == "5.2"
    # Two priors for arm A plus the single Gamma(2, 0.9) fit of arm B.
    assert len(result.fits) == 3
    assert sum(key.startswith("A: ") for key in result.fits) == 2
    assert sum(key.startswith("B: ") for key in result.fits) == 1
    assert {path.name for path in result.figures} == {
        "figure_4.png",
        "figure_5.png",
    }
    for path in result.figures:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert result.meta["t_max_a"] == result.meta["t_max_b"] == result.meta["t_max"]


def test_example_5_2_always_fits_the_reported_comparison_prior(
    tmp_path: Path,
) -> None:
    """Figure 5 compares ``Gamma(2, 0.9)`` with itself on both arms."""
    result = paper_examples.example_5_2(
        preset=TINY_PRESET,
        prior_indices=(2,),
        seed=13,
        out_dir=tmp_path,
    )
    assert result.meta["prior_indices"] == (0, 2)
    assert "A: nu ~ Gamma(2, 0.9)" in result.fits
    assert "B: nu ~ Gamma(2, 0.9)" in result.fits
