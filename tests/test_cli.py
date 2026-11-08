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

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import paper_artifacts, paper_examples
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


@pytest.fixture
def tiny_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the tiny preset so the CLI can drive a real, cheap run."""
    monkeypatch.setitem(paper_examples.PRESETS, "tiny", TINY_PRESET)


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


# ---------------------------------------------------------------------------
# Artifact-backed driver and CLI (plan §6.2)
# ---------------------------------------------------------------------------
def test_cli_out_defaults_to_none_and_keeps_the_figure_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--out`` is optional at the parser and resolved in ``main``."""
    parser = paper_examples.build_parser()
    assert parser.parse_args(["--example", "5.1"]).out is None

    calls: list[dict[str, object]] = []

    def fake_example_5_1(**kwargs: object) -> paper_examples.DemoResult:
        calls.append(kwargs)
        return paper_examples.DemoResult(label="5.1", fits={}, figures=())

    monkeypatch.setattr(paper_examples, "example_5_1", fake_example_5_1)

    assert paper_examples.main(["--example", "5.1", "--preset", "quick"]) == 0
    assert calls[0]["out_dir"] == paper_examples.DEFAULT_FIGURE_DIR
    assert calls[0]["save_dir"] is None

    explicit = tmp_path / "explicit-figures"
    assert (
        paper_examples.main(
            ["--example", "5.1", "--preset", "quick", "--out", str(explicit)]
        )
        == 0
    )
    assert calls[1]["out_dir"] == explicit

    save_dir = tmp_path / "run" / "5.1"
    assert (
        paper_examples.main(
            [
                "--example",
                "5.1",
                "--preset",
                "quick",
                "--save-dir",
                str(save_dir),
            ]
        )
        == 0
    )
    # The omitted --out keeps its documented meaning; the driver ignores it
    # because --save-dir suppresses figures entirely.
    assert calls[2]["out_dir"] == paper_examples.DEFAULT_FIGURE_DIR
    assert calls[2]["save_dir"] == save_dir


def test_example_5_1_save_dir_round_trips_the_fits(tmp_path: Path) -> None:
    """The driver saves every fit and the band, and writes no figure."""
    save_dir = tmp_path / "5.1"
    result = paper_examples.example_5_1(
        preset=TINY_PRESET,
        prior_indices=(0,),
        seed=31,
        save_dir=save_dir,
    )

    assert result.figures == ()
    assert not (save_dir / "figures").exists()
    assert (save_dir / "meta.json").is_file()
    assert (tmp_path / "logs" / "events-5.1.jsonl").is_file()

    loaded = paper_artifacts.load_demo_result(save_dir)
    assert loaded.label == "5.1"
    assert loaded.figures == ()
    assert loaded.meta == result.meta
    assert loaded.meta["prior_indices"] == (0,)
    assert set(loaded.fits) == set(result.fits)

    key = next(iter(result.fits))
    original = result.fits[key]
    restored = loaded.fits[key]
    assert restored.label == original.label
    assert restored.prior == original.prior
    assert np.array_equal(restored.data.time, original.data.time)
    assert np.array_equal(restored.data.event, original.data.event)
    assert restored.data.t_max == original.data.t_max
    assert np.array_equal(restored.t_grid, original.t_grid)
    assert np.array_equal(restored.functionals.S, original.functionals.S)
    assert np.array_equal(restored.functionals.f, original.functionals.f)
    assert restored.prior_band is not None
    assert original.prior_band is not None
    assert np.array_equal(restored.prior_band.S, original.prior_band.S)

    fit_dir = save_dir / "draws" / paper_artifacts.slug(original.label)
    assert (fit_dir / "functionals.npz").is_file()
    assert (fit_dir / "prior-band.npz").is_file()
    assert (fit_dir / "fit-meta.json").is_file()
    for index in range(original.n_chains):
        assert (fit_dir / f"chain-{index}.npz").is_file()


def test_cli_save_dir_writes_the_section_tree(
    tmp_path: Path, tiny_cli: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI writes draws, functionals, fit meta and events, but no figure."""
    save_dir = tmp_path / "5.1"
    assert (
        paper_examples.main(
            [
                "--example",
                "5.1",
                "--preset",
                "tiny",
                "--prior",
                "1",
                "--seed",
                "41",
                "--save-dir",
                str(save_dir),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "example 5.1: 1 fit(s)\n"

    loaded = paper_artifacts.load_demo_result(save_dir)
    assert loaded.label == "5.1"
    assert loaded.meta["prior_indices"] == (0,)
    assert not (save_dir / "figures").exists()

    fit = next(iter(loaded.fits.values()))
    fit_dir = save_dir / "draws" / paper_artifacts.slug(fit.label)
    assert (fit_dir / "functionals.npz").is_file()
    assert (fit_dir / "fit-meta.json").is_file()
    assert len(list(fit_dir.glob("chain-*.npz"))) == fit.n_chains

    payload = json.loads((fit_dir / "fit-meta.json").read_text(encoding="utf-8"))
    assert payload["extra"]["timing"] >= 0.0

    log = tmp_path / "logs" / "events-5.1.jsonl"
    records = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["kind"] for record in records] == [
        "run_start",
        "fit_start",
        "fit_done",
        "band_done",
        "run_end",
    ]
    for record in records:
        assert {"ts", "t_monotonic", "kind"} <= set(record)
    assert records[2]["timing"] >= 0.0


def test_cli_save_dir_puts_arm_b_in_its_own_directory(
    tmp_path: Path, tiny_cli: None
) -> None:
    """Section 5.2 persists arm B under its own ``B-<slug>`` directory."""
    save_dir = tmp_path / "5.2"
    assert (
        paper_examples.main(
            [
                "--example",
                "5.2",
                "--preset",
                "tiny",
                "--prior",
                "1",
                "--seed",
                "42",
                "--save-dir",
                str(save_dir),
            ]
        )
        == 0
    )

    loaded = paper_artifacts.load_demo_result(save_dir)
    assert set(loaded.fits) == {
        "A: nu ~ Gamma(2, 0.9)",
        "B: nu ~ Gamma(2, 0.9)",
    }
    assert loaded.meta["prior_indices"] == (0,)
    for key in loaded.fits:
        arm, label = key.split(": ", maxsplit=1)
        fit_dir = save_dir / "draws" / f"{arm}-{paper_artifacts.slug(label)}"
        assert (fit_dir / "fit-meta.json").is_file()
        payload = json.loads((fit_dir / "fit-meta.json").read_text(encoding="utf-8"))
        assert payload["arm"] == arm
    assert (tmp_path / "logs" / "events-5.2.jsonl").is_file()
    assert not (save_dir / "figures").exists()


@pytest.mark.parametrize("tracked", ["docs/x", "src/x", "tests/x"])
def test_cli_rejects_a_tracked_save_dir(
    tracked: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 3 GB artifact tree may never land in a tracked directory."""
    save_dir = REPO_ROOT / tracked
    assert (
        paper_examples.main(
            ["--example", "5.1", "--preset", "quick", "--save-dir", str(save_dir)]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "tracked path" in captured.err
    assert str(save_dir.resolve()) in captured.err
    assert not save_dir.exists()


def test_cli_unwritable_save_dir_exits_nonzero(
    tmp_path: Path,
    tiny_cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unwritable run root is one message and a nonzero exit."""
    original_mkdir = Path.mkdir

    def deny_mkdir(
        self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if self == tmp_path or tmp_path in self.parents:
            raise PermissionError(f"read-only filesystem: {self}")
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    save_dir = tmp_path / "5.1"
    assert (
        paper_examples.main(
            [
                "--example",
                "5.1",
                "--preset",
                "tiny",
                "--prior",
                "1",
                "--save-dir",
                str(save_dir),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert captured.err.startswith("error: ")
    assert "read-only filesystem" in captured.err
    assert captured.err.count("\n") == 1
