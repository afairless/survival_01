"""Legacy wrapper smoke tests (plan Section 9, step 17).

``src/kottas_2002_5_1.py`` no longer carries its own data generator, demo
script or half-working body; it is a thin wrapper over
:func:`src.paper_examples.example_5_1` (plan decision 10).  These tests pin that
contract: the module imports, it still re-exports the canonical
``TwoLognormalMix``, ``main`` forwards the parsed arguments to the driver, and
one tiny real run produces the Section 5.1 figures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import kottas_2002_5_1, paper_examples
from src.dp_functionals import FunctionalSettings
from src.dp_gibbs import DPSettings
from src.paper_data import TwoLognormalMix

#: Same economy as ``tests/test_cli.py``: enough sweeps to produce stored draws,
#: not enough to say anything about the posterior.
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


def test_wrapper_surface_is_only_the_driver() -> None:
    """The module re-exports the data class and exposes the wrapper API."""
    assert kottas_2002_5_1.__all__ == ["TwoLognormalMix", "build_parser", "main", "run"]
    assert kottas_2002_5_1.TwoLognormalMix is TwoLognormalMix
    assert kottas_2002_5_1.example_5_1 is paper_examples.example_5_1

    # The half-working body is gone: no local sampling, plotting or prior code.
    for stale in ("sample_dp_prior", "sample_prior_hyperparameters", "plt", "theta"):
        assert not hasattr(kottas_2002_5_1, stale), stale


def test_main_forwards_parsed_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_example_5_1(
        *,
        preset: object,
        prior_indices: object = None,
        seed: int | None = None,
        out_dir: Path | str = "",
        max_workers: int | None = None,
    ) -> paper_examples.DemoResult:
        calls.update(preset=preset, seed=seed, out_dir=out_dir, max_workers=max_workers)
        return paper_examples.DemoResult(label="5.1", fits={}, figures=())

    monkeypatch.setattr(kottas_2002_5_1, "example_5_1", fake_example_5_1)

    assert (
        kottas_2002_5_1.main(
            ["--preset", "paper", "--seed", "7", "--out", str(tmp_path)]
        )
        == 0
    )
    assert calls == {
        "preset": "paper",
        "seed": 7,
        "out_dir": tmp_path,
        "max_workers": None,
    }


def test_main_defaults_to_the_documented_seed_and_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_example_5_1(**kwargs: object) -> paper_examples.DemoResult:
        calls.update(kwargs)
        return paper_examples.DemoResult(label="5.1", fits={}, figures=())

    monkeypatch.setattr(kottas_2002_5_1, "example_5_1", fake_example_5_1)

    assert kottas_2002_5_1.main([]) == 0
    assert calls["preset"] == "quick"
    assert calls["seed"] == paper_examples.SEED_5_1
    assert calls["out_dir"] == paper_examples.DEFAULT_FIGURE_DIR


def test_main_rejects_an_unknown_preset() -> None:
    with pytest.raises(SystemExit) as excinfo:
        kottas_2002_5_1.main(["--preset", "quickk"])
    assert excinfo.value.code != 0


def test_wrapper_run_writes_the_section_5_1_figures(tmp_path: Path) -> None:
    result = kottas_2002_5_1.run(preset=TINY_PRESET, seed=21, out_dir=tmp_path)

    assert result.label == "5.1"
    assert {path.name for path in result.figures} == {
        "figure_1.png",
        "figure_2.png",
        "figure_3.png",
    }
    for path in result.figures:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert result.meta["preset"] == "tiny"
