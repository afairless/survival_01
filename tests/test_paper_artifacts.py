"""Plan §6.1: the artifact primitives and the chain/functional/fit/run round-trips.

Step 1 of the replication-artifact plan introduces the pieces every later
writer and loader share, so this module pins them in isolation:

* ``SCHEMA_VERSION`` is a positive integer and the two error types keep the
  documented relationship -- :class:`ArtifactIntegrityError` is an
  :class:`ArtifactError`, so a verifier can catch either level;
* ``slug`` implements the single directory-name rule of plan §5: every run of
  characters outside ``[A-Za-z0-9._-]`` collapses to one ``_``, the mapping is
  deterministic and idempotent, and an empty result is an error instead of a
  directory that silently cannot be named;
* ``_jsonable`` converts the NumPy scalars, arrays, tuples and nested dicts
  that metadata actually contains into JSON values, and rejects anything with
  no JSON representation instead of emitting a stringified repr;
* ``atomic_savez`` writes through ``<path>.tmp``, flushes and ``fsync``-es the
  handle, and only then renames with ``os.replace``; a failure mid-write
  leaves neither the temporary file nor a readable-but-wrong archive, and an
  existing archive survives a failed overwrite.

What each test would catch:

* a slug that keeps ``(``/`,``/spaces (paths the plan names differently), that
  is not idempotent (a second slug pass changes the directory), or that maps
  two distinct labels to the same empty name -- the exact-example, property and
  empty-label tests;
* ``_jsonable`` returning ``np.intp`` (which ``json.dumps`` rejects), or
  dropping tuple order and nested dict keys -- the unit matrix and the
  randomised round-trip test;
* the temporary-name bug of plan §5: handing ``.tmp`` to ``np.savez`` produces
  ``chain-0.npz.tmp.npz``, so the file-name test asserts the exact path and
  that no extra sibling exists;
* a write that is not atomic: no ``fsync`` before ``os.replace``, a leftover
  ``.tmp`` after a simulated failure, or a destroyed previous archive -- the
  ordering test, the failing-writer test and the overwrite test;
* silent compression drift: the default must be ``savez_compressed`` (R8's
  starting point) while ``compressed=False`` stays a supported escape hatch.

Step 3 adds the functional round-trip of plan §5.1.  ``functionals.npz`` and
``prior-band.npz`` share one schema -- a band is a ``Functionals`` -- so the
same writer serves both and the provenance dict is stored *verbatim* rather
than through a field list (``method``/``J_common``/``L`` for the posterior,
``sample``/``J_prior``/``max_nu`` for the band).  The tests pin the array
contract the loader enforces before the container is built (a transposed
``S``, a shape disagreement between two of the four arrays, a grid whose size
does not match the column count), that an infinite hazard survives bit-exactly
(it is the correct limit where ``S`` underflows), and that the one-draw /
one-grid-point case round-trips -- this is the container that introduces the
single-grid-point edge case.

Step 2 adds the chain round-trip of plan §5.1.  ``chain-<b>.npz`` is the
storage contract a reviewer loads without the fitting process, so the tests
pin the whole schema boundary: every array is bit-exact, the ``meta`` blob is
self-contained (schema version, provenance, prior, settings), and the loader
rejects a wrong key set, dtype, shape, draw count, schema version or metadata
field *before* ``DPChains`` is built.  The one-draw/one-grid-point and
all-censored cases are covered here because this is the container that
introduces them.

Step 4 adds the two container round-trips the long runs are written with.
``save_fit`` owns one directory whose ``fit-meta.json`` is the only place the
**un-slugged** ``label`` and the ``arm`` survive -- the directory name is a
slug -- and the tests pin both, together with the two ``meta`` dicts stored
verbatim and the prior band that is re-attached only when its file is listed.
Every loader check is exercised by tampering with a saved artifact: a
transposed ``S``, a wrong draw count, a shape or dtype that drifted from
``fit-meta.json``, a grid summary that disagrees with the stored grid, a
recorded-but-missing band, an unlisted band, and a metadata path escaping the
fit directory.  The writer's own contract is pinned too: a crash before
``fit-meta.json`` leaves no readable fit, and a re-save removes a stale band or
stale chain files instead of leaving arrays no loader would read.

``save_demo_result``/``load_demo_result`` are the section-level pair.  The
section ``meta.json`` keeps the fit keys in the driver's order and
``DemoResult.meta`` verbatim; the loader restores the types JSON loses
(``prior_indices`` back to a tuple, ``seed`` back to ``None``) because the
paper preconditions compare ``run.meta`` against the in-memory dict.  A fit
directory the section file does not name is rejected, so ``result.fits`` is the
whole tree a manifest can hash.  The None/One ends are covered here too: no
fits at all, one chain with one draw on a one-point grid, and an all-censored
dataset.

One integration test closes the loop the plan asks for: a ``quick``-preset
§5.2 arm-A fit is saved, loaded back, and must still feed ``r_hat()``/``ess()``
with bit-identical values while ``run.meta == result.meta``.

Step 5 adds the incremental writer the long runs use.  A section cannot be
held in memory, so ``ArtifactWriter`` writes each fit as soon as it is fitted,
attaches the prior band to the reference fit (rewriting only the two band
fields of ``fit-meta.json``, because an unlisted band file is an integrity
error) and writes the section ``meta.json`` last.  The tests pin the plan §5
layout, the append-only ``logs/events-<section>.jsonl`` stream (one JSON
object per line with ``ts``/``t_monotonic``/``kind``, extended rather than
truncated by a second writer, fields validated *before* the fit directory is
written), the defaults a fit takes from the section meta, the fit/band target
resolution, and the refusal to write a section meta that names a fit directory
the writer can no longer see.  The stream is the only run-level file this unit
writes: ``MANIFEST.json`` belongs to the manifest unit.

The property-based tests use seeded NumPy generators; the repository has no
Hypothesis dependency.  The inputs are random, the invariants are
deterministic.
"""

import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import paper_artifacts
from src.dp_functionals import Functionals, FunctionalSettings
from src.dp_gibbs import DPChains, DPPrior, DPSettings, SurvData, ess, r_hat
from src.paper_artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactWriter,
    _jsonable,
    atomic_savez,
    load_chain,
    load_demo_result,
    load_fit,
    load_functionals,
    save_chain,
    save_demo_result,
    save_fit,
    save_functionals,
    slug,
)
from src.paper_examples import (
    SEED_5_2,
    DemoResult,
    Fit,
    Preset,
    data_5_2,
    fit_dataset,
    make_prior,
    prior_band,
    time_grid,
)

ALLOWED_SLUG = re.compile(r"^[A-Za-z0-9._-]+$")

# A spread of slug-able and non-slug-able characters, including non-ASCII.
LABEL_ALPHABET = list("abcXYZ019 ._-()~,:/\\[]{}!@#%^&*+=|;'\"<>?\t\nνλα€")


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_schema_version_is_a_positive_integer() -> None:
    """Every ``meta`` blob and manifest stores this value; it must be usable."""
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_integrity_error_is_an_artifact_error() -> None:
    """Callers may catch the base class; the integrity type refines it."""
    assert issubclass(ArtifactIntegrityError, ArtifactError)
    assert not issubclass(ArtifactError, ArtifactIntegrityError)


def test_artifact_module_does_not_pull_in_matplotlib() -> None:
    """Plan §6.1: the artifact I/O module is pure NumPy and the standard library.

    ``Fit``/``DemoResult`` live in ``paper_examples``; the import is done inside
    the fit/run functions, so importing this module must stay clear of
    ``matplotlib`` (whose import in ``paper_examples`` is lazy anyway) and of an
    import cycle with the driver.
    """
    code = "\n".join(
        [
            "import sys",
            "import src.paper_artifacts",
            "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'",
            "assert 'src.paper_examples' not in sys.modules, "
            "       'paper_artifacts imported the driver at module scope'",
            "print('ok')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# slug
# ---------------------------------------------------------------------------


def test_slug_matches_the_plan_examples() -> None:
    """Plan §5: the prior label and the arm-prefixed directory names."""
    label = "nu ~ Gamma(2, 0.9)"
    assert slug(label) == "nu_Gamma_2_0.9"
    assert f"A-{slug(label)}" == "A-nu_Gamma_2_0.9"
    assert f"B-{slug(label)}" == "B-nu_Gamma_2_0.9"


def test_slug_collapses_each_run_to_one_underscore() -> None:
    """Runs, not characters: ``a  ~~  b`` is ``a_b``, not ``a____b``."""
    assert slug("a  ~~  b") == "a_b"
    assert slug("  leading and trailing  ") == "leading_and_trailing"


def test_slug_preserves_allowed_characters_verbatim() -> None:
    """Dots, dashes and underscores are legal directory characters."""
    assert slug("5.1 / arm_B-2") == "5.1_arm_B-2"
    assert slug("figure_4.png") == "figure_4.png"


def test_slug_rejects_an_empty_label() -> None:
    """An empty slug would name the output root itself; refuse it."""
    with pytest.raises(ArtifactError, match="empty slug"):
        slug("")
    with pytest.raises(ArtifactError, match="empty slug"):
        slug("   ~()  ")


def test_slug_rejects_a_non_string_label() -> None:
    """A list or number labels a fit only by accident; fail loudly."""
    with pytest.raises(ArtifactError, match="must be a string"):
        slug(51)  # type: ignore[arg-type]


def test_slug_property_allowed_charset_idempotence_and_length() -> None:
    """Random labels: fixed charset, idempotent, never longer than input."""
    rng = np.random.default_rng(20261108)
    for _ in range(200):
        length = int(rng.integers(1, 24))
        label = "x" + "".join(rng.choice(LABEL_ALPHABET, size=length))
        slugged = slug(label)
        assert ALLOWED_SLUG.fullmatch(slugged), (label, slugged)
        assert len(slugged) <= len(label)
        assert slug(slugged) == slugged
        assert slug(label) == slugged, "slug must be deterministic"


def test_slug_property_all_allowed_input_is_unchanged_up_to_boundary_trims() -> None:
    """Over the allowed alphabet only the boundary trims can change a label."""
    rng = np.random.default_rng(20261109)
    alphabet = list("abcXYZ019._-")
    for _ in range(100):
        length = int(rng.integers(1, 20))
        label = "x" + "".join(rng.choice(alphabet, size=length))
        assert slug(label) == label.strip("_")


# ---------------------------------------------------------------------------
# _jsonable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (7, 7),
        (1.5, 1.5),
        ("nu", "nu"),
        (np.intp(7), 7),
        (np.int64(-3), -3),
        (np.float64(1.25), 1.25),
        (np.bool_(False), False),
        (np.array(3.0), 3.0),
        (np.array([1.0, 2.0]), [1.0, 2.0]),
        (np.array([1, 2], dtype=np.int64), [1, 2]),
        (np.array(["a", "b"]), ["a", "b"]),
        (np.array([[True, False]]), [[True, False]]),
        ((1, np.intp(2)), [1, 2]),
        (
            {"a": (1, np.int64(2)), "b": {"c": np.array([True])}},
            {"a": [1, 2], "b": {"c": [True]}},
        ),
        (Path("/tmp/run"), "/tmp/run"),
    ],
)
def test_jsonable_unit_matrix(value: object, expected: object) -> None:
    """Scalars, arrays and containers map to their JSON equivalents."""
    converted = _jsonable(value)
    assert converted == expected
    assert json.loads(json.dumps(converted)) == expected


def test_jsonable_stringifies_mapping_keys() -> None:
    """JSON objects require string keys; the conversion says so explicitly."""
    assert _jsonable({1: "one", np.intp(2): "two"}) == {"1": "one", "2": "two"}


@pytest.mark.parametrize("value", [object(), {1, 2}, complex(1.0, 2.0)])
def test_jsonable_rejects_values_without_a_json_form(value: object) -> None:
    """Do not smuggle unsupported objects through ``str(value)``."""
    with pytest.raises(ArtifactError, match=type(value).__name__):
        _jsonable(value)


def _random_jsonable(rng: np.random.Generator, depth: int = 0) -> tuple[object, object]:
    """Return ``(value_with_numpy, expected_plain_json)`` for one random value."""
    kinds = ["none", "bool", "int", "float", "str", "scalar", "array"]
    if depth < 3:
        kinds += ["list", "tuple", "dict"]
    kind = str(rng.choice(kinds))
    if kind == "none":
        return None, None
    if kind == "bool":
        value = bool(rng.integers(0, 2))
        return value, value
    if kind == "int":
        value = int(rng.integers(-(10**6), 10**6))
        return value, value
    if kind == "float":
        value = float(rng.normal())
        return value, value
    if kind == "str":
        value = "".join(rng.choice(["a", "b", " ", "ν", "_"], size=4))
        return value, value
    if kind == "scalar":
        if rng.integers(0, 2):
            plain = int(rng.integers(-100, 100))
            return np.int64(plain), plain
        plain = float(rng.normal())
        return np.float64(plain), plain
    if kind == "array":
        shape = [(1,), (3,), (2, 2), ()][int(rng.integers(0, 4))]
        plain_array = np.round(rng.normal(size=shape), 6)
        plain = plain_array.tolist()
        return plain_array, plain
    if kind in {"list", "tuple"}:
        items = [
            _random_jsonable(rng, depth + 1) for _ in range(int(rng.integers(0, 4)))
        ]
        values = [value for value, _ in items]
        expected = [plain for _, plain in items]
        if kind == "tuple":
            return tuple(values), expected
        return values, expected
    items = [_random_jsonable(rng, depth + 1) for _ in range(int(rng.integers(0, 4)))]
    keys = [f"k{index}_{int(rng.integers(0, 100))}" for index in range(len(items))]
    return {key: value for key, (value, _) in zip(keys, items, strict=True)}, {
        key: plain for key, (_, plain) in zip(keys, items, strict=True)
    }


def test_jsonable_property_round_trips_random_nested_values() -> None:
    """Random nested provenance dicts survive ``json.dumps``/``json.loads``."""
    rng = np.random.default_rng(20261110)
    for _ in range(150):
        value, expected = _random_jsonable(rng)
        converted = _jsonable(value)
        assert converted == expected
        assert json.loads(json.dumps(converted)) == expected
        assert _jsonable(converted) == expected, "conversion must be idempotent"


# ---------------------------------------------------------------------------
# atomic_savez
# ---------------------------------------------------------------------------


def _example_arrays(seed: int) -> dict[str, np.ndarray]:
    """The array kinds a stored chain and its metadata exercise."""
    rng = np.random.default_rng(seed)
    return {
        "theta": rng.normal(size=(2, 3, 2)),
        "nu": rng.normal(size=2),
        "n_star": np.array([2, 3], dtype=np.int64),
        "event": np.array([True, False]),
        "t_max": np.array(4.5518),
        "meta": np.array('{"schema_version": 1, "label": "nu ~ Gamma(2, 0.9)"}'),
    }


def test_atomic_savez_round_trips_every_array_bit_exactly(tmp_path: Path) -> None:
    """Shape, dtype and every element survive; loading needs no pickle."""
    arrays = _example_arrays(20261111)
    path = atomic_savez(tmp_path / "draws" / "chain-0.npz", arrays)
    assert path == tmp_path / "draws" / "chain-0.npz"
    assert path.exists()
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, value in arrays.items():
            assert loaded[key].dtype == value.dtype, key
            assert loaded[key].shape == value.shape, key
            assert np.array_equal(loaded[key], value), key


def test_atomic_savez_names_the_exact_path_and_leaves_no_siblings(
    tmp_path: Path,
) -> None:
    """Plan §5: no ``*.npz.npz`` and no leftover ``*.npz.tmp``."""
    path = atomic_savez(tmp_path / "chain-0.npz", {"nu": np.array([1.0])})
    assert [entry.name for entry in tmp_path.iterdir()] == ["chain-0.npz"]
    assert path.name == "chain-0.npz"


def test_atomic_savez_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Per-fit directories are created by the writer, not the driver."""
    path = atomic_savez(
        tmp_path / "5.2" / "draws" / "A-nu_Gamma_2_0.9" / "chain-3.npz",
        {"nu": np.array([2.0])},
    )
    assert path.parent.is_dir()


@pytest.mark.parametrize(
    ("member", "expected_compression"),
    [
        ("chain-compressed.npz", zipfile.ZIP_DEFLATED),
        ("chain-stored.npz", zipfile.ZIP_STORED),
    ],
)
def test_atomic_savez_compression_is_explicit(
    tmp_path: Path, member: str, expected_compression: int
) -> None:
    """Compressed is the default (R8); ``compressed=False`` is available."""
    path = atomic_savez(
        tmp_path / member,
        {"theta": np.zeros((8, 8), dtype=np.float64)},
        compressed=member.startswith("chain-compressed"),
    )
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        assert infos
        assert all(info.compress_type == expected_compression for info in infos)


def test_atomic_savez_fsyncs_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temporary file is durable before ``os.replace`` publishes it."""
    events: list[object] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fake_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def fake_replace(source: object, destination: object) -> None:
        events.append(("replace", Path(str(source)).exists()))
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(paper_artifacts.os, "fsync", fake_fsync)
    monkeypatch.setattr(paper_artifacts.os, "replace", fake_replace)
    atomic_savez(tmp_path / "chain-0.npz", {"nu": np.array([1.0])})
    assert events == ["fsync", ("replace", True)]


def test_failed_write_leaves_no_temporary_file_and_no_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulated disk-full after some bytes must leave nothing behind."""
    path = tmp_path / "chain-0.npz"

    def explode(handle: object, arrays: object, *, compressed: bool) -> None:
        handle.write(b"PK\x03\x04 partial zip")  # type: ignore[attr-defined]
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(paper_artifacts, "_savez", explode)
    with pytest.raises(RuntimeError, match="disk full"):
        atomic_savez(path, {"nu": np.array([1.0])})
    assert list(tmp_path.iterdir()) == []


def test_failed_overwrite_preserves_the_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old file is complete until the new one is complete."""
    path = atomic_savez(tmp_path / "chain-0.npz", {"nu": np.array([1.0])})

    def explode(handle: object, arrays: object, *, compressed: bool) -> None:
        handle.write(b"PK\x03\x04 partial zip")  # type: ignore[attr-defined]
        raise OSError("simulated write error")

    monkeypatch.setattr(paper_artifacts, "_savez", explode)
    with pytest.raises(OSError, match="simulated write error"):
        atomic_savez(path, {"nu": np.array([2.0])})
    assert [entry.name for entry in tmp_path.iterdir()] == ["chain-0.npz"]
    with np.load(path, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["nu"], np.array([1.0]))


def test_successful_overwrite_replaces_the_archive(tmp_path: Path) -> None:
    """A second save publishes the new arrays and removes the temporary file."""
    path = tmp_path / "chain-0.npz"
    atomic_savez(path, {"nu": np.array([1.0])})
    atomic_savez(path, {"nu": np.array([2.0])})
    assert [entry.name for entry in tmp_path.iterdir()] == ["chain-0.npz"]
    with np.load(path, allow_pickle=False) as loaded:
        assert np.array_equal(loaded["nu"], np.array([2.0]))


@pytest.mark.parametrize("name", ["chain-0", "chain-0.npz.tmp", "chain-0.npy"])
def test_atomic_savez_rejects_a_non_npz_path(tmp_path: Path, name: str) -> None:
    """The §5 layout is ``*.npz``; refuse to invent another extension."""
    with pytest.raises(ArtifactError, match=r"must end in '\.npz'"):
        atomic_savez(tmp_path / name, {"nu": np.array([1.0])})


def test_atomic_savez_property_random_archives_round_trip(tmp_path: Path) -> None:
    """Random shapes and dtypes: still bit-exact and still one file."""
    rng = np.random.default_rng(20261112)
    dtypes = [np.float64, np.int64, np.bool_]
    for index in range(25):
        dtype = dtypes[int(rng.integers(0, len(dtypes)))]
        shape = [(1,), (4,), (2, 3), ()][int(rng.integers(0, 4))]
        if np.issubdtype(dtype, np.floating):
            array = rng.normal(size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            array = rng.integers(-10, 10, size=shape).astype(dtype)
        else:
            array = rng.integers(0, 2, size=shape).astype(dtype)
        directory = tmp_path / f"case-{index}"
        path = atomic_savez(directory / "array.npz", {"value": array})
        assert [entry.name for entry in directory.iterdir()] == ["array.npz"]
        with np.load(path, allow_pickle=False) as loaded:
            assert loaded["value"].dtype == array.dtype
            assert np.array_equal(loaded["value"], array)


# ---------------------------------------------------------------------------
# Chain round-trip (plan §5.1)
# ---------------------------------------------------------------------------

CHAIN_PRIOR = DPPrior(
    a_nu=2.0,
    b_nu=0.9,
    a_gamma=1.0,
    b_gamma=0.232235,
    a_phi=2.0,
    b_phi=1.0,
    d=2,
)


def make_chain(
    *,
    n_draws: int = 4,
    n: int = 6,
    seed: int = 20240601,
    event: np.ndarray | None = None,
    prior: DPPrior = CHAIN_PRIOR,
    burn_in: int = 5,
    thin: int = 1,
    n_alpha: int = 16,
    epsilon: float = 1e-4,
    n_chains: int = 4,
    settings_seed: int | None = 20240601,
    preset: str = "quick",
) -> DPChains:
    """A small chain that satisfies every ``DPChains`` invariant.

    ``alpha`` stays below one and ``phi`` above the prior's ``b_phi``, so the
    Section 2.1 bounds hold for the default and for every random prior the
    property test builds (``b_phi <= 1``).
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(1.05, 1.9, n)
    if event is None:
        event = np.ones(n, dtype=bool)
    data = SurvData(time=time, event=np.asarray(event, dtype=bool), t_max=3.75)
    alpha = rng.uniform(0.4, 0.9, size=(n_draws, n))
    phi = rng.uniform(1.2, 2.0, size=n_draws)
    lam = rng.uniform(0.5, 3.0, size=(n_draws, n))
    theta = np.stack([alpha, lam], axis=2)
    settings = DPSettings(
        burn_in=burn_in,
        n_save=n_draws,
        thin=thin,
        n_alpha=n_alpha,
        epsilon=epsilon,
        n_chains=n_chains,
        seed=settings_seed,
        preset=preset,
    )
    return DPChains(
        data=data,
        prior=prior,
        settings=settings,
        theta=theta,
        nu=rng.uniform(0.5, 3.0, size=n_draws),
        gamma=rng.uniform(0.2, 1.5, size=n_draws),
        phi=phi,
        n_star=rng.integers(1, n + 1, size=n_draws).astype(np.intp),
        log_post=-rng.uniform(1.0, 10.0, size=n_draws),
    )


def chain_grid(size: int = 5) -> np.ndarray:
    """An original-scale grid of ``size`` strictly positive points."""
    return np.linspace(0.25, 4.0, size)


def assert_chains_equal(loaded: DPChains, expected: DPChains) -> None:
    """Every stored value, dtype and provenance scalar survives bit-exactly."""
    assert loaded.prior == expected.prior
    assert loaded.settings == expected.settings
    assert loaded.data.t_max == expected.data.t_max
    for name in ("time", "event"):
        np.testing.assert_array_equal(
            getattr(loaded.data, name), getattr(expected.data, name)
        )
    for name in ("theta", "nu", "gamma", "phi", "n_star", "log_post"):
        actual = getattr(loaded, name)
        wanted = getattr(expected, name)
        assert actual.dtype == wanted.dtype, name
        np.testing.assert_array_equal(actual, wanted, err_msg=name)


def archive_members(path: Path) -> dict[str, np.ndarray]:
    """Read every member of an ``npz`` archive into memory."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rewrite_archive(
    path: Path, *, drop: tuple[str, ...] = (), **changes: np.ndarray
) -> None:
    """Replace or drop archive members, keeping the write atomic."""
    arrays = archive_members(path)
    for key in drop:
        arrays.pop(key)
    arrays.update(changes)
    atomic_savez(path, arrays)


#: The chain tests read naturally with this name; the helper is archive-generic.
rewrite_chain = rewrite_archive


def rewrite_meta(path: Path, mutate: object) -> None:
    """Apply ``mutate`` to the JSON ``meta`` object and rewrite the archive."""
    arrays = archive_members(path)
    record = json.loads(str(arrays["meta"][()]))
    mutate(record)  # type: ignore[operator]
    arrays["meta"] = np.asarray(json.dumps(record))
    atomic_savez(path, arrays)


#: See :data:`rewrite_chain`.
rewrite_chain_meta = rewrite_meta


def saved_chain(tmp_path: Path, **kwargs: object) -> tuple[DPChains, Path]:
    """Save :func:`make_chain` and return the chain and its path."""
    chain = make_chain(**kwargs)  # type: ignore[arg-type]
    path = tmp_path / "draws" / "A-nu_Gamma_2_0.9" / "chain-0.npz"
    save_chain(
        path,
        chain,
        t_grid=chain_grid(),
        label="nu ~ Gamma(2, 0.9)",
        chain_index=0,
    )
    return chain, path


# --- round-trip ------------------------------------------------------------


def test_save_chain_round_trips_every_stored_array(tmp_path: Path) -> None:
    """All seven draw arrays plus the dataset and ``t_grid`` are bit-exact."""
    chain = make_chain()
    grid = chain_grid()
    path = tmp_path / "chain-0.npz"
    written = save_chain(
        path,
        chain,
        t_grid=grid,
        label="nu ~ Gamma(2, 0.9)",
        chain_index=3,
    )
    assert written == path

    loaded = load_chain(path)
    assert_chains_equal(loaded, chain)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["t_grid"], grid)
        assert archive["n_star"].dtype == np.int64
        assert archive["event"].dtype == np.bool_
        assert archive["t_max"].shape == ()


def test_saved_chain_matches_the_plan_schema(tmp_path: Path) -> None:
    """Plan §5.1: the exact key set, dtypes and shapes of ``chain-<b>.npz``."""
    chain, path = saved_chain(tmp_path)
    expected_shapes = {
        "theta": (chain.n_draws, chain.n, 2),
        "nu": (chain.n_draws,),
        "gamma": (chain.n_draws,),
        "phi": (chain.n_draws,),
        "n_star": (chain.n_draws,),
        "log_post": (chain.n_draws,),
        "time": (chain.n,),
        "event": (chain.n,),
        "t_max": (),
        "t_grid": (5,),
        "meta": (),
    }
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(expected_shapes)
        for key, shape in expected_shapes.items():
            assert archive[key].shape == shape, key
        assert archive["theta"].dtype == np.float64
        assert archive["nu"].dtype == np.float64
        assert archive["gamma"].dtype == np.float64
        assert archive["phi"].dtype == np.float64
        assert np.issubdtype(archive["n_star"].dtype, np.integer)
        assert archive["log_post"].dtype == np.float64
        assert archive["time"].dtype == np.float64
        assert archive["event"].dtype == np.bool_
        assert archive["t_max"].dtype == np.float64
        assert archive["t_grid"].dtype == np.float64
        assert archive["meta"].dtype.kind == "U"


def test_chain_meta_round_trips_prior_settings_and_provenance(
    tmp_path: Path,
) -> None:
    """The per-chain blob is self-contained and reconstructs exactly."""
    chain, path = saved_chain(tmp_path)
    record = json.loads(str(archive_members(path)["meta"][()]))
    assert set(record) == {
        "schema_version",
        "label",
        "chain_index",
        "prior",
        "settings",
    }
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["label"] == "nu ~ Gamma(2, 0.9)"
    assert record["chain_index"] == 0
    assert record["prior"] == {
        "a_nu": 2.0,
        "b_nu": 0.9,
        "a_gamma": 1.0,
        "b_gamma": 0.232235,
        "a_phi": 2.0,
        "b_phi": 1.0,
        "d": 2,
    }
    assert record["settings"] == {
        "burn_in": 5,
        "n_save": chain.n_draws,
        "thin": 1,
        "n_alpha": 16,
        "epsilon": 1e-4,
        "n_chains": 4,
        "seed": 20240601,
        "preset": "quick",
    }
    loaded = load_chain(path)
    assert loaded.prior == chain.prior
    assert loaded.settings == chain.settings


def test_load_chain_needs_no_side_information(tmp_path: Path) -> None:
    """A chain copied out of its fit directory still round-trips."""
    chain, path = saved_chain(tmp_path)
    elsewhere = tmp_path / "standalone" / "chain-2.npz"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(path.read_bytes())
    assert path.exists()
    path.unlink()
    assert_chains_equal(load_chain(elsewhere), chain)


def test_save_chain_records_null_provenance_when_omitted(tmp_path: Path) -> None:
    """A bare chain has no fit context; ``null`` is the honest record."""
    chain = make_chain()
    path = tmp_path / "chain-1.npz"
    save_chain(path, chain, t_grid=chain_grid())
    record = json.loads(str(archive_members(path)["meta"][()]))
    assert record["label"] is None
    assert record["chain_index"] is None
    assert_chains_equal(load_chain(path), chain)


def test_chain_round_trip_with_one_draw_and_one_grid_point(
    tmp_path: Path,
) -> None:
    """None/One edge case: ``n_save=1``, one observation, one grid point."""
    chain = make_chain(n_draws=1, n=1, event=np.array([False]))
    path = tmp_path / "chain-0.npz"
    save_chain(path, chain, t_grid=np.array([0.5]))
    loaded = load_chain(path)
    assert loaded.n_draws == 1
    assert loaded.n == 1
    assert_chains_equal(loaded, chain)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["t_grid"].shape == (1,)
        assert archive["theta"].shape == (1, 1, 2)


def test_chain_round_trip_all_censored_dataset(tmp_path: Path) -> None:
    """A dataset with no events is valid and must stay valid on disk."""
    chain = make_chain(n=7, event=np.zeros(7, dtype=bool))
    path = tmp_path / "chain-0.npz"
    save_chain(path, chain, t_grid=chain_grid(3))
    loaded = load_chain(path)
    assert not loaded.data.event.any()
    assert_chains_equal(loaded, chain)


def test_save_chain_does_not_mutate_the_chain(tmp_path: Path) -> None:
    """Saving is a read of the container, never a conversion in place."""
    chain = make_chain()
    before = chain.theta.copy()
    save_chain(tmp_path / "chain-0.npz", chain, t_grid=chain_grid())
    np.testing.assert_array_equal(chain.theta, before)
    assert chain.theta.dtype == np.float64
    assert chain.n_star.dtype == np.intp


# --- loader validation -----------------------------------------------------


def test_load_chain_rejects_a_missing_file(tmp_path: Path) -> None:
    """The error names the path; a silent empty chain must be impossible."""
    path = tmp_path / "does-not-exist.npz"
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_chain(path)
    assert str(path) in str(excinfo.value)


def test_load_chain_rejects_a_directory(tmp_path: Path) -> None:
    """A directory is not an archive, and the message says so."""
    with pytest.raises(ArtifactError, match="is not a file"):
        load_chain(tmp_path)


def test_load_chain_rejects_a_non_npz_file(tmp_path: Path) -> None:
    """A truncated or renamed file is an integrity error naming the path."""
    path = tmp_path / "chain-0.npz"
    path.write_bytes(b"not an npz archive")
    with pytest.raises(ArtifactIntegrityError, match="npz") as excinfo:
        load_chain(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize("key", ["theta", "nu", "n_star", "time", "meta"])
def test_load_chain_rejects_a_missing_array(tmp_path: Path, key: str) -> None:
    """A missing schema member is caught before any container is built."""
    _, path = saved_chain(tmp_path)
    rewrite_chain(path, drop=(key,))
    with pytest.raises(ArtifactIntegrityError, match=key):
        load_chain(path)


def test_load_chain_rejects_an_unexpected_array(tmp_path: Path) -> None:
    """An extra member means the schema moved; refuse to guess."""
    _, path = saved_chain(tmp_path)
    rewrite_chain(path, surprise=np.array([1.0]))
    with pytest.raises(ArtifactIntegrityError, match="surprise"):
        load_chain(path)


def test_load_chain_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    """There is no migration path (plan §5.1): the version must match."""
    _, path = saved_chain(tmp_path)

    def bump(record: dict[str, object]) -> None:
        record["schema_version"] = SCHEMA_VERSION + 1

    rewrite_chain_meta(path, bump)
    with pytest.raises(ArtifactIntegrityError, match="schema version"):
        load_chain(path)


@pytest.mark.parametrize(
    ("key", "convert"),
    [
        ("theta", lambda value: value.astype(np.float32)),
        ("nu", lambda value: value.astype(np.int64)),
        ("gamma", lambda value: value.astype(np.float32)),
        ("phi", lambda value: value.astype(np.float32)),
        ("n_star", lambda value: value.astype(np.float64)),
        ("log_post", lambda value: value.astype(np.float32)),
        ("time", lambda value: value.astype(np.float32)),
        ("event", lambda value: value.astype(np.int64)),
        ("t_max", lambda value: value.astype(np.float32)),
        ("t_grid", lambda value: value.astype(np.float32)),
    ],
)
def test_load_chain_rejects_a_wrong_array_dtype(
    tmp_path: Path, key: str, convert: object
) -> None:
    """The plan's dtype table is enforced before NumPy can widen silently."""
    _, path = saved_chain(tmp_path)
    value = archive_members(path)[key]
    rewrite_chain(path, **{key: convert(value)})  # type: ignore[operator]
    with pytest.raises(ArtifactIntegrityError, match="dtype"):
        load_chain(path)


@pytest.mark.parametrize(
    ("key", "broken"),
    [
        ("theta", lambda value: value.transpose(1, 0, 2)),
        ("theta", lambda value: value[:, :, :1]),
        ("nu", lambda value: value[:-1]),
        ("n_star", lambda value: value.reshape(-1, 1)),
        ("event", lambda value: value[:-1]),
        ("time", lambda value: value.reshape(-1, 1)),
        ("t_max", lambda value: value.reshape(1)),
        ("t_grid", lambda value: value.reshape(1, -1)),
    ],
)
def test_load_chain_rejects_a_wrong_array_shape(
    tmp_path: Path, key: str, broken: object
) -> None:
    """A transposed array is the failure mode the schema check exists for."""
    _, path = saved_chain(tmp_path, n_draws=4, n=6)
    value = archive_members(path)[key]
    rewrite_chain(path, **{key: broken(value)})  # type: ignore[operator]
    with pytest.raises(ArtifactIntegrityError, match="shape|dtype"):
        load_chain(path)


def test_load_chain_rejects_a_draw_count_not_matching_settings(
    tmp_path: Path,
) -> None:
    """``settings.n_save`` is the declared draw count; a short trace fails."""
    _, path = saved_chain(tmp_path, n_draws=4)
    arrays = archive_members(path)
    rewrite_chain(
        path,
        theta=arrays["theta"][:-1],
        nu=arrays["nu"][:-1],
        gamma=arrays["gamma"][:-1],
        phi=arrays["phi"][:-1],
        n_star=arrays["n_star"][:-1],
        log_post=arrays["log_post"][:-1],
    )
    with pytest.raises(ArtifactIntegrityError, match="expected \\(4"):
        load_chain(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(meta="not-an-object"),
        lambda record: record.update(prior="not-an-object"),
    ],
)
def test_load_chain_rejects_a_structurally_broken_meta(
    tmp_path: Path, mutate: object
) -> None:
    """A nested field with the wrong structure must not reach a constructor."""
    _, path = saved_chain(tmp_path)

    def apply(record: dict[str, object]) -> None:
        mutate(record)  # type: ignore[operator]

    rewrite_chain_meta(path, apply)
    with pytest.raises(ArtifactIntegrityError, match="meta|schema"):
        load_chain(path)


def test_load_chain_rejects_a_non_object_meta_blob(tmp_path: Path) -> None:
    """The blob is a JSON object, not an array or a string."""
    _, path = saved_chain(tmp_path)
    rewrite_chain(path, meta=np.asarray("[1, 2]"))
    with pytest.raises(ArtifactIntegrityError, match="JSON object"):
        load_chain(path)


def test_load_chain_rejects_invalid_meta_json(tmp_path: Path) -> None:
    """Invalid JSON is an integrity error, not an unhandled ``JSONDecodeError``."""
    _, path = saved_chain(tmp_path)
    rewrite_chain(path, meta=np.asarray("{not json"))
    with pytest.raises(ArtifactIntegrityError, match="valid JSON"):
        load_chain(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["prior"].update(unknown_field=1.0),
        lambda record: record["prior"].pop("a_nu"),
        lambda record: record["prior"].update(d=2.5),
        lambda record: record["settings"].update(unknown_field=1),
        lambda record: record["settings"].pop("seed"),
        lambda record: record["settings"].update(epsilon=2.0),
        lambda record: record["settings"].update(preset=""),
        lambda record: record["settings"].update(seed=-1),
        lambda record: record.update(label=7),
        lambda record: record.update(chain_index=-1),
        lambda record: record.pop("settings"),
    ],
)
def test_load_chain_rejects_invalid_metadata_fields(
    tmp_path: Path, mutate: object
) -> None:
    """Every meta field is validated against its declared type and bounds."""
    _, path = saved_chain(tmp_path)

    def apply(record: dict[str, object]) -> None:
        mutate(record)  # type: ignore[operator]

    rewrite_chain_meta(path, apply)
    with pytest.raises(ArtifactIntegrityError, match="meta|settings|prior|label"):
        load_chain(path)


def test_load_chain_rejects_a_container_invariant_violation(
    tmp_path: Path,
) -> None:
    """Arrays of the right shape can still violate Section 2.1; report it."""
    _, path = saved_chain(tmp_path)
    arrays = archive_members(path)
    rewrite_chain(path, phi=np.full(arrays["phi"].shape, 0.5))
    with pytest.raises(ArtifactIntegrityError, match="DPChains contract"):
        load_chain(path)


# --- writer validation -----------------------------------------------------


def test_save_chain_rejects_a_non_chain(tmp_path: Path) -> None:
    """A dict with the right keys is not a chain; fail at the writer."""
    with pytest.raises(ArtifactError, match="DPChains"):
        save_chain(tmp_path / "chain-0.npz", object(), t_grid=chain_grid())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0.5, 1.0]]),
        np.array([]),
        np.array([0.0, 1.0]),
        np.array([0.5, np.nan]),
        np.array([0.5, -1.0]),
    ],
)
def test_save_chain_rejects_an_invalid_grid(tmp_path: Path, grid: np.ndarray) -> None:
    """The grid is a fit-level input; validate it before writing bytes."""
    with pytest.raises(ArtifactError, match="t_grid"):
        save_chain(tmp_path / "chain-0.npz", make_chain(), t_grid=grid)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("label", "chain_index"),
    [
        (7, None),
        (None, -1),
        (None, True),
        (None, 1.5),
    ],
)
def test_save_chain_rejects_invalid_provenance(
    tmp_path: Path, label: object, chain_index: object
) -> None:
    """Provenance fields are checked with the same care as the arrays."""
    with pytest.raises(ArtifactError, match="label|chain_index"):
        save_chain(
            tmp_path / "chain-0.npz",
            make_chain(),
            t_grid=chain_grid(),
            label=label,  # type: ignore[arg-type]
            chain_index=chain_index,  # type: ignore[arg-type]
        )


def test_save_chain_requires_an_npz_path(tmp_path: Path) -> None:
    """The plan's layout is ``*.npz``; never invent another extension."""
    with pytest.raises(ArtifactError, match=r"must end in '\.npz'"):
        save_chain(tmp_path / "chain-0", make_chain(), t_grid=chain_grid())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epsilon", 2.0),
        ("n_alpha", 1),
        ("preset", ""),
        ("burn_in", -1),
    ],
)
def test_save_chain_rejects_settings_outside_the_schema(
    tmp_path: Path, field: str, value: object
) -> None:
    """The writer validates provenance with the loader's own rules."""
    chain = make_chain()
    broken = replace(chain, settings=replace(chain.settings, **{field: value}))
    with pytest.raises(ArtifactError, match="metadata"):
        save_chain(tmp_path / "chain-0.npz", broken, t_grid=chain_grid())
    assert list(tmp_path.iterdir()) == []


def test_save_chain_rejects_a_non_finite_prior(tmp_path: Path) -> None:
    """A NaN hyperparameter must not be written as an unloadable archive."""
    chain = make_chain()
    broken = replace(chain, prior=replace(chain.prior, a_nu=float("nan")))
    with pytest.raises(ArtifactError, match="metadata"):
        save_chain(tmp_path / "chain-0.npz", broken, t_grid=chain_grid())
    assert list(tmp_path.iterdir()) == []


# --- property-based --------------------------------------------------------


def test_chain_round_trip_property_random_chains(tmp_path: Path) -> None:
    """Random sizes, censoring patterns and grids all round-trip bit-exactly."""
    rng = np.random.default_rng(20261113)
    for case in range(20):
        n_draws = int(rng.integers(1, 7))
        n = int(rng.integers(1, 9))
        n_grid = int(rng.integers(1, 7))
        event = rng.integers(0, 2, size=n).astype(bool)
        chain = make_chain(
            n_draws=n_draws,
            n=n,
            event=event,
            seed=int(rng.integers(0, 10**9)),
            n_chains=int(rng.integers(1, 6)),
            settings_seed=None
            if bool(rng.integers(0, 2))
            else int(rng.integers(0, 10**9)),
        )
        grid = np.sort(rng.uniform(0.05, 6.0, size=n_grid))
        path = tmp_path / f"case-{case}" / "chain-0.npz"
        save_chain(path, chain, t_grid=grid, chain_index=case)
        loaded = load_chain(path)
        assert_chains_equal(loaded, chain)
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["t_grid"], grid)


def test_chain_round_trip_property_random_provenance(tmp_path: Path) -> None:
    """Random priors and sampler knobs survive the JSON meta round-trip."""
    rng = np.random.default_rng(20261114)
    for case in range(20):
        prior = DPPrior(
            a_nu=float(rng.uniform(0.1, 5.0)),
            b_nu=float(rng.uniform(0.01, 3.0)),
            a_gamma=float(rng.uniform(0.1, 4.0)),
            b_gamma=float(rng.uniform(0.01, 2.0)),
            a_phi=float(rng.uniform(0.1, 4.0)),
            b_phi=float(rng.uniform(0.01, 1.0)),
            d=int(rng.integers(1, 5)),
        )
        n_draws = int(rng.integers(1, 5))
        chain = make_chain(
            n_draws=n_draws,
            n=int(rng.integers(1, 8)),
            prior=prior,
            burn_in=int(rng.integers(0, 1000)),
            thin=int(rng.integers(1, 10)),
            n_alpha=int(rng.integers(2, 65)),
            epsilon=float(rng.uniform(1e-6, 0.5)),
            n_chains=int(rng.integers(1, 6)),
            settings_seed=None
            if bool(rng.integers(0, 2))
            else int(rng.integers(0, 10**9)),
            preset=str(rng.choice(["quick", "paper", "paper-lite", "custom"])),
            seed=int(rng.integers(0, 10**9)),
        )
        path = tmp_path / f"case-{case}.npz"
        save_chain(path, chain, t_grid=chain_grid(int(rng.integers(1, 5))))
        loaded = load_chain(path)
        assert loaded.prior == prior
        assert loaded.settings == chain.settings
        assert_chains_equal(loaded, chain)


# ---------------------------------------------------------------------------
# Functional round-trip (plan §5.1)
# ---------------------------------------------------------------------------

#: The per-chain seeds a fit records; JSON has to turn the tuple into a list.
CHAIN_SEEDS: tuple[int, ...] = (20240601, 20240602, 20240603, 20240604)

#: The posterior provenance dict of a §5.1/§5.2 fit.
FUNCTIONALS_META: dict[str, object] = {
    "method": "exact",
    "J_common": 8,
    "atom_tol": 1e-6,
    "seed": 20240601,
    "epsilon": 1e-4,
    "n_chains": 4,
    "chain_seeds": CHAIN_SEEDS,
    "n_draws": 4,
    "t_max": 4.5518,
}

#: The prior-band provenance dict of a §5.1 fit.
PRIOR_BAND_META: dict[str, object] = {
    "sample": "prior",
    "method": "exact",
    "J_prior": 12,
    "max_nu": 27.5,
    "epsilon": 1e-4,
    "atom_tol": 1e-6,
    "seed": 20240613,
    "n_draws": 100,
    "t_max": 4.5518,
}


def make_functionals(
    *,
    n_draws: int = 4,
    n_grid: int = 5,
    seed: int = 20261115,
    meta: dict[str, object] | None = None,
    hazard_inf: bool = False,
) -> Functionals:
    """A small ``Functionals`` that satisfies every container invariant.

    ``S`` and ``F = 1 - S`` stay in ``[0, 1]``, ``f`` and ``h`` are
    non-negative and ``S``/``F``/``f`` finite, so the only way to make the
    container reject these arrays is to change them deliberately.
    """
    rng = np.random.default_rng(seed)
    survival = rng.uniform(0.05, 0.95, size=(n_draws, n_grid))
    density = rng.uniform(0.0, 2.0, size=(n_draws, n_grid))
    hazard = rng.uniform(0.0, 3.0, size=(n_draws, n_grid))
    if hazard_inf:
        # The documented limit where S underflows to zero while f > 0.
        hazard[0, -1] = np.inf
    return Functionals(
        t_grid=np.linspace(0.5, 3.5, n_grid),
        S=survival,
        F=1.0 - survival,
        f=density,
        h=hazard,
        meta=dict(FUNCTIONALS_META if meta is None else meta),
    )


def saved_functionals(
    tmp_path: Path, *, name: str = "functionals.npz", **kwargs: object
) -> tuple[Functionals, Path]:
    """Save :func:`make_functionals` into a per-fit draws directory."""
    func = make_functionals(**kwargs)  # type: ignore[arg-type]
    path = tmp_path / "5.1" / "draws" / "A-nu_Gamma_2_0.9" / name
    save_functionals(path, func)
    return func, path


def assert_functionals_equal(loaded: Functionals, expected: Functionals) -> None:
    """Every stored array and the reconstructed grid survive bit-exactly."""
    np.testing.assert_array_equal(loaded.t_grid, expected.t_grid)
    assert loaded.t_grid.dtype == expected.t_grid.dtype
    for name in ("S", "F", "f", "h"):
        actual = getattr(loaded, name)
        wanted = getattr(expected, name)
        assert actual.dtype == wanted.dtype, name
        np.testing.assert_array_equal(actual, wanted, err_msg=name)


# --- round-trip -----------------------------------------------------------


def test_save_functionals_round_trips_every_array(tmp_path: Path) -> None:
    """``t_grid`` and the four ``(B, M)`` arrays are bit-exact."""
    func = make_functionals()
    path = tmp_path / "functionals.npz"
    written = save_functionals(path, func)
    assert written == path

    loaded = load_functionals(path)
    assert_functionals_equal(loaded, func)
    assert loaded.n_draws == func.n_draws
    assert loaded.n_grid == func.n_grid


def test_saved_functionals_match_the_plan_schema(tmp_path: Path) -> None:
    """Plan §5.1: the exact key set, dtypes, shapes and a 0-d ``meta``."""
    func, path = saved_functionals(tmp_path)
    expected_shapes = {
        "t_grid": (func.n_grid,),
        "S": (func.n_draws, func.n_grid),
        "F": (func.n_draws, func.n_grid),
        "f": (func.n_draws, func.n_grid),
        "h": (func.n_draws, func.n_grid),
        "meta": (),
    }
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(expected_shapes)
        for key, shape in expected_shapes.items():
            assert archive[key].shape == shape, key
        for key in ("t_grid", "S", "F", "f", "h"):
            assert archive[key].dtype == np.float64, key
        assert archive["meta"].dtype.kind == "U"


def test_functionals_meta_blob_wraps_the_provenance_dict(tmp_path: Path) -> None:
    """``{"schema_version", "meta"}``: the version cannot shadow a meta key."""
    _, path = saved_functionals(tmp_path)
    record = json.loads(str(archive_members(path)["meta"][()]))
    assert set(record) == {"schema_version", "meta"}
    assert record["schema_version"] == SCHEMA_VERSION
    assert isinstance(record["meta"], dict)


def test_functionals_meta_round_trips_verbatim(tmp_path: Path) -> None:
    """The posterior dict survives without a hand-written field list."""
    func, path = saved_functionals(tmp_path)
    loaded = load_functionals(path)
    assert loaded.meta == _jsonable(FUNCTIONALS_META)
    # A tuple is the one documented conversion: JSON has no tuple type.
    assert loaded.meta["chain_seeds"] == list(CHAIN_SEEDS)
    assert isinstance(loaded.meta["chain_seeds"], list)


def test_prior_band_round_trips_with_its_own_meta(tmp_path: Path) -> None:
    """``prior-band.npz`` is a ``Functionals``: one schema serves both."""
    func, path = saved_functionals(
        tmp_path, name="prior-band.npz", meta=PRIOR_BAND_META
    )
    loaded = load_functionals(path)
    assert_functionals_equal(loaded, func)
    assert loaded.meta == _jsonable(PRIOR_BAND_META)
    assert set(loaded.meta) == set(PRIOR_BAND_META)


def test_functionals_meta_survives_numpy_scalars_and_nesting(tmp_path: Path) -> None:
    """Provenance holds NumPy scalars and nested dicts in practice."""
    meta: dict[str, object] = {
        "seed": np.int64(7),
        "atom_tol": np.float64(1e-6),
        "nested": {"seeds": (np.intp(1), np.intp(2)), "flag": np.bool_(True)},
    }
    func, path = saved_functionals(tmp_path, meta=meta)
    loaded = load_functionals(path)
    assert_functionals_equal(loaded, func)
    assert loaded.meta == {
        "seed": 7,
        "atom_tol": 1e-6,
        "nested": {"seeds": [1, 2], "flag": True},
    }


def test_functionals_round_trip_preserves_an_infinite_hazard(
    tmp_path: Path,
) -> None:
    """``h = inf`` where ``S`` underflows is the correct limit, not a NaN."""
    func = make_functionals(hazard_inf=True)
    path = tmp_path / "functionals.npz"
    save_functionals(path, func)
    loaded = load_functionals(path)
    assert np.isinf(loaded.h[0, -1])
    assert_functionals_equal(loaded, func)
    assert np.array_equal(loaded.h, func.h)


def test_functionals_round_trip_with_one_draw_and_one_grid_point(
    tmp_path: Path,
) -> None:
    """None/One edge case: ``n_save=1`` and a single grid point."""
    func = make_functionals(n_draws=1, n_grid=1)
    path = tmp_path / "functionals.npz"
    save_functionals(path, func)
    loaded = load_functionals(path)
    assert loaded.n_draws == 1
    assert loaded.n_grid == 1
    assert_functionals_equal(loaded, func)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["t_grid"].shape == (1,)
        assert archive["S"].shape == (1, 1)


def test_save_functionals_does_not_mutate_the_input(tmp_path: Path) -> None:
    """Saving reads the container; it never converts arrays in place."""
    func = make_functionals()
    before = func.S.copy()
    save_functionals(tmp_path / "functionals.npz", func)
    np.testing.assert_array_equal(func.S, before)
    assert func.S.dtype == np.float64
    assert func.t_grid.dtype == np.float64


def test_functionals_loader_needs_no_side_information(tmp_path: Path) -> None:
    """The archive is self-contained: no ``fit-meta.json`` is required."""
    func, path = saved_functionals(tmp_path)
    elsewhere = tmp_path / "standalone" / "functionals.npz"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(path.read_bytes())
    path.unlink()
    assert_functionals_equal(load_functionals(elsewhere), func)


# --- loader validation -----------------------------------------------------


def test_load_functionals_rejects_a_missing_file(tmp_path: Path) -> None:
    """The error names the path; a silently empty result is impossible."""
    path = tmp_path / "does-not-exist.npz"
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_functionals(path)
    assert str(path) in str(excinfo.value)


def test_load_functionals_rejects_a_directory(tmp_path: Path) -> None:
    """A directory is not an archive, and the message says so."""
    with pytest.raises(ArtifactError, match="is not a file"):
        load_functionals(tmp_path)


def test_load_functionals_rejects_a_non_npz_file(tmp_path: Path) -> None:
    """A truncated or renamed file is an integrity error naming the path."""
    path = tmp_path / "functionals.npz"
    path.write_bytes(b"not an npz archive")
    with pytest.raises(ArtifactIntegrityError, match="npz") as excinfo:
        load_functionals(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize("key", ["t_grid", "S", "F", "f", "h", "meta"])
def test_load_functionals_rejects_a_missing_array(tmp_path: Path, key: str) -> None:
    """A missing schema member is caught before any container is built."""
    _, path = saved_functionals(tmp_path)
    rewrite_archive(path, drop=(key,))
    with pytest.raises(ArtifactIntegrityError, match=key):
        load_functionals(path)


def test_load_functionals_rejects_an_unexpected_array(tmp_path: Path) -> None:
    """An extra member means the schema moved; refuse to guess."""
    _, path = saved_functionals(tmp_path)
    rewrite_archive(path, surprise=np.array([1.0]))
    with pytest.raises(ArtifactIntegrityError, match="surprise"):
        load_functionals(path)


def test_load_functionals_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    """There is no migration path (plan §5.1): the version must match."""
    _, path = saved_functionals(tmp_path)

    def bump(record: dict[str, object]) -> None:
        record["schema_version"] = SCHEMA_VERSION + 1

    rewrite_meta(path, bump)
    with pytest.raises(ArtifactIntegrityError, match="schema version"):
        load_functionals(path)


@pytest.mark.parametrize("key", ["t_grid", "S", "F", "f", "h"])
def test_load_functionals_rejects_a_wrong_array_dtype(tmp_path: Path, key: str) -> None:
    """The plan's dtype table is enforced before NumPy can widen silently."""
    _, path = saved_functionals(tmp_path)
    arrays = archive_members(path)
    arrays[key] = arrays[key].astype(np.float32)
    atomic_savez(path, arrays)
    with pytest.raises(ArtifactIntegrityError, match="dtype"):
        load_functionals(path)


@pytest.mark.parametrize("key", ["S", "F", "f", "h"])
def test_load_functionals_rejects_a_transposed_array(tmp_path: Path, key: str) -> None:
    """A transposed ``S`` is the failure mode this schema check exists for."""
    _, path = saved_functionals(tmp_path, n_draws=4, n_grid=5)
    arrays = archive_members(path)
    arrays[key] = arrays[key].transpose(1, 0)
    atomic_savez(path, arrays)
    with pytest.raises(ArtifactIntegrityError, match="shape|columns") as excinfo:
        load_functionals(path)
    assert key in str(excinfo.value)


@pytest.mark.parametrize(
    ("key", "broken"),
    [
        ("S", lambda value: value.reshape(-1)),
        ("F", lambda value: value[:, :-1]),
        ("f", lambda value: value[:, :-1]),
        ("h", lambda value: value[:, :-1]),
    ],
)
def test_load_functionals_rejects_a_shape_disagreement(
    tmp_path: Path, key: str, broken: object
) -> None:
    """The four arrays must share one ``(n_draws, n_grid)`` shape."""
    _, path = saved_functionals(tmp_path, n_draws=4, n_grid=5)
    value = archive_members(path)[key]
    rewrite_archive(path, **{key: broken(value)})  # type: ignore[operator]
    with pytest.raises(ArtifactIntegrityError, match="shape|columns"):
        load_functionals(path)


def test_load_functionals_rejects_a_grid_size_mismatch(tmp_path: Path) -> None:
    """``n_grid`` is declared by ``t_grid``; one extra point must fail."""
    _, path = saved_functionals(tmp_path, n_draws=4, n_grid=5)
    grid = archive_members(path)["t_grid"]
    rewrite_archive(path, t_grid=np.append(grid, 4.0))
    with pytest.raises(ArtifactIntegrityError, match="t_grid point"):
        load_functionals(path)


def test_load_functionals_rejects_an_empty_draw_count(tmp_path: Path) -> None:
    """A zero-row functional has no mean or quantile; refuse it."""
    _, path = saved_functionals(tmp_path, n_draws=4, n_grid=5)
    arrays = archive_members(path)
    rewrite_archive(
        path,
        S=arrays["S"][:0],
        F=arrays["F"][:0],
        f=arrays["f"][:0],
        h=arrays["h"][:0],
    )
    with pytest.raises(ArtifactIntegrityError, match="at least one draw"):
        load_functionals(path)


@pytest.mark.parametrize(
    "t_grid",
    [
        np.array([]),
        np.array([[0.5, 1.0]]),
        np.array([0.0, 1.0]),
        np.array([0.5, np.nan]),
        np.array([0.5, -1.0]),
    ],
)
def test_load_functionals_rejects_an_invalid_grid(
    tmp_path: Path, t_grid: np.ndarray
) -> None:
    """The grid must be 1-D, finite and strictly positive on write and read."""
    _, path = saved_functionals(tmp_path)
    rewrite_archive(path, t_grid=t_grid)
    with pytest.raises(ArtifactIntegrityError, match="t_grid"):
        load_functionals(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(meta="not-an-object"),
        lambda record: record.update(meta=[1, 2]),
        lambda record: record.update(meta=None),
        lambda record: record.pop("meta"),
        lambda record: record.update(unknown=1),
    ],
)
def test_load_functionals_rejects_a_broken_meta_record(
    tmp_path: Path, mutate: object
) -> None:
    """A nested field with the wrong structure must not reach a constructor."""
    _, path = saved_functionals(tmp_path)

    def apply(record: dict[str, object]) -> None:
        mutate(record)  # type: ignore[operator]

    rewrite_meta(path, apply)
    with pytest.raises(ArtifactIntegrityError, match="meta"):
        load_functionals(path)


def test_load_functionals_rejects_a_non_object_meta_blob(tmp_path: Path) -> None:
    """The blob is a JSON object, not an array or a string."""
    _, path = saved_functionals(tmp_path)
    rewrite_archive(path, meta=np.asarray("[1, 2]"))
    with pytest.raises(ArtifactIntegrityError, match="JSON object"):
        load_functionals(path)


def test_load_functionals_rejects_invalid_meta_json(tmp_path: Path) -> None:
    """Invalid JSON is an integrity error, not a bare ``JSONDecodeError``."""
    _, path = saved_functionals(tmp_path)
    rewrite_archive(path, meta=np.asarray("{not json"))
    with pytest.raises(ArtifactIntegrityError, match="valid JSON"):
        load_functionals(path)


def test_load_functionals_rejects_a_container_invariant_violation(
    tmp_path: Path,
) -> None:
    """Right shapes can still violate the container; report it, do not crash."""
    _, path = saved_functionals(tmp_path)
    arrays = archive_members(path)
    rewrite_archive(path, S=np.full(arrays["S"].shape, 1.5))
    with pytest.raises(ArtifactIntegrityError, match="Functionals contract"):
        load_functionals(path)


# --- writer validation -----------------------------------------------------


def test_save_functionals_rejects_a_non_functionals(tmp_path: Path) -> None:
    """A dict with the right keys is not a ``Functionals``; fail at the writer."""
    with pytest.raises(ArtifactError, match="Functionals"):
        save_functionals(tmp_path / "functionals.npz", object())  # type: ignore[arg-type]


def test_save_functionals_requires_an_npz_path(tmp_path: Path) -> None:
    """The plan's layout is ``*.npz``; never invent another extension."""
    with pytest.raises(ArtifactError, match=r"must end in '\.npz'"):
        save_functionals(tmp_path / "functionals", make_functionals())


@pytest.mark.parametrize(
    "meta",
    [
        {"bad": object()},
        {"bad": {1, 2}},
        [("method", "exact")],
    ],
)
def test_save_functionals_rejects_unusable_meta(tmp_path: Path, meta: object) -> None:
    """A meta the loader could not read back must not be written."""
    func = make_functionals()
    func.meta = meta  # type: ignore[assignment]
    with pytest.raises(ArtifactError, match="meta"):
        save_functionals(tmp_path / "functionals.npz", func)
    assert list(tmp_path.iterdir()) == []


def test_save_functionals_rejects_arrays_outside_the_schema(
    tmp_path: Path,
) -> None:
    """The writer applies the loader's dtype/shape rules, so they cannot drift."""
    func = make_functionals()
    func.S = func.S.astype(np.float32)
    with pytest.raises(ArtifactError, match="artifact schema"):
        save_functionals(tmp_path / "functionals.npz", func)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0.5, 1.0]]),
        np.array([]),
        np.array([0.0, 1.0]),
        np.array([0.5, np.nan]),
        np.array([0.5, -1.0]),
    ],
)
def test_save_functionals_rejects_an_invalid_grid(
    tmp_path: Path, grid: np.ndarray
) -> None:
    """The grid is validated before any bytes are written."""
    func = make_functionals()
    func.t_grid = grid
    with pytest.raises(ArtifactError, match="t_grid"):
        save_functionals(tmp_path / "functionals.npz", func)
    assert list(tmp_path.iterdir()) == []


# --- property-based --------------------------------------------------------


def test_functionals_round_trip_property_random_arrays(tmp_path: Path) -> None:
    """Random draw counts, grid sizes and hazards round-trip bit-exactly."""
    rng = np.random.default_rng(20261116)
    for case in range(20):
        n_draws = int(rng.integers(1, 7))
        n_grid = int(rng.integers(1, 8))
        func = make_functionals(
            n_draws=n_draws,
            n_grid=n_grid,
            seed=int(rng.integers(0, 10**9)),
            hazard_inf=bool(rng.integers(0, 2)),
            meta={"case": case, "seeds": (case, case + 1)},
        )
        # A random grid whose size matches the arrays, strictly positive.
        func.t_grid = np.sort(rng.uniform(0.01, 9.0, size=n_grid))
        path = tmp_path / f"case-{case}" / "functionals.npz"
        save_functionals(path, func)
        loaded = load_functionals(path)
        assert_functionals_equal(loaded, func)
        assert loaded.meta == _jsonable(func.meta)


def test_functionals_meta_property_round_trips_random_dicts(
    tmp_path: Path,
) -> None:
    """Random nested provenance dicts survive the verbatim JSON round-trip."""
    rng = np.random.default_rng(20261117)
    for case in range(20):
        value, expected = _random_jsonable(rng)
        meta = {"provenance": value}
        func = make_functionals(meta=meta)
        path = tmp_path / f"case-{case}.npz"
        save_functionals(path, func)
        loaded = load_functionals(path)
        assert loaded.meta == {"provenance": expected}
        assert loaded.meta == _jsonable(meta)
        assert_functionals_equal(loaded, func)


# ---------------------------------------------------------------------------
# Fit round-trip (plan §5.1, §6.1)
# ---------------------------------------------------------------------------

#: The readable label the §5.1 driver records; the directory only holds its slug.
FIT_LABEL = "nu ~ Gamma(2, 0.9)"


def make_fit(
    *,
    n_chains: int = 2,
    n_draws: int = 4,
    n_grid: int = 5,
    n: int = 6,
    seed: int = 20240601,
    event: np.ndarray | None = None,
    label: str = FIT_LABEL,
    with_band: bool = False,
    band_draws: int = 3,
    functionals_meta: dict[str, object] | None = None,
    band_meta: dict[str, object] | None = None,
) -> Fit:
    """A small but internally consistent ``Fit`` (chains + functionals).

    Every chain shares one prior, one settings record and one dataset, and the
    functionals hold exactly ``n_chains * n_draws`` rows on the fit's grid --
    the invariants ``save_fit`` checks before it writes anything.
    """
    chains = [
        make_chain(
            n_draws=n_draws,
            n=n,
            seed=seed + index,
            event=event,
            n_chains=n_chains,
            settings_seed=seed,
        )
        for index in range(n_chains)
    ]
    grid = chain_grid(n_grid)
    functionals = make_functionals(
        n_draws=n_chains * n_draws,
        n_grid=n_grid,
        meta=functionals_meta,
    )
    functionals.t_grid = grid
    band = None
    if with_band:
        band = make_functionals(
            n_draws=band_draws,
            n_grid=n_grid,
            seed=20261118,
            meta=band_meta if band_meta is not None else PRIOR_BAND_META,
        )
        band.t_grid = grid
    return Fit(
        label=label,
        data=chains[0].data,
        prior=chains[0].prior,
        t_grid=grid,
        chains=chains,
        functionals=functionals,
        prior_band=band,
    )


def saved_fit(
    tmp_path: Path, *, name: str = "A-nu_Gamma_2_0.9", **kwargs: object
) -> tuple[Fit, Path]:
    """Save :func:`make_fit` into a plan-shaped per-fit directory."""
    fit = make_fit(**kwargs)  # type: ignore[arg-type]
    directory = tmp_path / "5.1" / "draws" / name
    save_fit(directory, fit, meta={"arm": "A", "data_seed": 721383})
    return fit, directory


def fit_meta(directory: Path) -> dict[str, Any]:
    """Read the raw ``fit-meta.json`` object."""
    return json.loads((directory / "fit-meta.json").read_text(encoding="utf-8"))


def rewrite_fit_meta(directory: Path, mutate: Any) -> None:
    """Apply ``mutate`` to ``fit-meta.json`` and write it back."""
    payload = fit_meta(directory)
    mutate(payload)  # type: ignore[operator]
    (directory / "fit-meta.json").write_text(json.dumps(payload), encoding="utf-8")


def demo_meta(directory: Path) -> dict[str, Any]:
    """Read the raw section ``meta.json`` object."""
    return json.loads((directory / "meta.json").read_text(encoding="utf-8"))


def assert_fits_equal(loaded: Fit, expected: Fit) -> None:
    """Every chain, functional array, grid and label survives bit-exactly."""
    assert loaded.label == expected.label
    assert loaded.prior == expected.prior
    np.testing.assert_array_equal(loaded.t_grid, expected.t_grid)
    assert len(loaded.chains) == len(expected.chains)
    for actual, wanted in zip(loaded.chains, expected.chains, strict=True):
        assert_chains_equal(actual, wanted)
    assert_functionals_equal(loaded.functionals, expected.functionals)
    assert loaded.functionals.meta == _jsonable(expected.functionals.meta)
    if expected.prior_band is None:
        assert loaded.prior_band is None
    else:
        assert loaded.prior_band is not None
        assert_functionals_equal(loaded.prior_band, expected.prior_band)
        assert loaded.prior_band.meta == _jsonable(expected.prior_band.meta)


# --- round-trip ------------------------------------------------------------


def test_save_fit_writes_the_plan_layout(tmp_path: Path) -> None:
    """A fit directory holds every chain, one functional file and its meta."""
    _, directory = saved_fit(tmp_path)
    assert sorted(path.name for path in directory.iterdir()) == [
        "chain-0.npz",
        "chain-1.npz",
        "fit-meta.json",
        "functionals.npz",
    ]
    payload = fit_meta(directory)
    assert set(payload) == {
        "schema_version",
        "label",
        "arm",
        "prior",
        "settings",
        "data_seed",
        "shared_t_max",
        "n_chains",
        "n_draws",
        "t_grid",
        "chains",
        "functionals",
        "prior_band",
        "functionals_meta",
        "prior_band_meta",
        "extra",
    }
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["n_chains"] == 2
    assert payload["n_draws"] == 8


def test_fit_round_trips_every_chain_and_functional_array(tmp_path: Path) -> None:
    """Chains, functionals, grid and label are bit-exact after a round trip."""
    fit, directory = saved_fit(tmp_path)
    loaded = load_fit(directory)
    assert_fits_equal(loaded, fit)
    assert loaded.n_chains == 2
    np.testing.assert_array_equal(loaded.data.time, fit.data.time)
    np.testing.assert_array_equal(loaded.data.event, fit.data.event)
    assert loaded.data.t_max == fit.data.t_max


def test_fit_meta_keeps_the_un_slugged_label_and_arm(tmp_path: Path) -> None:
    """The directory is a slug; ``fit-meta.json`` carries the readable label."""
    _, directory = saved_fit(tmp_path)
    payload = fit_meta(directory)
    assert directory.name == "A-nu_Gamma_2_0.9"
    assert payload["label"] == FIT_LABEL
    assert payload["arm"] == "A"
    assert payload["data_seed"] == 721383
    assert payload["t_grid"] == {
        "size": 5,
        "first": float(chain_grid()[0]),
        "last": float(chain_grid()[-1]),
    }
    assert payload["chains"] == [
        {"path": "chain-0.npz", "dtype": "float64", "shape": [4, 6, 2]},
        {"path": "chain-1.npz", "dtype": "float64", "shape": [4, 6, 2]},
    ]
    assert payload["functionals"] == {
        "path": "functionals.npz",
        "dtype": "float64",
        "shape": [8, 5],
    }
    assert load_fit(directory).label == FIT_LABEL


def test_fit_meta_records_both_meta_dicts_verbatim(tmp_path: Path) -> None:
    """The posterior and prior-band provenance dicts are stored, not enumerated."""
    _, directory = saved_fit(tmp_path, with_band=True)
    payload = fit_meta(directory)
    assert payload["functionals_meta"] == _jsonable(FUNCTIONALS_META)
    assert payload["prior_band_meta"] == _jsonable(PRIOR_BAND_META)


def test_fit_round_trips_the_prior_band(tmp_path: Path) -> None:
    """A band is one more ``Functionals`` file and re-attaches on load."""
    fit, directory = saved_fit(tmp_path, with_band=True)
    assert (directory / "prior-band.npz").exists()
    loaded = load_fit(directory)
    assert loaded.prior_band is not None
    assert loaded.prior_band.meta == _jsonable(PRIOR_BAND_META)
    assert_fits_equal(loaded, fit)


def test_fit_round_trips_without_a_prior_band(tmp_path: Path) -> None:
    """No band file, no ``prior_band``: the loader does not invent one."""
    _, directory = saved_fit(tmp_path)
    assert not (directory / "prior-band.npz").exists()
    assert fit_meta(directory)["prior_band"] is None
    assert fit_meta(directory)["prior_band_meta"] is None
    assert load_fit(directory).prior_band is None


def test_save_fit_records_extra_metadata_verbatim(tmp_path: Path) -> None:
    """Timings and chain seeds go under ``extra`` so they never need a bump."""
    fit = make_fit()
    directory = tmp_path / "fit"
    save_fit(
        directory,
        fit,
        meta={
            "arm": None,
            "data_seed": None,
            "chain_seeds": (20240601, 20240602),
            "timing_seconds": 1.5,
        },
    )
    payload = fit_meta(directory)
    assert payload["arm"] is None
    assert payload["shared_t_max"] == float(fit.data.t_max)
    assert payload["extra"] == {
        "chain_seeds": [20240601, 20240602],
        "timing_seconds": 1.5,
    }


def test_fit_round_trip_with_one_draw_one_chain_one_grid_point(
    tmp_path: Path,
) -> None:
    """The None/One end of the contract: one chain, one draw, one grid point."""
    fit, directory = saved_fit(
        tmp_path,
        n_chains=1,
        n_draws=1,
        n_grid=1,
        with_band=True,
        band_draws=1,
    )
    loaded = load_fit(directory)
    assert_fits_equal(loaded, fit)
    assert loaded.t_grid.size == 1
    assert len(loaded.chains) == 1
    assert loaded.chains[0].nu.size == 1
    assert loaded.functionals.n_grid == 1


def test_fit_round_trip_all_censored_dataset(tmp_path: Path) -> None:
    """An all-censored dataset is a legal fit and survives the round trip."""
    event = np.zeros(6, dtype=bool)
    fit, directory = saved_fit(tmp_path, event=event)
    loaded = load_fit(directory)
    assert not loaded.data.event.any()
    assert_fits_equal(loaded, fit)


def test_save_fit_clears_a_stale_prior_band(tmp_path: Path) -> None:
    """Re-saving without a band must not leave the old file looking listed."""
    fit, directory = saved_fit(tmp_path, with_band=True)
    assert (directory / "prior-band.npz").exists()
    fit.prior_band = None
    save_fit(directory, fit, meta={"arm": "A", "data_seed": 721383})
    assert not (directory / "prior-band.npz").exists()
    assert load_fit(directory).prior_band is None


def test_failed_functionals_write_leaves_no_fit_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fit-meta.json`` is written last, so a crash leaves no readable fit."""
    fit = make_fit()
    directory = tmp_path / "fit"
    real_savez = paper_artifacts._savez

    def failing_savez(handle: object, arrays: object, *, compressed: bool) -> None:
        if isinstance(arrays, dict) and "S" in arrays:
            raise RuntimeError("disk full")
        real_savez(handle, arrays, compressed=compressed)  # type: ignore[arg-type]

    monkeypatch.setattr(paper_artifacts, "_savez", failing_savez)
    with pytest.raises(RuntimeError, match="disk full"):
        save_fit(directory, fit, meta={"arm": "A"})
    assert not (directory / "fit-meta.json").exists()
    assert not list(directory.glob("*.tmp"))
    with pytest.raises(ArtifactError, match="does not exist"):
        load_fit(directory)


# --- loader validation -----------------------------------------------------


def test_load_fit_rejects_a_missing_directory(tmp_path: Path) -> None:
    """A missing fit directory is an artifact error naming the path."""
    missing = tmp_path / "nowhere"
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_fit(missing)
    assert str(missing) in str(excinfo.value)


def test_load_fit_rejects_a_file(tmp_path: Path) -> None:
    """A file where a fit directory belongs is not silently opened."""
    path = tmp_path / "fit"
    path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ArtifactError, match="is not a directory"):
        load_fit(path)


def test_load_fit_rejects_a_missing_fit_meta(tmp_path: Path) -> None:
    """Chain files without ``fit-meta.json`` are not a fit."""
    directory = tmp_path / "fit"
    directory.mkdir()
    (directory / "chain-0.npz").write_bytes(b"")
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_fit(directory)
    assert "fit-meta.json" in str(excinfo.value)


def test_load_fit_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    """The schema has no migration path; an unknown version is an error."""
    _, directory = saved_fit(tmp_path)
    rewrite_fit_meta(
        directory,
        lambda payload: payload.__setitem__("schema_version", SCHEMA_VERSION + 1),
    )
    with pytest.raises(ArtifactIntegrityError, match="schema version"):
        load_fit(directory)


def test_load_fit_rejects_a_transposed_functionals_array(tmp_path: Path) -> None:
    """A transposed ``S`` is caught by the functional loader, not by plotting."""
    _, directory = saved_fit(tmp_path)
    path = directory / "functionals.npz"
    arrays = archive_members(path)
    rewrite_archive(path, S=np.ascontiguousarray(arrays["S"].T))
    with pytest.raises(ArtifactIntegrityError, match="shape|columns"):
        load_fit(directory)


def test_load_fit_rejects_a_wrong_draw_count(tmp_path: Path) -> None:
    """``n_draws`` must equal ``n_chains * n_save``; a stale count is an error."""
    _, directory = saved_fit(tmp_path)
    rewrite_fit_meta(directory, lambda payload: payload.__setitem__("n_draws", 7))
    with pytest.raises(ArtifactIntegrityError, match="n_draws"):
        load_fit(directory)


def test_load_fit_rejects_a_shape_disagreeing_with_fit_meta(tmp_path: Path) -> None:
    """A functional shape that drifted from ``fit-meta.json`` is caught."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        record = payload["functionals"]
        assert isinstance(record, dict)
        record["shape"] = [8, 6]

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="shape"):
        load_fit(directory)


def test_load_fit_rejects_a_dtype_disagreeing_with_fit_meta(tmp_path: Path) -> None:
    """The recorded dtype is part of the schema, not a comment."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        record = payload["functionals"]
        assert isinstance(record, dict)
        record["dtype"] = "float32"

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="dtype"):
        load_fit(directory)


def test_load_fit_rejects_a_chain_shape_disagreeing_with_fit_meta(
    tmp_path: Path,
) -> None:
    """A chain record that disagrees with the archive is an integrity error."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        records = payload["chains"]
        assert isinstance(records, list)
        records[0]["shape"] = [4, 7, 2]

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="shape"):
        load_fit(directory)


def test_load_fit_rejects_a_grid_summary_disagreement(tmp_path: Path) -> None:
    """The readable grid summary is checked against the stored grid."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        record = payload["t_grid"]
        assert isinstance(record, dict)
        record["last"] = float(record["last"]) + 1.0

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="t_grid"):
        load_fit(directory)


def test_load_fit_rejects_a_functionals_meta_disagreement(tmp_path: Path) -> None:
    """The stored meta dict must match the one recorded in ``fit-meta.json``."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        record = payload["functionals_meta"]
        assert isinstance(record, dict)
        record["method"] = "mc"

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="functionals_meta"):
        load_fit(directory)


def test_load_fit_rejects_a_recorded_band_without_a_file(tmp_path: Path) -> None:
    """A recorded prior band is part of the fit; a missing file is an error."""
    _, directory = saved_fit(tmp_path, with_band=True)
    (directory / "prior-band.npz").unlink()
    with pytest.raises(ArtifactError, match="does not exist"):
        load_fit(directory)


def test_load_fit_rejects_an_unlisted_band_file(tmp_path: Path) -> None:
    """An unrecorded band file would be hashed but never loaded; reject it."""
    fit, directory = saved_fit(tmp_path)
    band = make_functionals(n_grid=5, seed=20261119)
    band.t_grid = fit.t_grid
    save_functionals(directory / "prior-band.npz", band)
    with pytest.raises(ArtifactIntegrityError, match="unlisted prior band"):
        load_fit(directory)


def test_load_fit_rejects_a_meta_path_outside_the_directory(tmp_path: Path) -> None:
    """A recorded path may not escape the fit directory."""
    _, directory = saved_fit(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        records = payload["chains"]
        assert isinstance(records, list)
        records[0]["path"] = "../chain-0.npz"

    rewrite_fit_meta(directory, mutate)
    with pytest.raises(ArtifactIntegrityError, match="inside the fit directory"):
        load_fit(directory)


# --- writer validation -----------------------------------------------------


def test_save_fit_rejects_a_non_fit(tmp_path: Path) -> None:
    """A dict with the right attributes is not a ``Fit``."""
    with pytest.raises(ArtifactError, match="Fit"):
        save_fit(tmp_path / "fit", object())  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "meta",
    [
        {"arm": 7},
        {"arm": ""},
        {"data_seed": -1},
        {"data_seed": 1.5},
        {"shared_t_max": 0.0},
        {"shared_t_max": "soon"},
        {"timing": object()},
        [("arm", "A")],
    ],
)
def test_save_fit_rejects_invalid_provenance(tmp_path: Path, meta: object) -> None:
    """Provenance the loader could not read back must not be written."""
    fit = make_fit()
    with pytest.raises(ArtifactError):
        save_fit(tmp_path / "fit", fit, meta=meta)  # type: ignore[arg-type]
    assert not (tmp_path / "fit" / "fit-meta.json").exists()


def test_save_fit_rejects_a_chain_outside_the_fit(tmp_path: Path) -> None:
    """Every chain must share the fit's prior, settings and dataset."""
    fit = make_fit()
    fit.chains[1] = make_chain(seed=1, settings_seed=1)
    with pytest.raises(ArtifactError, match="settings"):
        save_fit(tmp_path / "fit", fit, meta={"arm": "A"})
    assert not (tmp_path / "fit" / "fit-meta.json").exists()


def test_save_fit_rejects_a_functionals_grid_disagreement(tmp_path: Path) -> None:
    """The functional grid and the fit grid are one grid, not two."""
    fit = make_fit()
    fit.functionals.t_grid = np.linspace(0.5, 9.0, 5)
    with pytest.raises(ArtifactError, match="t_grid"):
        save_fit(tmp_path / "fit", fit, meta={"arm": "A"})


def test_save_fit_rejects_a_draw_count_mismatch(tmp_path: Path) -> None:
    """``functionals`` must hold ``n_chains * n_save`` rows."""
    fit = make_fit()
    fit.functionals = make_functionals(n_draws=3, n_grid=5)
    fit.functionals.t_grid = fit.t_grid
    with pytest.raises(ArtifactError, match="n_chains \\* n_save"):
        save_fit(tmp_path / "fit", fit, meta={"arm": "A"})


# --- demo-result round-trip ------------------------------------------------


def demo_result(*, with_band: bool = True, seed: int | None = 20240602) -> DemoResult:
    """A two-arm result shaped like §5.2, with the §5.1 meta keys as well."""
    fit_a = make_fit(with_band=with_band)
    fit_b = make_fit(seed=20240603)
    return DemoResult(
        label="5.2",
        fits={
            f"A: {FIT_LABEL}": fit_a,
            f"B: {FIT_LABEL}": fit_b,
        },
        meta={
            "preset": "quick",
            "seed": seed,
            "data_seed": 721383,
            "prior_indices": (0, 2),
            "n_chains": 2,
            "burn_in": 5,
            "n_save": 4,
            "thin": 1,
            "t_max": 4.5518,
        },
    )


def test_demo_result_round_trips_fits_keys_and_meta(tmp_path: Path) -> None:
    """Every fit and the whole meta dict come back, in the driver's order."""
    result = demo_result()
    directory = save_demo_result(tmp_path / "5.2", result)
    run = load_demo_result(directory)
    assert run.label == "5.2"
    assert list(run.fits) == list(result.fits)
    assert run.meta == result.meta
    assert run.figures == ()
    for key, fit in result.fits.items():
        assert_fits_equal(run.fits[key], fit)
    assert fit_meta(directory / "draws" / "A-nu_Gamma_2_0.9")["arm"] == "A"
    assert fit_meta(directory / "draws" / "B-nu_Gamma_2_0.9")["arm"] == "B"
    assert (directory / "draws" / "A-nu_Gamma_2_0.9" / "prior-band.npz").exists()


def test_demo_result_meta_restores_prior_indices_as_a_tuple(tmp_path: Path) -> None:
    """JSON has no tuples; the paper preconditions compare ``prior_indices``."""
    result = demo_result(with_band=False, seed=None)
    directory = save_demo_result(tmp_path / "5.2", result)
    stored = demo_meta(directory)["meta"]
    assert stored["prior_indices"] == [0, 2]
    loaded = load_demo_result(directory)
    assert loaded.meta["prior_indices"] == (0, 2)
    assert isinstance(loaded.meta["prior_indices"], tuple)
    assert loaded.meta["seed"] is None
    assert loaded.meta["t_max"] == 4.5518
    assert loaded.meta == result.meta


def test_demo_result_round_trips_an_empty_run(tmp_path: Path) -> None:
    """The None end of the contract: no fits still round-trips its meta."""
    result = DemoResult(label="5.1", fits={}, meta={"preset": "quick"})
    directory = save_demo_result(tmp_path / "5.1", result)
    run = load_demo_result(directory)
    assert run.fits == {}
    assert run.meta == {"preset": "quick"}
    assert demo_meta(directory)["fit_keys"] == []


def test_load_demo_result_rejects_a_missing_directory(tmp_path: Path) -> None:
    """A missing section directory is named in the error."""
    missing = tmp_path / "nowhere"
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_demo_result(missing)
    assert str(missing) in str(excinfo.value)


def test_load_demo_result_rejects_a_missing_meta(tmp_path: Path) -> None:
    """Fits without a section ``meta.json`` are not a run."""
    directory = tmp_path / "5.2"
    directory.mkdir()
    with pytest.raises(ArtifactError, match="does not exist") as excinfo:
        load_demo_result(directory)
    assert "meta.json" in str(excinfo.value)


def test_load_demo_result_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    """The section schema is versioned like every other artifact."""
    directory = save_demo_result(tmp_path / "5.2", demo_result())
    payload = demo_meta(directory)
    payload["schema_version"] = SCHEMA_VERSION + 1
    (directory / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="schema version"):
        load_demo_result(directory)


def test_load_demo_result_rejects_an_unlisted_fit_directory(tmp_path: Path) -> None:
    """A fit directory the section metadata does not name is an error."""
    directory = save_demo_result(tmp_path / "5.2", demo_result())
    stray = directory / "draws" / "C-nu_Gamma_2_0.9"
    save_fit(stray, make_fit(), meta={"arm": "C"})
    with pytest.raises(ArtifactIntegrityError, match="unlisted fit directory"):
        load_demo_result(directory)


def test_load_demo_result_rejects_fits_in_a_section_that_names_none(
    tmp_path: Path,
) -> None:
    """An empty run has no ``draws/``; a stray fit there is still unlisted."""
    directory = save_demo_result(
        tmp_path / "5.1", DemoResult(label="5.1", fits={}, meta={})
    )
    save_fit(directory / "draws" / "nu_Gamma_2_0.9", make_fit(), meta={"arm": None})
    with pytest.raises(ArtifactIntegrityError, match="unlisted fit directory"):
        load_demo_result(directory)


def test_load_demo_result_rejects_a_key_mismatch(tmp_path: Path) -> None:
    """The fit directory must hold the fit the section key names."""
    directory = save_demo_result(tmp_path / "5.2", demo_result())
    rewrite_fit_meta(
        directory / "draws" / "A-nu_Gamma_2_0.9",
        lambda payload: payload.__setitem__("label", "nu ~ Gamma(3, 0.05)"),
    )
    with pytest.raises(ArtifactIntegrityError, match="disagree with the section key"):
        load_demo_result(directory)


def test_save_demo_result_rejects_duplicate_directory_names(tmp_path: Path) -> None:
    """Two labels that slug to the same name cannot both own a directory."""
    first = make_fit(label="nu ~ Gamma(2, 0.9)")
    second = make_fit(label="nu  ~ Gamma(2, 0.9)")
    result = DemoResult(
        label="5.1", fits={first.label: first, second.label: second}, meta={}
    )
    with pytest.raises(ArtifactError, match="both map to the directory"):
        save_demo_result(tmp_path / "5.1", result)
    assert not (tmp_path / "5.1" / "meta.json").exists()


def test_save_demo_result_rejects_a_key_label_mismatch(tmp_path: Path) -> None:
    """A key that is not the fit's (arm-prefixed) label is a caller bug."""
    fit = make_fit()
    result = DemoResult(label="5.2", fits={"A: something else": fit}, meta={})
    with pytest.raises(ArtifactError, match="does not match the fit's label"):
        save_demo_result(tmp_path / "5.2", result)


def test_save_demo_result_rejects_a_non_result(tmp_path: Path) -> None:
    """A duck-typed object is not a ``DemoResult``."""
    with pytest.raises(ArtifactError, match="DemoResult"):
        save_demo_result(tmp_path / "5.2", object())  # type: ignore[arg-type]


# --- integration -----------------------------------------------------------

#: A quick-preset fit of the §5.2 arm A is the cheapest *real* sampler run:
#: 2 chains x 1 000 stored draws over 20 observations (about 30 s).
QUICK_PRESET = Preset(
    name="quick",
    sampler=DPSettings(
        burn_in=1_000,
        n_save=1_000,
        thin=10,
        n_alpha=256,
        epsilon=1e-4,
        n_chains=2,
        seed=None,
        preset="quick",
    ),
    functionals=FunctionalSettings(method="exact", L=200, atom_tol=1e-6, n_t_grid=300),
    n_prior_draws=200,
)


def test_quick_preset_round_trip_feeds_r_hat_and_ess(tmp_path: Path) -> None:
    """A quick-preset result survives save/load and still feeds the diagnostics.

    This is the §6.1 acceptance check for the fit/run unit: the loaded chains
    must reproduce ``r_hat()``/``ess()`` exactly, the functionals (and the
    quick preset's prior band) must be bit-exact, and ``DemoResult.meta`` must
    compare equal to the in-memory dict that the paper preconditions use.
    """
    from src.paper_data import leukemia_5_2

    raw = leukemia_5_2()
    shared_t_max = max(float(np.max(raw[arm][0])) for arm in ("A", "B"))
    data = data_5_2("A", t_max=shared_t_max)
    grid = time_grid(shared_t_max, 300)
    fit = fit_dataset(
        data,
        make_prior(0),
        preset=QUICK_PRESET,
        seed=SEED_5_2,
        t_grid=grid,
        max_workers=2,
    )
    fit.prior_band = prior_band(fit, preset=QUICK_PRESET, seed=SEED_5_2)
    key = f"A: {fit.label}"
    result = DemoResult(
        label="5.2",
        fits={key: fit},
        meta={
            "preset": "quick",
            "seed": SEED_5_2,
            "prior_indices": (0,),
            "n_chains": 2,
            "burn_in": 1_000,
            "n_save": 1_000,
            "thin": 10,
            "t_max": shared_t_max,
            "t_max_a": data.t_max,
            "t_max_b": data.t_max,
        },
    )

    directory = save_demo_result(tmp_path / "5.2", result)
    run = load_demo_result(directory)
    assert run.meta == result.meta
    loaded = run.fits[key]
    assert_fits_equal(loaded, fit)

    for name in ("nu", "n_star"):
        assert r_hat(loaded.chains, name) == r_hat(fit.chains, name)
        assert ess(loaded.chains, name) == ess(fit.chains, name)
    assert r_hat(loaded.chains, "S").keys() == r_hat(fit.chains, "S").keys()
    for label in r_hat(loaded.chains, "S"):
        np.testing.assert_array_equal(
            r_hat(loaded.chains, "S")[label], r_hat(fit.chains, "S")[label]
        )
    for label in ess(loaded.chains, "S"):
        np.testing.assert_array_equal(
            ess(loaded.chains, "S")[label], ess(fit.chains, "S")[label]
        )


# ---------------------------------------------------------------------------
# Incremental run writer (plan §6.1)
# ---------------------------------------------------------------------------

#: The section provenance a §5.2 driver records; ``t_max`` is the shared grid
#: maximum, which is what makes arm B's fit record the common scale.
WRITER_META: dict[str, Any] = {
    "preset": "quick",
    "seed": 20240602,
    "data_seed": 721383,
    "prior_indices": (0, 2),
    "n_chains": 2,
    "burn_in": 5,
    "n_save": 4,
    "thin": 1,
    "t_max": 4.5518,
    "t_max_a": 4.5518,
    "t_max_b": 3.5,
}


def artifact_writer(
    tmp_path: Path, *, section: str = "5.2", **kwargs: Any
) -> ArtifactWriter:
    """An :class:`ArtifactWriter` rooted at ``tmp_path/run``."""
    kwargs.setdefault("meta", dict(WRITER_META))
    return ArtifactWriter(tmp_path / "run", section=section, **kwargs)


def writer_band(fit: Fit, *, n_draws: int = 3) -> Functionals:
    """A prior band on ``fit``'s grid, shaped like the §5.1 band."""
    band = make_functionals(
        n_draws=n_draws,
        n_grid=fit.t_grid.size,
        seed=20261118,
        meta=dict(PRIOR_BAND_META),
    )
    band.t_grid = np.array(fit.t_grid, copy=True)
    return band


def event_lines(path: Path) -> list[dict[str, Any]]:
    """Parse an event stream into its JSON records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- construction ----------------------------------------------------------


def test_artifact_writer_uses_the_plan_layout(tmp_path: Path) -> None:
    """Root, section directory and event stream follow plan §5 exactly."""
    root = tmp_path / "run"
    writer = ArtifactWriter(root, section="5.2", meta={"preset": "quick"})
    assert writer.root == root
    assert writer.section == "5.2"
    assert writer.section_dir == root / "5.2"
    assert writer.event_log == root / "logs" / "events-5.2.jsonl"
    assert writer.fit_keys == ()
    assert not root.exists()


@pytest.mark.parametrize("section", ["", ".", "..", "5/1", "5 1", "5\\1", 5, None])
def test_artifact_writer_rejects_an_unusable_section(
    tmp_path: Path, section: object
) -> None:
    """The section becomes a directory and file name, so it is validated."""
    with pytest.raises(ArtifactError, match="section"):
        ArtifactWriter(tmp_path / "run", section=section)  # type: ignore[arg-type]


@pytest.mark.parametrize("root", ["", None, 3])
def test_artifact_writer_rejects_an_unusable_root(root: object) -> None:
    """An empty or non-path run root is named in the error."""
    with pytest.raises(ArtifactError, match="run root"):
        ArtifactWriter(root, section="5.1")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "meta",
    [object(), [1, 2], {"bad": object()}, {"bad": complex(1.0, 2.0)}],
)
def test_artifact_writer_rejects_unusable_section_meta(
    tmp_path: Path, meta: object
) -> None:
    """The section meta is stored verbatim, so it must have a JSON form."""
    with pytest.raises(ArtifactError, match="section meta"):
        ArtifactWriter(tmp_path / "run", section="5.1", meta=meta)  # type: ignore[arg-type]


def test_artifact_writer_label_defaults_to_the_section(tmp_path: Path) -> None:
    """``DemoResult.label`` is the section name unless a caller overrides it."""
    writer = artifact_writer(tmp_path)
    writer.write_meta()
    assert load_demo_result(writer.section_dir).label == "5.2"


def test_artifact_writer_label_override_survives_write_meta(tmp_path: Path) -> None:
    """A readable label is kept verbatim rather than slugged."""
    writer = ArtifactWriter(
        tmp_path / "run", section="5.1", meta={}, label="Section 5.1"
    )
    writer.write_meta()
    assert load_demo_result(writer.section_dir).label == "Section 5.1"


# --- event stream ----------------------------------------------------------


def test_event_appends_one_json_object_per_line(tmp_path: Path) -> None:
    """Each call appends one record with the documented keys."""
    writer = artifact_writer(tmp_path)
    first = writer.event("run_start", stage="driver", preset="quick")
    second = writer.event("fit_start", key=FIT_LABEL, prior_indices=(0, 2))
    lines = event_lines(writer.event_log)
    assert len(lines) == 2
    assert lines[0] == first
    assert lines[1] == second
    assert lines[0]["kind"] == "run_start"
    assert lines[0]["stage"] == "driver"
    assert lines[1]["prior_indices"] == [0, 2]
    assert isinstance(lines[0]["t_monotonic"], float)
    assert lines[1]["t_monotonic"] >= lines[0]["t_monotonic"]


def test_event_stream_is_append_only_across_writers(tmp_path: Path) -> None:
    """A second writer of the same section extends the stream, never truncates."""
    first = artifact_writer(tmp_path)
    first.event("run_start")
    first.event("fit_done", key=FIT_LABEL)
    log = first.event_log
    before = log.read_text(encoding="utf-8")
    second = artifact_writer(tmp_path)
    second.event("run_end")
    after = log.read_text(encoding="utf-8")
    assert after.startswith(before)
    assert [record["kind"] for record in event_lines(log)] == [
        "run_start",
        "fit_done",
        "run_end",
    ]


def test_event_stamp_uses_the_injected_clocks(tmp_path: Path) -> None:
    """``ts`` and ``t_monotonic`` are the documented clocks, not local time."""
    writer = artifact_writer(
        tmp_path,
        wall_clock=lambda: "2026-11-08T00:00:01.250Z",
        monotonic=lambda: 42.5,
    )
    record = writer.event("run_start")
    assert record["ts"] == "2026-11-08T00:00:01.250Z"
    assert record["t_monotonic"] == 42.5
    parsed = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)


def test_default_event_stamp_is_iso_and_utc(tmp_path: Path) -> None:
    """Without a seam the writer stamps an ISO 8601 UTC wall clock."""
    writer = artifact_writer(tmp_path)
    record = writer.event("run_start")
    parsed = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
    assert parsed.utcoffset() == timedelta(0)
    assert isinstance(record["t_monotonic"], float)
    assert np.isfinite(record["t_monotonic"])


@pytest.mark.parametrize("field", ["ts", "t_monotonic"])
def test_event_rejects_a_reserved_field(tmp_path: Path, field: str) -> None:
    """A caller field may not shadow the log's fixed schema."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="shadow"):
        writer.event("run_start", **{field: "x"})
    assert not writer.event_log.exists()


@pytest.mark.parametrize("kind", ["", None, 3])
def test_event_rejects_an_invalid_kind(tmp_path: Path, kind: object) -> None:
    """Every line must name the stage it records."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="event kind"):
        writer.event(kind)  # type: ignore[arg-type]
    assert not writer.event_log.exists()


def test_event_rejects_fields_without_a_json_form(tmp_path: Path) -> None:
    """A non-serialisable field fails instead of being stringified."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="JSON-serialisable"):
        writer.event("run_start", payload=object())
    assert not writer.event_log.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wall_clock": lambda: ""},
        {"wall_clock": lambda: 3},  # type: ignore[dict-item]
        {"monotonic": lambda: float("inf")},
        {"monotonic": lambda: float("nan")},
    ],
)
def test_event_rejects_an_unusable_clock(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    """An empty stamp or a non-finite tick would corrupt the ordering."""
    writer = artifact_writer(tmp_path, **kwargs)
    with pytest.raises(ArtifactError, match="clock"):
        writer.event("run_start")
    assert not writer.event_log.exists()


# --- fit_done --------------------------------------------------------------


def test_fit_done_writes_a_plan_shaped_fit_directory(tmp_path: Path) -> None:
    """The directory name, fit-meta defaults and the event follow plan §5/§5.1."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit, timing=1.5, arm="A")
    assert directory == writer.section_dir / "draws" / "A-nu_Gamma_2_0.9"
    assert_fits_equal(load_fit(directory), fit)
    stored = fit_meta(directory)
    assert stored["arm"] == "A"
    assert stored["label"] == FIT_LABEL
    assert stored["data_seed"] == 721383
    assert stored["shared_t_max"] == 4.5518
    assert stored["extra"]["timing"] == 1.5
    assert writer.fit_keys == (f"A: {FIT_LABEL}",)
    records = event_lines(writer.event_log)
    assert len(records) == 1
    assert records[0]["kind"] == "fit_done"
    assert records[0]["key"] == f"A: {FIT_LABEL}"
    assert records[0]["path"] == "5.2/draws/A-nu_Gamma_2_0.9"
    assert records[0]["n_chains"] == fit.n_chains
    assert records[0]["n_draws"] == fit.functionals.n_draws
    assert records[0]["timing"] == 1.5


def test_fit_done_without_an_arm_uses_the_five_one_directory(tmp_path: Path) -> None:
    """§5.1 fits are keyed by the bare label and get no arm prefix."""
    writer = artifact_writer(tmp_path, section="5.1")
    fit = make_fit()
    directory = writer.fit_done(fit)
    assert directory == writer.section_dir / "draws" / "nu_Gamma_2_0.9"
    assert fit_meta(directory)["arm"] is None
    assert writer.fit_keys == (FIT_LABEL,)
    assert event_lines(writer.event_log)[0]["key"] == FIT_LABEL


def test_fit_done_provenance_overrides_the_section_defaults(tmp_path: Path) -> None:
    """Explicit provenance wins over the section meta; the rest is ``extra``."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(
        fit,
        timing={"chains": 0.5},
        arm="A",
        data_seed=11,
        shared_t_max=9.5,
        chain_seeds=(1, 2),
    )
    stored = fit_meta(directory)
    assert stored["data_seed"] == 11
    assert stored["shared_t_max"] == 9.5
    assert stored["extra"] == {"timing": {"chains": 0.5}, "chain_seeds": [1, 2]}


def test_fit_done_rejects_a_duplicate_fit(tmp_path: Path) -> None:
    """Writing the same key twice would silently replace a finished fit."""
    writer = artifact_writer(tmp_path)
    writer.fit_done(make_fit(), arm="A")
    with pytest.raises(ArtifactError, match="already written"):
        writer.fit_done(make_fit(), arm="A")
    assert len(event_lines(writer.event_log)) == 1


def test_fit_done_rejects_a_slug_collision(tmp_path: Path) -> None:
    """Two labels that slug to one name cannot both own a directory."""
    writer = artifact_writer(tmp_path, section="5.1")
    writer.fit_done(make_fit(label="nu ~ Gamma(2, 0.9)"))
    with pytest.raises(ArtifactError, match="both map to the directory"):
        writer.fit_done(make_fit(label="nu  ~ Gamma(2, 0.9)"))


def test_fit_done_rejects_a_non_fit(tmp_path: Path) -> None:
    """A duck-typed object is not a ``Fit``."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="Fit"):
        writer.fit_done(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("arm", ["", "A/B", "A B", "5A", 1])
def test_fit_done_rejects_an_unusable_arm(tmp_path: Path, arm: object) -> None:
    """The arm prefixes a directory name, so ``A/B`` cannot be accepted."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="arm"):
        writer.fit_done(make_fit(), arm=arm)  # type: ignore[arg-type]


def test_fit_done_rejects_provenance_without_a_json_form(tmp_path: Path) -> None:
    """Fields are validated before the fit directory is written."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="JSON-serialisable"):
        writer.fit_done(make_fit(), bad=object())
    assert not (writer.section_dir / "draws" / "nu_Gamma_2_0.9").exists()
    assert not writer.event_log.exists()


# --- band_done -------------------------------------------------------------


def test_band_done_attaches_the_band_to_the_first_fit(tmp_path: Path) -> None:
    """The §5.1 reference fit carries the band, and the metadata lists it."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit, arm="A")
    band = writer_band(fit)
    path = writer.band_done(band, timing=0.25)
    assert path == directory / "prior-band.npz"
    loaded = load_fit(directory)
    assert loaded.prior_band is not None
    assert_functionals_equal(loaded.prior_band, band)
    assert loaded.prior_band.meta == _jsonable(PRIOR_BAND_META)
    stored = fit_meta(directory)
    assert stored["prior_band"] == {
        "path": "prior-band.npz",
        "dtype": "float64",
        "shape": [3, int(fit.t_grid.size)],
    }
    assert stored["prior_band_meta"] == _jsonable(PRIOR_BAND_META)
    assert stored["extra"]["band_timing"] == 0.25
    records = event_lines(writer.event_log)
    assert [record["kind"] for record in records] == ["fit_done", "band_done"]
    assert records[1]["key"] == f"A: {FIT_LABEL}"
    assert records[1]["n_draws"] == 3
    assert records[1]["path"] == "5.2/draws/A-nu_Gamma_2_0.9/prior-band.npz"
    assert records[1]["timing"] == 0.25


def test_band_done_targets_an_explicit_fit(tmp_path: Path) -> None:
    """An object, a label and an arm each select the band's fit directory."""
    writer = artifact_writer(tmp_path)
    fit_a = make_fit()
    fit_b = make_fit(seed=20240603)
    directory_a = writer.fit_done(fit_a, arm="A")
    directory_b = writer.fit_done(fit_b, arm="B")
    band = writer_band(fit_b)
    assert writer.band_done(band, fit=fit_b, arm="B") == directory_b / "prior-band.npz"
    assert load_fit(directory_b).prior_band is not None
    assert load_fit(directory_a).prior_band is None
    assert writer.band_done(band, arm="A") == directory_a / "prior-band.npz"
    assert load_fit(directory_a).prior_band is not None
    with pytest.raises(ArtifactError, match="several arms"):
        writer.band_done(band, fit_label=FIT_LABEL)


def test_band_done_rejects_an_unknown_target(tmp_path: Path) -> None:
    """A label no fit was written under names the miss, and writes nothing."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit, arm="A")
    band = writer_band(fit)
    with pytest.raises(ArtifactError, match="no fit with label"):
        writer.band_done(band, fit_label="nu ~ Gamma(3, 0.05)")
    assert not (directory / "prior-band.npz").exists()
    assert fit_meta(directory)["prior_band"] is None


def test_band_done_requires_a_written_fit(tmp_path: Path) -> None:
    """The band is attached to a fit, so there must be one to attach it to."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="call fit_done"):
        writer.band_done(make_functionals())


def test_band_done_rejects_a_non_functionals(tmp_path: Path) -> None:
    """A duck-typed band is refused before any write."""
    writer = artifact_writer(tmp_path)
    writer.fit_done(make_fit())
    with pytest.raises(ArtifactError, match="Functionals"):
        writer.band_done(object())  # type: ignore[arg-type]


def test_band_done_rejects_a_grid_disagreement(tmp_path: Path) -> None:
    """A band on a different grid would be rejected by ``load_fit``; refuse early."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit)
    band = writer_band(fit)
    band.t_grid = band.t_grid * 1.5
    with pytest.raises(ArtifactError, match="grid"):
        writer.band_done(band)
    assert not (directory / "prior-band.npz").exists()
    assert fit_meta(directory)["prior_band"] is None


def test_band_done_rejects_arrays_outside_the_schema(tmp_path: Path) -> None:
    """A shape disagreement between the four arrays is caught before writing."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit)
    band = writer_band(fit)
    band.S = band.S[:, :2]
    with pytest.raises(ArtifactError, match="outside the artifact schema"):
        writer.band_done(band)
    assert not (directory / "prior-band.npz").exists()
    assert fit_meta(directory)["prior_band"] is None


def test_band_done_rejects_a_changed_fit_directory(tmp_path: Path) -> None:
    """A fit directory resaved under another label is not the recorded fit."""
    writer = artifact_writer(tmp_path)
    fit = make_fit()
    directory = writer.fit_done(fit, arm="A")
    rewrite_fit_meta(directory, lambda payload: payload.__setitem__("arm", "B"))
    with pytest.raises(ArtifactError, match="no longer match"):
        writer.band_done(writer_band(fit))


# --- write_meta ------------------------------------------------------------


def test_write_meta_round_trips_through_load_demo_result(tmp_path: Path) -> None:
    """A writer-built section loads as the driver's ``DemoResult``."""
    writer = artifact_writer(tmp_path)
    fit_a = make_fit()
    fit_b = make_fit(seed=20240603)
    band = writer_band(fit_a)
    writer.fit_done(fit_a, timing=2.0, arm="A")
    writer.fit_done(fit_b, timing=1.0, arm="B")
    writer.band_done(band, fit=fit_a, arm="A", timing=0.1)
    meta_path = writer.write_meta()
    assert meta_path == writer.section_dir / "meta.json"
    assert not (meta_path.with_suffix(".json.tmp")).exists()
    run = load_demo_result(writer.section_dir)
    keys = [f"A: {FIT_LABEL}", f"B: {FIT_LABEL}"]
    assert list(run.fits) == keys
    assert run.label == "5.2"
    assert run.meta == WRITER_META
    assert run.meta["prior_indices"] == (0, 2)
    assert run.figures == ()
    assert_fits_equal(run.fits[keys[0]], replace(fit_a, prior_band=band))
    assert_fits_equal(run.fits[keys[1]], fit_b)
    assert demo_meta(writer.section_dir)["fit_keys"] == keys


def test_write_meta_round_trips_an_empty_section(tmp_path: Path) -> None:
    """The None end: no fits still produces a loadable section."""
    writer = artifact_writer(
        tmp_path, section="5.1", meta={"preset": "quick", "seed": None}
    )
    writer.write_meta()
    run = load_demo_result(writer.section_dir)
    assert run.fits == {}
    assert run.label == "5.1"
    assert run.meta == {"preset": "quick", "seed": None}
    assert demo_meta(writer.section_dir)["fit_keys"] == []


def test_write_meta_rejects_a_missing_fit_directory(tmp_path: Path) -> None:
    """A section that names a fit it no longer holds is refused."""
    writer = artifact_writer(tmp_path)
    directory = writer.fit_done(make_fit())
    (directory / "fit-meta.json").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        writer.write_meta()
    assert not (writer.section_dir / "meta.json").exists()


def test_write_meta_accepts_a_replacement_meta(tmp_path: Path) -> None:
    """The driver may supply the final provenance when the section ends."""
    writer = artifact_writer(tmp_path)
    writer.write_meta(meta={"preset": "paper", "seed": None})
    assert load_demo_result(writer.section_dir).meta == {
        "preset": "paper",
        "seed": None,
    }


@pytest.mark.parametrize("meta", [{"bad": object()}, [1, 2], 3])
def test_write_meta_rejects_unusable_meta(tmp_path: Path, meta: object) -> None:
    """A replacement meta is held to the same JSON contract."""
    writer = artifact_writer(tmp_path)
    with pytest.raises(ArtifactError, match="section meta"):
        writer.write_meta(meta=meta)  # type: ignore[arg-type]


def test_writer_is_closed_after_the_context_manager(tmp_path: Path) -> None:
    """Every writer method refuses to write once the block has exited."""
    root = tmp_path / "run"
    with ArtifactWriter(root, section="5.2", meta={"preset": "quick"}) as writer:
        assert writer.root == root
        writer.event("run_start")
    with pytest.raises(ArtifactError, match="closed"):
        writer.event("run_end")
    with pytest.raises(ArtifactError, match="closed"):
        writer.fit_done(make_fit())
    with pytest.raises(ArtifactError, match="closed"):
        writer.band_done(make_functionals())
    with pytest.raises(ArtifactError, match="closed"):
        writer.write_meta()
