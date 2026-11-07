"""Plan §6.1: the artifact primitives and the chain round-trip.

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

Step 2 adds the chain round-trip of plan §5.1.  ``chain-<b>.npz`` is the
storage contract a reviewer loads without the fitting process, so the tests
pin the whole schema boundary: every array is bit-exact, the ``meta`` blob is
self-contained (schema version, provenance, prior, settings), and the loader
rejects a wrong key set, dtype, shape, draw count, schema version or metadata
field *before* ``DPChains`` is built.  The one-draw/one-grid-point and
all-censored cases are covered here because this is the container that
introduces them.

The property-based tests use seeded NumPy generators; the repository has no
Hypothesis dependency.  The inputs are random, the invariants are
deterministic.
"""

import json
import os
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src import paper_artifacts
from src.dp_gibbs import DPChains, DPPrior, DPSettings, SurvData
from src.paper_artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    ArtifactIntegrityError,
    _jsonable,
    atomic_savez,
    load_chain,
    save_chain,
    slug,
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


def rewrite_chain(
    path: Path, *, drop: tuple[str, ...] = (), **changes: np.ndarray
) -> None:
    """Replace or drop archive members, keeping the write atomic."""
    arrays = archive_members(path)
    for key in drop:
        arrays.pop(key)
    arrays.update(changes)
    atomic_savez(path, arrays)


def rewrite_chain_meta(path: Path, mutate: object) -> None:
    """Apply ``mutate`` to the JSON ``meta`` object and rewrite the archive."""
    arrays = archive_members(path)
    record = json.loads(str(arrays["meta"][()]))
    mutate(record)  # type: ignore[operator]
    arrays["meta"] = np.asarray(json.dumps(record))
    atomic_savez(path, arrays)


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
