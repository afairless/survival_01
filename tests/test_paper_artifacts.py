"""Plan §6.1: the artifact primitives (``src/paper_artifacts.py``).

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

The property-based tests use seeded NumPy generators; the repository has no
Hypothesis dependency.  The inputs are random, the invariants are
deterministic.
"""

import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src import paper_artifacts
from src.paper_artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    ArtifactIntegrityError,
    _jsonable,
    atomic_savez,
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
