"""Serialisation primitives for the artifact-backed replication run.

The replication run of ``docs/research/dp_gibbs_replication_artifacts_plan.md``
stores every draw, functional array and summary it produces under a
git-ignored ``output/`` tree (§5 of that plan).  A reviewer must be able to
load those files without the fitting process, so the loaders of this module
will reconstruct containers explicitly (never via pickle) and validate every
array against the declared schema first.

This unit contributes the schema-independent primitives that the chain,
functional, fit and run writers are built on, plus the chain round-trip:

* :data:`SCHEMA_VERSION` -- stamped into every ``meta`` blob, section
  ``meta.json`` and run ``MANIFEST.json``; a loader rejects an unknown value
  rather than guessing a migration;
* :class:`ArtifactError` and :class:`ArtifactIntegrityError` -- the error
  vocabulary of the module (the latter is raised when a file disagrees with
  its schema or its checksum);
* :func:`slug` -- the one directory-name rule of plan §5, so an in-memory
  prior label ``"nu ~ Gamma(2, 0.9)"`` maps to ``"nu_Gamma_2_0.9"`` (arm A
  is the prefix ``"A-"``);
* :func:`_jsonable` -- converts NumPy scalars, arrays and the metadata
  containers into values ``json.dumps`` accepts, because the meta dicts are
  stored verbatim rather than through a hand-written field list;
* :func:`atomic_savez` -- an ``npz`` write that is either complete or absent;
* :func:`save_chain` / :func:`load_chain` -- one self-contained ``DPChains``
  (plan §5.1): every stored array, the dataset, the prior and the settings,
  so a reviewer can load a chain without the fitting process;
* :func:`save_functionals` / :func:`load_functionals` -- one ``Functionals``
  (plan §5.1): ``t_grid`` and the four ``(B, M)`` arrays plus the provenance
  dict verbatim, so a prior band and a posterior functional of the same shape
  are stored by one rule;
* :func:`save_fit` / :func:`load_fit` -- one per-fit directory (plan §5.1):
  ``chain-<b>.npz`` for every chain, ``functionals.npz``, an optional
  ``prior-band.npz`` and a ``fit-meta.json`` holding the **un-slugged**
  ``label``, the ``arm``, both meta dicts verbatim and the recorded array
  shapes/dtypes;
* :func:`save_demo_result` / :func:`load_demo_result` -- a whole section
  (plan §5): the section ``meta.json`` plus one fit directory per fit under
  ``draws/``, reconstructing ``DemoResult.meta`` with its original types
  (``prior_indices`` stays a tuple);
* :class:`ArtifactWriter` -- the *incremental* writer the drivers use (plan
  §6.1): each fit is written as soon as it is fitted, the prior band is
  attached to the reference fit, and the section ``meta.json`` is written last,
  so a crashed run never claims artifacts it does not hold.  It owns the
  append-only ``logs/events-<section>.jsonl`` stream (``ts``/
  ``t_monotonic``/``kind``) and never writes ``MANIFEST.json`` -- the manifest
  unit is layered on top;
* :func:`finalize_run` -- the single run-level ``MANIFEST.json`` of plan §6.1:
  one SHA-256 and size per file in the whole tree (draws, summaries, reports,
  figures, logs), plus the code/environment provenance and the section
  ``meta.json`` payloads.  The manifest never lists itself, because a file
  cannot carry its own hash; the report records that hash externally;
* :func:`verify_manifest` -- the reviewer entry point: re-hash every listed
  file, fail on a flipped byte, a truncated or missing file and an unlisted
  extra, and return a :class:`ManifestReport` naming what was checked.

Why the write helper exists
---------------------------
Handing ``chain-0.npz.tmp`` to :func:`numpy.savez` does not work: NumPy
appends ``.npz`` to names that lack it, so the temporary file would become
``chain-0.npz.tmp.npz`` and ``os.replace`` would rename the wrong name.  The
helper therefore opens ``path.with_suffix(path.suffix + ".tmp")`` itself,
writes through the handle, flushes and :func:`os.fsync`\\ es it, closes, and
only then renames with :func:`os.replace`.  A crash mid-save never leaves a
readable-but-truncated archive and never leaves a stray ``*.npz.npz`` behind.

Scope
-----
This module imports NumPy, the standard library and the container types it
reconstructs on load (``DPChains`` and friends from ``dp_gibbs``, and
``Functionals`` from ``dp_functionals``).  It must never import ``matplotlib``
or any plotting code (plan §6.1), and it never runs the sampler or evaluates a
functional -- it only owns the containers.  ``Fit`` and ``DemoResult`` live in
``src.paper_examples``; importing that module is cheap (its ``matplotlib``
import is lazy), but the import is done **inside** the fit/run functions rather
than at module scope, so the driver can import this module in either order
without a cycle.  The manifest unit of plan §6.1 is layered on top of the
primitives here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import IO, TYPE_CHECKING, Any

import numpy as np

from src.dp_functionals import Functionals
from src.dp_gibbs import DPChains, DPPrior, DPSettings, SurvData

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at run time
    from src.paper_examples import DemoResult, Fit

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactWriter",
    "ManifestReport",
    "atomic_savez",
    "finalize_run",
    "load_chain",
    "load_demo_result",
    "load_fit",
    "load_functionals",
    "manifest_path",
    "save_chain",
    "save_demo_result",
    "save_fit",
    "save_functionals",
    "slug",
    "verify_manifest",
]

#: Version of the on-disk artifact schema (plan §5.1).  Bumped only when a
#: loader can no longer interpret an older ``meta`` blob; this plan has no
#: migration path, so a mismatch is an error rather than a conversion.
SCHEMA_VERSION = 1

#: Every run of characters outside this set becomes a single ``_``.
_NON_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactError(Exception):
    """Base error for building, writing or reading replication artifacts."""


class ArtifactIntegrityError(ArtifactError):
    """An artifact disagrees with its declared schema, size or checksum."""


def slug(label: str) -> str:
    """Return the plan §5 directory name for ``label``.

    Every run of characters outside ``[A-Za-z0-9._-]`` is replaced by one
    underscore and boundary underscores are trimmed, so
    ``"nu ~ Gamma(2, 0.9)"`` becomes ``"nu_Gamma_2_0.9"`` (not
    ``"nu_Gamma_2_0.9_"``) and an arm prefix simply prepends to the result
    (``"A-nu_Gamma_2_0.9"``).  The slug is deliberately not invertible: the
    original label is stored in ``fit-meta.json`` and is what the loaders key
    ``DemoResult.fits`` on.

    The mapping is idempotent (``_`` is itself allowed) and deterministic.  An
    empty result would name a directory that cannot be created, so it is
    rejected instead of producing a silently wrong path.

    Raises
    ------
    ArtifactError
        If ``label`` is not a string or has no slug-able characters.
    """
    if not isinstance(label, str):
        raise ArtifactError(
            f"artifact label must be a string, got {type(label).__name__}"
        )
    slugged = _NON_SLUG.sub("_", label).strip("_")
    if not slugged:
        raise ArtifactError(f"artifact label {label!r} yields an empty slug")
    return slugged


def _jsonable(value: Any) -> Any:
    """Convert ``value`` into a structure :func:`json.dumps` accepts.

    NumPy scalars become the corresponding Python scalar, arrays become nested
    lists, mappings and sequences are converted recursively, and ``None``,
    ``bool``, ``int``, ``float`` and ``str`` pass through unchanged.  Mapping
    keys are stringified because JSON objects cannot carry other key types.

    The conversion is used for provenance and for the ``meta`` dicts that must
    round-trip verbatim (``Functionals.meta``, the per-chain blob and the
    per-fit metadata).  Non-finite floats are left as floats: Python's
    :mod:`json` writes and reads ``NaN``/``Infinity`` by default, and turning
    them into strings would change the value on the way back in.

    Raises
    ------
    ArtifactError
        If ``value`` contains something with no JSON representation (for
        example ``object()`` or a ``complex``).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise ArtifactError(
        f"value of type {type(value).__name__} is not JSON-serialisable"
    )


def _savez(handle: IO[bytes], arrays: Mapping[str, Any], *, compressed: bool) -> None:
    """Write ``arrays`` into an open binary handle (test seam for failures)."""
    if compressed:
        np.savez_compressed(handle, **arrays)
    else:
        np.savez(handle, **arrays)


def atomic_savez(
    path: str | os.PathLike[str],
    arrays: Mapping[str, Any],
    *,
    compressed: bool = True,
) -> Path:
    """Write ``arrays`` to ``path`` as an ``.npz`` archive, atomically.

    The archive is written to ``<path>.tmp`` through an open binary handle
    (never by giving NumPy a temporary *name*), flushed, ``fsync``-ed, closed
    and only then moved into place with :func:`os.replace`.  If anything fails
    -- including a failure after some bytes were written -- the temporary file
    is removed and the previous contents of ``path`` (if any) are untouched.

    Parameters
    ----------
    path:
        Destination ending in ``.npz``.  Missing parent directories are
        created; the helper never appends an extension of its own.
    arrays:
        Archive members.  Arrays are written one at a time by NumPy from the
        objects that already own them, so the caller's arrays are not copied
        up front (plan §6.1: never hold two extra copies of a large trace).
    compressed:
        Use :func:`numpy.savez_compressed` (the default, subject to the R8
        size/CPU measurement) instead of :func:`numpy.savez`.

    Returns
    -------
    pathlib.Path
        The path actually written.

    Raises
    ------
    ArtifactError
        If ``path`` does not end in ``.npz``.
    """
    target = Path(path)
    if target.suffix != ".npz":
        raise ArtifactError(f"npz artifact path must end in '.npz': {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            _savez(handle, arrays, compressed=compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# Chain serialisation (plan §5.1)
# ---------------------------------------------------------------------------

#: The ``DPPrior`` fields stored in a ``meta`` blob, in declaration order.
_PRIOR_FIELDS = ("a_nu", "b_nu", "a_gamma", "b_gamma", "a_phi", "b_phi", "d")

#: The ``DPSettings`` fields stored in a ``meta`` blob.
_SETTINGS_FIELDS = (
    "burn_in",
    "n_save",
    "thin",
    "n_alpha",
    "epsilon",
    "n_chains",
    "seed",
    "preset",
)

#: Keys of one ``chain-<b>.npz`` (plan §5.1).  Extra keys are rejected so a
#: typo cannot hide behind an ignored array.
_CHAIN_KEYS = frozenset(
    {
        "theta",
        "nu",
        "gamma",
        "phi",
        "n_star",
        "log_post",
        "time",
        "event",
        "t_max",
        "t_grid",
        "meta",
    }
)

#: Keys of the per-chain ``meta`` JSON object.
_CHAIN_META_KEYS = frozenset(
    {"schema_version", "label", "chain_index", "prior", "settings"}
)

#: The array fields ``DPChains`` stores per draw; all are ``float64``.
_DRAW_FLOAT_FIELDS = ("nu", "gamma", "phi", "log_post")


def _open_npz(path: str | os.PathLike[str]) -> np.lib.npyio.NpzFile:
    """Open ``path`` as an ``npz`` archive without ever unpickling.

    The file is checked first, so a missing path produces an
    :class:`ArtifactError` naming it instead of NumPy's bare ``FileNotFoundError``
    from the middle of a loader.  A file that is not a readable ``npz`` -- a
    truncated download, a plain ``.npy`` renamed to ``.npz`` -- is an
    :class:`ArtifactIntegrityError`, because the archive disagrees with the
    declared schema.
    """
    target = Path(path)
    if not target.exists():
        raise ArtifactError(f"artifact does not exist: {target}")
    if not target.is_file():
        raise ArtifactError(f"artifact is not a file: {target}")
    try:
        archive = np.load(target, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ArtifactIntegrityError(
            f"{target}: cannot read npz archive: {exc}"
        ) from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ArtifactIntegrityError(
            f"{target}: not an npz archive (np.load returned {type(archive).__name__})"
        )
    return archive


def _require_exact_keys(
    archive: np.lib.npyio.NpzFile,
    expected: frozenset[str],
    target: Path,
    *,
    kind: str,
) -> None:
    """Reject an archive whose member names disagree with the schema."""
    present = set(archive.files)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ArtifactIntegrityError(
            f"{target}: {kind} schema mismatch ({', '.join(details)})"
        )


def _member(archive: np.lib.npyio.NpzFile, key: str, target: Path) -> np.ndarray:
    """Read one array member, turning a corrupt member into an artifact error."""
    try:
        return archive[key]
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as exc:
        raise ArtifactIntegrityError(
            f"{target}: cannot read array {key!r}: {exc}"
        ) from exc


def _require_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    target: Path,
    *,
    dtype: type,
    ndim: int,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate one array's dtype, rank and shape *before* it is used.

    The dtype is checked against the plan §5.1 table: ``float64`` for every
    real-valued array, ``bool`` for ``event`` and any integer type for
    ``n_star`` (Linux ``intp`` is ``int64``; the container casts back).  Shape
    checks are performed against the header only, so a corrupt archive cannot
    make the loader allocate an arbitrarily large array.
    """
    array = _member(archive, key, target)
    if dtype is np.integer:
        dtype_ok = np.issubdtype(array.dtype, np.integer)
        expected_dtype = "an integer dtype"
    else:
        dtype_ok = array.dtype == np.dtype(dtype)
        expected_dtype = str(np.dtype(dtype))
    if not dtype_ok:
        raise ArtifactIntegrityError(
            f"{target}: array {key!r} has dtype {array.dtype}, "
            f"expected {expected_dtype}"
        )
    if array.ndim != ndim:
        raise ArtifactIntegrityError(
            f"{target}: array {key!r} has shape {array.shape}, "
            f"expected a {ndim}-D array"
        )
    if expected_shape is not None and array.shape != expected_shape:
        raise ArtifactIntegrityError(
            f"{target}: array {key!r} has shape {array.shape}, "
            f"expected {expected_shape}"
        )
    return array


def _require_string_scalar(
    archive: np.lib.npyio.NpzFile, key: str, target: Path
) -> str:
    """Return the 0-d unicode member ``key`` as a Python string."""
    array = _member(archive, key, target)
    if array.dtype.kind != "U" or array.ndim != 0:
        raise ArtifactIntegrityError(
            f"{target}: array {key!r} must be a 0-d unicode string, "
            f"got dtype {array.dtype} and shape {array.shape}"
        )
    return str(array[()])


def _as_meta_number(value: Any, name: str, target: Path) -> float | int:
    """Validate one finite numeric metadata value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be a number, got {value!r}"
        )
    if not np.isfinite(float(value)):
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be finite, got {value!r}"
        )
    return value


def _as_meta_int(
    value: Any, name: str, target: Path, *, minimum: int | None = None
) -> int:
    """Validate one integer metadata value, optionally bounded below."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be an integer, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be at least {minimum}, got {value!r}"
        )
    return value


def _require_record(
    value: Any, expected: tuple[str, ...] | frozenset[str], name: str, target: Path
) -> Mapping[str, Any]:
    """Validate a nested JSON object against an exact key set."""
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be a JSON object, "
            f"got {type(value).__name__}"
        )
    keys = set(value)
    expected_keys = set(expected)
    missing = sorted(expected_keys - keys)
    extra = sorted(keys - expected_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} schema mismatch ({', '.join(details)})"
        )
    return value


def _prior_from_record(record: Any, target: Path) -> DPPrior:
    """Validate a stored ``DPPrior`` record and rebuild it."""
    record = _require_record(record, _PRIOR_FIELDS, "prior", target)
    values: dict[str, Any] = {
        name: _as_meta_number(record[name], f"prior.{name}", target)
        for name in _PRIOR_FIELDS[:-1]
    }
    values["d"] = _as_meta_int(record["d"], "prior.d", target, minimum=1)
    return DPPrior(**values)


def _settings_from_record(record: Any, target: Path) -> DPSettings:
    """Validate a stored ``DPSettings`` record and rebuild it."""
    record = _require_record(record, _SETTINGS_FIELDS, "settings", target)
    values: dict[str, Any] = {
        "burn_in": _as_meta_int(
            record["burn_in"], "settings.burn_in", target, minimum=0
        ),
        "n_save": _as_meta_int(record["n_save"], "settings.n_save", target, minimum=1),
        "thin": _as_meta_int(record["thin"], "settings.thin", target, minimum=1),
        "n_alpha": _as_meta_int(
            record["n_alpha"], "settings.n_alpha", target, minimum=2
        ),
        "epsilon": _as_meta_number(record["epsilon"], "settings.epsilon", target),
        "n_chains": _as_meta_int(
            record["n_chains"], "settings.n_chains", target, minimum=1
        ),
        "seed": None
        if record["seed"] is None
        else _as_meta_int(record["seed"], "settings.seed", target, minimum=0),
        "preset": record["preset"],
    }
    epsilon = float(values["epsilon"])
    if not 0.0 < epsilon < 1.0:
        raise ArtifactIntegrityError(
            f"{target}: meta field 'settings.epsilon' must lie strictly "
            f"between 0 and 1, got {epsilon!r}"
        )
    preset = values["preset"]
    if not isinstance(preset, str) or not preset:
        raise ArtifactIntegrityError(
            f"{target}: meta field 'settings.preset' must be a non-empty "
            f"string, got {preset!r}"
        )
    return DPSettings(**values)


def _read_chain_meta(
    archive: np.lib.npyio.NpzFile, target: Path
) -> tuple[str | None, int | None, DPPrior, DPSettings]:
    """Parse and validate the self-contained per-chain metadata blob."""
    text = _require_string_scalar(archive, "meta", target)
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"{target}: meta blob is not valid JSON: {exc}"
        ) from exc
    record = _require_record(record, _CHAIN_META_KEYS, "meta", target)

    version = _as_meta_int(
        record["schema_version"], "schema_version", target, minimum=1
    )
    if version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"{target}: schema version {version} is not the supported "
            f"version {SCHEMA_VERSION} (no migration path)"
        )

    label = record["label"]
    if label is not None and not isinstance(label, str):
        raise ArtifactIntegrityError(
            f"{target}: meta field 'label' must be a string or null, got {label!r}"
        )
    chain_index = record["chain_index"]
    if chain_index is not None:
        chain_index = _as_meta_int(chain_index, "chain_index", target, minimum=0)

    return (
        label,
        chain_index,
        _prior_from_record(record["prior"], target),
        _settings_from_record(record["settings"], target),
    )


def _validated_t_grid(t_grid: object) -> np.ndarray:
    """Validate an original-scale evaluation grid before it is written."""
    try:
        grid = np.asarray(t_grid, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"t_grid must be a 1-D array of times: {exc}") from exc
    if grid.ndim != 1 or grid.size == 0:
        raise ArtifactError(
            f"t_grid must be a non-empty 1-D array, got shape {grid.shape}"
        )
    if not np.all(np.isfinite(grid)):
        raise ArtifactError("t_grid must contain only finite times")
    if np.any(grid <= 0.0):
        raise ArtifactError(
            "t_grid must be strictly positive on the original time scale"
        )
    return np.array(grid, dtype=np.float64, copy=True)


def save_chain(
    path: str | os.PathLike[str],
    chain: DPChains,
    *,
    t_grid: object,
    label: str | None = None,
    chain_index: int | None = None,
) -> Path:
    """Write one :class:`DPChains` as a self-contained ``chain-<b>.npz``.

    The archive holds every stored array of the chain plus ``time``, ``event``,
    ``t_max`` and the fit's ``t_grid``, and a ``meta`` string with the schema
    version, the provenance (``label``, ``chain_index``), the seven
    ``DPPrior`` scalars and the eight ``DPSettings`` knobs.  :func:`load_chain`
    therefore needs no side information and no pickle.

    ``t_grid`` is a ``Fit`` field rather than a ``DPChains`` field (plan §5.1),
    so it is explicit keyword-only input; ``save_fit`` passes the fit's grid.
    ``label`` and ``chain_index`` are optional provenance only -- they are not
    part of ``DPChains`` and are recorded so a bare chain file can be traced
    back to its fit and chain number.

    Parameters
    ----------
    path:
        Destination ending in ``.npz``.  The write is atomic (see
        :func:`atomic_savez`).
    chain:
        The chain to store.  It is not mutated.
    t_grid:
        Original-scale evaluation grid of the fit, 1-D, finite and strictly
        positive.
    label:
        The fit's un-slugged prior label, e.g. ``"nu ~ Gamma(2, 0.9)"``.
    chain_index:
        Index of this chain within its fit (``0``-based), or ``None`` when the
        caller has no fit context.

    Returns
    -------
    pathlib.Path
        The path actually written.

    Raises
    ------
    ArtifactError
        If ``chain`` is not a :class:`DPChains`, ``t_grid`` is not a valid
        original-scale grid, ``label``/``chain_index`` have the wrong type, or
        ``path`` does not end in ``.npz``.
    """
    if not isinstance(chain, DPChains):
        raise ArtifactError(
            f"save_chain expects a DPChains, got {type(chain).__name__}"
        )
    grid = _validated_t_grid(t_grid)
    if label is not None and not isinstance(label, str):
        raise ArtifactError(
            f"label must be a string or None, got {type(label).__name__}"
        )
    if chain_index is not None:
        if (
            isinstance(chain_index, bool)
            or not isinstance(chain_index, (int, np.integer))
            or int(chain_index) < 0
        ):
            raise ArtifactError(
                f"chain_index must be a non-negative integer or None, "
                f"got {chain_index!r}"
            )
        chain_index = int(chain_index)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "chain_index": chain_index,
        "prior": {
            name: _jsonable(getattr(chain.prior, name)) for name in _PRIOR_FIELDS
        },
        "settings": {
            name: _jsonable(getattr(chain.settings, name)) for name in _SETTINGS_FIELDS
        },
    }
    # Validate the provenance with the same rules the loader applies, so the
    # writer cannot produce an archive that ``load_chain`` would reject.
    try:
        _prior_from_record(meta["prior"], Path(path))
        _settings_from_record(meta["settings"], Path(path))
    except ArtifactIntegrityError as exc:
        raise ArtifactError(
            f"chain metadata is outside the artifact schema: {exc}"
        ) from exc

    arrays = {
        "theta": chain.theta,
        "nu": chain.nu,
        "gamma": chain.gamma,
        "phi": chain.phi,
        "n_star": np.asarray(chain.n_star, dtype=np.int64),
        "log_post": chain.log_post,
        "time": chain.data.time,
        "event": chain.data.event,
        "t_max": np.asarray(chain.data.t_max, dtype=np.float64),
        "t_grid": grid,
        "meta": np.asarray(json.dumps(_jsonable(meta), sort_keys=True)),
    }
    return atomic_savez(path, arrays)


def load_chain(path: str | os.PathLike[str]) -> DPChains:
    """Load a chain written by :func:`save_chain`.

    Every member is checked against the plan §5.1 schema -- exact key set,
    dtype, rank and shape -- **before** any container is built, so a corrupt
    header fails with an :class:`ArtifactIntegrityError` instead of reaching
    ``DPChains.__post_init__`` or allocating a wrongly-shaped trace.  The
    dataset, prior and settings are reconstructed explicitly from their stored
    scalars (no pickle, ``allow_pickle=False``), then ``DPChains`` validates
    its own Section 2.1 invariants.  A container violation is reported as an
    artifact error naming ``path``.

    The provenance fields ``label`` and ``chain_index`` are validated but not
    returned: they are not part of ``DPChains``, and the fit-level loader reads
    them from ``fit-meta.json``.

    Raises
    ------
    ArtifactError
        If ``path`` is missing, is not a file, or is not a readable ``npz``.
    ArtifactIntegrityError
        If the archive disagrees with the declared schema (keys, dtypes,
        shapes, metadata fields, schema version) or the stored arrays violate
        the ``DPChains`` invariants.
    """
    chain, _ = _load_chain_with_grid(path)
    return chain


def _load_chain_with_grid(
    path: str | os.PathLike[str],
) -> tuple[DPChains, np.ndarray]:
    """Load a chain file and return the fit grid stored alongside it.

    ``t_grid`` is a ``Fit`` field, not a ``DPChains`` field, so the fit loader
    needs both halves of the archive.  Keeping one implementation means the
    stand-alone :func:`load_chain` and :func:`load_fit` cannot drift apart in
    what they accept.
    """
    target = Path(path)
    with _open_npz(target) as archive:
        _require_exact_keys(archive, _CHAIN_KEYS, target, kind="chain")
        _, _, prior, settings = _read_chain_meta(archive, target)

        time = _require_array(archive, "time", target, dtype=np.float64, ndim=1)
        if time.size == 0:
            raise ArtifactIntegrityError(f"{target}: array 'time' must not be empty")
        event = _require_array(
            archive,
            "event",
            target,
            dtype=np.bool_,
            ndim=1,
            expected_shape=(time.size,),
        )
        t_max = _require_array(archive, "t_max", target, dtype=np.float64, ndim=0)
        grid = _require_array(archive, "t_grid", target, dtype=np.float64, ndim=1)
        if grid.size == 0:
            raise ArtifactIntegrityError(f"{target}: array 't_grid' must not be empty")
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise ArtifactIntegrityError(
                f"{target}: array 't_grid' must contain only finite, "
                "strictly positive times"
            )

        n_draws = int(settings.n_save)
        theta = _require_array(
            archive,
            "theta",
            target,
            dtype=np.float64,
            ndim=3,
            expected_shape=(n_draws, int(time.size), 2),
        )
        draw_shape = (n_draws,)
        draw_arrays = {
            name: _require_array(
                archive,
                name,
                target,
                dtype=np.float64,
                ndim=1,
                expected_shape=draw_shape,
            )
            for name in _DRAW_FLOAT_FIELDS
        }
        n_star = _require_array(
            archive,
            "n_star",
            target,
            dtype=np.integer,
            ndim=1,
            expected_shape=draw_shape,
        )

        try:
            data = SurvData(
                time=np.array(time, dtype=np.float64, copy=True),
                event=np.array(event, dtype=bool, copy=True),
                t_max=float(t_max[()]),
            )
            chain = DPChains(
                data=data,
                prior=prior,
                settings=settings,
                theta=theta,
                nu=draw_arrays["nu"],
                gamma=draw_arrays["gamma"],
                phi=draw_arrays["phi"],
                n_star=n_star,
                log_post=draw_arrays["log_post"],
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"{target}: stored chain violates the DPChains contract: {exc}"
            ) from exc
    return chain, np.array(grid, dtype=np.float64, copy=True)


# ---------------------------------------------------------------------------
# Functional serialisation (plan §5.1)
# ---------------------------------------------------------------------------

#: Keys of one ``functionals.npz`` (plan §5.1).  The same five array keys are
#: used by ``prior-band.npz``: a band is a ``Functionals`` and needs no second
#: schema.
_FUNCTIONALS_KEYS = frozenset({"t_grid", "S", "F", "f", "h", "meta"})

#: Top-level keys of the ``meta`` blob.  The container's own ``meta`` dict is
#: stored *inside* ``meta`` rather than merged with these, so a provenance key
#: named ``schema_version`` cannot shadow the version stamp.
_FUNCTIONALS_META_KEYS = frozenset({"schema_version", "meta"})

#: The four per-draw ``(n_draws, n_grid)`` ``float64`` arrays.
_FUNCTIONAL_ARRAY_FIELDS = ("S", "F", "f", "h")


def _require_jsonable_meta(
    meta: object, target: Path, *, name: str = "functionals meta"
) -> dict[str, Any]:
    """Convert a container's ``meta`` dict to the JSON object that is stored.

    The dict is stored verbatim (plan §5.1): its keys are not enumerated,
    because a posterior functional carries ``method``/``J_common``/``L`` and a
    prior band carries ``sample``/``J_prior``/``max_nu``.  :func:`_jsonable`
    is the only conversion applied, so NumPy scalars become Python scalars and
    a tuple becomes a JSON list; a value with no JSON form is rejected instead
    of being stringified.

    ``name`` labels the field in the error message, because the same helper
    validates ``Functionals.meta``, the two ``fit-meta.json`` meta records and
    the caller-supplied ``extra`` provenance.

    Raises
    ------
    ArtifactIntegrityError
        If ``meta`` is not a mapping or holds a value with no JSON form.
    """
    if not isinstance(meta, Mapping):
        raise ArtifactIntegrityError(
            f"{target}: {name} must be a JSON object, got {type(meta).__name__}"
        )
    try:
        converted = _jsonable(dict(meta))
    except ArtifactError as exc:
        raise ArtifactIntegrityError(
            f"{target}: {name} is not JSON-serialisable: {exc}"
        ) from exc
    if not isinstance(converted, dict):
        raise ArtifactIntegrityError(
            f"{target}: {name} must be a JSON object, got {type(converted).__name__}"
        )
    return converted


def _check_functional_arrays(
    arrays: Mapping[str, np.ndarray], grid: np.ndarray, target: Path
) -> tuple[int, int]:
    """Validate the dtype/shape contract the loader and writer share.

    Every one of ``S``, ``F``, ``f`` and ``h`` must be a ``float64`` matrix of
    the same ``(n_draws, n_grid)`` shape, with ``n_grid == t_grid.size``.  The
    check is made against the archive headers *before* the container is built,
    so a transposed ``S`` fails here rather than inside ``Functionals``, and it
    is run by the writer too, so the two directions cannot drift apart.

    Returns
    -------
    tuple of int
        ``(n_draws, n_grid)``.

    Raises
    ------
    ArtifactIntegrityError
        If a member's dtype or shape disagrees with the schema.
    """
    for name in _FUNCTIONAL_ARRAY_FIELDS:
        array = arrays[name]
        if array.dtype != np.dtype(np.float64):
            raise ArtifactIntegrityError(
                f"{target}: array {name!r} has dtype {array.dtype}, expected float64"
            )
        if array.ndim != 2:
            raise ArtifactIntegrityError(
                f"{target}: array {name!r} has shape {array.shape}, "
                f"expected a 2-D (n_draws, n_grid) array"
            )
    n_draws = int(arrays["S"].shape[0])
    if n_draws < 1:
        raise ArtifactIntegrityError(
            f"{target}: functional arrays must hold at least one draw, "
            f"got shape {arrays['S'].shape}"
        )
    expected = (n_draws, int(grid.size))
    for name in _FUNCTIONAL_ARRAY_FIELDS:
        if arrays[name].shape != expected:
            raise ArtifactIntegrityError(
                f"{target}: array {name!r} has shape {arrays[name].shape}, "
                f"expected {expected} (one column per t_grid point)"
            )
    return n_draws, int(grid.size)


def _read_functionals_meta(
    archive: np.lib.npyio.NpzFile, target: Path
) -> dict[str, Any]:
    """Parse the ``meta`` blob and return the stored provenance dict."""
    text = _require_string_scalar(archive, "meta", target)
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"{target}: meta blob is not valid JSON: {exc}"
        ) from exc
    record = _require_record(record, _FUNCTIONALS_META_KEYS, "meta", target)

    version = _as_meta_int(
        record["schema_version"], "schema_version", target, minimum=1
    )
    if version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"{target}: schema version {version} is not the supported "
            f"version {SCHEMA_VERSION} (no migration path)"
        )
    return _require_jsonable_meta(record["meta"], target)


def save_functionals(path: str | os.PathLike[str], func: Functionals) -> Path:
    """Write one :class:`Functionals` as a ``functionals.npz`` archive.

    The archive holds the evaluation grid, the four ``(n_draws, n_grid)``
    arrays and the provenance dict, which is stored **verbatim** through
    :func:`_jsonable` inside a ``{"schema_version", "meta"}`` record.  The same
    writer stores a §5.1 prior band (``prior-band.npz``): a band is a
    ``Functionals`` like any other, so it needs no second schema and its
    ``sample``/``J_prior``/``max_nu`` keys survive without a hand-written
    field list.

    The write is atomic (see :func:`atomic_savez`) and the input is not
    mutated: the arrays are written from the object that owns them, without an
    up-front copy.

    Parameters
    ----------
    path:
        Destination ending in ``.npz``.
    func:
        The functional arrays to store.

    Returns
    -------
    pathlib.Path
        The path actually written.

    Raises
    ------
    ArtifactError
        If ``func`` is not a :class:`Functionals`, its grid or arrays are
        outside the declared schema, its ``meta`` has no JSON form, or
        ``path`` does not end in ``.npz``.
    """
    if not isinstance(func, Functionals):
        raise ArtifactError(
            f"save_functionals expects a Functionals, got {type(func).__name__}"
        )
    target = Path(path)
    grid = _validated_t_grid(func.t_grid)
    arrays = {
        "t_grid": grid,
        "S": func.S,
        "F": func.F,
        "f": func.f,
        "h": func.h,
    }
    try:
        _check_functional_arrays(arrays, grid, target)
        inner_meta = _require_jsonable_meta(func.meta, target)
    except ArtifactIntegrityError as exc:
        raise ArtifactError(
            f"functionals are outside the artifact schema: {exc}"
        ) from exc

    meta = {"schema_version": SCHEMA_VERSION, "meta": inner_meta}
    return atomic_savez(
        target, {**arrays, "meta": np.asarray(json.dumps(meta, sort_keys=True))}
    )


def load_functionals(path: str | os.PathLike[str]) -> Functionals:
    """Load a functional archive written by :func:`save_functionals`.

    Every member is checked against the plan §5.1 schema -- exact key set,
    dtype, rank and shape -- **before** the container is built, so a corrupt
    header fails with an :class:`ArtifactIntegrityError` instead of reaching
    ``Functionals.__post_init__``.  ``S``, ``F``, ``f`` and ``h`` must share
    one ``(n_draws, n_grid)`` shape and ``n_grid`` must equal ``t_grid.size``;
    that is what rejects the transposed-``S`` failure mode the plan calls out.

    The provenance dict is returned exactly as stored, so a posterior
    functional keeps ``method``/``J_common``/``chain_seeds`` and a prior band
    keeps ``sample``/``J_prior``/``max_nu``.  Values that were tuples in the
    original dict are JSON lists, which is the documented consequence of
    storing the dict verbatim.

    Raises
    ------
    ArtifactError
        If ``path`` is missing, is not a file, or is not a readable ``npz``.
    ArtifactIntegrityError
        If the archive disagrees with the declared schema (keys, dtypes,
        shapes, metadata fields, schema version) or the stored arrays violate
        the ``Functionals`` contract.
    """
    target = Path(path)
    with _open_npz(target) as archive:
        _require_exact_keys(archive, _FUNCTIONALS_KEYS, target, kind="functionals")
        meta = _read_functionals_meta(archive, target)

        grid = _require_array(archive, "t_grid", target, dtype=np.float64, ndim=1)
        if grid.size == 0:
            raise ArtifactIntegrityError(f"{target}: array 't_grid' must not be empty")
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise ArtifactIntegrityError(
                f"{target}: array 't_grid' must contain only finite, "
                "strictly positive times"
            )
        arrays = {
            name: _member(archive, name, target) for name in _FUNCTIONAL_ARRAY_FIELDS
        }
        _check_functional_arrays(arrays, grid, target)

        try:
            return Functionals(
                t_grid=np.array(grid, dtype=np.float64, copy=True),
                S=arrays["S"],
                F=arrays["F"],
                f=arrays["f"],
                h=arrays["h"],
                meta=meta,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"{target}: stored functionals violate the Functionals contract: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Fit serialisation (plan §5.1, §6.1)
# ---------------------------------------------------------------------------

#: File names of one per-fit artifact directory (plan §5).
_FIT_META_NAME = "fit-meta.json"
_FUNCTIONALS_NAME = "functionals.npz"
_PRIOR_BAND_NAME = "prior-band.npz"
_CHAIN_NAME = "chain-{index}.npz"

#: Section-level file names (plan §5): the section ``meta.json`` and the
#: directory holding one fit directory per fit.
_DEMO_META_NAME = "meta.json"
_DRAWS_DIR = "draws"

#: Keys of one ``fit-meta.json``.  The shape/dtype records and the two meta
#: dicts are the plan §5.1 field list; ``extra`` holds caller provenance
#: (timings, chain seeds) verbatim, so a new provenance key never forces a
#: schema bump.
_FIT_META_KEYS = frozenset(
    {
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
)

#: One archive record inside ``fit-meta.json``: a file name, the array dtype
#: and the shape.  The dtype is recorded rather than inferred so a loader can
#: report a disagreement between an archive and its metadata in the schema's
#: own terms.
_FIT_ARRAY_KEYS = frozenset({"path", "dtype", "shape"})

#: ``t_grid`` record: point count and both endpoints.  The full grid is
#: rebuilt from the archives and checked against these three numbers, so
#: ``fit-meta.json`` stays readable instead of holding 300 floats.
_FIT_GRID_KEYS = frozenset({"size", "first", "last"})

#: ``save_fit`` metadata keys that are lifted into the fixed part of
#: ``fit-meta.json``; every other key is stored under ``extra``.
_FIT_META_RESERVED = frozenset({"arm", "data_seed", "shared_t_max"})

#: Keys of the section ``meta.json``.  ``fit_keys`` records the driver's fit
#: order, which a JSON object of fits would lose.
_DEMO_META_KEYS = frozenset({"schema_version", "label", "fit_keys", "meta"})

#: ``"A: nu ~ Gamma(2, 0.9)"`` -> arm ``"A"``, label ``"nu ~ Gamma(2, 0.9)"``.
_ARM_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9]*): (.+)$")


@dataclass(frozen=True)
class _FitMetadata:
    """Validated contents of one ``fit-meta.json`` (internal)."""

    directory: Path
    path: Path
    label: str
    arm: str | None
    prior: DPPrior
    settings: DPSettings
    data_seed: int | None
    shared_t_max: float | None
    n_chains: int
    n_draws: int
    n_grid: int
    grid_first: float
    grid_last: float
    chains: tuple[tuple[str, tuple[int, ...]], ...]
    functionals: tuple[str, tuple[int, ...]]
    prior_band: tuple[str, tuple[int, ...]] | None
    functionals_meta: dict[str, Any]
    prior_band_meta: dict[str, Any] | None
    extra: dict[str, Any]


def _write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write one JSON artifact atomically.

    Same contract as :func:`atomic_savez`: write ``<path>.tmp``, flush,
    ``fsync``, close, then :func:`os.replace`.  A crash leaves either the old
    file or the new one, never a half-written ``fit-meta.json`` that a loader
    would have to interpret.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _read_json_object(path: str | os.PathLike[str], *, kind: str) -> Any:
    """Read a JSON file, naming it in every failure mode."""
    target = Path(path)
    if not target.exists():
        raise ArtifactError(f"artifact does not exist: {target}")
    if not target.is_file():
        raise ArtifactError(f"artifact is not a file: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactIntegrityError(f"{target}: cannot read {kind}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"{target}: {kind} is not valid JSON: {exc}"
        ) from exc


def _as_optional_int(
    value: Any, name: str, target: Path, *, minimum: int = 0
) -> int | None:
    """Validate an integer metadata value that may be ``null``."""
    if value is None:
        return None
    return _as_meta_int(value, name, target, minimum=minimum)


def _as_optional_float(value: Any, name: str, target: Path) -> float | None:
    """Validate a finite numeric metadata value that may be ``null``."""
    if value is None:
        return None
    return float(_as_meta_number(value, name, target))


def _as_label(value: Any, name: str, target: Path) -> str:
    """Validate the un-slugged label (the key ``DemoResult.fits`` uses)."""
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be a non-empty string, got {value!r}"
        )
    return value


def _as_arm(value: Any, name: str, target: Path) -> str | None:
    """Validate the optional arm prefix (``"A"``/``"B"`` in §5.2)."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} must be a non-empty string or null, "
            f"got {value!r}"
        )
    return value


def _member_path(directory: Path, name: Any, *, field: str, target: Path) -> Path:
    """Resolve a recorded file name inside ``directory``.

    Only a plain file name is accepted: an absolute path, a parent reference or
    a nested path would let a tampered ``fit-meta.json`` point a loader outside
    the fit directory it belongs to.
    """
    if not isinstance(name, str) or not name:
        raise ArtifactIntegrityError(
            f"{target}: meta field {field!r} must record a non-empty file name, "
            f"got {name!r}"
        )
    candidate = Path(name)
    if candidate.name != name or name in {".", ".."}:
        raise ArtifactIntegrityError(
            f"{target}: meta field {field!r} must name a file inside the fit "
            f"directory, got {name!r}"
        )
    return directory / name


def _array_record(
    value: Any, name: str, target: Path, *, ndim: int
) -> tuple[str, tuple[int, ...]]:
    """Validate one ``{"path", "dtype", "shape"}`` archive record."""
    record = _require_record(value, _FIT_ARRAY_KEYS, name, target)
    path = _member_path(Path("."), record["path"], field=f"{name}.path", target=target)
    dtype = record["dtype"]
    if dtype != "float64":
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} records dtype {dtype!r}, expected 'float64'"
        )
    shape = record["shape"]
    if not isinstance(shape, list) or len(shape) != ndim:
        raise ArtifactIntegrityError(
            f"{target}: meta field {name!r} shape must be a list of {ndim} "
            f"integers, got {shape!r}"
        )
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ArtifactIntegrityError(
                f"{target}: meta field {name!r} shape must hold positive "
                f"integers, got {shape!r}"
            )
        dims.append(int(dim))
    return path.name, tuple(dims)


def _grid_record(value: Any, target: Path) -> tuple[int, float, float]:
    """Validate the readable ``t_grid`` summary stored in ``fit-meta.json``."""
    record = _require_record(value, _FIT_GRID_KEYS, "t_grid", target)
    size = _as_meta_int(record["size"], "t_grid.size", target, minimum=1)
    first = float(_as_meta_number(record["first"], "t_grid.first", target))
    last = float(_as_meta_number(record["last"], "t_grid.last", target))
    if first <= 0.0 or last <= 0.0:
        raise ArtifactIntegrityError(
            f"{target}: meta field 't_grid' endpoints must be strictly positive, "
            f"got {first!r} and {last!r}"
        )
    return size, first, last


def _chain_records(
    value: Any, *, n_chains: int, n_save: int, target: Path
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Validate the per-chain archive records of one fit."""
    if not isinstance(value, list):
        raise ArtifactIntegrityError(
            f"{target}: meta field 'chains' must be a list, got {type(value).__name__}"
        )
    if len(value) != n_chains:
        raise ArtifactIntegrityError(
            f"{target}: meta field 'chains' lists {len(value)} files, "
            f"expected {n_chains}"
        )
    records: list[tuple[str, tuple[int, ...]]] = []
    for index, item in enumerate(value):
        name, shape = _array_record(item, f"chains[{index}]", target, ndim=3)
        if shape[0] != n_save or shape[2] != 2:
            raise ArtifactIntegrityError(
                f"{target}: meta field 'chains[{index}]' shape {shape} "
                f"disagrees with (n_save, n, 2) = ({n_save}, n, 2)"
            )
        if records and shape[1] != records[0][1][1]:
            raise ArtifactIntegrityError(
                f"{target}: meta field 'chains[{index}]' shape {shape} has a "
                f"different observation count than chains[0] "
                f"({records[0][1][0]} rows)"
            )
        records.append((name, shape))
    return tuple(records)


def _meta_digest(value: Any) -> str:
    """Canonical comparison form of a metadata dict.

    JSON turns a tuple into a list, so comparing the reconstructed dict with
    the recorded one must compare their JSON forms; the sorted dump also makes
    a non-finite float (``NaN != NaN``) compare equal to itself, which keeps
    the verbatim round-trip check honest instead of accidentally strict.
    """
    try:
        return json.dumps(_jsonable(value), sort_keys=True)
    except ArtifactError:  # pragma: no cover - validated before comparison
        return repr(value)


def _same_dataset(left: SurvData, right: SurvData) -> bool:
    """True when two chains were fitted to the same stored dataset."""
    return (
        float(left.t_max) == float(right.t_max)
        and np.array_equal(left.time, right.time)
        and np.array_equal(left.event, right.event)
    )


def _split_fit_key(key: str) -> tuple[str | None, str]:
    """Split a ``DemoResult.fits`` key into ``(arm, label)``.

    ``"nu ~ Gamma(2, 0.9)"`` has no arm (§5.1); ``"A: nu ~ Gamma(2, 0.9)"``
    does (§5.2).  The label half must equal ``Fit.label``, which is what keeps
    the un-slugged label recoverable from ``fit-meta.json``.
    """
    match = _ARM_KEY.match(key)
    if match is None:
        return None, key
    return match.group(1), match.group(2)


def _fit_directory_name(arm: str | None, label: str) -> str:
    """The plan §5 fit directory name: ``A-<slug>`` or just ``<slug>``."""
    return f"{arm}-{slug(label)}" if arm else slug(label)


def _fit_provenance(
    meta: Mapping[str, Any] | None, target: Path
) -> tuple[str | None, int | None, float | None, dict[str, Any]]:
    """Split ``save_fit``'s metadata into its typed keys and ``extra``."""
    if meta is None:
        meta = {}
    if not isinstance(meta, Mapping):
        raise ArtifactError(f"fit meta must be a mapping, got {type(meta).__name__}")
    arm = meta.get("arm")
    if arm is not None and (not isinstance(arm, str) or not arm):
        raise ArtifactError(
            f"fit meta 'arm' must be a non-empty string or None, got {arm!r}"
        )
    raw_seed = meta.get("data_seed")
    data_seed: int | None = None
    if raw_seed is not None:
        if (
            isinstance(raw_seed, bool)
            or not isinstance(raw_seed, (int, np.integer))
            or int(raw_seed) < 0
        ):
            raise ArtifactError(
                f"fit meta 'data_seed' must be a non-negative integer or None, "
                f"got {raw_seed!r}"
            )
        data_seed = int(raw_seed)
    shared_t_max = meta.get("shared_t_max")
    resolved_shared: float | None = None
    if shared_t_max is not None:
        try:
            shared_value = float(shared_t_max)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"fit meta 'shared_t_max' must be a number or None, "
                f"got {shared_t_max!r}"
            ) from exc
        if not np.isfinite(shared_value) or shared_value <= 0.0:
            raise ArtifactError(
                f"fit meta 'shared_t_max' must be finite and positive, "
                f"got {shared_t_max!r}"
            )
        resolved_shared = shared_value
    extra = {key: value for key, value in meta.items() if key not in _FIT_META_RESERVED}
    try:
        converted = _jsonable(extra)
    except ArtifactError as exc:
        raise ArtifactError(f"fit meta is not JSON-serialisable: {exc}") from exc
    if not isinstance(converted, dict):  # pragma: no cover - extra is a dict
        raise ArtifactError("fit meta 'extra' must be a JSON object")
    return arm, data_seed, resolved_shared, converted


def save_fit(
    directory: str | os.PathLike[str],
    fit: Fit,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write one :class:`Fit` into a self-contained per-fit directory.

    Layout (plan §5)::

        <directory>/
          chain-0.npz ... chain-<B-1>.npz
          functionals.npz
          prior-band.npz        # only when fit.prior_band is set
          fit-meta.json         # written last

    ``fit-meta.json`` records the **un-slugged** ``fit.label`` (the directory
    name is only a slug), the ``arm``, the seven prior scalars, the eight
    settings knobs, the dataset seed and shared ``t_max``, the readable
    ``t_grid`` summary, the archive dtype/shape records, and the posterior and
    prior-band meta dicts **verbatim**.  ``meta`` supplies the provenance the
    fit itself does not hold (``arm``, ``data_seed``, ``shared_t_max``) and any
    other JSON values (timings, chain seeds), which are stored under
    ``extra``.

    The metadata file is written last and atomically: a crashed save leaves
    chain/functional archives without a ``fit-meta.json``, so :func:`load_fit`
    rejects the directory instead of reconstructing a half-written fit.  A
    re-save removes a stale ``prior-band.npz`` and stale ``chain-*.npz`` files
    from an earlier, longer fit so the directory never holds unlisted arrays.

    Parameters
    ----------
    directory:
        Destination fit directory.  Missing parents are created.
    fit:
        The fit to store.  It is not mutated.
    meta:
        Provenance mapping.  ``arm`` is a non-empty string or ``None``,
        ``data_seed`` a non-negative integer or ``None``, ``shared_t_max`` a
        positive finite number or ``None`` (defaults to ``fit.data.t_max``).

    Returns
    -------
    pathlib.Path
        The directory actually written.

    Raises
    ------
    ArtifactError
        If ``fit`` is not a :class:`Fit`, its chains, dataset, prior, settings,
        grid or meta dicts disagree with each other, a chain's ``theta`` is not
        ``float64``, or ``meta`` holds a value with no JSON form.
    """
    from src.paper_examples import Fit as FitType

    if not isinstance(fit, FitType):
        raise ArtifactError(f"save_fit expects a Fit, got {type(fit).__name__}")
    target = Path(directory)
    arm, data_seed, shared_t_max, extra = _fit_provenance(meta, target)
    if not isinstance(fit.label, str) or not fit.label:
        raise ArtifactError(f"fit label must be a non-empty string, got {fit.label!r}")
    label = fit.label
    grid = _validated_t_grid(fit.t_grid)
    if not isinstance(fit.prior, DPPrior):
        raise ArtifactError(
            f"fit.prior must be a DPPrior, got {type(fit.prior).__name__}"
        )
    if not isinstance(fit.data, SurvData):
        raise ArtifactError(
            f"fit.data must be a SurvData, got {type(fit.data).__name__}"
        )
    if not fit.chains:
        raise ArtifactError("a fit must hold at least one chain")
    first = fit.chains[0]
    if not isinstance(first, DPChains):
        raise ArtifactError(f"fit chains must be DPChains, got {type(first).__name__}")
    settings = first.settings
    if settings.n_chains != len(fit.chains):
        raise ArtifactError(
            f"fit holds {len(fit.chains)} chains but its settings declare "
            f"{settings.n_chains}"
        )
    for index, chain in enumerate(fit.chains):
        if not isinstance(chain, DPChains):
            raise ArtifactError(
                f"fit chains must be DPChains, got {type(chain).__name__}"
            )
        if chain.prior != fit.prior:
            raise ArtifactError(f"chain {index} prior disagrees with fit.prior")
        if chain.settings != settings:
            raise ArtifactError(
                f"chain {index} settings disagree with chain 0's settings"
            )
        if not _same_dataset(chain.data, fit.data):
            raise ArtifactError(f"chain {index} dataset disagrees with fit.data")
        if chain.theta.dtype != np.dtype(np.float64):
            raise ArtifactError(
                f"chain {index} theta has dtype {chain.theta.dtype}, expected float64"
            )
    n_draws = int(settings.n_save) * len(fit.chains)

    functionals = fit.functionals
    if not isinstance(functionals, Functionals):
        raise ArtifactError(
            f"fit.functionals must be a Functionals, got {type(functionals).__name__}"
        )
    if not np.array_equal(functionals.t_grid, grid):
        raise ArtifactError("fit.functionals.t_grid disagrees with fit.t_grid")
    if functionals.n_draws != n_draws:
        raise ArtifactError(
            f"fit.functionals holds {functionals.n_draws} draws, expected "
            f"{n_draws} (n_chains * n_save)"
        )
    try:
        _check_functional_arrays(
            {name: getattr(functionals, name) for name in _FUNCTIONAL_ARRAY_FIELDS},
            grid,
            target,
        )
        functionals_meta = _require_jsonable_meta(functionals.meta, target)
    except ArtifactIntegrityError as exc:
        raise ArtifactError(
            f"functionals are outside the artifact schema: {exc}"
        ) from exc

    band = fit.prior_band
    band_record: dict[str, Any] | None = None
    band_meta: dict[str, Any] | None = None
    if band is not None:
        if not isinstance(band, Functionals):
            raise ArtifactError(
                f"fit.prior_band must be a Functionals or None, "
                f"got {type(band).__name__}"
            )
        if not np.array_equal(band.t_grid, grid):
            raise ArtifactError("fit.prior_band.t_grid disagrees with fit.t_grid")
        try:
            _check_functional_arrays(
                {name: getattr(band, name) for name in _FUNCTIONAL_ARRAY_FIELDS},
                grid,
                target,
            )
            band_meta = _require_jsonable_meta(band.meta, target)
        except ArtifactIntegrityError as exc:
            raise ArtifactError(
                f"prior band is outside the artifact schema: {exc}"
            ) from exc

    if shared_t_max is None:
        shared_t_max = float(fit.data.t_max)

    target.mkdir(parents=True, exist_ok=True)
    # A re-save into a longer fit's directory must not leave unlisted arrays
    # behind: the manifest hashes every file it finds, so a stale chain or band
    # would look like part of the fit.
    for stale in target.glob("chain-*.npz"):
        match = re.fullmatch(r"chain-(\d+)\.npz", stale.name)
        if match is None or int(match.group(1)) >= len(fit.chains):
            stale.unlink()
    if band is None:
        (target / _PRIOR_BAND_NAME).unlink(missing_ok=True)

    chain_records: list[dict[str, Any]] = []
    for index, chain in enumerate(fit.chains):
        name = _CHAIN_NAME.format(index=index)
        save_chain(
            target / name,
            chain,
            t_grid=grid,
            label=label,
            chain_index=index,
        )
        chain_records.append(
            {
                "path": name,
                "dtype": "float64",
                "shape": [int(dim) for dim in chain.theta.shape],
            }
        )

    save_functionals(target / _FUNCTIONALS_NAME, functionals)
    if band is not None:
        save_functionals(target / _PRIOR_BAND_NAME, band)
        band_record = {
            "path": _PRIOR_BAND_NAME,
            "dtype": "float64",
            "shape": [int(dim) for dim in band.S.shape],
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "arm": arm,
        "prior": {name: _jsonable(getattr(fit.prior, name)) for name in _PRIOR_FIELDS},
        "settings": {
            name: _jsonable(getattr(settings, name)) for name in _SETTINGS_FIELDS
        },
        "data_seed": data_seed,
        "shared_t_max": shared_t_max,
        "n_chains": len(fit.chains),
        "n_draws": n_draws,
        "t_grid": {
            "size": int(grid.size),
            "first": float(grid[0]),
            "last": float(grid[-1]),
        },
        "chains": chain_records,
        "functionals": {
            "path": _FUNCTIONALS_NAME,
            "dtype": "float64",
            "shape": [n_draws, int(grid.size)],
        },
        "prior_band": band_record,
        "functionals_meta": functionals_meta,
        "prior_band_meta": band_meta,
        "extra": extra,
    }
    _write_json(target / _FIT_META_NAME, payload)
    return target


def _read_fit_meta(directory: str | os.PathLike[str]) -> _FitMetadata:
    """Read and validate ``fit-meta.json`` without loading any array."""
    target = Path(directory)
    if not target.exists():
        raise ArtifactError(f"artifact does not exist: {target}")
    if not target.is_dir():
        raise ArtifactError(f"artifact is not a directory: {target}")
    path = target / _FIT_META_NAME
    record = _read_json_object(path, kind="fit metadata")
    record = _require_record(record, _FIT_META_KEYS, "fit metadata", path)

    version = _as_meta_int(record["schema_version"], "schema_version", path, minimum=1)
    if version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"{path}: schema version {version} is not the supported "
            f"version {SCHEMA_VERSION} (no migration path)"
        )
    label = _as_label(record["label"], "label", path)
    arm = _as_arm(record["arm"], "arm", path)
    prior = _prior_from_record(record["prior"], path)
    settings = _settings_from_record(record["settings"], path)
    data_seed = _as_optional_int(record["data_seed"], "data_seed", path)
    shared_t_max = _as_optional_float(record["shared_t_max"], "shared_t_max", path)
    if shared_t_max is not None and shared_t_max <= 0.0:
        raise ArtifactIntegrityError(
            f"{path}: meta field 'shared_t_max' must be strictly positive, "
            f"got {shared_t_max!r}"
        )
    n_chains = _as_meta_int(record["n_chains"], "n_chains", path, minimum=1)
    n_draws = _as_meta_int(record["n_draws"], "n_draws", path, minimum=1)
    if n_chains != settings.n_chains:
        raise ArtifactIntegrityError(
            f"{path}: meta field 'n_chains' ({n_chains}) disagrees with "
            f"settings.n_chains ({settings.n_chains})"
        )
    expected_draws = int(settings.n_chains) * int(settings.n_save)
    if n_draws != expected_draws:
        raise ArtifactIntegrityError(
            f"{path}: meta field 'n_draws' ({n_draws}) is not "
            f"n_chains * n_save ({expected_draws})"
        )
    n_grid, grid_first, grid_last = _grid_record(record["t_grid"], path)
    chains = _chain_records(
        record["chains"],
        n_chains=n_chains,
        n_save=int(settings.n_save),
        target=path,
    )
    func_name, func_shape = _array_record(
        record["functionals"], "functionals", path, ndim=2
    )
    if func_shape != (n_draws, n_grid):
        raise ArtifactIntegrityError(
            f"{path}: meta field 'functionals' shape {func_shape} disagrees "
            f"with (n_draws, n_grid) = {(n_draws, n_grid)}"
        )

    raw_band = record["prior_band"]
    raw_band_meta = record["prior_band_meta"]
    if raw_band is None:
        if raw_band_meta is not None:
            raise ArtifactIntegrityError(
                f"{path}: meta field 'prior_band_meta' is set but 'prior_band' is null"
            )
        band: tuple[str, tuple[int, ...]] | None = None
        band_meta: dict[str, Any] | None = None
    else:
        if raw_band_meta is None:
            raise ArtifactIntegrityError(
                f"{path}: meta field 'prior_band' is set but 'prior_band_meta' is null"
            )
        band_name, band_shape = _array_record(raw_band, "prior_band", path, ndim=2)
        if band_shape[1] != n_grid:
            raise ArtifactIntegrityError(
                f"{path}: meta field 'prior_band' shape {band_shape} disagrees "
                f"with the fit's {n_grid}-point t_grid"
            )
        band = (band_name, band_shape)
        band_meta = _require_jsonable_meta(
            raw_band_meta, path, name="fit metadata 'prior_band_meta'"
        )

    return _FitMetadata(
        directory=target,
        path=path,
        label=label,
        arm=arm,
        prior=prior,
        settings=settings,
        data_seed=data_seed,
        shared_t_max=shared_t_max,
        n_chains=n_chains,
        n_draws=n_draws,
        n_grid=n_grid,
        grid_first=grid_first,
        grid_last=grid_last,
        chains=chains,
        functionals=(func_name, func_shape),
        prior_band=band,
        functionals_meta=_require_jsonable_meta(
            record["functionals_meta"], path, name="fit metadata 'functionals_meta'"
        ),
        prior_band_meta=band_meta,
        extra=_require_jsonable_meta(
            record["extra"], path, name="fit metadata 'extra'"
        ),
    )


def _load_fit_from_meta(meta: _FitMetadata) -> Fit:
    """Load the archives of an already-validated fit directory."""
    from src.paper_examples import Fit as FitType

    target = meta.directory
    chains: list[DPChains] = []
    grid: np.ndarray | None = None
    data: SurvData | None = None
    for index, (name, shape) in enumerate(meta.chains):
        path = _member_path(target, name, field=f"chains[{index}]", target=meta.path)
        chain, chain_grid = _load_chain_with_grid(path)
        if list(chain.theta.shape) != list(shape):
            raise ArtifactIntegrityError(
                f"{path}: theta shape {chain.theta.shape} disagrees with the "
                f"fit metadata shape {shape}"
            )
        if chain.prior != meta.prior:
            raise ArtifactIntegrityError(
                f"{path}: stored prior disagrees with fit-meta.json"
            )
        if chain.settings != meta.settings:
            raise ArtifactIntegrityError(
                f"{path}: stored settings disagree with fit-meta.json"
            )
        if grid is None:
            grid = chain_grid
            data = chain.data
        else:
            if not np.array_equal(chain_grid, grid):
                raise ArtifactIntegrityError(f"{path}: t_grid disagrees with chains[0]")
            if not _same_dataset(chain.data, data):  # type: ignore[arg-type]
                raise ArtifactIntegrityError(
                    f"{path}: dataset disagrees with chains[0]"
                )
        chains.append(chain)
    if grid is None or data is None:  # pragma: no cover - n_chains >= 1
        raise ArtifactIntegrityError(f"{meta.path}: fit records no chains")
    if (
        int(grid.size) != meta.n_grid
        or float(grid[0]) != meta.grid_first
        or float(grid[-1]) != meta.grid_last
    ):
        raise ArtifactIntegrityError(
            f"{target}: stored t_grid ({grid.size} points, "
            f"[{float(grid[0])!r}, {float(grid[-1])!r}]) disagrees with "
            f"fit-meta.json ({meta.n_grid} points, "
            f"[{meta.grid_first!r}, {meta.grid_last!r}])"
        )

    func_path = _member_path(
        target, meta.functionals[0], field="functionals", target=meta.path
    )
    functionals = load_functionals(func_path)
    if functionals.S.shape != meta.functionals[1]:
        raise ArtifactIntegrityError(
            f"{func_path}: shape {functionals.S.shape} disagrees with the fit "
            f"metadata shape {meta.functionals[1]}"
        )
    if not np.array_equal(functionals.t_grid, grid):
        raise ArtifactIntegrityError(
            f"{func_path}: t_grid disagrees with the chains' t_grid"
        )
    if _meta_digest(functionals.meta) != _meta_digest(meta.functionals_meta):
        raise ArtifactIntegrityError(
            f"{func_path}: metadata disagrees with fit-meta.json 'functionals_meta'"
        )

    band: Functionals | None = None
    if meta.prior_band is not None:
        band_path = _member_path(
            target, meta.prior_band[0], field="prior_band", target=meta.path
        )
        band = load_functionals(band_path)
        if band.S.shape != meta.prior_band[1]:
            raise ArtifactIntegrityError(
                f"{band_path}: shape {band.S.shape} disagrees with the fit "
                f"metadata shape {meta.prior_band[1]}"
            )
        if not np.array_equal(band.t_grid, grid):
            raise ArtifactIntegrityError(
                f"{band_path}: t_grid disagrees with the chains' t_grid"
            )
        if _meta_digest(band.meta) != _meta_digest(meta.prior_band_meta):
            raise ArtifactIntegrityError(
                f"{band_path}: metadata disagrees with fit-meta.json 'prior_band_meta'"
            )
    else:
        stray = target / _PRIOR_BAND_NAME
        if stray.exists():
            raise ArtifactIntegrityError(
                f"{stray}: unlisted prior band (fit-meta.json records none)"
            )

    return FitType(
        label=meta.label,
        data=data,
        prior=meta.prior,
        t_grid=functionals.t_grid,
        chains=chains,
        functionals=functionals,
        prior_band=band,
    )


def load_fit(directory: str | os.PathLike[str]) -> Fit:
    """Load a fit directory written by :func:`save_fit`.

    ``fit-meta.json`` is validated first -- schema version, every metadata
    field, the recorded array shapes/dtypes against the settings, and the
    recorded ``t_grid`` summary.  Then each archive is checked against its
    record **and** its schema: chains must agree with the fit's prior, settings
    and dataset, every chain must share one ``t_grid``, and the functional and
    band arrays must hold one column per grid point.  A discrepancy between an
    archive and its metadata is an :class:`ArtifactIntegrityError` naming the
    file, never a silently reconstructed fit.

    The prior band is attached only when ``fit-meta.json`` records one and the
    band file is present; a recorded-but-missing band and an unlisted band file
    are both errors.

    Raises
    ------
    ArtifactError
        If ``directory`` is missing, is not a directory, or has no
        ``fit-meta.json``.
    ArtifactIntegrityError
        If the metadata or any archive disagrees with the declared schema.
    """
    return _load_fit_from_meta(_read_fit_meta(directory))


# ---------------------------------------------------------------------------
# Demo-result serialisation (plan §5, §6.1)
# ---------------------------------------------------------------------------


def _restore_demo_meta(meta: dict[str, Any], target: Path) -> dict[str, Any]:
    """Rebuild ``DemoResult.meta`` with the types the driver produced.

    JSON has no tuples and stores every number as a float or int, so the
    fields the paper preconditions compare are restored explicitly:
    ``prior_indices`` becomes a tuple of ints and the ``t_max`` values become
    floats.  ``seed``/``data_seed`` stay int-or-``None`` so a run started from
    the default seed is not turned into ``0``.
    """
    restored = dict(meta)
    if "prior_indices" in restored:
        value = restored["prior_indices"]
        if not isinstance(value, list):
            raise ArtifactIntegrityError(
                f"{target}: meta field 'prior_indices' must be a list, got {value!r}"
            )
        indices: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ArtifactIntegrityError(
                    f"{target}: meta field 'prior_indices' must hold "
                    f"non-negative integers, got {value!r}"
                )
            indices.append(int(item))
        restored["prior_indices"] = tuple(indices)
    for name in ("t_max", "t_max_a", "t_max_b"):
        if name in restored and restored[name] is not None:
            restored[name] = float(
                _as_meta_number(restored[name], f"meta.{name}", target)
            )
    for name in ("seed", "data_seed"):
        if name in restored:
            restored[name] = _as_optional_int(restored[name], f"meta.{name}", target)
    return restored


def save_demo_result(directory: str | os.PathLike[str], result: DemoResult) -> Path:
    """Write one :class:`DemoResult` as a section directory (plan §5).

    ``<directory>/meta.json`` holds the section label, the fit keys in the
    driver's order and ``result.meta`` verbatim; ``<directory>/draws/`` holds
    one :func:`save_fit` directory per fit, named ``<arm>-<slug(label)>``.  No
    figures are written: plan R7 makes the report stage the only figure writer.

    The section ``meta.json`` is written **after** every fit, so a crashed
    save never produces a section that claims fits it does not hold.  A fit
    key of the form ``"A: <label>"`` becomes the ``A-`` prefix and records
    ``arm="A"``; a bare key is the §5.1 form and records ``arm=None``.

    Raises
    ------
    ArtifactError
        If ``result`` is not a :class:`DemoResult`, a key disagrees with its
        fit's label, two keys map to the same directory, or its meta dict has
        no JSON form.
    """
    from src.paper_examples import DemoResult as DemoResultType

    if not isinstance(result, DemoResultType):
        raise ArtifactError(
            f"save_demo_result expects a DemoResult, got {type(result).__name__}"
        )
    target = Path(directory)
    if not isinstance(result.label, str) or not result.label:
        raise ArtifactError(
            f"demo label must be a non-empty string, got {result.label!r}"
        )
    label = result.label
    if not isinstance(result.meta, Mapping):
        raise ArtifactError(
            f"demo meta must be a mapping, got {type(result.meta).__name__}"
        )
    try:
        section_meta = _jsonable(dict(result.meta))
    except ArtifactError as exc:
        raise ArtifactError(f"demo meta is not JSON-serialisable: {exc}") from exc
    if not isinstance(section_meta, dict):  # pragma: no cover - dict in, dict out
        raise ArtifactError("demo meta must be a JSON object")
    data_seed = section_meta.get("data_seed")
    if data_seed is not None and (
        isinstance(data_seed, bool) or not isinstance(data_seed, int)
    ):
        raise ArtifactError(
            f"demo meta 'data_seed' must be an integer or None, got {data_seed!r}"
        )

    # Validate the whole key set before writing anything, so a duplicate slug
    # cannot leave a half-written section behind.
    plan: list[tuple[str, str | None, str]] = []
    directories: dict[str, str] = {}
    for key, fit in result.fits.items():
        if not isinstance(key, str) or not key:
            raise ArtifactError(f"demo fit keys must be non-empty strings, got {key!r}")
        arm, fit_label = _split_fit_key(key)
        fit_label_attr = getattr(fit, "label", None)
        if fit_label != fit_label_attr:
            raise ArtifactError(
                f"demo fit key {key!r} does not match the fit's label "
                f"{fit_label_attr!r}"
            )
        name = _fit_directory_name(arm, fit_label)
        if name in directories:
            raise ArtifactError(
                f"demo fit keys {directories[name]!r} and {key!r} both map to "
                f"the directory {name!r}"
            )
        directories[name] = key
        plan.append((key, arm, name))

    fit_keys: list[str] = []
    for key, arm, name in plan:
        fit = result.fits[key]
        save_fit(
            target / _DRAWS_DIR / name,
            fit,
            meta={
                "arm": arm,
                "data_seed": data_seed,
                "shared_t_max": float(fit.data.t_max),
            },
        )
        fit_keys.append(key)

    _write_json(
        target / _DEMO_META_NAME,
        {
            "schema_version": SCHEMA_VERSION,
            "label": label,
            "fit_keys": fit_keys,
            "meta": section_meta,
        },
    )
    return target


def load_demo_result(directory: str | os.PathLike[str]) -> DemoResult:
    """Load a section directory written by :func:`save_demo_result`.

    The section ``meta.json`` names the fits and their order; every named
    directory is loaded with :func:`load_fit` and its recorded ``label``/``arm``
    must match the key it is filed under.  A fit directory that the section
    metadata does not name is an error, so a reviewer can trust that
    ``result.fits`` is the whole tree.

    ``DemoResult.meta`` is reconstructed with its original types --
    ``prior_indices`` as a tuple, ``seed``/``data_seed`` as int or ``None``,
    ``t_max`` as floats -- and ``figures`` is empty, because plan R7 moves
    figures to the report stage.

    Raises
    ------
    ArtifactError
        If ``directory`` is missing, is not a directory, or has no
        ``meta.json``.
    ArtifactIntegrityError
        If a fit disagrees with the section metadata or its own schema.
    """
    from src.paper_examples import DemoResult as DemoResultType

    target = Path(directory)
    if not target.exists():
        raise ArtifactError(f"artifact does not exist: {target}")
    if not target.is_dir():
        raise ArtifactError(f"artifact is not a directory: {target}")
    meta_path = target / _DEMO_META_NAME
    record = _read_json_object(meta_path, kind="section metadata")
    record = _require_record(record, _DEMO_META_KEYS, "section metadata", meta_path)

    version = _as_meta_int(
        record["schema_version"], "schema_version", meta_path, minimum=1
    )
    if version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"{meta_path}: schema version {version} is not the supported "
            f"version {SCHEMA_VERSION} (no migration path)"
        )
    label = _as_label(record["label"], "label", meta_path)
    raw_keys = record["fit_keys"]
    if not isinstance(raw_keys, list):
        raise ArtifactIntegrityError(
            f"{meta_path}: meta field 'fit_keys' must be a list, "
            f"got {type(raw_keys).__name__}"
        )
    fit_keys: list[str] = []
    for key in raw_keys:
        if not isinstance(key, str) or not key:
            raise ArtifactIntegrityError(
                f"{meta_path}: meta field 'fit_keys' must hold non-empty "
                f"strings, got {key!r}"
            )
        if key in fit_keys:
            raise ArtifactIntegrityError(
                f"{meta_path}: meta field 'fit_keys' lists {key!r} twice"
            )
        fit_keys.append(key)
    section_meta = _require_jsonable_meta(
        record["meta"], meta_path, name="section meta"
    )

    fits: dict[str, Fit] = {}
    expected: dict[str, str] = {}
    for key in fit_keys:
        arm, fit_label = _split_fit_key(key)
        name = _fit_directory_name(arm, fit_label)
        if name in expected:
            raise ArtifactIntegrityError(
                f"{meta_path}: fit keys {expected[name]!r} and {key!r} both map "
                f"to the directory {name!r}"
            )
        expected[name] = key
        fit_meta = _read_fit_meta(target / _DRAWS_DIR / name)
        if fit_meta.label != fit_label or fit_meta.arm != arm:
            raise ArtifactIntegrityError(
                f"{fit_meta.path}: label/arm ({fit_meta.label!r}, "
                f"{fit_meta.arm!r}) disagree with the section key {key!r}"
            )
        fits[key] = _load_fit_from_meta(fit_meta)

    draws = target / _DRAWS_DIR
    if draws.exists():
        # An empty run legitimately has no ``draws/``; if the directory exists,
        # every entry must be a fit the section metadata names -- including the
        # case of a section that names no fits at all.
        if not draws.is_dir():
            raise ArtifactError(f"artifact is not a directory: {draws}")
        for child in sorted(draws.iterdir()):
            if child.name not in expected:
                raise ArtifactIntegrityError(
                    f"{child}: unlisted fit directory (not named by {meta_path})"
                )

    return DemoResultType(
        label=label,
        fits=fits,
        figures=(),
        meta=_restore_demo_meta(section_meta, meta_path),
    )


# ---------------------------------------------------------------------------
# Incremental run writer (plan §6.1)
# ---------------------------------------------------------------------------

#: Event-record keys the writer owns; a caller field may not shadow one, so a
#: provenance key cannot silently replace the stamp or the kind.
_EVENT_RESERVED = frozenset({"ts", "t_monotonic", "kind"})

#: Run-root subdirectory holding the log files (plan §5).
_LOGS_DIR = "logs"

#: Event-stream name: one append-only stream per section.
_EVENTS_NAME = "events-{section}.jsonl"

#: An arm prefix must be a plain identifier.  It becomes part of a directory
#: name, so ``"A/B"`` would put a fit outside ``draws/``.
_ARM_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def _utc_timestamp() -> str:
    """Wall-clock stamp of one event line (ISO 8601, UTC, milliseconds)."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validated_root(root: object) -> Path:
    """Validate the run root of an :class:`ArtifactWriter`."""
    if isinstance(root, os.PathLike):
        return Path(root)
    if isinstance(root, str) and root:
        return Path(root)
    raise ArtifactError(f"artifact run root must be a non-empty path, got {root!r}")


def _validated_section(section: object) -> str:
    """Validate a section name: one plain directory name (``"5.1"``)."""
    if not isinstance(section, str) or not section:
        raise ArtifactError(
            f"artifact section must be a non-empty string, got {section!r}"
        )
    if section in {".", ".."} or _NON_SLUG.search(section):
        raise ArtifactError(
            f"artifact section {section!r} must be a single directory name of "
            "[A-Za-z0-9._-] characters"
        )
    return section


def _validated_arm(arm: object) -> str | None:
    """Validate the optional arm prefix (``"A"``/``"B"`` in §5.2)."""
    if arm is None:
        return None
    if not isinstance(arm, str) or not _ARM_NAME.match(arm):
        raise ArtifactError(
            f"artifact arm must be None or a plain identifier, got {arm!r}"
        )
    return arm


def _validated_text(value: object, *, name: str) -> str:
    """Validate a required non-empty string field of the writer."""
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _writer_meta(meta: object, *, name: str) -> dict[str, Any]:
    """Convert a caller's provenance mapping into the stored JSON object."""
    if meta is None:
        return {}
    if not isinstance(meta, Mapping):
        raise ArtifactError(f"{name} must be a mapping, got {type(meta).__name__}")
    try:
        converted = _jsonable(dict(meta))
    except ArtifactError as exc:
        raise ArtifactError(f"{name} is not JSON-serialisable: {exc}") from exc
    if not isinstance(converted, dict):  # pragma: no cover - dict in, dict out
        raise ArtifactError(f"{name} must be a JSON object")
    return converted


@dataclass(frozen=True)
class _WrittenFit:
    """One fit the writer has already written (internal bookkeeping)."""

    key: str
    arm: str | None
    label: str
    directory: Path


class ArtifactWriter:
    """Write one replication section incrementally (plan §6.1).

    A driver builds one writer per section and calls :meth:`fit_done` after
    every fit, :meth:`band_done` once the prior band exists and
    :meth:`write_meta` when the section is complete.  The long runs cannot hold
    a whole section in memory, so nothing is batched: a fit is written before
    the next one starts, and the section ``meta.json`` is written last, after
    every fit directory it names exists.

    Layout, relative to ``root`` (plan §5)::

        logs/events-<section>.jsonl        # append-only event stream
        <section>/draws/<arm->slug>/       # one save_fit directory per fit
        <section>/meta.json                # section metadata, written last

    ``root`` is the run root, not the section directory, because the event
    stream is run-level: both sections of one replication share ``logs/``.
    The writer never writes ``MANIFEST.json`` (plan R7 gives that to
    ``finalize_run``) and never writes figures (also R7).

    Each :meth:`event` line is one JSON object with the writer's ``ts`` (ISO
    8601 UTC), ``t_monotonic`` (:func:`time.perf_counter`, so events can be
    ordered without trusting the wall clock) and ``kind`` fields plus the
    caller's fields.  The line is appended, flushed and ``fsync``-ed before the
    call returns, so a crash keeps every completed event and never truncates an
    earlier one.  Event fields are validated *before* the artifact they
    describe is written, so a non-serialisable provenance value cannot leave a
    fit directory without its log line.

    Parameters
    ----------
    root:
        Run root directory.  Created lazily by the first write.
    section:
        Section name, a single directory name of ``[A-Za-z0-9._-]``
        (``"5.1"``/``"5.2"``).  It names both ``<root>/<section>`` and
        ``logs/events-<section>.jsonl``.
    meta:
        Section provenance stored verbatim in ``meta.json`` -- the
        ``DemoResult.meta`` dict of the driver (preset, seeds,
        ``prior_indices``, chain count, ``t_max``).  Defaults to an empty
        object.
    label:
        The section label (``DemoResult.label``); defaults to ``section``.
    monotonic:
        Clock for ``t_monotonic``; defaults to :func:`time.perf_counter`.
        A test seam only.
    wall_clock:
        Clock for ``ts``; defaults to an ISO 8601 UTC stamp.  A test seam only.

    Raises
    ------
    ArtifactError
        If ``root``/``section``/``label`` have the wrong type or shape, or
        ``meta`` has no JSON form.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        section: str,
        meta: Mapping[str, Any] | None = None,
        label: str | None = None,
        monotonic: Callable[[], float] | None = None,
        wall_clock: Callable[[], str] | None = None,
    ) -> None:
        self._root = _validated_root(root)
        self._section = _validated_section(section)
        self._label = (
            self._section
            if label is None
            else _validated_text(label, name="artifact section label")
        )
        self._meta = _writer_meta(meta, name="section meta")
        self._monotonic = perf_counter if monotonic is None else monotonic
        self._wall_clock = _utc_timestamp if wall_clock is None else wall_clock
        self._fits: dict[str, _WrittenFit] = {}
        self._closed = False

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> ArtifactWriter:
        """Return the writer; writes stay allowed until the block exits."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        """Close the writer; a later write is an error, not a silent append."""
        self._closed = True
        return False

    # -- public state ------------------------------------------------------

    @property
    def root(self) -> Path:
        """The run root this writer writes under."""
        return self._root

    @property
    def section(self) -> str:
        """The section name (``"5.1"``/``"5.2"``)."""
        return self._section

    @property
    def section_dir(self) -> Path:
        """The section directory, ``<root>/<section>``."""
        return self._root / self._section

    @property
    def event_log(self) -> Path:
        """The append-only event stream, ``<root>/logs/events-<section>.jsonl``."""
        return self._root / _LOGS_DIR / _EVENTS_NAME.format(section=self._section)

    @property
    def fit_keys(self) -> tuple[str, ...]:
        """The fit keys written so far, in write order."""
        return tuple(self._fits)

    # -- events ------------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Append one event line and return the record that was written.

        ``kind`` names the stage (``"run_start"``, ``"fit_done"``,
        ``"band_done"``, ``"run_end"``, ...).  ``fields`` are stored verbatim
        through :func:`_jsonable`; they may not shadow ``ts``,
        ``t_monotonic`` or ``kind``, because those three keys are the log's
        fixed schema.  The file is opened in append mode for each call, so
        two writers of the same section -- a resumed run, a test -- extend one
        stream instead of truncating it.

        Raises
        ------
        ArtifactError
            If the writer is closed, ``kind`` is not a non-empty string, a
            field shadows a reserved key, or a field has no JSON form.
        """
        payload = self._event_payload(kind, fields)
        self._append_event(payload)
        return payload

    def _event_payload(self, kind: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and assemble one event record without writing it."""
        self._require_open()
        if not isinstance(kind, str) or not kind:
            raise ArtifactError(f"event kind must be a non-empty string, got {kind!r}")
        clashing = sorted(_EVENT_RESERVED.intersection(fields))
        if clashing:
            raise ArtifactError(
                f"event fields may not shadow the writer keys "
                f"{sorted(_EVENT_RESERVED)}: got {clashing}"
            )
        try:
            converted = _jsonable(dict(fields))
        except ArtifactError as exc:
            raise ArtifactError(
                f"event {kind!r} fields are not JSON-serialisable: {exc}"
            ) from exc
        if not isinstance(converted, dict):  # pragma: no cover - dict in, dict out
            raise ArtifactError(f"event {kind!r} fields must be a JSON object")
        stamp = self._wall_clock()
        if not isinstance(stamp, str) or not stamp:
            raise ArtifactError(
                f"event wall clock must return a non-empty string, got {stamp!r}"
            )
        tick = float(self._monotonic())
        if not np.isfinite(tick):
            raise ArtifactError(f"event monotonic clock must be finite, got {tick!r}")
        return {"ts": stamp, "t_monotonic": tick, "kind": kind, **converted}

    def _append_event(self, payload: Mapping[str, Any]) -> Path:
        """Append one JSON object as a line, flushed and ``fsync``-ed."""
        target = self.event_log
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target

    # -- fits --------------------------------------------------------------

    def fit_done(
        self,
        fit: Fit,
        timing: Any = None,
        *,
        arm: str | None = None,
        **provenance: Any,
    ) -> Path:
        """Write one finished fit as a :func:`save_fit` directory.

        The directory is ``<root>/<section>/draws/<arm->slug(label)>`` -- the
        plan §5 name, so a reviewer never needs the writer to find it.  The
        fit's own ``prior_band`` is stored when it has one; the driver usually
        attaches the band later, with :meth:`band_done`.

        Two defaults come from the section ``meta`` because the fit object does
        not hold them: ``data_seed`` is recorded when the section meta has one
        (it is a demo-level seed) and ``shared_t_max`` is taken from the
        section ``t_max``, which is the common grid maximum of §5.2.  Anything
        in ``provenance`` overrides both; everything that is not a reserved
        fit key (``arm``, ``data_seed``, ``shared_t_max``) is stored under
        ``fit-meta.json``'s ``extra``, next to ``timing``.

        Parameters
        ----------
        fit:
            The finished :class:`Fit`; it is not mutated.
        timing:
            Optional wall time (seconds) or timing mapping.  Stored in
            ``fit-meta.json`` as ``extra.timing`` and repeated in the event.
        arm:
            The §5.2 arm prefix, ``None`` for §5.1.
        **provenance:
            Any further JSON provenance (chain seeds, functional settings).

        Returns
        -------
        pathlib.Path
            The fit directory that was written.

        Raises
        ------
        ArtifactError
            If ``fit`` is not a :class:`Fit`, the arm/key/directory was
            already written, or a provenance value has no JSON form.
        """
        self._require_open()
        from src.paper_examples import Fit as FitType

        if not isinstance(fit, FitType):
            raise ArtifactError(f"fit_done expects a Fit, got {type(fit).__name__}")
        label = _validated_text(fit.label, name="fit label")
        arm = _validated_arm(arm)
        key = f"{arm}: {label}" if arm else label
        if key in self._fits:
            raise ArtifactError(f"fit {key!r} was already written by this writer")
        name = _fit_directory_name(arm, label)
        directory = self.section_dir / _DRAWS_DIR / name
        for record in self._fits.values():
            if record.directory == directory:
                raise ArtifactError(
                    f"fits {record.key!r} and {key!r} both map to the directory "
                    f"{name!r}"
                )

        fields: dict[str, Any] = dict(provenance)
        fields.update(
            {
                "key": key,
                "path": self._relative(directory),
                "n_chains": int(fit.n_chains),
                "n_draws": int(fit.functionals.n_draws),
            }
        )
        if timing is not None:
            fields["timing"] = timing
        payload = self._event_payload("fit_done", fields)

        meta: dict[str, Any] = {"arm": arm}
        if "data_seed" not in provenance and "data_seed" in self._meta:
            meta["data_seed"] = self._meta["data_seed"]
        if "shared_t_max" not in provenance and "t_max" in self._meta:
            meta["shared_t_max"] = self._meta["t_max"]
        meta.update(provenance)
        if timing is not None:
            meta["timing"] = timing

        save_fit(directory, fit, meta=meta)
        self._fits[key] = _WrittenFit(
            key=key, arm=arm, label=label, directory=directory
        )
        self._append_event(payload)
        return directory

    # -- prior band --------------------------------------------------------

    def band_done(
        self,
        band: Functionals,
        timing: Any = None,
        *,
        fit: Fit | None = None,
        fit_label: str | None = None,
        arm: str | None = None,
        **provenance: Any,
    ) -> Path:
        """Attach one prior band to an already written fit directory.

        The band is written as ``prior-band.npz`` and the fit's
        ``fit-meta.json`` is then updated to record it, because
        :func:`load_fit` refuses an unlisted band file.  The band file is
        written first: if the process dies between the two writes, the
        directory holds a band the metadata does not name, which is an
        integrity error rather than a fit that silently carries a stale band.

        The target fit defaults to the first fit this writer wrote -- the
        reference prior's fit, exactly where the §5.1 driver leaves the band.
        ``fit`` (the in-memory object) contributes its label, so in a two-arm
        section pass ``arm`` as well: a label that matches several arms is an
        error rather than a guess, because the wrong fit would silently carry
        the band.

        Parameters
        ----------
        band:
            The prior :class:`Functionals`.  Its grid must be the target fit's
            grid (same point count and endpoints) and its arrays must match
            the functional schema.
        timing:
            Optional wall time for the band, stored as
            ``fit-meta.json``'s ``extra.band_timing``.
        fit:
            The fit object the band belongs to (its ``label`` selects the
            directory).
        fit_label:
            The un-slugged label of the target fit, when no object is at hand.
        arm:
            The target arm, when the label exists in more than one arm.
        **provenance:
            Any further JSON provenance, merged into the fit's ``extra``.

        Returns
        -------
        pathlib.Path
            The ``prior-band.npz`` that was written.

        Raises
        ------
        ArtifactError
            If ``band`` is not a :class:`Functionals`, no fit was written yet,
            the target is unknown or ambiguous, the band's grid disagrees with
            the fit's, or the band is outside the artifact schema.
        """
        self._require_open()
        if not isinstance(band, Functionals):
            raise ArtifactError(
                f"band_done expects a Functionals, got {type(band).__name__}"
            )
        record = self._resolve_fit(fit=fit, fit_label=fit_label, arm=arm)
        fit_meta = _read_fit_meta(record.directory)
        if fit_meta.label != record.label or fit_meta.arm != record.arm:
            raise ArtifactError(
                f"{fit_meta.path}: label/arm ({fit_meta.label!r}, "
                f"{fit_meta.arm!r}) no longer match the fit this writer "
                f"recorded ({record.key!r})"
            )

        grid = _validated_t_grid(band.t_grid)
        if (
            int(grid.size) != fit_meta.n_grid
            or float(grid[0]) != fit_meta.grid_first
            or float(grid[-1]) != fit_meta.grid_last
        ):
            raise ArtifactError(
                f"prior band grid ({grid.size} points, "
                f"[{float(grid[0])!r}, {float(grid[-1])!r}]) disagrees with "
                f"the fit's grid ({fit_meta.n_grid} points, "
                f"[{fit_meta.grid_first!r}, {fit_meta.grid_last!r}])"
            )
        arrays = {name: getattr(band, name) for name in _FUNCTIONAL_ARRAY_FIELDS}
        try:
            _check_functional_arrays(arrays, grid, record.directory)
            band_meta = _require_jsonable_meta(
                band.meta, record.directory, name="prior band meta"
            )
        except ArtifactIntegrityError as exc:
            raise ArtifactError(
                f"prior band is outside the artifact schema: {exc}"
            ) from exc

        extra = _writer_meta(provenance, name="band provenance")
        if timing is not None:
            extra["band_timing"] = timing
        try:
            extra = _jsonable(extra)
        except ArtifactError as exc:
            raise ArtifactError(
                f"band provenance is not JSON-serialisable: {exc}"
            ) from exc

        band_path = record.directory / _PRIOR_BAND_NAME
        fields: dict[str, Any] = dict(provenance)
        fields.update(
            {
                "key": record.key,
                "path": self._relative(band_path),
                "n_draws": int(band.n_draws),
            }
        )
        if timing is not None:
            fields["timing"] = timing
        payload = self._event_payload("band_done", fields)

        save_functionals(band_path, band)

        stored = _read_json_object(fit_meta.path, kind="fit metadata")
        stored_record = dict(
            _require_record(stored, _FIT_META_KEYS, "fit metadata", fit_meta.path)
        )
        stored_extra = stored_record["extra"]
        if not isinstance(stored_extra, dict):  # pragma: no cover - validated above
            raise ArtifactIntegrityError(
                f"{fit_meta.path}: meta field 'extra' must be a JSON object"
            )
        stored_extra.update(extra)
        stored_record["prior_band"] = {
            "path": _PRIOR_BAND_NAME,
            "dtype": "float64",
            "shape": [int(arrays["S"].shape[0]), int(grid.size)],
        }
        stored_record["prior_band_meta"] = band_meta
        _write_json(fit_meta.path, stored_record)
        self._append_event(payload)
        return band_path

    def _resolve_fit(
        self,
        *,
        fit: Fit | None,
        fit_label: str | None,
        arm: str | None,
    ) -> _WrittenFit:
        """Select the fit directory a prior band belongs to."""
        from src.paper_examples import Fit as FitType

        if fit is not None:
            if not isinstance(fit, FitType):
                raise ArtifactError(
                    f"band_done expects a Fit or None, got {type(fit).__name__}"
                )
            object_label = _validated_text(fit.label, name="fit label")
            if fit_label is not None and fit_label != object_label:
                raise ArtifactError(
                    f"band_done got fit {object_label!r} but fit_label {fit_label!r}"
                )
            fit_label = object_label
        if fit_label is not None:
            fit_label = _validated_text(fit_label, name="fit label")
        arm = _validated_arm(arm)
        if not self._fits:
            raise ArtifactError(
                "band_done needs a fit written by this writer first; "
                "call fit_done before the prior band"
            )
        if fit_label is None:
            if arm is None:
                return next(iter(self._fits.values()))
            candidates = [record for record in self._fits.values() if record.arm == arm]
            if not candidates:
                raise ArtifactError(f"no fit in arm {arm!r} was written by this writer")
            if len(candidates) > 1:
                raise ArtifactError(
                    f"arm {arm!r} holds {len(candidates)} fits; pass fit_label "
                    "to choose the prior band's fit"
                )
            return candidates[0]
        candidates = [
            record
            for record in self._fits.values()
            if record.label == fit_label and (arm is None or record.arm == arm)
        ]
        if not candidates:
            raise ArtifactError(
                f"no fit with label {fit_label!r} and arm {arm!r} was written "
                "by this writer"
            )
        if len(candidates) > 1:
            arms = ", ".join(repr(record.arm) for record in candidates)
            raise ArtifactError(
                f"fit label {fit_label!r} was written for several arms "
                f"({arms}); pass arm to choose the prior band's fit"
            )
        return candidates[0]

    # -- section metadata --------------------------------------------------

    def write_meta(self, *, meta: Mapping[str, Any] | None = None) -> Path:
        """Write the section ``meta.json`` once the section is complete.

        The file names the fits in write order (a JSON object of fits would
        lose it) through ``fit_keys``, so :func:`load_demo_result` rebuilds the
        same key order.  It is written last and atomically: a crashed run has
        no ``meta.json``, which the loader reports instead of accepting a
        section that names fits it does not hold.  A recorded fit directory
        that no longer exists (deleted by hand, removed by a cleanup) is an
        error for the same reason.

        Parameters
        ----------
        meta:
            Optional replacement for the constructor's section provenance.

        Returns
        -------
        pathlib.Path
            ``<root>/<section>/meta.json``.

        Raises
        ------
        ArtifactError
            If the writer is closed, a recorded fit directory is missing its
            ``fit-meta.json``, or the new ``meta`` has no JSON form.
        """
        self._require_open()
        if meta is not None:
            self._meta = _writer_meta(meta, name="section meta")
        missing = [
            record.key
            for record in self._fits.values()
            if not (record.directory / _FIT_META_NAME).is_file()
        ]
        if missing:
            raise ArtifactError(
                f"fit artifacts are missing for {missing}; refusing to write a "
                "section meta.json that names them"
            )
        return _write_json(
            self.section_dir / _DEMO_META_NAME,
            {
                "schema_version": SCHEMA_VERSION,
                "label": self._label,
                "fit_keys": list(self._fits),
                "meta": self._meta,
            },
        )

    # -- internals ---------------------------------------------------------

    def _require_open(self) -> None:
        """Reject writes after the context manager has exited."""
        if self._closed:
            raise ArtifactError(
                f"artifact writer for section {self._section!r} is closed"
            )

    def _relative(self, path: Path) -> str:
        """Return ``path`` relative to the run root, as a POSIX-style string."""
        return Path(os.path.relpath(path, self._root)).as_posix()


# ---------------------------------------------------------------------------
# Run manifest (plan §6.1)
# ---------------------------------------------------------------------------

#: The one run-level manifest.  It is written at the root of a replication run
#: and never lists itself: a file cannot carry its own hash, so the report
#: records ``sha256(MANIFEST.json)`` externally (plan §8 item 1).
_MANIFEST_NAME = "MANIFEST.json"

#: Shape of a stored SHA-256 digest (lowercase hexadecimal).
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

#: Read size used while hashing.  One MiB keeps a 96 MB trace at ~100 reads and
#: the peak memory of a verification at one chunk per file.
_HASH_CHUNK = 1024 * 1024

#: Package versions recorded as run provenance.  ``importlib.metadata`` reads
#: the installed distributions; nothing is imported, so the artifact module
#: still keeps ``matplotlib`` out of ``sys.modules`` (plan §6.1).
_PROVENANCE_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "matplotlib",
)

#: Keys of the run-level manifest.  The set is exact so a typo cannot hide
#: behind an ignored field.
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "created_at",
        "code_sha",
        "code_dirty",
        "pixi_lock_sha256",
        "packages",
        "sections",
        "meta",
        "file_count",
        "total_bytes",
        "files",
    }
)

#: Keys of one ``files`` entry: the digest and the byte size of one artifact.
_FILE_RECORD_KEYS = frozenset({"sha256", "size"})


@dataclass(frozen=True)
class ManifestReport:
    """What :func:`verify_manifest` checked and found.

    A report is returned only when every listed artifact matches its recorded
    size and digest, listed paths are safe, and no unlisted artifact exists
    (unless the caller allowed extras).  Anything else raises
    :class:`ArtifactIntegrityError`, so a returned report is a pass.

    Attributes
    ----------
    root:
        The run root that was verified.
    manifest:
        ``<root>/MANIFEST.json``.
    schema_version:
        The version the manifest claims; it equals :data:`SCHEMA_VERSION`.
    created_at:
        The manifest's ISO 8601 UTC stamp, as written by :func:`finalize_run`.
    code_sha / code_dirty:
        The git revision and dirty flag recorded at finalisation, or ``None``
        when the run was finalised outside a repository.
    file_count / total_bytes:
        Number of artifacts listed and their summed size (the manifest's own
        file is never part of either).
    paths:
        The verified artifact paths, relative to ``root``, in sorted order.
    extra:
        Unlisted artifacts found under ``root``.  Empty unless the caller
        passed ``allow_extra=True``.
    """

    root: Path
    manifest: Path
    schema_version: int
    created_at: str
    code_sha: str | None
    code_dirty: bool | None
    file_count: int
    total_bytes: int
    paths: tuple[str, ...]
    extra: tuple[str, ...] = ()


def manifest_path(root: str | os.PathLike[str]) -> Path:
    """Return the run-level manifest path of ``root`` without touching disk.

    ``root`` need not exist yet; this is the path :func:`finalize_run` writes
    and :func:`verify_manifest` reads.

    Raises
    ------
    ArtifactError
        If ``root`` is not a non-empty path.
    """
    return _validated_root(root) / _MANIFEST_NAME


def _existing_run_root(root: object) -> Path:
    """Validate a run root that must already exist as a directory."""
    target = _validated_root(root)
    if not target.exists():
        raise ArtifactError(f"artifact run root does not exist: {target}")
    if not target.is_dir():
        raise ArtifactError(f"artifact run root is not a directory: {target}")
    return target


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file, read in chunks."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactIntegrityError(f"{path}: cannot hash artifact: {exc}") from exc
    return digest.hexdigest()


def _run_tree_files(root: Path, manifest: Path) -> list[Path]:
    """List every file under ``root`` except the manifest itself.

    The walk is deterministic (sorted directories and names) and does not
    follow directory symlinks, so two finalisations of the same tree list the
    same paths.  Directories are not artifacts: an empty directory is not
    part of the checksum record.
    """
    paths: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories.sort()
        for name in sorted(filenames):
            path = Path(directory, name)
            if path == manifest:
                continue
            paths.append(path)
    return paths


def _git_output(root: Path, *arguments: str) -> str | None:
    """Run one read-only git command, returning ``None`` when it cannot work.

    Finalising a run outside a repository, or on a machine without git, must
    still produce a manifest; the provenance field then stays ``null``.  No
    command here writes to the repository.
    """
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _installed_versions() -> dict[str, str]:
    """Return the installed versions of the packages the run depends on."""
    versions: dict[str, str] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = _package_version(name)
        except _PackageNotFoundError:
            continue
    return versions


def _run_provenance(root: Path) -> dict[str, Any]:
    """Collect the code and environment provenance of a run root.

    The run root normally lives inside the repository (``output/`` is
    git-ignored), so the git calls work from there.  ``code_dirty`` counts
    every path ``git status --porcelain`` reports, which never includes the
    ignored ``output/`` tree itself; ``pixi_lock_sha256`` is the digest of the
    repository's ``pixi.lock``.  Each field degrades to ``None`` instead of
    failing the finalisation.
    """
    revision = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--porcelain")
    toplevel = _git_output(root, "rev-parse", "--show-toplevel")

    lock_sha: str | None = None
    if toplevel is not None and toplevel.strip():
        lock = Path(toplevel.strip()) / "pixi.lock"
        if lock.is_file():
            lock_sha = _sha256_file(lock)

    return {
        "code_sha": (revision.strip() or None) if revision is not None else None,
        "code_dirty": bool(status.strip()) if status is not None else None,
        "pixi_lock_sha256": lock_sha,
        "packages": _installed_versions(),
    }


def _section_metas(root: Path) -> dict[str, Any]:
    """Return the section ``meta.json`` payloads found under ``root``.

    One replication root holds one directory per section (``5.1/``,
    ``5.2/``); each is named by its ``meta.json``.  The manifests keeps those
    payloads verbatim so the run-level record carries the preset, seeds and
    timings without depending on section files that a later cleanup could
    remove.  A section file at another schema version is an integrity error,
    exactly as it is for the loaders.
    """
    sections: dict[str, Any] = {}
    for entry in sorted(root.iterdir()):
        meta_file = entry / _DEMO_META_NAME
        if not entry.is_dir() or not meta_file.is_file():
            continue
        payload = _read_json_object(meta_file, kind="section metadata")
        if not isinstance(payload, Mapping):
            raise ArtifactIntegrityError(
                f"{meta_file}: section metadata must be a JSON object, "
                f"got {type(payload).__name__}"
            )
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ArtifactIntegrityError(
                f"{meta_file}: unsupported schema_version {version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        sections[entry.name] = dict(payload)
    return sections


def finalize_run(
    root: str | os.PathLike[str], *, meta: Mapping[str, Any] | None = None
) -> Path:
    """Hash a completed run tree and write its single ``MANIFEST.json``.

    Plan §6.1: called once after the report stage, over the whole run root, so
    ``summary/``, ``report/`` and ``figures/`` are covered together with every
    chain and functional.  A later stage (the §5.1 run joining the §5.2 tree,
    R6) simply calls this again: the manifest is rewritten atomically and now
    covers both sections.

    The manifest is one JSON object::

        schema_version, created_at,
        code_sha, code_dirty, pixi_lock_sha256, packages,
        sections:  {"5.1": <section meta.json>, "5.2": ...},
        meta:      <caller provenance, verbatim>,
        file_count, total_bytes,
        files:     {"<posix relative path>": {"sha256": ..., "size": ...}}

    ``MANIFEST.json`` never lists itself (a file cannot carry its own hash);
    every other regular file under ``root`` is listed, including log files
    and the report.  The file counts and sizes are consistent with the
    ``files`` object by construction and are re-checked by
    :func:`verify_manifest`.

    Parameters
    ----------
    root:
        The existing run root.  It is never created: finalising a mistyped
        path would produce a manifest for an empty run.
    meta:
        Optional JSON-serialisable provenance (run stamp, harness notes)
        stored under ``meta``.  Section payloads and code provenance are
        collected automatically.

    Returns
    -------
    pathlib.Path
        The written ``MANIFEST.json``.

    Raises
    ------
    ArtifactError
        If ``root`` is unusable or ``meta`` has no JSON form.
    ArtifactIntegrityError
        If a file disappears or cannot be read while hashing, or a section
        ``meta.json`` is malformed or at another schema version.
    """
    target = _existing_run_root(root)
    manifest = target / _MANIFEST_NAME
    caller_meta = _writer_meta(meta, name="run manifest meta")

    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in _run_tree_files(target, manifest):
        relative = path.relative_to(target).as_posix()
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"{path}: cannot stat artifact: {exc}"
            ) from exc
        files[relative] = {"sha256": _sha256_file(path), "size": size}
        total_bytes += size

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_timestamp(),
        **_run_provenance(target),
        "sections": _section_metas(target),
        "meta": caller_meta,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    return _write_json(manifest, payload)


def _manifest_mapping(value: Any, name: str, target: Path) -> Mapping[str, Any]:
    """Validate one JSON object field of a manifest."""
    if not isinstance(value, Mapping):
        raise ArtifactIntegrityError(
            f"{target}: manifest field {name!r} must be a JSON object, "
            f"got {type(value).__name__}"
        )
    return value


def _manifest_string_map(value: Any, name: str, target: Path) -> dict[str, str]:
    """Validate a manifest object whose values are all strings."""
    mapping = _manifest_mapping(value, name, target)
    converted: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(item, str):
            raise ArtifactIntegrityError(
                f"{target}: manifest field {name}[{key!r}] must be a string, "
                f"got {type(item).__name__}"
            )
        converted[key] = item
    return converted


def _manifest_text(value: Any, name: str, target: Path) -> str:
    """Validate a required non-empty string field of a manifest."""
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"{target}: manifest field {name!r} must be a non-empty string, "
            f"got {value!r}"
        )
    return value


def _manifest_optional_string(value: Any, name: str, target: Path) -> str | None:
    """Validate an optional non-empty string field of a manifest."""
    if value is None:
        return None
    return _manifest_text(value, name, target)


def _manifest_optional_bool(value: Any, name: str, target: Path) -> bool | None:
    """Validate an optional boolean field of a manifest."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ArtifactIntegrityError(
            f"{target}: manifest field {name!r} must be a boolean or null, "
            f"got {value!r}"
        )
    return value


def _manifest_relative_path(value: Any, target: Path) -> str:
    """Validate a listed artifact path as a canonical relative POSIX path.

    The path is later joined to the run root, so an absolute path, a ``..``
    component, an empty/dot path or a non-canonical spelling (``a//b``) is
    rejected rather than normalised: two spellings of one file would let a
    manifest name one artifact twice.  A manifest naming itself is rejected
    too -- that is the hash cycle the design forbids.
    """
    if not isinstance(value, str) or not value:
        raise ArtifactIntegrityError(
            f"{target}: manifest file path must be a non-empty string, got {value!r}"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ArtifactIntegrityError(
            f"{target}: manifest file path {value!r} is not a canonical path "
            "relative to the run root"
        )
    if value == _MANIFEST_NAME:
        raise ArtifactIntegrityError(f"{target}: the run manifest may not list itself")
    return value


def _manifest_file_record(value: Any, name: str, target: Path) -> tuple[str, int]:
    """Validate one ``files`` entry and return ``(sha256, size)``."""
    record = _require_record(value, _FILE_RECORD_KEYS, f"files[{name!r}]", target)
    digest = record["sha256"]
    if not isinstance(digest, str) or not _SHA256_HEX.match(digest):
        raise ArtifactIntegrityError(
            f"{target}: files[{name!r}].sha256 must be a 64-character "
            f"lowercase hexadecimal digest, got {digest!r}"
        )
    size = _as_meta_int(record["size"], f"files[{name!r}].size", target, minimum=0)
    return digest, size


def verify_manifest(
    root: str | os.PathLike[str], *, allow_extra: bool = False
) -> ManifestReport:
    """Re-hash a run tree against its ``MANIFEST.json``.

    The reviewer entry point of plan §9 layer 1.  It reads the manifest,
    validates its schema (including :data:`SCHEMA_VERSION`), then checks every
    listed artifact:

    * a missing file or a listed path that is not a regular file;
    * a size mismatch (a truncated file);
    * a SHA-256 mismatch (a flipped byte, or any edit of the same length).

    It also walks the tree in the other direction: a file under the root that
    the manifest does not list is an integrity failure unless the caller
    passed ``allow_extra=True`` (the verifier re-renders figures into a
    scratch directory and may legitimately add files there).

    Every failure is collected and reported in one :class:`ArtifactIntegrityError`
    naming the manifest and each offending path, so a broken tree is fixed in
    one pass instead of one exception per run.

    Parameters
    ----------
    root:
        The run root holding ``MANIFEST.json``.
    allow_extra:
        Accept unlisted files under ``root``.  They are listed in
        :attr:`ManifestReport.extra`; listed files are still fully verified.

    Returns
    -------
    ManifestReport
        The verified manifest's provenance and artifact inventory.

    Raises
    ------
    ArtifactError
        If the root or the manifest is missing or unreadable.
    ArtifactIntegrityError
        If the manifest schema is invalid, a listed artifact fails its size or
        digest check, or an unlisted artifact exists while ``allow_extra`` is
        false.
    """
    target = _existing_run_root(root)
    manifest = target / _MANIFEST_NAME
    payload = _read_json_object(manifest, kind="run manifest")
    record = _require_record(payload, _MANIFEST_KEYS, "run manifest", manifest)

    schema_version = _as_meta_int(
        record["schema_version"], "schema_version", manifest, minimum=1
    )
    if schema_version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"{manifest}: unsupported schema_version {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    created_at = _manifest_text(record["created_at"], "created_at", manifest)
    code_sha = _manifest_optional_string(record["code_sha"], "code_sha", manifest)
    code_dirty = _manifest_optional_bool(record["code_dirty"], "code_dirty", manifest)
    _manifest_optional_string(record["pixi_lock_sha256"], "pixi_lock_sha256", manifest)
    _manifest_string_map(record["packages"], "packages", manifest)
    _manifest_mapping(record["sections"], "sections", manifest)
    _manifest_mapping(record["meta"], "meta", manifest)

    expected_count = _as_meta_int(
        record["file_count"], "file_count", manifest, minimum=0
    )
    expected_bytes = _as_meta_int(
        record["total_bytes"], "total_bytes", manifest, minimum=0
    )
    files = _manifest_mapping(record["files"], "files", manifest)
    if expected_count != len(files):
        raise ArtifactIntegrityError(
            f"{manifest}: file_count is {expected_count} but the manifest "
            f"lists {len(files)} files"
        )

    listed: dict[str, tuple[str, int]] = {}
    for name, value in files.items():
        relative = _manifest_relative_path(name, manifest)
        listed[relative] = _manifest_file_record(value, relative, manifest)
    total_bytes = sum(size for _, size in listed.values())
    if expected_bytes != total_bytes:
        raise ArtifactIntegrityError(
            f"{manifest}: total_bytes is {expected_bytes} but the listed "
            f"files total {total_bytes}"
        )

    failures: list[str] = []
    for relative, (digest, size) in listed.items():
        path = target / relative
        if not path.is_file():
            state = "is not a file" if path.exists() else "is missing"
            failures.append(f"{path}: listed artifact {state}")
            continue
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            failures.append(f"{path}: cannot stat artifact: {exc}")
            continue
        if actual_size != size:
            failures.append(
                f"{path}: size mismatch; manifest says {size}, found {actual_size}"
            )
            continue
        actual_digest = _sha256_file(path)
        if actual_digest != digest:
            failures.append(
                f"{path}: sha256 mismatch; manifest says {digest}, "
                f"found {actual_digest}"
            )
    if failures:
        raise ArtifactIntegrityError(
            f"{manifest}: integrity check failed for {len(failures)} "
            f"artifact(s):\n" + "\n".join(failures)
        )

    extra = tuple(
        sorted(
            path.relative_to(target).as_posix()
            for path in _run_tree_files(target, manifest)
            if path.relative_to(target).as_posix() not in listed
        )
    )
    if extra and not allow_extra:
        raise ArtifactIntegrityError(
            f"{manifest}: {len(extra)} unlisted artifact(s) under {target}: "
            f"{list(extra)}"
        )

    return ManifestReport(
        root=target,
        manifest=manifest,
        schema_version=schema_version,
        created_at=created_at,
        code_sha=code_sha,
        code_dirty=code_dirty,
        file_count=len(listed),
        total_bytes=total_bytes,
        paths=tuple(sorted(listed)),
        extra=extra,
    )
