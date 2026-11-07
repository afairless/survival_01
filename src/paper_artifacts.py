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
  are stored by one rule.

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
functional -- it only owns the containers.  The fit, run-writer and manifest
units of plan §6.1 are layered on top of the primitives here.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np

from src.dp_functionals import Functionals
from src.dp_gibbs import DPChains, DPPrior, DPSettings, SurvData

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactIntegrityError",
    "atomic_savez",
    "load_chain",
    "load_functionals",
    "save_chain",
    "save_functionals",
    "slug",
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
    return chain


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


def _require_jsonable_meta(meta: object, target: Path) -> dict[str, Any]:
    """Convert a container's ``meta`` dict to the JSON object that is stored.

    The dict is stored verbatim (plan §5.1): its keys are not enumerated,
    because a posterior functional carries ``method``/``J_common``/``L`` and a
    prior band carries ``sample``/``J_prior``/``max_nu``.  :func:`_jsonable`
    is the only conversion applied, so NumPy scalars become Python scalars and
    a tuple becomes a JSON list; a value with no JSON form is rejected instead
    of being stringified.

    Raises
    ------
    ArtifactIntegrityError
        If ``meta`` is not a mapping or holds a value with no JSON form.
    """
    if not isinstance(meta, Mapping):
        raise ArtifactIntegrityError(
            f"{target}: functionals meta must be a JSON object, "
            f"got {type(meta).__name__}"
        )
    try:
        converted = _jsonable(dict(meta))
    except ArtifactError as exc:
        raise ArtifactIntegrityError(
            f"{target}: functionals meta is not JSON-serialisable: {exc}"
        ) from exc
    if not isinstance(converted, dict):
        raise ArtifactIntegrityError(
            f"{target}: functionals meta must be a JSON object, "
            f"got {type(converted).__name__}"
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
