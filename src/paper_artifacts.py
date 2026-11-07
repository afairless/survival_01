"""Serialisation primitives for the artifact-backed replication run.

The replication run of ``docs/research/dp_gibbs_replication_artifacts_plan.md``
stores every draw, functional array and summary it produces under a
git-ignored ``output/`` tree (§5 of that plan).  A reviewer must be able to
load those files without the fitting process, so the loaders of this module
will reconstruct containers explicitly (never via pickle) and validate every
array against the declared schema first.

This unit contributes the schema-independent primitives that the chain,
functional, fit and run writers are built on:

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
* :func:`atomic_savez` -- an ``npz`` write that is either complete or absent.

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
Pure I/O plus NumPy: this module must not import ``matplotlib`` or any
plotting code (plan §6.1), and it must not import the sampler or the
functional modules.  The chain, functional, fit, run-writer and manifest
units of plan §6.1 are layered on top of the primitives here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactIntegrityError",
    "atomic_savez",
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
