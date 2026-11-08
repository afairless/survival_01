"""Test 33: the replication and verification scripts (plan Section 6.5).

Two shell entry points wrap the artifact pipeline so the runbook is one line:

``scripts/replicate.sh <5.1|5.2> [run_root]``
    the run: environment log, driver, report stage, a manifest, the
    artifact-mode acceptance replay, then the final ``finalize_run`` and
    ``verify_manifest``.
``scripts/verify_replication.sh <run_root>``
    the "show me" check: manifest verification, byte-for-byte figure
    re-rendering into a scratch directory, and the artifact-mode replay, with
    a final integrity re-check proving the tree was not written to.

These tests pin what a Python unit test cannot: that both files are executable
bash that parses, that the ``pixi`` tasks point at them, that the argument
surface fails closed (an unknown section or a mistyped run root exits non-zero
instead of starting a long run), and -- in one end-to-end smoke test -- that a
real run of the driver, report, finaliser and verifier agrees with itself.

The smoke test runs Section 5.2 at the ``quick`` preset with a single prior,
which is a machinery check, not an acceptance check: the acceptance clauses
need the paper preset and all three priors, so the replay stage is skipped
with ``REPLICATE_ACCEPTANCE=0``.  The run root lives under the git-ignored
``output/`` tree because ``finalize_run`` records the checkout's git SHA from
the run's own directory -- a root outside the repository records ``null``,
which the acceptance replay would then refuse.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from src import paper_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
REPLICATE = SCRIPTS_DIR / "replicate.sh"
VERIFY = SCRIPTS_DIR / "verify_replication.sh"

#: Long enough for one ``quick`` Section 5.2 run in the test environment.
SMOKE_TIMEOUT = 900


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_script(
    script: Path,
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one script through bash and capture its output.

    ``REPLICATE_PYTHON`` points at the interpreter running the tests, so the
    tests never depend on ``pixi`` being on ``PATH``; the scripts default to
    ``pixi run python``, which is the runbook's launcher.
    """
    env = dict(os.environ)
    env["REPLICATE_PYTHON"] = sys.executable
    if environment is not None:
        env.update(environment)
    return subprocess.run(
        ["bash", str(script), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=SMOKE_TIMEOUT,
        check=False,
        env=env,
    )


@pytest.fixture
def output_root() -> Iterator[Path]:
    """An empty, git-ignored directory under ``output/``, removed afterwards."""
    root = REPO_ROOT / "output" / f"pytest-scripts-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def make_run_root(tmp_path: Path, *, section: str = "5.2") -> Path:
    """A run root with one plank figure, ready for ``finalize_run``."""
    root = tmp_path / "run"
    figure = root / section / "figures" / "figure_4.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"\x89PNG\r\n\x1a\nstored")
    return root


# ---------------------------------------------------------------------------
# The files themselves
# ---------------------------------------------------------------------------
def test_scripts_are_executable_and_parse() -> None:
    for script in (REPLICATE, VERIFY):
        assert script.is_file(), f"{script} is missing"
        assert (script.stat().st_mode & 0o111) != 0, f"{script} is not executable"
        lines = script.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#!/usr/bin/env bash"
        assert "set -euo pipefail" in lines
        parsed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_pixi_tasks_wrap_the_scripts() -> None:
    with (REPO_ROOT / "pixi.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    tasks = manifest["tasks"]
    assert tasks["replicate-5.2"] == "scripts/replicate.sh 5.2"
    assert tasks["replicate-5.1"] == "scripts/replicate.sh 5.1"
    assert tasks["verify-paper"] == "scripts/verify_replication.sh"


def test_replicate_runs_the_stages_in_the_plan_order() -> None:
    """§6.5: driver, report, a manifest, the replay, finalise, verify."""
    text = REPLICATE.read_text(encoding="utf-8")
    labels = [
        "stage 1/6 driver",
        "stage 2/6 report",
        "stage 3/6 finalize_run",
        "stage 4/6 artifact-mode pytest",
        "stage 5/6 finalize_run",
        "stage 6/6 verify_manifest",
    ]
    start = 0
    for label in labels:
        index = text.find(label, start)
        assert index >= 0, f"{label!r} is not announced in {REPLICATE.name}"
        start = index + len(label)

    # ... and each label is followed by the command it names, in the same
    # order, in the execution half of the script.
    body = text.split('stage "run root: $run_root"', 1)[1]
    assert body.count("finalize_run") >= 2, (
        "the manifest is written before the replay and again at the end"
    )
    for command in (
        "src.paper_examples",
        "src.paper_report",
        "--write",
        "finalize_run",
        "DP_PAPER_ARTIFACTS=",
        "DP_PAPER_PRESET=",
        "verify_manifest",
    ):
        assert command in body, f"{command!r} is not run in {REPLICATE.name}"
    # The replay is the committed acceptance module, in artifact mode only.
    assert "-m pytest" in body
    assert "-m paper" in body
    assert 'test_module="tests/test_paper_' in text


# ---------------------------------------------------------------------------
# Argument surface: fail closed, before any fitting
# ---------------------------------------------------------------------------
def test_replicate_usage_names_both_sections() -> None:
    completed = run_script(REPLICATE)
    assert completed.returncode == 2
    assert "usage:" in completed.stderr
    assert "5.1" in completed.stderr and "5.2" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_replicate_rejects_an_unknown_section(tmp_path: Path) -> None:
    completed = run_script(REPLICATE, "5.3", str(tmp_path / "run"))
    assert completed.returncode == 2
    assert "unknown section '5.3'" in completed.stderr
    assert "usage:" in completed.stderr


def test_replicate_refuses_to_create_a_given_run_root(tmp_path: Path) -> None:
    """A given root is reused (R6), never created: a typo must not start a run."""
    missing = tmp_path / "typo-root"
    completed = run_script(REPLICATE, "5.2", str(missing))
    assert completed.returncode == 1
    assert "does not exist" in completed.stderr
    assert "reused" in completed.stderr
    assert not missing.exists()


def test_verify_requires_an_existing_run_root(tmp_path: Path) -> None:
    completed = run_script(VERIFY, str(tmp_path / "nope"))
    assert completed.returncode == 1
    assert "is not a directory" in completed.stderr

    usage = run_script(VERIFY)
    assert usage.returncode == 2
    assert "usage:" in usage.stderr


def test_verify_refuses_a_root_without_a_manifest(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    completed = run_script(VERIFY, str(root))
    assert completed.returncode == 1
    assert "MANIFEST.json is missing" in completed.stderr


def test_verify_refuses_a_root_without_sections(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "env.txt").write_text("git_head: deadbeef\n", encoding="utf-8")
    paper_artifacts.finalize_run(root)

    completed = run_script(VERIFY, str(root), environment={"REPLICATE_ACCEPTANCE": "0"})
    assert completed.returncode == 1
    assert "no section directory" in completed.stderr


def test_verify_detects_a_tampered_tree(tmp_path: Path) -> None:
    root = make_run_root(tmp_path)
    figure = root / "5.2" / "figures" / "figure_4.png"
    paper_artifacts.finalize_run(root)
    assert paper_artifacts.verify_manifest(root).file_count == 1

    figure.write_bytes(b"\x89PNG\r\n\x1a\nflipped")
    completed = run_script(VERIFY, str(root), environment={"REPLICATE_ACCEPTANCE": "0"})
    assert completed.returncode == 1
    assert "figure_4.png" in completed.stderr
    assert "mismatch" in completed.stderr


# ---------------------------------------------------------------------------
# Smoke: a real run, from the driver to a verified manifest
# ---------------------------------------------------------------------------
def test_replicate_then_verify_round_trip_at_quick(output_root: Path) -> None:
    """The whole pipeline runs, and the verifier reproduces its figures.

    ``quick`` with one prior is a machinery check (see the module docstring),
    so the acceptance replay is skipped; every other stage runs, including the
    report stage that owns ``figures/`` and both ``finalize_run`` calls.  The
    verifier then re-renders both figures from the stored draws and compares
    them byte-for-byte, and its final integrity check proves the scratch
    re-render left the tree untouched.
    """
    run_root = output_root / "run"
    run_root.mkdir()
    replicate = run_script(
        REPLICATE,
        "5.2",
        str(run_root),
        environment={
            "REPLICATE_PRESET": "quick",
            "REPLICATE_PRIORS": "1",
            "REPLICATE_ACCEPTANCE": "0",
        },
    )
    assert replicate.returncode == 0, replicate.stderr

    section = run_root / "5.2"
    assert (run_root / "logs" / "env.txt").is_file()
    assert (run_root / "logs" / "run-5.2.log").is_file()
    assert (run_root / "logs" / "events-5.2.jsonl").is_file()
    assert not (run_root / "logs" / "pytest-paper-5.2.log").exists()
    assert (section / "meta.json").is_file()
    assert (section / "summary" / "acceptance.json").is_file()
    assert (section / "report" / "tables.md").is_file()
    assert sorted(path.name for path in (section / "figures").glob("*.png")) == [
        "figure_4.png",
        "figure_5.png",
    ]

    env_text = (run_root / "logs" / "env.txt").read_text(encoding="utf-8")
    for field in ("git_head:", "pixi_lock_sha256:", "nproc:", "python"):
        assert field in env_text

    manifest = json.loads((run_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in (
        "5.2/meta.json",
        "5.2/report/tables.md",
        "5.2/summary/acceptance.json",
        "5.2/figures/figure_4.png",
        "logs/env.txt",
        "logs/run-5.2.log",
    ):
        assert name in manifest["files"], f"{name} is not covered by MANIFEST.json"
    assert manifest["code_sha"] is not None, (
        "the run root must live inside the checkout so finalize_run can record "
        "the git SHA the acceptance replay compares against"
    )

    verify = run_script(
        VERIFY, str(run_root), environment={"REPLICATE_ACCEPTANCE": "0"}
    )
    assert verify.returncode == 0, verify.stderr
    assert verify.stdout.count("byte-identical") == 2
    assert json.loads((run_root / "MANIFEST.json").read_text(encoding="utf-8")) == (
        manifest
    )
