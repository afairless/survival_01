#!/usr/bin/env bash
#
# One Kottas (2002) replication stage, from the driver to a verified manifest.
#
# Plan: docs/research/dp_gibbs_replication_artifacts_plan.md §6.5 and §7.
#
#   scripts/replicate.sh <5.1|5.2> [run_root]
#
# With no ``run_root`` the script stamps ``output/replication-<UTC>``.  With
# one it **reuses the existing root** (plan R6): that is how the §5.1 run joins
# the §5.2 tree that Step 1 created, so the run-level ``MANIFEST.json`` ends up
# covering both sections.  A given root is never created -- a mistyped path
# would otherwise produce a manifest over an empty run.
#
# Stages, in order, each exiting non-zero on failure (``set -euo pipefail``):
#
#   1. ``logs/env.txt`` -- git SHA/dirty flag, pixi lock digest, package
#      versions, host, cores, memory, disk (appended, so a reused root records
#      both stages);
#   2. the driver, ``python -m src.paper_examples --save-dir <root>/<section>``,
#      teeing to ``logs/run-<section>.log``;
#   3. the report stage, ``python -m src.paper_report --write``, which is the
#      only writer of the section's ``figures/`` (plan R7);
#   4. ``finalize_run`` -- early, and again at the end: the artifact-mode
#      acceptance suite asserts that the manifest's ``code_sha`` is this
#      checkout's ``HEAD``, so a manifest must exist before the replay;
#   5. ``pytest -m paper tests/test_paper_5_<n>.py`` with
#      ``DP_PAPER_ARTIFACTS`` and ``DP_PAPER_PRESET`` set, teeing to
#      ``logs/pytest-paper-<section>.log`` -- no fitting, the saved draws are
#      replayed;
#   6. ``finalize_run`` again (now the manifest also covers the pytest log) and
#      ``verify_manifest``, so the run cannot finish with a broken tree.
#
# Environment (all optional):
#
#   REPLICATE_PRESET     run length preset (default: paper)
#   REPLICATE_PRIORS     --prior value: 1, 2, 3 or all (default: all)
#   REPLICATE_ACCEPTANCE 0 skips stage 5, for a machinery smoke run only: the
#                        quick preset and a single prior do not satisfy the
#                        acceptance clauses, so a smoke run must not claim to
#                        have replayed them
#   REPLICATE_PYTHON     the interpreter launcher (default: ``pixi run python``)
#
# Examples:
#
#   scripts/replicate.sh 5.2                                   # ~4 h, §5.2
#   scripts/replicate.sh 5.1 output/replication-2026-11-08T00-00-00Z
#                                                              # reuse (R6)
#
# Artifacts live under the git-ignored ``output/`` tree (plan R5); nothing
# here writes a tracked path.
set -euo pipefail

readonly SECTION_CHOICES="5.1 5.2"
readonly DEFAULT_PRESET="paper"
readonly DEFAULT_PRIORS="all"

usage() {
    cat <<'EOF'
usage: scripts/replicate.sh <5.1|5.2> [run_root]

Run one Kottas (2002) replication stage: environment log, driver, report,
artifact-mode acceptance replay, finalize_run and verify_manifest.

With no run_root the root is stamped as output/replication-<UTC>; with one an
existing root is reused and joined (plan R6).

Environment:
  REPLICATE_PRESET     run length preset (default: paper)
  REPLICATE_PRIORS     --prior value: 1, 2, 3 or all (default: all)
  REPLICATE_ACCEPTANCE 0 skips the acceptance replay (smoke runs only)
  REPLICATE_PYTHON     interpreter launcher (default: pixi run python)
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage >&2
    exit 2
fi

section="$1"
case " $SECTION_CHOICES " in
    *" $section "*) ;;
    *)
        echo "error: unknown section '$section'; expected one of: $SECTION_CHOICES" >&2
        usage >&2
        exit 2
        ;;
esac

# Resolve the repository root from this script's own location, so the script
# works from any working directory and always logs the right checkout.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

preset="${REPLICATE_PRESET:-$DEFAULT_PRESET}"
priors="${REPLICATE_PRIORS:-$DEFAULT_PRIORS}"
# Unquoted on purpose: this is a command with arguments ("pixi run python"),
# not a single path.
# shellcheck disable=SC2086
python_cmd="${REPLICATE_PYTHON:-pixi run python}"
acceptance="${REPLICATE_ACCEPTANCE:-1}"

if [ "$#" -eq 2 ]; then
    run_root="$2"
    if [ ! -d "$run_root" ]; then
        echo "error: run root '$run_root' does not exist; a given root is" \
            "reused, never created (plan R6)" >&2
        exit 1
    fi
else
    stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
    run_root="output/replication-$stamp"
    if [ -e "$run_root" ]; then
        echo "error: run root '$run_root' already exists; pass it explicitly" \
            "to reuse it (plan R6)" >&2
        exit 1
    fi
    mkdir -p "$run_root"
fi

run_root="$(cd -- "$run_root" && pwd)"
log_dir="$run_root/logs"
section_dir="$run_root/$section"
mkdir -p "$log_dir"
run_log="$log_dir/run-$section.log"
pytest_log="$log_dir/pytest-paper-$section.log"
env_log="$log_dir/env.txt"
test_module="tests/test_paper_$(printf '%s' "$section" | tr '.' '_').py"

# ``stage`` records a stage in both the console and the run log; ``say``
# writes to the console only.  Everything that touches the run log must happen
# *before* the final ``finalize_run``: the manifest hashes the log, so a line
# appended afterwards would make the tree it just described stale.  That is
# also why the finalise/verify stages below use ``say``.
stage() {
    printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$section" "$*" |
        tee -a "$run_log"
}

say() {
    printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$section" "$*"
}

write_env_log() {
    local dirty
    {
        echo "# --- section $section: $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
        echo "run_root: $run_root"
        echo "section: $section"
        echo "preset: $preset"
        echo "priors: $priors"
        echo "python_command: $python_cmd"
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        dirty="$(git status --porcelain 2>/dev/null | wc -l)"
        echo "git_dirty_paths: $dirty"
        if [ -f pixi.lock ]; then
            echo "pixi_lock_sha256: $(sha256sum pixi.lock | cut -d' ' -f1)"
        else
            echo "pixi_lock_sha256: unknown"
        fi
        echo "host: $(hostname 2>/dev/null || echo unknown)"
        echo "platform: $(uname -srm)"
        echo "nproc: $(nproc 2>/dev/null || echo unknown)"
        if [ -r /proc/meminfo ]; then
            echo "mem_total: $(awk '/^MemTotal:/ {print $2 " kB"}' /proc/meminfo)"
        fi
        echo "disk:"
        df -h . 2>/dev/null | sed 's/^/  /'
        echo "versions:"
        # shellcheck disable=SC2086
        $python_cmd -c 'import platform, numpy, scipy, matplotlib
print("  python", platform.python_version())
print("  numpy", numpy.__version__)
print("  scipy", scipy.__version__)
print("  matplotlib", matplotlib.__version__)'
    } >>"$env_log"
}

# One small ``-c`` program per manifest call: the artifact helpers live in
# Python, and the shell must not re-implement hashing or validation.
finalize_run() {
    # shellcheck disable=SC2086
    $python_cmd -c 'import sys
from src.paper_artifacts import finalize_run
print("manifest:", finalize_run(sys.argv[1]))' "$run_root"
}

verify_manifest() {
    # shellcheck disable=SC2086
    $python_cmd -c 'import sys
from src.paper_artifacts import verify_manifest
report = verify_manifest(sys.argv[1])
print(
    "verified:",
    report.file_count,
    "files,",
    report.total_bytes,
    "bytes, code_sha",
    report.code_sha,
)' "$run_root"
}

stage "run root: $run_root"
write_env_log
stage "wrote environment log: $env_log"

stage "stage 1/6 driver: --example $section --preset $preset --prior $priors" \
    "--save-dir $section_dir"
# shellcheck disable=SC2086
$python_cmd -u -m src.paper_examples \
    --example "$section" \
    --preset "$preset" \
    --prior "$priors" \
    --save-dir "$section_dir" 2>&1 | tee -a "$run_log"

stage "stage 2/6 report: --from-artifacts $section_dir --write"
# shellcheck disable=SC2086
$python_cmd -m src.paper_report \
    --from-artifacts "$section_dir" --write 2>&1 | tee -a "$run_log"

stage "stage 3/6 finalize_run (pre-replay: the replay checks code_sha)"
finalize_run | tee -a "$run_log"

if [ "$acceptance" = "0" ]; then
    stage "stage 4/6 artifact-mode pytest: skipped (REPLICATE_ACCEPTANCE=0;" \
        "a smoke run must not claim an acceptance replay)"
else
    stage "stage 4/6 artifact-mode pytest: $test_module (preset $preset)"
    # shellcheck disable=SC2086
    DP_PAPER_ARTIFACTS="$section_dir" \
        DP_PAPER_PRESET="$preset" \
        $python_cmd -m pytest -m paper "$test_module" -q 2>&1 | tee "$pytest_log"
fi

stage "stage 5/6 finalize_run (covers every file above, including the log)"
finalize_run

say "stage 6/6 verify_manifest"
verify_manifest

say "done: $run_root"
