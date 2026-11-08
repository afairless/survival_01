#!/usr/bin/env bash
#
# Verify a finished replication tree without fitting anything.
#
# Plan: docs/research/dp_gibbs_replication_artifacts_plan.md §6.5 and §9.
#
#   scripts/verify_replication.sh <run_root>
#
# The reviewer entry point.  Three layers, in this order:
#
#   1. integrity -- ``verify_manifest`` re-hashes every file the run-level
#      ``MANIFEST.json`` lists, so a flipped byte, a truncated file, a missing
#      file or an unlisted extra is named and the script exits non-zero;
#   2. recomputation -- each section's figures are re-rendered from the stored
#      draws **into a scratch directory** and compared byte-for-byte with the
#      tree's ``figures/``.  The scratch copy is deliberate: writing into the
#      verified tree would invalidate the manifest that was just checked;
#   3. acceptance replay -- ``pytest -m paper`` for every section present, in
#      artifact mode (``DP_PAPER_ARTIFACTS`` plus the preset recorded in the
#      section's ``meta.json``), which loads the saved draws instead of
#      refitting.  A plain refit gives 11/11 and 7/7; artifact mode gives
#      12/12 and 8/8, the extra assertion being the manifest ``code_sha``
#      check against this checkout's ``HEAD``.
#
# The manifest is verified again at the end, which proves that neither the
# re-rendering nor the replay wrote anything into the verified tree.
#
# Environment (both optional):
#
#   REPLICATE_ACCEPTANCE 0 skips layer 3, for a machinery smoke run only
#   REPLICATE_PYTHON     the interpreter launcher (default: ``pixi run python``)
#
# ``DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH=1`` is inherited by the replay, so an
# older run can be inspected from a newer checkout without pretending its
# draws came from ``HEAD``.
#
# Exits 0 only when all three layers pass.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/verify_replication.sh <run_root>

Verify a replication run without fitting: MANIFEST.json integrity, figures
re-rendered byte-for-byte into a scratch directory, and the artifact-mode
acceptance replay for every section under the root.

Environment:
  REPLICATE_ACCEPTANCE 0 skips the acceptance replay (smoke runs only)
  REPLICATE_PYTHON     interpreter launcher (default: pixi run python)
EOF
}

if [ "$#" -ne 1 ]; then
    usage >&2
    exit 2
fi

run_root="$1"
if [ ! -d "$run_root" ]; then
    echo "error: run root '$run_root' is not a directory" >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Unquoted on purpose: this is a command with arguments ("pixi run python").
# shellcheck disable=SC2086
python_cmd="${REPLICATE_PYTHON:-pixi run python}"
acceptance="${REPLICATE_ACCEPTANCE:-1}"

run_root="$(cd -- "$run_root" && pwd)"
manifest="$run_root/MANIFEST.json"
if [ ! -f "$manifest" ]; then
    echo "error: $manifest is missing; finalise the run with finalize_run" \
        "before verifying it" >&2
    exit 1
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/verify-replication-XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

stage() {
    printf '%s [verify] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
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

# Re-render one section's figures into a scratch tree.  ``_render_figures`` is
# the report stage's own chooser (plan R7: one writer per artifact), so what is
# compared here is exactly what produced the committed figures.
render_figures() {
    # shellcheck disable=SC2086
    $python_cmd - "$1" "$2" <<'PY'
import sys
from pathlib import Path

from src.paper_artifacts import load_demo_result
from src.paper_report import _render_figures

section, target = Path(sys.argv[1]), Path(sys.argv[2])
_render_figures(target, load_demo_result(section))
PY
}

# The preset is recorded in the section's ``meta.json`` under ``meta`` (the
# driver's ``DemoResult.meta`` verbatim); ``DP_PAPER_PRESET`` must name the same
# preset or the replay rejects the run's settings.
preset_of() {
    # shellcheck disable=SC2086
    $python_cmd -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["meta"]["preset"])' "$1"
}

stage "1/3 manifest integrity: $manifest"
verify_manifest

# Section trees are the directories named after a paper section that carry a
# ``meta.json``; everything else under the root is a log directory.
section_dirs=()
for entry in "$run_root"/*/; do
    name="$(basename "$entry")"
    case "$name" in
        5.1 | 5.2) ;;
        *) continue ;;
    esac
    if [ -f "$entry/meta.json" ]; then
        section_dirs+=("$entry")
    fi
done
if [ "${#section_dirs[@]}" -eq 0 ]; then
    echo "error: $run_root holds no section directory (5.1/ or 5.2/ with a" \
        "meta.json); there is nothing to verify" >&2
    exit 1
fi

for section_dir in "${section_dirs[@]}"; do
    name="$(basename "$section_dir")"
    stage "2/3 figure re-render: section $name -> $scratch/$name"
    render_figures "$section_dir" "$scratch/$name"

    shopt -s nullglob
    stored=("$section_dir"/figures/*.png)
    rendered=("$scratch/$name"/figures/*.png)
    shopt -u nullglob
    if [ "${#stored[@]}" -eq 0 ]; then
        echo "error: $section_dir/figures holds no PNG; the report stage did" \
            "not run for this section" >&2
        exit 1
    fi
    if [ "${#stored[@]}" -ne "${#rendered[@]}" ]; then
        echo "error: section $name stored ${#stored[@]} figure(s) but the" \
            "stored draws render ${#rendered[@]}" >&2
        exit 1
    fi
    for png in "${stored[@]}"; do
        figure="$(basename "$png")"
        if ! cmp -s "$png" "$scratch/$name/figures/$figure"; then
            echo "error: re-rendered $name/figures/$figure differs from the" \
                "stored PNG; the figures are not reproducible from the draws" >&2
            exit 1
        fi
        stage "2/3 $name/figures/$figure: byte-identical"
    done
done

if [ "$acceptance" = "0" ]; then
    stage "3/3 acceptance replay: skipped (REPLICATE_ACCEPTANCE=0)"
else
    for section_dir in "${section_dirs[@]}"; do
        name="$(basename "$section_dir")"
        preset="$(preset_of "$section_dir/meta.json")"
        test_module="tests/test_paper_$(printf '%s' "$name" | tr '.' '_').py"
        stage "3/3 acceptance replay: section $name (preset $preset)"
        # shellcheck disable=SC2086
        DP_PAPER_ARTIFACTS="$section_dir" \
            DP_PAPER_PRESET="$preset" \
            $python_cmd -m pytest -m paper "$test_module" -q
    done
fi

stage "integrity re-check: neither stage may have written into the tree"
verify_manifest

stage "ok: $run_root"
