# Implementation Plan: Artifact-Backed Kottas (2002) Replication Run

Source: `docs/research/dp_gibbs_replication_artifacts_plan.md` (the `Plan`
column cites its section numbers)

Supporting documents:

- `docs/research/dp_gibbs_plan.md` — original implementation plan (settings
  §6.2, acceptance criteria §6.3, decisions §10)
- `docs/research/dp_gibbs_results.md` — existing results note; the run must
  reproduce its first-500-draw numbers before §5.1 starts
- `docs/research/dp_gibbs_code_guide.md`, `docs/research/dp_weibull_code_guide.md`
  — the modules being exercised and extended

Pre-work: branch from a clean `master` (the only uncommitted items are this
plan and `TODO.md`, committed first). Every step runs the gate
`pixi run test`, `pixi run lint`, `pixi run typecheck`, updates
`docs/research/dp_gibbs_code_guide.md` when it adds or changes a module, then
commits the message in column 2 verbatim. Artifacts under `output/` are never
committed.

## Steps

| # | Commit message | Logical unit | Key deliverables | Tests | Plan |
| --- | --- | --- | --- | --- | --- |
| 1 | feat: add artifact schema helpers and atomic npz writes | Artifact primitives | `src/paper_artifacts.py` module skeleton: `SCHEMA_VERSION`, `ArtifactError`, `ArtifactIntegrityError`, `slug()`, `_jsonable()`, and the fsync + `os.replace` npz write helper; `.gitignore` entry for `output/`; code-guide section | Unit, Property-based | §6.1 |
| 2 | feat: add DPChains artifact round-trip | Chain serialisation | `src/paper_artifacts.py` `save_chain(path, chain, *, t_grid)` / `load_chain(path)` with the self-contained per-chain `meta` blob; shape/dtype/schema validation before `DPChains` construction; `tests/test_paper_artifacts.py` | Unit, Property-based | §6.1 |
| 3 | feat: add Functionals artifact round-trip | Functional serialisation | `src/paper_artifacts.py` `save_functionals` / `load_functionals`, arrays plus the meta dict verbatim; tests for transposed `S` rejection; `tests/test_paper_artifacts.py` | Unit, Property-based | §6.1 |
| 4 | feat: add Fit and DemoResult artifact round-trip | Fit and run serialisation | `src/paper_artifacts.py` `save_fit` / `load_fit` (un-slugged `label`, `arm`, both meta dicts, prior band re-attach) and `save_demo_result` / `load_demo_result` (`prior_indices` as a tuple); compression measurement recorded as R8's outcome in plan §2; quick-preset round-trip feeding `r_hat()`/`ess()` | Unit, Integration | §6.1, §5.1 |
| 5 | feat: add run event log and section artifact writer | Incremental writer | `src/paper_artifacts.py` `ArtifactWriter` (`fit_done`, `band_done`, `event`, `write_meta`) and the append-only `logs/events-<section>.jsonl` event stream with `ts`/`t_monotonic`/`kind`; `tests/test_paper_artifacts.py` | Unit | §6.1 |
| 6 | feat: add run manifest finalisation and integrity verification | Run manifest | `src/paper_artifacts.py` `finalize_run` (single run-level `MANIFEST.json`, hashes every section file, never lists itself) and `verify_manifest` (flipped byte, truncated, missing, unlisted extra, schema version); `tests/test_paper_artifacts.py` | Unit, Property-based | §6.1 |
| 7 | feat: persist replication artifacts | Driver and CLI wiring | `src/paper_examples.py` `save_dir` on `example_5_1` / `example_5_2`, per-fit save before the next fit, no driver-written figures under `save_dir`, arm B in its own directory; CLI `--save-dir`, `--out` default `None`, tracked-path rejection; `tests/test_cli.py`; code guide | Unit, Smoke | §6.2 |
| 8 | feat: add artifact report summary payloads | Report payloads | `src/paper_report.py` `summarize_5_1`, `summarize_5_2`, `diagnostics` (full draws and the 500-draw slice via `dataclasses.replace` with full-trace `J`), `delta_vs_baseline`, acceptance target constant mirroring the paper tests; `docs/research/dp_gibbs_results_baseline.json`; `tests/test_paper_report.py` | Unit | §6.3 |
| 9 | feat: add artifact report renderer and CLI | Report rendering | `src/paper_report.py` `render_markdown`, `write_report_artifacts` (`summary/stats.json`, `rhat_ess.csv`, `quantiles.csv`, `acceptance.json`, `report/tables.md`, fixed-metadata PNG figures in `<section>/figures/`), `main` (`--from-artifacts`, `--write`, `--baseline`), lazy `_pyplot`; byte-identical figure test; code guide | Unit, Integration | §6.3 |
| 10 | test: replay the paper acceptance suite from saved artifacts | Artifact-mode fixtures | `tests/test_paper_5_1.py`, `tests/test_paper_5_2.py`: `DP_PAPER_ARTIFACTS` fixture branch, manifest `code_sha` vs `HEAD` test with `DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH`, artifact-mode counts 12/12 and 8/8 | Integration | §6.4 |
| 11 | chore: add replication and verification scripts | Run scripts and tasks | `scripts/replicate.sh` (env log, driver, report, artifact-mode pytest, `finalize_run`, `verify_manifest`; reuses a given run root), `scripts/verify_replication.sh` (no fitting; scratch re-render compared byte-for-byte), `pixi.toml` tasks `replicate-5.2`, `replicate-5.1`, `verify-paper` | Smoke | §6.5 |
| 12 | docs: add artifact-backed Kottas replication report | Replication report | `docs/research/dp_gibbs_replication_report.md`: generated tables transcribed from the run's `report/tables.md` plus the §8 prose sections; one-line pointer at the top of `docs/research/dp_gibbs_results.md` | — | §7 Step 4, §8 |
| 13 | docs: record the artifact run in the implementation plan | Plan amendment | `docs/research/dp_gibbs_plan.md` §10 decision 8 amended with the new output paths, run stamp and manifest hash | — | §7 Step 4 |

## Runbook (no commits)

Steps 1–11 are implemented and committed first. The long compute stages below
produce only git-ignored artifacts under `output/`, so they are not table rows;
they are the gate between step 11 and step 12. Full commands are in plan §7.

1. **Pre-flight + dry run** (plan §7 Step 0, after steps 1–11): check clean
   tree, free disk, RAM, `nproc` and package versions; then run the
   `quick`-preset smoke chain
   (`--save-dir output/smoke/5.2` → `paper_report --write` → `finalize_run` →
   `DP_PAPER_ARTIFACTS=output/smoke/5.2 DP_PAPER_PRESET=quick pytest -m paper
   tests/test_paper_5_2.py`). All three stages must run; delete `output/smoke`.
2. **§5.2 paper run** (~4 h, plan §7 Step 1):
   `scripts/replicate.sh 5.2` under `systemd-inhibit`/`tmux`; expect 4 fit
   directories, 16 chain files, 2 figures and a green acceptance log.
3. **§5.2 gate** (plan §7 Step 2): compare the first-500 column against
   `dp_gibbs_results.md`; the prior-band width and `n_eta_above_grid` rows are
   `not_comparable`; do not start §5.1 on a comparable mismatch.
   *Gate outcome (2026-11-08, run `replication-2026-11-08T02-26-53Z`):* the
   chains replay exactly (all seven fits' first-500 R̂/ESS match the note to
   its displayed precision), but the first-500 *summaries* did not: the column
   sliced the stored ``Functionals`` and so kept the 10 000-draw trace's
   `J_common`, which left every Figure 4-5 row 0.5-1 % off. Recomputing the
   functionals from the truncated chains (the slice's own `J_common`)
   reproduces all 16 comparable rows. Fixed in `src/paper_report.py`
   (`slice_result` now recomputes; `diagnostics` records `J_common_slice`),
   commit `fix: recompute slice functionals for the first-500 column`; the
   §5.2 report artifacts were re-rendered from the same draws.
   *§5.1 gate (2026-11-09, same run root):* 28/28 acceptance clauses pass and
   19 of 20 comparable first-500 rows match the note to displayed precision.
   The exception is `figure_1.nu_prior_95_width`: the report uses the exact
   gamma quantiles (5.92 / 53.29 / 132.12), the note's 6.1 / 61 / 117 are
   sample estimates from the probe's 200 prior draws (72nd / 95th / 14th
   percentile of that Monte Carlo distribution). Marked `not_comparable` with
   the reason in the baseline JSON, commit `fix: mark the note's prior-width
   row not comparable`; all draws-based rows are unaffected.
4. **§5.1 paper run** (~8–12 h overnight, plan §7 Step 3):
   `scripts/replicate.sh 5.1 output/replication-<stamp>` reusing Step 1's root
   (R6); `MANIFEST.json` is rewritten to cover both sections.
5. **Verify** (plan §7 Step 4): `scripts/verify_replication.sh
   output/replication-<stamp>` exits 0 without fitting, then write the report
   for step 12 from `output/.../report/tables.md`.

## Notes

- Steps 2–6 split plan §6.1's single module into independently testable units
  (primitives → chain → functionals → fit/run → writer → manifest); steps 8–9
  split §6.3 into payload computation and rendering/CLI. The plan's §6.2, §6.4
  and §6.5 commit messages are kept as written.
- The one-draw / None-One-Many edge cases (`n_save=1`, one chain, one grid
  point, all-censored data) belong to the step that introduces the affected
  container, not a later catch-all test.
- Step 12 requires the completed paper runs; it must not be started with
  `quick` artifacts or with a `not_comparable` gate failure.
- Artifacts and figures are never committed; the committed record is the code,
  the report, its manifest reference and the checksums (R5).
