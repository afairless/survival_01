# Plan: Artifact-Backed Kottas (2002) Replication Run

Status: proposed, not started.
Scope: re-run the Section 5.1 / 5.2 reproduction at the `paper` preset with the
committed seeds, persist every stored draw and every summary to a git-untracked
`output/` directory, and produce a new results report from those artifacts.

Related documents:

- `docs/research/dp_gibbs_plan.md` — the implementation plan (settings §6.2,
  acceptance criteria §6.3, decisions §10).
- `docs/research/dp_gibbs_results.md` — the existing results note; it records
  numbers from the first 500 stored draws of each chain, not from the full
  draws, and no artifacts were kept.
- `docs/research/dp_weibull_code_guide.md`, `docs/research/dp_gibbs_code_guide.md`
  — the code being exercised.

## 1. Why this run

The existing note is good prose but it is not evidence that can be audited: it
carries no draws, no logs, no checksums and no figures, and the figures
directory it names (`docs/research/figures/`) does not exist. The only survivor
of the accepted §5.2 run is a pair of PNGs in a pytest temp directory
(`/tmp/pytest-of-tr/pytest-104/section_5_2_paper0/`).

This run is meant to answer "show me": given a git SHA, a seed table and a
pixi lock, a reviewer must be able to (a) see the chains, (b) see the figures
those chains produce, (c) recompute every acceptance value without fitting
anything, and (d) re-run the acceptance suite against the saved draws instead
of against a fresh 12-hour fit.

## 2. Decisions taken (this plan)

| # | Decision | Consequence |
| --- | --- | --- |
| R1 | `paper` preset, staged: §5.2 first (~4 h), then §5.1 (~8–12 h) | Save/report pipeline is validated on the short section before the long one. |
| R2 | Reuse the committed seeds (`SEED_5_1=20240601`, `SEED_5_2=20240602`, `DATA_SEED_5_1=721383`) | Deterministic replay: the chains behind `dp_gibbs_results.md` are regenerated, so the new report can carry a delta table against it. |
| R3 | Persist everything: every stored draw (θ, ν, γ, φ, n\*, log-post), all per-draw functionals (S, F, f, h), prior bands, plus summaries, figures, logs | ≈2.5–3 GB under `output/`. Disk has 53 GB free. Dropped arrays cannot be recovered without another 12-hour run. |
| R4 | Report = script-generated tables + curated prose | Numbers and tables are produced by code from the artifacts; the narrative lives in a committed markdown file. |
| R5 | `output/` is git-ignored and never committed | Binaries and draws stay local; the committed record is the code, the report, its manifest reference and the checksums. |
| R6 | One run root for both sections; Step 3 reuses Step 1's `run_root` | A single `MANIFEST.json` covers 7 fits and both section trees, so the report has one path and one hash to cite. |
| R7 | Figures are written **only** by the report stage, from the stored draws; PNG only, no PDF | One writer per artifact: the manifest hashes exactly what a reviewer sees. PDF is dropped because matplotlib embeds a creation date, which breaks byte-level determinism and serves no reviewer purpose. |
| R8 | `savez` vs `savez_compressed` is settled by a measurement in 6.1 and recorded here; `--reuse-complete` is out of scope | Per-fit saves already bound the loss of an interrupted run, so resumption is not needed for acceptance. Compression is a measured size/CPU trade, not an open question. **Outcome (measured in the fit/run unit, step 4):** one `quick` §5.1 fit (2 chains × 1 000 draws × 200 observations plus the `(2 000, 300)` functionals, 24.51 MiB uncompressed) wrote in 0.04 s with `np.savez` and 0.61 s with `np.savez_compressed`, at 24.51 MiB (25 697 642 bytes) vs 17.75 MiB (18 615 305 bytes) — 27.6 % fewer bytes for 0.57 s of extra CPU (~25 ms/MiB, ≈12 s per paper §5.1 fit against a ~1 h fit). The ≥ 5 % rule passes, so `savez_compressed` stays the default of `atomic_savez`. |

These amend `dp_gibbs_plan.md` §10 decision 8 ("figures go to the git-ignored
`docs/research/figures/`; numbers go in `dp_gibbs_results.md`") in one respect:
the reproduction now writes a *complete* run directory under `output/`, and a
new report (`dp_gibbs_replication_report.md`) supersedes the note as the
artifact-backed record. The old note stays as history and gains a one-line
pointer at the top.

## 3. Non-goals

- No changes to the sampler, the prior constants, the time transform, the
  presets, the data or any acceptance threshold. If a criterion fails, that is
  a finding in the report, not a reason to edit `src/`.
- No model comparison (`log_post` stays a diagnostic, plan decision 4).
- No claim of pixel-identical reproduction of the paper's figures; the standing
  caveats of `dp_gibbs_results.md` §6 still apply.
- No artifact storage in git, no Git LFS, no external object store.

## 4. Current state and gaps

What exists today (`src/paper_examples.py`, 958 lines):

- `preset` table (`quick`, `paper`, `paper-lite`); `paper` = burn-in 5 000,
  `n_save` 10 000, thin 150, 4 chains, `n_alpha` 256, `n_t_grid` 300.
- `fit_dataset()` → `Fit` (chains + `Functionals`), `example_5_1()`,
  `example_5_2()`, the five figure recipes, CLI `--example/--preset/--prior/
  --seed/--out/--max-workers`.
- `DPChains` (`src/dp_gibbs.py` §3128) with `theta (B, n, 2)`, `nu`, `gamma`,
  `phi`, `n_star`, `log_post`, plus `data`, `prior`, `settings`. At the paper
  preset θ is 32 MB per §5.1 chain and 3.2 MB per §5.2 chain.
- `Functionals` (`src/dp_functionals.py` §228) with `t_grid`, `S`, `F`, `f`,
  `h`, `meta`. Per fit the four arrays are `(4 × n_save, 300)` = 384 MB.
- `r_hat()` and `ess()` accept saved `DPChains`, so diagnostics are recomputable.
- `compare()` returns the whole Figure 5 payload (hazard ratio, CDF difference,
  η draws, `prob_eta_a_gt_eta_b`) plus `meta["n_eta_above_grid"]`, the per-fit
  counts of quantiles censored to `+inf`.
- `DPChains.__post_init__` enforces `n_draws == settings.n_save`, so a
  "first 500 draws" slice of a loaded chain must also replace the settings
  (`dataclasses.replace(settings, n_save=500)`); slicing the arrays alone is
  rejected at the storage boundary.
- `pixi run test` excludes `paper`; the acceptance modules are
  `tests/test_paper_5_1.py` (11 assertions) and `tests/test_paper_5_2.py`
  (7 assertions), both with a module-scoped fixture that refits.

What is missing:

1. No serialisation of `Fit`/`DPChains`/`Functionals`/`DemoResult`; nothing is
   written except PNGs.
2. No run manifest: no git SHA, seeds, timings, checksums or environment record.
3. No way to run the acceptance suite against saved draws; today a
   `pytest -m paper` "verification" costs another full fit.
4. No reporting code: the old note's numbers were computed by ad-hoc scripts in
   `/tmp` (e.g. `/tmp/probe_lite.py`), so the note cannot be regenerated.
5. No `.gitignore` entry for `output/`.

## 5. Artifact layout

One directory per replication, timestamped in UTC, with a section subdirectory
each. `output/` is ignored by git.

```text
output/
  replication-2026-11-08T00-00-00Z/
    MANIFEST.json                  # run-level: SHA, seeds, env, checksums, timings
    logs/
      env.txt                      # git SHA/dirty flag, pixi lock sha, python/numpy/scipy,
                                   # matplotlib, hostname, nproc, MemTotal, df output
      run-5.2.log                  # driver stdout+stderr, timestamped
      run-5.1.log
      events-5.2.jsonl             # structured per-fit/per-stage events
      events-5.1.jsonl
      pytest-paper-5.2.log         # acceptance suite replayed against artifacts
      pytest-paper-5.1.log
      pytest-unit.log              # `pixi run test`, lint, typecheck (short)
    5.2/
      meta.json                    # DemoResult.meta + preset + seeds + timings + code SHA
      draws/
        A-Gamma_2_0.9/
          chain-0.npz ... chain-3.npz
          functionals.npz
          fit-meta.json            # prior, settings, data, grid, seeds, shapes, dtypes
        A-Gamma_2_0.1/ ...
        A-Gamma_3_0.05/ ...
        B-Gamma_2_0.9/ ...
      summary/
        stats.json                 # every §6.3 value, full draws and first-500 draws
        rhat_ess.csv
        quantiles.csv              # per-fit ν, n*, median quantiles at 2.5/50/97.5%
        acceptance.json            # clause -> target, observed, pass/fail
      figures/
        figure_4.png, figure_5.png
      report/
        tables.md                  # generated tables (source for the committed report)
    5.1/
      ... same shape, figures 1-3, plus one prior-band.npz under the reference
      (first-selected) prior's fit directory
```

Naming: `slug(label)` replaces every run of characters outside `[A-Za-z0-9._-]`
with one `_`, so `"nu ~ Gamma(2, 0.9)"` → `nu_Gamma_2_0.9`; arms are prefixed,
giving `A-nu_Gamma_2_0.9`, `B-nu_Gamma_2_0.9`. The slug is **not** inverted:
`fit-meta.json` stores the original `label` and `arm`, and `load_demo_result`
keys `DemoResult.fits` from those, which is exactly what the paper tests index
(`"nu ~ Gamma(2, 0.9)"`, `"A: nu ~ Gamma(2, 0.9)"`).

The §5.1 prior band belongs to the first selected prior (as in `example_5_1`);
it is stored once, in that fit's directory, and re-attached on load as
`fits[prior_label(first_index)].prior_band`.

There is **one** run root for both sections (R6): `scripts/replicate.sh 5.1
<run_root>` is given the root Step 1 created, and `MANIFEST.json` is written
once, at the end of the §5.1 run, covering both section trees.

### 5.1 Serialisation schema (`np.savez_compressed`)

`chain-<b>.npz`:

| key | dtype | shape | source |
| --- | --- | --- | --- |
| `theta` | float64 | `(n_save, n, 2)` | `chain.theta` |
| `nu` | float64 | `(n_save,)` | `chain.nu` |
| `gamma` | float64 | `(n_save,)` | `chain.gamma` |
| `phi` | float64 | `(n_save,)` | `chain.phi` |
| `n_star` | int64 | `(n_save,)` | `chain.n_star` |
| `log_post` | float64 | `(n_save,)` | `chain.log_post` |
| `time` | float64 | `(n,)` | `chain.data.time` (transformed scale) |
| `event` | bool | `(n,)` | `chain.data.event` |
| `t_max` | float64 | `()` | `chain.data.t_max` |
| `t_grid` | float64 | `(n_t_grid,)` | the fit's grid, passed in by `save_fit` |
| `meta` | `<U` (JSON string) | `()` | `schema_version`, `label`, `chain_index`, `prior`, `settings` |

`functionals.npz`: `t_grid`, `S`, `F`, `f`, `h` (each `(n_chains*n_save, 300)`,
float64). `prior-band.npz`: the same five array keys (a `Functionals` returned by
`prior_functionals` holds `F` too) plus its own `meta`.

`fit-meta.json` (per fit): `label` (the un-slugged `prior_label()` string),
`arm`, `prior` (7 scalars), `settings` (`burn_in`, `n_save`, `thin`,
`n_alpha`, `epsilon`, `n_chains`, `seed`, `preset`), `data_seed` (null for
§5.2), `shared_t_max`, `t_grid` size and endpoints, `n_draws = n_chains *
n_save`, array dtypes/shapes, and **`functionals_meta` / `prior_band_meta`
stored verbatim** (whatever keys the object holds: `method`, `J_common`,
`n_draws`, `n_chains`, `seed`, `chain_seeds`, `t_max`, `atom_tol`, `epsilon`
for the posterior; `sample`, `method`, `J_prior`, `max_nu`, `epsilon`,
`atom_tol`, `seed` for the band). The schema does not enumerate meta keys;
it round-trips the dict and lets `Functionals` validate the arrays.

Each `chain-<b>.npz` also carries a `meta` key: a 0-d `<U` string holding the
JSON of `schema_version`, `label`, `chain_index`, `prior` and `settings`, so
`load_chain` really is self-contained. `t_grid` is a fit field, not a
`DPChains` field: `save_chain(path, chain, *, t_grid)` takes it as an explicit
argument, and `save_fit` passes the fit's grid.

`meta.json` (per section) and `MANIFEST.json` (per run) carry `schema_version`,
`code_sha`, `code_dirty`, `pixi_lock_sha256`, package versions, `preset`, seed
table (base seed, per-chain seeds `seed+b`), data seed, timings, file sizes and
SHA-256 per artifact, and the wall-clock start/end with ISO timestamps.

Loading is by explicit reconstruction, **not pickle**: `SurvData(time, event,
t_max)`, `DPPrior(**scalars)`, `DPSettings(**scalars)`, `DPChains(...)`,
`Functionals(t_grid, S, F, f, h, meta)`. `np.load(..., allow_pickle=False)` is
explicit, and every array's shape/dtype is checked against the declared schema
**before** the container is built (so a corrupt header cannot allocate
arbitrarily). The containers then validate in `__post_init__`.

Writes are atomic, but not by handing a `.tmp` name to `np.savez_compressed`:
numpy appends `.npz` to names that lack it, so `chain-0.npz.tmp` would become
`chain-0.npz.tmp.npz`. Instead each writer opens `path.with_suffix(path.suffix
+ ".tmp")` as a binary file object, passes the handle to `numpy.savez`, calls
`flush()` + `os.fsync()`, closes, then `os.replace(tmp, path)`. A crashed save
never leaves a readable-but-wrong file, and never leaves a stray `*.npz.npz`.

## 6. Code changes

Five small, separately committed units. Every unit ends with the gate
`pixi run test && pixi run lint && pixi run typecheck` (per `TODO.md`), and every
unit that adds a module also updates `docs/research/dp_gibbs_code_guide.md` and
lists its step in `TODO.md` in the same commit (repo convention; documentation is
part of the change).

### 6.1 `src/paper_artifacts.py` (new, ~450 lines)

Commit: `feat: add replication artifact serialisation`.

Public API:

- `slug(label)` — the directory-name rule of §5 (runs of non-`[A-Za-z0-9._-]` →
  `_`); the inverse is not needed because the label is stored in `fit-meta.json`.
- `save_chain(path, chain, *, t_grid)`, `load_chain(path)` — round-trip one
  `DPChains` together with its `meta` blob, so `load_chain` needs no side
  information.
- `save_functionals(path, func)`, `load_functionals(path)` — arrays **and** the
  meta dict (verbatim, via `_jsonable`).
- `save_fit(dir, fit, *, meta)`, `load_fit(dir)` — writes `fit-meta.json`
  including the un-slugged `label`, `arm` and both meta dicts; returns a `Fit`
  with `prior_band` attached only when a band file sits in that directory.
- `save_demo_result(dir, result)`, `load_demo_result(dir)` — used by tests and
  by the smoke run, **not** by the long runs (they write incrementally through
  `ArtifactWriter`: one writer per artifact). It writes `meta.json` and all
  fits; it writes no figures. `load_demo_result` reconstructs `DemoResult.meta`
  with the original types (`prior_indices` as a tuple, `seed` int or None,
  `t_max` floats), because the paper preconditions compare `run.meta` against
  the in-memory values.
- `ArtifactWriter(root, *, section, meta)` — context manager used by the
  drivers. Methods `fit_done(fit, timing)`, `band_done(...)`,
  `event(kind, **fields)` (appends one JSON object per line to
  `logs/events-<section>.jsonl` with `{"ts", "t_monotonic", "kind", ...}`),
  and `write_meta()` (writes the section `meta.json`). It does **not** write
  `MANIFEST.json`.
- `finalize_run(root)` — called once by `replicate.sh` after the report stage:
  walks the whole tree, hashes every file, writes the single run-level
  `MANIFEST.json` (R7, so `summary/`, `report/` and `figures/` are included).
- `verify_manifest(root)` — re-reads `MANIFEST.json`, re-hashes every listed
  file, fails on a size or hash mismatch, on a missing file, and (unless
  `allow_extra=True`, used by the verifier after re-rendering) on an unlisted
  file; returns a report or raises `ArtifactIntegrityError`.
- `_jsonable(value)` — numpy scalars/arrays → JSON; used for `Functionals.meta`
  and every provenance dict.

Design constraints:

- No matplotlib, no plotting; pure I/O + numpy. Keeps `test 32`'s import-hygiene
  contract intact.
- Compression (R8): write one representative fit both ways in step 1, keep
  `savez_compressed` only if it saves ≥ 5 % of bytes; either way the measured
  numbers and the choice go into §2 as R8's outcome before step 6.2 starts.
- Never hold two extra copies of a 96 MB array: write arrays one at a time from
  the object that already owns them, through the temp-file handle of §5.
- `schema_version` lives in each `meta` blob and in `meta.json`/
  `MANIFEST.json`; the loaders reject a mismatch (no migration path in this
  plan), and `verify_manifest` reports the version it checked.

Tests (`tests/test_paper_artifacts.py`, quick preset *and* one-draw edge cases,
`tmp_path`):

- round-trip equality for every array of `DPChains`, `Functionals`, `Fit`,
  `DemoResult` (bit-exact, `np.array_equal`), including `t_grid`, the prior
  band, `fit-meta.label`, and **both** meta dicts verbatim (posterior and
  band);
- the same round trip with `n_save=1`, one chain, one grid point, and an
  all-censored dataset (None/One/Many coverage);
- `load_fit` rejects a transposed `S`, a wrong draw count, and a shape/dtype
  that disagrees with `fit-meta.json`;
- atomic write: a simulated failure mid-save leaves no `.npz` and no `.npz.tmp`;
- `verify_manifest` detects a flipped byte, a truncated file, a missing file
  and an unlisted extra file;
- `finalize_run` covers `summary/`, `report/` and `figures/` and its own
  `MANIFEST.json` is *not* listed inside itself (the report hashes it
  externally);
- `ArtifactWriter.event` produces one valid JSON object per line with the
  documented keys, appending across calls;
- `_jsonable` round-trips `np.intp`, `np.float64`, arrays, `None`, nested dicts
  and tuples;
- `load_*` on a missing/unreadable directory raises an error naming the path;
- loader rejects `schema_version` mismatches;
- a quick-preset `save_demo_result`/`load_demo_result` round-trip feeds
  `r_hat()`/`ess()` and reproduces the in-memory values exactly, and
  `load_demo_result(...).meta == result.meta` including `prior_indices` as a
  tuple.

### 6.2 Wire saving into the drivers and CLI (`src/paper_examples.py`)

- `example_5_1(..., save_dir: Path | None = None)`: after each `fit_dataset()`
  returns, save the fit **immediately** (chains then functionals), before
  starting the next fit; after `prior_band()` save the band. Write
  `logs/events-5.1.jsonl` entries with `time.perf_counter()` timings. With
  `save_dir` set the driver writes **no figures** (R7: the report stage owns
  `figures/`, re-rendered from the stored draws) and returns `figures=()`;
  without it, behaviour is unchanged, including the returned paths.
- Same for `example_5_2`, section `5.2`. Arm B is saved as its own fit
  directory (`B-<slug>`).
- `save_dir` must resolve inside `output/` (or outside the repository); a path
  under `docs/`, `src/` or `tests/` is rejected with one clear message, so a
  3 GB artifact tree can never land in a tracked directory.
- CLI: add `--save-dir DIR`. `--out` becomes `default=None`: when `--out` is
  omitted it keeps today's meaning (`DEFAULT_FIGURE_DIR`) **unless** `--save-dir`
  is given, in which case no figures are written by the driver at all (R7).
  `main()` prints the resolved figure paths as it does today. A
  `--run-id` is not needed: the run root is the directory `--save-dir` names.
- Keep `--max-workers` defaulting to the chain count (4); the pool is
  chain-parallel and each worker is single-threaded numpy.
- No change to how fits are computed: the run must stay a pure replay.

Tests: extend `tests/test_cli.py` smoke cases — quick preset with `--save-dir`
writes the expected tree (four chains, functionals, `fit-meta.json`, events log,
**no** `figures/`) and `load_demo_result` reproduces the fixtures' values; a
missing `--out` still writes to `DEFAULT_FIGURE_DIR` semantics (asserted on the
returned paths, not on disk); `--save-dir docs/x` is rejected; an unwritable
`--save-dir` exits non-zero with one clear message (patched `Path.mkdir` rather
than a chmod, which a root user would defeat).

Commit: `feat: persist replication artifacts`.

### 6.3 `src/paper_report.py` (new, ~500 lines)

Consumes a loaded `DemoResult` and emits the report payload. Public API:

- `summarize_5_1(result) -> dict` — the §6.3 Figure 1–3 quantities:
  prior means, posterior mean/95 % CI/99th percentile of ν, `Pr(3 ≤ n* ≤ 15)`,
  n\* mean and observed range, prior vs posterior 95 % widths; mean
  `S(1.1727)`, `∫f dt` (trapezoid on the grid), prior vs posterior band width on
  `(0, 4]`, `f(3.26)` vs `f(2.4)`; smoothed-hazard sign changes in `(0, 5)` and
  the band ratio at 4.5 vs 2.5.
- `summarize_5_2(result) -> dict` — pairwise prior-curve agreement on `(0, 20)`
  both directions, `min(S_A − S_B)` on `(0, 20)`, the arm-B smoothed-hazard
  maximum in `[10, 20]` and the maximum above 20; `Pr(η_A > η_B)`, mean
  `F_B − F_A` minimum, `Pr(F_B > F_A)` range/mean on `(0, 20)`, mean hazard
  ratio at t=5 and t=25, `Pr(ratio_5 > ratio_25)`, 95 % ratio bands at both
  times, `n_eta_above_grid`.
- `diagnostics(result) -> dict` — `r_hat()`/`ess()` for ν, n\* and S at the five
  diagnostic times, computed twice: on all stored draws, and on the old note's
  slice. The slice is built with `dataclasses.replace(chain.settings,
  n_save=500)` (required by `DPChains.__post_init__`, `dp_gibbs.py:3205`) and
  reuses the **full-trace** `J_common`/`J_prior` (see §7 Step 2 and §9), so the
  first-500 column is the note's probe value, not a new fit.
- `delta_vs_baseline(summary, baseline)` — signed differences against
  `docs/research/dp_gibbs_results_baseline.json` (extracted once from the
  existing note in this step, with a header saying so, and with a
  `not_comparable` column for the rows listed in §7 Step 2).
- `render_markdown(payload) -> str` — the tables only (no prose): settings,
  diagnostics, Figure 1–5 tables, acceptance table, delta table, manifest table.
- `write_report_artifacts(save_dir, payload, result)` — writes
  `summary/stats.json`, `summary/rhat_ess.csv`, `summary/quantiles.csv`
  (per fit: ν mean/2.5/50/97.5 %, n\* mean/2.5/50/97.5 %, and the per-draw median
  η of each arm for §5.2), `summary/acceptance.json`, `report/tables.md`, and
  re-renders the figures **into `<section>/figures/`** from the loaded
  `Functionals` with a fixed matplotlib `metadata` dict (no `CreationDate`), PNG
  only, so the bytes are reproducible. It never writes outside its section
  directory.
- `main(argv)` — `python -m src.paper_report --from-artifacts <section_dir>
  --write [--baseline FILE]`, so the runbook and `replicate.sh` have one entry
  point (the CLI is part of the 6.3 API, not an afterthought).

Acceptance values in `acceptance.json` come from the same computation as the
committed tests; the expected target/label table is a module-level constant
mirroring `tests/test_paper_5_1.py` / `tests/test_paper_5_2.py` constants. A
unit test asserts that every constant in the tests' target tuples is present in
the report's target table (guards against drift between the two files).

Tests (`tests/test_paper_report.py`, quick preset): key presence and shapes;
`render_markdown` produces every table header; `delta_vs_baseline` sign/format
and the `not_comparable` list; the diagnostics slice keeps
`n_draws == settings.n_save` and reproduces `r_hat`/`ess` on a sliced chain;
re-rendering the same `Functionals` twice yields **byte-identical** PNGs (exact
bytes, not byte size); matplotlib is imported lazily (`_pyplot()` inside
`write_report_artifacts`, the same pattern as `paper_examples`), asserted the
same way `tests/test_cli.py::test_sampler_imports_do_not_pull_in_matplotlib`
asserts it for the sampler modules.

A `quick`-preset artifact-mode test enters the default gate too: save a tiny
`DemoResult`, point `DP_PAPER_ARTIFACTS` at it, and run the fixture branch logic
(not the whole paper module) so the loader path cannot rot between long runs.

Commit: `feat: add artifact-backed replication report`.

### 6.4 Artifact-mode acceptance fixtures (edit `tests/test_paper_5_1.py`, `tests/test_paper_5_2.py`)

Both module fixtures gain one branch:

```python
if os.environ.get("DP_PAPER_ARTIFACTS"):
    from src import paper_artifacts
    return paper_artifacts.load_demo_result(Path(os.environ["DP_PAPER_ARTIFACTS"]))
```

`DP_PAPER_PRESET=paper` (the default) must still match `run.meta["preset"]`, so
`test_run_uses_the_reproduction_settings` keeps guarding that the loaded tree is
the paper run, not a quick smoke run. `load_demo_result` must restore
`run.meta["prior_indices"]` as a **tuple** (`test_cli` asserts `== (0, 2)` in
memory, and the paper preconditions assert `== tuple(range(3))`), which JSON
would otherwise turn into a list.

New test in each module: when the env var is set, the manifest's `code_sha`
must equal `git rev-parse HEAD` (override variable
`DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH=1` for inspecting an older run); when it
is not set, the test skips. Because that is a twelfth and an eighth test, the
artifact-mode counts are **12/12 and 8/8**, while a plain refit run stays
11/11 and 7/7 (§8 item 6 and §12 state both).

Commit: `test: replay the paper acceptance suite from saved artifacts`.

### 6.5 Tooling

- `.gitignore`: `output/` is added in 6.1 (not here), so the smoke run of
  §7 Step 0 can never be committed; `docs/research/figures/` stays.
- `scripts/replicate.sh <5.1|5.2> [run_root]` (tracked, `set -euo pipefail`):
  with no `run_root` it stamps `output/replication-<UTC>`; with one it **reuses
  the existing root** (R6), which is how Step 3 joins Step 1's tree. It writes
  `logs/env.txt`, runs the driver under `pixi run python -u` with `tee`, then
  the report step (which writes the section's `figures/`), then the
  artifact-mode pytest, then `finalize_run <run_root>` — and only then
  `verify_manifest`, so the manifest covers every file the run produced.
  Nonzero exit on any stage.
- `scripts/verify_replication.sh <run_root>` (tracked): `verify_manifest` plus
  `pytest -m paper` in artifact mode, no fitting. Figure re-rendering goes to a
  scratch directory and is compared byte-for-byte; it must **not** rewrite the
  verified trees, or it would invalidate the manifest it just checked. This is
  the reviewer entry point.
- `pixi.toml` tasks: `replicate-5.2`, `replicate-5.1`, `verify-paper`
  (thin wrappers around the scripts, so the runbook is one line).
- `output/README.md` is **not** created (it would be the only tracked file in an
  ignored tree); the layout is documented in this plan and in the report.

Commit: `chore: add replication and verification scripts`.

## 7. Runbook

### Step 0 — pre-flight (15 min, no commit)

```bash
cd /home/tr/Documents/survival_01
git status --short --branch           # expect clean, record the SHA
df -h .                               # need >= 30 GB free (expect ~2.5-3 GB)
free -g                               # 15 GB RAM; 3 fits in RAM ~1.5 GB
nproc                                 # 16
pixi run python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```

Then a cheap end-to-end dry run of the new machinery at `quick` (seconds). It
runs **after 6.3 and 6.4** (see §11), because it exercises the report CLI and
the artifact-mode fixture branch:

```bash
pixi run python -m src.paper_examples --example 5.2 --preset quick \
  --save-dir output/smoke/5.2
pixi run python -m src.paper_report --from-artifacts output/smoke/5.2 --write
pixi run python -c "from src.paper_artifacts import finalize_run; finalize_run('output/smoke/5.2')"
DP_PAPER_ARTIFACTS=output/smoke/5.2 DP_PAPER_PRESET=quick \
  pixi run pytest -m paper tests/test_paper_5_2.py
```

`DP_PAPER_PRESET=quick` is required, not optional: the smoke root records
`"preset": "quick"`, and `test_run_uses_the_reproduction_settings` compares
`run.meta["preset"]` against the preset named by that variable. The smoke run
is a machinery check, not an acceptance check — its §6.3 assertions may fail at
`quick`, in which case only the load/report/diagnostics path is being verified.
All three stages must run; delete `output/smoke` afterwards.

### Step 1 — §5.2 at the paper preset (~4 h)

```bash
nohup systemd-inhibit --what=sleep:idle --why="paper replication" \
  scripts/replicate.sh 5.2 > output/replicate-5.2.out 2>&1 &
```

Interactive alternative: run the same command inside `tmux new -s replicate`.
The script:

1. writes `logs/env.txt` (SHA, dirty flag, lock hash, versions, host, cores,
   memory, disk);
2. runs `python -m src.paper_examples --example 5.2 --preset paper
   --save-dir <run_root>/5.2`, teeing to `logs/run-5.2.log`;
3. runs `python -m src.paper_report --from-artifacts <run_root>/5.2 --write`
   (writes `summary/`, `report/` and the section's `figures/`);
4. runs `pytest -m paper tests/test_paper_5_2.py` with `DP_PAPER_ARTIFACTS`;
5. runs `finalize_run <run_root>` and then `verify_manifest <run_root>`.

Artifacts are written per fit (arm A prior 1, 2, 3, then arm B), so a crash
loses at most one fit. Expected end state: 4 fit directories, **16** chain
files, 4 `functionals.npz`, 2 PNG figures, manifest, green acceptance log.
The root stays open for Step 3, which writes `<run_root>/5.1` into the same
tree and re-runs `finalize_run` (R6).

### Step 2 — inspect §5.2 (30 min, gate before the long run)

- `MANIFEST.json` sizes: expect ~1.5 GB for §5.2.
- `summary/acceptance.json`: all 7 clauses pass.
- `summary/stats.json` has three columns per row: full draws, first-500 draws,
  and the old note's value. **Full draws are compared against the §6.3 target
  windows plus a stated Monte Carlo tolerance** (they are a sharper estimate,
  not the probe values); the **first-500 column is the like-for-like check**
  against the note, and only it is a hard gate.
- The first-500 column is computed with the slice's own `r_hat`/`ess` but the
  **full trace's** `J_common`/`J_prior`, because the note's probe values were
  read from the full run's chains (§4 of the note). Both J values are recorded
  in `stats.json` so a future reader can tell which was used.
- Rows that **cannot** match by construction are marked `not_comparable` and
  are not part of the gate: the Figure 2 prior-band width (the probe drew 200
  prior bands, the run draws 1 000) and `n_eta_above_grid` (a count, not a
  summary). Everything else must match to the displayed precision; a listed
  row that does not is a finding.
- Sanity: `r_hat < 1.05` everywhere, ESS ≥ 1 400 per fit **on the first-500
  slice** (on 40 000 draws ESS is an order of magnitude larger, so that column
  alone would be a vacuous gate), `n_eta_above_grid` recorded.
- If a comparable first-500 row differs, do **not** start §5.1 — investigate
  the drift (code change, preset, numpy version) first.

### Step 3 — §5.1 at the paper preset (~8–12 h, overnight)

```bash
nohup systemd-inhibit --what=sleep:idle --why="paper replication" \
  scripts/replicate.sh 5.1 output/replication-<stamp> > output/replicate-5.1.out 2>&1 &
```

The `<stamp>` is Step 1's root, verbatim (R6). The §5.2 tree is left untouched
and the run-level `MANIFEST.json` is rewritten to cover both sections.

Same stages, `tests/test_paper_5_1.py`. Per fit: 4 chains × ≈2.5 h sampling
(4 in parallel) + ≈45 min functionals + save. Three fits sequentially.
Expected artifacts ≈1.5 GB. The previous run was interrupted by a workstation
suspend for part of its wall clock; `systemd-inhibit` prevents the repeat, and
per-fit saving bounds the loss if it happens anyway.

### Step 4 — verify and report (1–2 h)

```bash
scripts/verify_replication.sh output/replication-<stamp>   # no fitting
```

Then write `docs/research/dp_gibbs_replication_report.md` from
`output/.../report/tables.md` plus the prose sections listed in §8, and commit:

1. `docs: add artifact-backed Kottas replication report`
   (report + baseline JSON + the one-line pointer in `dp_gibbs_results.md`);
2. `chore: ignore replication output directory` (if not already in the feature
   commits);
3. `docs: record the artifact run in the implementation plan`
   (amend `dp_gibbs_plan.md` §10 decision 8 with the new paths).

Nothing from `output/` is committed. The report cites the run path, the git SHA,
the manifest checksum and the artifact table so a later reader can ask for the
directory.

## 8. Report contents (`docs/research/dp_gibbs_replication_report.md`)

Mirrors the structure of `dp_gibbs_results.md` so the two can be read side by
side, but every table is generated from the artifacts and labelled with the run:

1. **Provenance** — git SHA, dirty flag, pixi lock hash, package versions, host,
   cores, RAM, start/end timestamps, run path, manifest SHA-256 (computed after
   `finalize_run`, since a file cannot hash itself).
2. **Settings and seeds** — preset table, the seed table incl. every per-chain
   seed `seed+b`, data seeds, sample statistics (median/IQR/max/n above 4).
3. **Chain diagnostics** — split R̂ and ESS for ν, n\* and S at the five times,
   full draws and first-500 draws, per fit (7 fits).
4. **§5.1 achieved values** — Figures 1–3 tables, each row: quantity, target
   (with the amendment label A1/A2 where applicable), observed, PASS/FAIL.
5. **§5.2 achieved values** — Figures 4–5 tables, same shape, incl. the
   hazard-ratio band gap and `n_eta_above_grid`.
6. **Acceptance replay** — the `pytest -m paper` output in artifact mode:
   12/12 and 8/8 (the eleventh/seventh refit assertion plus the manifest-SHA
   check), quoting the assertion names and the log path, and noting that a
   plain refit run gives 11/11 and 7/7.
7. **Agreement with the previous note** — delta table (first-500 summaries vs
   the note's values, with the `not_comparable` rows marked) and an explicit
   statement of what is identical (chains), what sharpens (full-draw summaries)
   and what moved (diagnostics on 10 000 draws vs 500, and the J values used).
8. **Artifact inventory** — per-directory file counts, shapes, dtypes, sizes,
   SHA-256 of `MANIFEST.json`; how to verify with `scripts/verify_replication.sh`.
9. **Caveats** — carried over from the old note: prior scale (decision 2), our
   sample vs the paper's, amendments A1–A4, performance-budget misses,
   `log_post` is not a marginal likelihood. Plus: this is Kottas (2002) §5.1/5.2
   only; no full-paper claim.
10. **Reproduction cost** — measured per-sweep and per-draw functional cost from
    the run's own timings, and the wall clock, so the next run can be planned.

Length target: comparable to the old note (~260 lines), with generated tables
and hand-written prose around them.

## 9. Verification strategy

Three independent layers, none of which requires a refit:

1. **Integrity** — `MANIFEST.json` checksums vs files; schema version; shapes and
   dtypes; container `__post_init__` validation on load.
2. **Acceptance replay** — the committed `paper` tests, unmodified thresholds,
   run against the loaded artifacts via `DP_PAPER_ARTIFACTS`. This is the same
   suite that produced "11/11 and 7/7" in the old note, now provably checking
   the same arrays the report summarises.
3. **Recomputation** — `src/paper_report.py` recomputes every table value from
   the arrays and re-renders the figures from the stored `Functionals`; the
   committed report is a transcription of `report/tables.md`. A cheap
   `quick`-preset test asserts the report module and the acceptance constants
   stay in sync.

Determinism check (R2, §7 Step 2): the chains are a replay, so the posterior
summaries computed from the *first 500 draws* — using the full trace's
`J_common`/`J_prior` — must match the old note to its displayed precision, for
every row that is not in the `not_comparable` list. The rows that cannot match
(prior-band width, 200 vs 1 000 prior draws; `n_eta_above_grid`, a count) are
listed once in the plan and once in the report rather than silently excluded.
Any mismatch in a comparable row is a finding, investigated before the report
is accepted.

## 10. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A crash at hour 11 loses the run | Per-fit save in `example_5_1/5_2`; each fit's artifacts are complete before the next fit starts. Resumption is explicitly out of scope (R8): a lost fit is refitted from the same seed, and the partial tree is deleted first. |
| Workstation suspends mid-run (happened last time) | `systemd-inhibit --what=sleep:idle`; `tmux`; per-fit saves bound any loss. |
| Disk fills (53 GB free, ~3 GB needed) | `df -h` pre-flight gate; `verify_manifest` fails loudly on truncated files; size budget printed after each fit. |
| Memory pressure (15 GB RAM) | Fits are sequential (≈0.5 GB each), 4 workers, no extra array copies; `n_save` unchanged. |
| Silent divergence from the old note | First-500 summary comparison is a hard gate in Step 2/4, with the `not_comparable` rows named in advance so the gate cannot fire spuriously. |
| Report/test thresholds drift | Acceptance target table in `paper_report.py` mirrored from the test constants with a unit test over the mirror. |
| Compression cost/size surprise | Measured in 6.1 on one fit; the ≥ 5 % rule and the numbers land in §2 as R8's outcome before 6.2 (R8). |
| Accidental commit of artifacts | `output/` git-ignored from 6.1; `--save-dir` refuses tracked paths; the runbook ends with `git status` showing only the intended files. |
| A figure written twice, or a manifest that misses late files | One writer per artifact (R7) and `finalize_run` at the very end (§6.1, §6.5). |

## 11. Effort and timeline

| Work | Estimate |
| --- | --- |
| 6.1 artifacts I/O + tests | 3–4 h |
| 6.2 driver/CLI wiring + tests | 1.5–2 h |
| 6.3 report module + tests | 3–4 h |
| 6.4 artifact-mode fixtures | 1 h |
| 6.5 scripts, gitignore, docs pointers | 1 h |
| Dry run (`quick`) | 15 min |
| §5.2 paper run | ≈4 h wall |
| §5.1 paper run | ≈8–12 h wall (overnight) |
| Verify + write report | 1–2 h |
| Total code | ≈10–12 h, in five commits (6.1–6.5) |
| Total compute | ≈12–16 h wall, unattended |

Sequencing constraint: 6.1 → 6.2 → 6.3 → 6.4 → 6.5 → dry run (Step 0) → Step 1
→ Step 2 gate → Step 3 → Step 4. The dry run moved here because it exercises
the report CLI (6.3) and the artifact-mode fixture branch (6.4); running it
earlier would test machinery that does not exist yet.

## 12. Definition of done

- `output/replication-<stamp>/` exists with the layout of §5, `MANIFEST.json`
  validates, and both section trees contain every draw (7 fits, 28 chains,
  280 000 stored draws total) and every functional array.
- `scripts/verify_replication.sh <run_root>` exits 0 with no fitting:
  integrity (`MANIFEST.json` covers both section trees), 12/12 + 8/8
  artifact-mode replays, figures re-rendered byte-identically into a scratch
  directory, artifacts unmodified.
- `docs/research/dp_gibbs_replication_report.md` is committed, generated tables
  match `report/tables.md` byte-for-byte, and it states the run path, SHA and
  manifest hash.
- `docs/research/dp_gibbs_results.md` points to the new report.
- `git status --short` shows no tracked artifact and no uncommitted change
  outside the intended commits; `pixi run test`, `lint` and `typecheck` are
  green.

## 13. Resolved questions

Asked and answered before writing this plan:

1. Preset/staging → `paper`, §5.2 then §5.1 (R1).
2. Seeds → committed seeds, deterministic replay (R2).
3. Artifact volume → everything, ≈2.5–3 GB (R3).
4. Report production → generated tables + curated prose (R4).

Resolved during the plan review (2026-11-08), no longer open:

1. Run root → one root for both sections; Step 3 reuses Step 1's (R6).
2. Manifest → written once, by `finalize_run`, after the report stage (R7).
3. First-500 gate → full-trace J, per-row tolerance, explicit
   `not_comparable` rows (§7 Step 2, §9).
4. Compression → measured in 6.1 against the ≥ 5 % rule and recorded as R8's
   outcome (R8); resumption (`--reuse-complete`) dropped from scope (R8);
   report filename fixed to `docs/research/dp_gibbs_replication_report.md`, next
   to the old note.

No open questions remain. Anything raised during execution is a plan change
and is recorded here, not silently decided in the runbook.
