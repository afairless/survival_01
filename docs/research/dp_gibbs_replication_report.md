# Artifact-Backed Replication Report: Kottas (2002) §5.1 and §5.2

This is the artifact-backed record of the paper reproduction. Unlike
`dp_gibbs_results.md`, every number below is regenerated from stored draws by
`src/paper_report.py`, and the draws, functionals, figures, logs and checksums
sit in one git-untracked run directory:

| item | value |
| --- | --- |
| run path | `output/replication-2026-11-08T02-26-53Z` |
| sections | 5.1 and 5.2, `paper` preset, all priors |
| files / bytes | 67 / 2 608 901 966 (2.43 GiB; the total excludes the manifest itself) |
| manifest | `MANIFEST.json` in the run root, `file_count: 67` |
| manifest SHA-256 | `2a3bebb4541ee42b77143d8d11062869072f2605c8986a6cf280f2a5519e7655` |
| manifest `created_at` | 2026-11-09T02:24:21Z |
| manifest `code_sha` | `0234f597…` (`fix: mark the note's prior-width row not comparable`) |

Verify without fitting anything:

```bash
scripts/verify_replication.sh output/replication-2026-11-08T02-26-53Z
```

That runs the three layers of the plan's §9: re-hash every file the manifest
lists, re-render every figure into a scratch directory and compare it
byte-for-byte, and replay both acceptance suites in artifact mode
(`DP_PAPER_ARTIFACTS`), then re-hash the tree to prove neither stage wrote into
it. It exits 0 only when all three pass; it did, for this tree.

The two documentation commits that follow the run (this report and the plan
amendment) leave `HEAD` ahead of the manifest's `code_sha`, so a replay from a
later checkout is run with `DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH=1`, or from
commit `0234f59`. The artifacts themselves are unchanged by them.

`dp_gibbs_results.md` remains the narrative note; this report supersedes it as
the auditable record and says in §7 exactly what reproduces and what does not.

## 1. Provenance

Recorded at the start of each section's run in `logs/env.txt` and at
finalisation in `MANIFEST.json`.

| field | §5.2 | §5.1 |
| --- | --- | --- |
| started | 2026-11-08T02:26:53Z | 2026-11-08T16:05:28Z |
| finished (driver) | 2026-11-08T06:20:35Z | 2026-11-09T01:36:11Z |
| finished (all six stages) | 2026-11-08T06:21:13Z | 2026-11-09T01:40:23Z |
| git HEAD at start | `4a3e6be` | `21caf16` |
| dirty paths at start | 0 | 0 |
| python | 3.15.2 | 3.15.2 |
| numpy / scipy / matplotlib | 2.5.4 / 1.18.1 / 3.11.2 | 2.5.4 / 1.18.1 / 3.11.2 |

The manifest additionally records `pandas 3.1.0`, `polars 2.1.0`,
`pyarrow 26.0.0` and the pixi lock hash
`0e943d139d76ffb7b6b34619713b76c9441d0de075b68cedcafc2ffa4c011e73`, identical
across both sections. Host: 16-core Linux workstation (`rt`), 15 GiB RAM, no
other load; `systemd-inhibit --what=sleep:idle` held the machine awake for both
sections, so — unlike the earlier run recorded in the old note — no suspend is
inside either wall clock.

Two code commits are relevant. `4a3e6be` produced the §5.2 draws.
`21caf16` (the first-500 estimator fix, §7 below) produced the §5.1 draws and
re-rendered the §5.2 report. `0234f59` re-rendered the §5.1 report after one
baseline row was reclassified. No draw was regenerated after `21caf16`.

## 2. Settings and seeds

Both sections use the same sampler knobs (plan §6.2); only the data and the
chain seed differ.

| knob | value |
| --- | --- |
| burn-in | 5 000 sweeps |
| saved draws per chain | 10 000 |
| thinning | 150 sweeps |
| sweeps per chain | 1 505 000 |
| chains per fit | 4 |
| α grid | 256 |
| truncation tolerance ε | 1e-4 |
| prior-band draws | 1 000 (reference prior of §5.1 only) |
| evaluation grid | 300 points on the original scale |

| section | fits | chains | chain seed | data seed |
| --- | --- | --- | --- | --- |
| §5.1 | 3 (one per ν prior) | 12 | 20240601 | 721383 |
| §5.2 | 4 (arm A × 3 priors, arm B) | 16 | 20240602 | none (transcribed data) |

Chain `b` of a group uses `seed + b`, so every draw is replayable from the
integers above; the per-chain seeds are also stored in each chain file's
`meta`. The three ν priors are Gamma(2, 0.9), Gamma(2, 0.1) and Gamma(3, 0.05),
with model constants `d=2`, `a_gamma=1`, `b_gamma=0.232235`, `a_phi=2`,
`b_phi=1.77522` (plan decision 1; the code uses `b_phi=1.0` in its
prior dataclass — see the caveats in §9 and the old note).

Sample statistics, recomputed from the stored `data` arrays:

| sample | n | censored | median | IQR | max | n above 4 |
| --- | --- | --- | --- | --- | --- | --- |
| §5.1 mixture sample | 200 | 0 | 1.217 | 0.817 – 2.351 | 5.139 | 2 |
| §5.2 arm A | 20 | 3 | 16.5 | 7.0 – 28.25 | 49 | 17 |
| §5.2 arm B | 20 | 2 | 10.0 | 3.75 – 18.75 | 44 | 14 |

§5.1's `t_max` is 5.139 (original scale); §5.2 uses one shared scale of 49
(the larger arm maximum) for both arms, because `compare` rejects two fits with
different `t_max`.

## 3. Chain diagnostics

Split R̂ and bulk ESS for ν, n\* and the five-time S trace, per fit: on the full
10 000 stored draws per chain, and on the old note's probe slice (first 500
draws, with the truncation level recomputed from the slice). Generated in
`<section>/summary/rhat_ess.csv` and the `Diagnostics` table of
`<section>/report/tables.md`.

| fit | draw set | R̂(ν) | ESS(ν) | R̂(n\*) | ESS(n\*) | R̂(S) min / max | ESS(S) min / max | J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| §5.1 Gamma(2, 0.9) | full | 1.00007 | 28 831 | 1.00005 | 22 483 | 1.00001 / 1.00030 | 10 681 / 37 317 | 1 921 |
| §5.1 Gamma(2, 0.9) | first 500 | 1.00208 | 1 569 | 1.00328 | 1 496 | 0.99894 / 1.00567 | 757 / 1 719 | 1 901 |
| §5.1 Gamma(2, 0.1) | full | 0.99999 | 30 349 | 1.00002 | 28 977 | 1.00001 / 1.00020 | 13 555 / 38 270 | 2 031 |
| §5.1 Gamma(2, 0.1) | first 500 | 1.00046 | 1 627 | 1.00233 | 1 511 | 0.99910 / 1.00362 | 933 / 1 963 | 1 982 |
| §5.1 Gamma(3, 0.05) | full | 0.99994 | 36 694 | 0.99996 | 34 870 | 0.99993 / 1.00011 | 17 751 / 39 705 | 2 198 |
| §5.1 Gamma(3, 0.05) | first 500 | 1.00023 | 1 794 | 0.99975 | 1 599 | 0.99872 / 1.00015 | 822 / 1 981 | 2 053 |
| §5.2 A Gamma(2, 0.9) | full | 1.00007 | 39 211 | 1.00000 | 39 589 | 0.99992 / 1.00009 | 38 470 / 38 839 | 328 |
| §5.2 A Gamma(2, 0.9) | first 500 | 1.00097 | 1 822 | 1.00057 | 1 972 | 0.99919 / 1.00019 | 1 724 / 1 983 | — |
| §5.2 A Gamma(2, 0.1) | full | 1.00009 | 39 708 | 1.00017 | 39 689 | 0.99997 / 1.00011 | 39 560 / 40 049 | 1 419 |
| §5.2 A Gamma(2, 0.1) | first 500 | 0.99967 | 1 421 | 0.99993 | 1 474 | 0.99991 / 1.00054 | 1 806 / 2 014 | — |
| §5.2 A Gamma(3, 0.05) | full | 1.00001 | 38 265 | 1.00006 | 38 324 | 0.99998 / 1.00014 | 38 895 / 40 222 | 3 157 |
| §5.2 A Gamma(3, 0.05) | first 500 | 1.00157 | 1 602 | 0.99992 | 1 874 | 0.99856 / 0.99969 | 1 928 / 2 030 | — |
| §5.2 B Gamma(2, 0.9) | full | 0.99998 | 39 731 | 1.00004 | 37 209 | 0.99994 / 1.00004 | 38 801 / 39 955 | 335 |
| §5.2 B Gamma(2, 0.9) | first 500 | 0.99962 | 1 893 | 0.99904 | 1 834 | 0.99893 / 0.99943 | 1 784 / 2 014 | — |

`J` is `J_common`, the shared truncation level derived from `max_b ν_b` of the
draws the functionals were computed from; the full `stats.json` records both
the stored arrays' level and the slice's. Every fit satisfies R̂ < 1.05 and,
on the first-500 slice, ESS ≥ 1 421 for ν and n\* — the acceptance gate of plan
§3.7, which is the strict version of the statistic.

## 4. §5.1 achieved values

Full draws (40 000 per fit); the note's first-500 probe values are in §7.
Figures 1–3 are `5.1/figures/figure_{1,2,3}.png`.

### Figure 1 — ν and n\*

| quantity | Gamma(2, 0.9) | Gamma(2, 0.1) | Gamma(3, 0.05) |
| --- | --- | --- | --- |
| prior mean of ν | 2.222 | 20 | 60 |
| posterior mean of ν | 1.523 | 2.678 | 4.057 |
| posterior 95 % interval | 0.363, 3.640 | 0.582, 7.019 | 1.016, 10.167 |
| prior 95 % width (exact) | 5.922 | 53.294 | 132.12 |
| posterior width | 3.277 | 6.436 | 9.151 |
| posterior 99th percentile of ν | 4.290 | 8.516 | 12.249 |
| posterior mean of n\* | 7.156 | 9.927 | 12.985 |
| Pr(3 ≤ n\* ≤ 15 \| D) | 0.983 | 0.885 | 0.717 |
| observed n\* range | 2 – 27 | 2 – 46 | 2 – 63 |

The per-prior floors in the acceptance table are amendment A1: the original
"≥ 0.85 under all three priors" reading was replaced by 0.85 / 0.85 / 0.70,
because the extreme Gamma(3, 0.05) prior has a visible tail past 15 in the
paper's own Figure 1. The observed 0.717 clears that floor by 0.017 and the
mean ordering 7.16 → 9.93 → 12.99 reproduces the paper's Figure 1 ordering.

### Figure 2 — S(t) and f(t)

| clause | target | Gamma(2, 0.9) | Gamma(2, 0.1) | Gamma(3, 0.05) |
| --- | --- | --- | --- | --- |
| mean S(1.1727) | [0.42, 0.58] | 0.4952 | 0.4953 | 0.4950 |
| ∫ f̂ dt over the grid | [0.95, 1.05] | 0.9755 | 0.9740 | 0.9722 |
| f̂(3.26) > f̂(2.4) | two modes | 0.1478 > 0.1287 | 0.1480 > 0.1283 | 0.1477 > 0.1290 |
| prior vs posterior 95 % band width on (0, 4], reference prior | prior ≥ posterior | 0.8587 vs 0.0936 | — | — |

### Figure 3 — smoothed hazard

| prior | change points on (0, 5) | band width at t=2.5 | at t=4.5 | far / near |
| --- | --- | --- | --- | --- |
| Gamma(2, 0.9) | 1.015, 2.137, 4.462 | 0.3867 | 4.6941 | 12.14 |
| Gamma(2, 0.1) | 1.015, 2.137, 4.245 | 0.3843 | 4.5174 | 11.75 |
| Gamma(3, 0.05) | 1.034, 2.137, 4.147 | 0.3841 | 4.1691 | 10.86 |

Exactly three sign changes under every prior, within the A2 tolerances
(0.25 / 0.25 / 0.35 around 1.022 / 2.366 / 4.454), and the far band is 10.9×
to 12.1× the near one against a 1.5× floor.

## 5. §5.2 achieved values

Full draws (40 000 per fit); arm A is the Gamma(2, 0.9) fit, as in the paper.
Figures 4–5 are `5.2/figures/figure_{4,5}.png`.

### Figure 4 — prior agreement and arm A vs arm B

| quantity | value |
| --- | --- |
| prior curves agreeing within each other's 95 % band, all ordered pairs | 1.000 of grid points in (0, 20) |
| min of mean S_A − S_B | 0.0137 (positive everywhere in the window) |
| arm-B smoothed-hazard peak | t = 11.8335, value 0.0738 |
| maximum of arm-B hazard above t = 20 | 0.0587 (< peak) |

### Figure 5 — hazard ratio and CDF difference

| quantity | value |
| --- | --- |
| Pr(η_A > η_B \| D) | 0.9014 |
| min of mean F_B − F_A | 0.0137 (non-negative in the window) |
| Pr(F_B > F_A \| D) range / mean | 0.5656 – 0.9121 / 0.8363 |
| mean hazard ratio at t = 5 / t = 25 | 2.3275 / 1.1592 |
| decline (near / far) | 2.008 (floor 1.3×) |
| Pr(ratio(5) > ratio(25) \| D) | 0.8219 (floor 0.75) |
| 95 % ratio band at t = 5 / t = 25 | 0.7120 – 5.5974 / 0.2806 – 3.0889 |
| n_eta_above_grid (fit_a / fit_b) | 1 / 0 |

The Pr(F_B > F_A) floor of 0.5 is amendment A3; observed minimum 0.5656.
`n_eta_above_grid` is a count, recorded but never gated.

## 6. Acceptance replay

`pytest -m paper` in artifact mode loads the saved draws instead of refitting:

| section | command | result | log |
| --- | --- | --- | --- |
| §5.1 | `DP_PAPER_ARTIFACTS=…/5.1 DP_PAPER_PRESET=paper pytest -m paper tests/test_paper_5_1.py` | **12 passed** | `logs/run-5.1.log` |
| §5.2 | `DP_PAPER_ARTIFACTS=…/5.2 DP_PAPER_PRESET=paper pytest -m paper tests/test_paper_5_2.py` | **8 passed** | `logs/run-5.2.log` |

Artifact mode gives 12/12 and 8/8 rather than the 11/11 and 7/7 of the old
note: the acceptance assertions are unchanged, and one extra test per module
compares the manifest's `code_sha` with `HEAD` (skippable with
`DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH=1` for an older tree). The report's own
clause tables agree with the tests clause-for-clause: **28/28** for §5.1 and
**10/10** for §5.2, from `5.1/summary/acceptance.json` and
`5.2/summary/acceptance.json` — the same thresholds, mirrored into
`ACCEPTANCE_TARGETS` and guarded by a unit test.

## 7. Agreement with the previous note

The gate of plan §7 Step 2 is the *first-500* column against the note, because
the note's probe read 500 stored draws per chain. Full-draw columns are a
sharper estimate of the same posteriors, not a second gate.

**Identical (the chains).** Every fit's first-500 R̂ and ESS reproduce §3 of the
note: ESS(ν) 1 569 / 1 627 / 1 794 / 1 822 / 1 421 / 1 602 / 1 893 against the
note's 1 569 / 1 627 / 1 794 / 1 822 / 1 421 / 1 602 / 1 893, ESS(n\*) likewise
to four digits. The sampler replay is exact; no numpy, preset or code drift is
involved.

**Sharpened (the summaries).** With 40 000 draws instead of 2 000, the full-draw
values above are the ones to quote: e.g. ν posterior means 1.523 / 2.678 /
4.057 against the probe's 1.50 / 2.71 / 3.98, Pr(η_A > η_B) 0.9014 against
0.89, hazard-ratio decline 2.008 against 1.95, and §5.1 ESS values an order of
magnitude larger.

**Moved (one estimator fix).** The first-500 column initially sliced the stored
functional arrays, which kept the 10 000-draw trace's `J_common` while the
chains were truncated. That left every Figure 4–5 row 0.5–1 % away from the note
(peak time 12.02 vs 11.83; Pr(η_A > η_B) 0.887 vs 0.89) with identical chains.
`slice_result` now recomputes the functionals from the slice, so the column uses
the slice's own truncation level, and every comparable row matches to the
note's displayed precision:

| row (first 500 draws) | note | this run |
| --- | --- | --- |
| ν posterior means | 1.50 / 2.71 / 3.98 | 1.500 / 2.711 / 3.982 |
| ν posterior 95 % | (0.38, 3.65) / (0.59, 7.08) / (1.03, 9.99) | 0.376–3.647 / 0.592–7.076 / 1.032–9.991 |
| ν posterior width | 3.27 / 6.49 / 8.96 | 3.271 / 6.484 / 8.959 |
| n\* mean | 7.02 / 9.88 / 12.78 | 7.021 / 9.878 / 12.775 |
| Pr(3 ≤ n\* ≤ 15) | 0.987 / 0.885 / 0.730 | 0.987 / 0.885 / 0.730 |
| mean S(1.1727) | 0.494 / 0.494 / 0.495 | 0.4943 / 0.4940 / 0.4952 |
| change points | 1.015, 2.137, 4.462 … | identical to 3 dp |
| arm-B peak time / value | 11.83 / 0.0735 | 11.8335 / 0.073475 |
| Pr(η_A > η_B) | 0.89 | 0.890 |
| mean hazard ratio at 5 / 25 | 2.298 / 1.176 | 2.2984 / 1.1764 |
| ratio decline | 1.95 | 1.954 |
| 95 % ratio band | 0.663–5.455 / 0.282–3.103 | 0.663–5.455 / 0.282–3.103 |

**Two rows cannot match by construction**, and are flagged `not_comparable`
with their reasons in `dp_gibbs_results_baseline.json`:

1. `figure_1.nu_prior_95_width` — the note's 6.1 / 61 / 117 is a sample estimate
   from the probe's 200 prior draws; the report computes the exact gamma
   quantiles, 5.922 / 53.294 / 132.12. The note's values are +3 %, +14.5 % and
   −11.4 % away and sit at the 72nd, 95th and 14th percentile of the 200-draw
   sampling distribution of that statistic (4 000 simulated probes). A 14.5 %
   error is unreachable for a correct quantile, and the acceptance clause it
   feeds (posterior narrower than prior) holds either way.
2. `figure_2.prior_band_width` — the probe drew 200 prior bands, the run 1 000:
   0.833 vs 0.859.

`n_eta_above_grid` is also flagged: it is a count (0 / 1 here), not a summary.
The remaining §5.2 delta rows and every other §5.1 row agree to the note's
displayed precision; the full 20-row and 16-row delta tables are in the
sections' `report/tables.md`.

## 8. Artifact inventory

| directory | files | bytes | contents |
| --- | --- | --- | --- |
| `5.1/draws/` | 3 fits × (4 chains + functionals + prior band for the reference) | 1 132 240 244 | chains, functionals, `fit-meta.json` |
| `5.2/draws/` | 4 fits × (4 chains + functionals) | 1 476 652 860 | chains, functionals, `fit-meta.json` |
| `5.1/` + `5.2/` summaries | 28 + 32 files total | — | `stats.json`, `rhat_ess.csv`, `quantiles.csv`, `acceptance.json`, `report/tables.md`, `figures/*.png` |
| `logs/` | 7 | 8 862 | `env.txt`, `events-{5.1,5.2}.jsonl`, `run-{5.1,5.2}.log`, `pytest-paper-{5.1,5.2}.log` |
| run root | 1 | 11 887 | `MANIFEST.json` |

Array shapes, verified on load (`dp_gibbs.py` / `dp_functionals.py` containers
re-validate on construction):

| artifact | keys | shape | dtype |
| --- | --- | --- | --- |
| `chain-b.npz` (§5.1) | `theta`, `nu`, `gamma`, `phi`, `n_star`, `log_post`, `time`, `event`, `t_max`, `t_grid`, `meta` | `theta (10000, 200, 2)`; traces `(10000,)` | float64, `n_star` int64 |
| `chain-b.npz` (§5.2) | same | `theta (10000, 20, 2)` | float64 |
| `functionals.npz` | `t_grid`, `S`, `F`, `f`, `h`, `meta` | `(40000, 300)` each | float64 |
| `prior-band.npz` | `t_grid`, `S`, `F`, `f`, `h`, `meta` | `(1000, 300)` each | float64 |

`MANIFEST.json` lists all 67 files with `sha256` and `size`, plus the code SHA,
dirty flag, pixi lock hash, package versions and the per-section settings.
`scripts/verify_replication.sh` is the reader's entry point; the manifest never
lists itself, and `finalize_run` re-stamps `created_at` on every run, so the
durable identity of a tree is its `code_sha` + `file_count` + `total_bytes`
(the SHA-256 above is the value of this finalisation).

## 9. Caveats

Carried over from `dp_gibbs_results.md` §6, unchanged:

- **Prior scale.** The priors are elicited on the original time scale but used
  on `t_tilde`; the prior median of the population median is about 0.61 under
  the reference prior, not the elicited guess. Every prior band is compared
  with `prior_functionals`, never with the elicitation guess.
- **Our sample, not the paper's.** §5.1 uses one i.i.d. sample of 200 from the
  lognormal mixture (seed 721383); §5.2 uses the transcribed leukemia times.
  "Pass" means every §6.3 inequality holds, not that the figures are
  pixel-identical to the paper's.
- **Amendments A1–A4** (per-prior n\* floors, change-point tolerances, the
  Pr(F_B > F_A) floor and the shared `t_max`) are part of the acceptance
  criteria and visible in the clause tables of §4–§6.
- **`log_post` is a diagnostic, not a marginal likelihood.** No model
  comparison, no Bayes factor, no claim beyond Kottas (2002) §5.1/§5.2.
- **The b_phi constant.** The model constants in the code are `a_phi = 2`,
  `b_phi = 1.0`; plan decision 1 records `b_phi = 1.77522`. The two differ, the
  difference is pre-existing in the sampler's prior dataclass and is not
  changed by this run (it would invalidate the comparison with the note). It is
  listed here so a reader does not mistake the reproduction for the paper's
  exact prior.
- **The `first_500` column is a probe emulation**, not a second fit: it
  truncates the stored chains and recomputes the functionals from them, so its
  `J_common` is smaller than the full trace's (see §3).

New in this run: the manifest's `code_sha` is a single value for both
sections, while the §5.2 draws were produced by `4a3e6be` and the §5.1 draws by
`21caf16`; `logs/run-5.2.log` records the §5.2 re-render, and §1 above records
the split.

## 10. Reproduction cost

Measured wall clock from `logs/events-{5.1,5.2}.jsonl` (one number per fit,
covering sampling, functionals and saving):

| section | fit | draws | seconds | per chain | implied per sweep |
| --- | --- | --- | --- | --- | --- |
| §5.2 | A Gamma(2, 0.9) | 40 000 | 2 422 | 40 min | 1.61 ms |
| §5.2 | A Gamma(2, 0.1) | 40 000 | 3 914 | 65 min | 2.60 ms |
| §5.2 | A Gamma(3, 0.05) | 40 000 | 5 124 | 85 min | 3.40 ms |
| §5.2 | B Gamma(2, 0.9) | 40 000 | 2 516 | 42 min | 1.67 ms |
| §5.1 | Gamma(2, 0.9) | 40 000 | 10 933 | 3 h 02 m | 7.26 ms |
| §5.1 | Gamma(2, 0.1) | 40 000 | 11 346 | 3 h 09 m | 7.54 ms |
| §5.1 | Gamma(3, 0.05) | 40 000 | 11 926 | 3 h 19 m | 7.92 ms |

Per sweep is `seconds / 1 505 000` (4 chains run in parallel, so it is also the
per-chain cost). §5.2 totals 3 h 54 m end to end (4 fits), §5.1 9 h 35 m
(3 fits), 13 h 29 m for the 7-fit run. The n=200 fits cost about 4.5× the n=20
fits per sweep, and the Gamma(3, 0.05) fits about 2× the Gamma(2, 0.9) fits
because the truncation level J grows with the prior's ν scale (328 → 3 157 in
§5.2, 1 921 → 2 198 in §5.1).

The functionals share is not separately logged, but it is visible in the
single-core phase after the four sampling workers exit: for §5.1's
Gamma(3, 0.05) fit that phase ran about an hour of the 3 h 19 m, which matches
the old note's "≈45 min of functionals per §5.1 fit". Extrapolating for the
next run: budget `0.7 × per-sweep × 1 505 000` for sampling and ~15 % of that
again for functionals, and remember that per-fit saving bounds any loss.
