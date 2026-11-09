# Artifact-Backed Replication Report: Kottas (2002) §5.1 and §5.2

This is the artifact-backed record of the paper reproduction. Unlike
`dp_gibbs_results.md`, every number below is regenerated from stored draws by
`src/paper_report.py`, and the draws, functionals, figures, logs and checksums
sit in one git-untracked run directory:

| item | value |
| --- | --- |
| run path | `output/replication-2026-11-08T02-26-53Z` |
| sections | 5.1 and 5.2, `paper` preset, all priors |
| fits / chains / stored draws | 7 / 28 / 280 000 |
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

The tables are spliced in verbatim from each section's `report/tables.md`, so a
reviewer can diff them against the run tree. `dp_gibbs_results.md` remains the
narrative note; this report supersedes it as the auditable record and says in
§7 exactly what reproduces and what does not.

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

Three code commits matter. `4a3e6be` produced the §5.2 draws. `21caf16` (the
first-500 estimator fix, §7 below) produced the §5.1 draws and re-rendered the
§5.2 report. `0234f59` re-rendered the §5.1 report after one baseline row was
reclassified. No draw was regenerated after `21caf16`.

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
`b_phi=1.77522` (plan decision 1; the sampler's prior dataclass uses
`b_phi=1.0` — the pre-existing discrepancy is listed in §9).

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
draws, with the truncation level recomputed from the slice). The `first_500`
`J_common` is the slice's own, so it is smaller than the full trace's; every
fit satisfies R̂ < 1.05 and, on the slice, ESS ≥ 1 421 for ν and n\* — the
acceptance gate of plan §3.7, which is the strict version of the statistic.

Verbatim from `5.1/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.1 --write`.

| fit | draw set | draws per chain | R-hat nu | ESS nu | R-hat n* | ESS n* | R-hat S | ESS S | J_common | J_prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nu ~ Gamma(2, 0.9) | full | 10000 | 1.00007 | 28831.4 | 1.00005 | 22482.8 | 1.00001, 1.00006, 1.0001, 1.0003, 1.00006 | 33134.2, 37317.2, 25665.8, 10681, 22933.2 | 1921 | 99 |
| nu ~ Gamma(2, 0.9) | first_500 | 500 | 1.00208 | 1569.04 | 1.00328 | 1495.58 | 0.998944, 0.998685, 1.00159, 1.00567, 0.999373 | 1673.92, 1719.01, 1373.36, 756.907, 1335.59 | 1901 | 99 |
| nu ~ Gamma(2, 0.1) | full | 10000 | 0.999993 | 30349.3 | 1.00002 | 28977.4 | 1.00001, 1.00002, 1.00004, 1.0002, 1.00007 | 35989, 38270.2, 26538.3, 13554.8, 28380.8 | 2031 | n/a |
| nu ~ Gamma(2, 0.1) | first_500 | 500 | 1.00046 | 1626.92 | 1.00233 | 1510.65 | 0.999733, 0.999295, 0.999099, 1.00362, 1.00062 | 1640.88, 1963.21, 1496.68, 933.492, 1337.07 | 1982 | n/a |
| nu ~ Gamma(3, 0.05) | full | 10000 | 0.999937 | 36694.2 | 0.999955 | 34869.9 | 1.00002, 0.999997, 1.00011, 1.00002, 0.999929 | 37502.3, 39704.5, 27934.8, 17750.7, 35452.3 | 2198 | n/a |
| nu ~ Gamma(3, 0.05) | first_500 | 500 | 1.00023 | 1793.53 | 0.999749 | 1598.57 | 1.00015, 0.998947, 0.998914, 0.998722, 1.00003 | 1980.98, 1896.05, 1584.55, 821.911, 1790.23 | 2053 | n/a |

The `first_500` block truncates each chain to 500 stored draws and recomputes the truncation level from the slice, so its `J_common` is smaller than the full trace's; `J_prior` belongs to the prior band, which is row-sliced. `R-hat S` and `ESS S` are the five monitored times in the order `nu` reports them.

Verbatim from `5.2/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.2 --write`.

| fit | draw set | draws per chain | R-hat nu | ESS nu | R-hat n* | ESS n* | R-hat S | ESS S | J_common | J_prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: nu ~ Gamma(2, 0.9) | full | 10000 | 1.00007 | 39211 | 1 | 39588.6 | 0.999921, 0.999955, 0.999991, 1.00004, 1.00009 | 38772.1, 38726, 38469.9, 38791.3, 38838.7 | 328 | n/a |
| A: nu ~ Gamma(2, 0.9) | first_500 | 500 | 1.00097 | 1822.29 | 1.00057 | 1971.56 | 1.00019, 0.999826, 0.999188, 0.999668, 0.999353 | 1976.98, 1841.88, 1983.01, 1802.58, 1724.36 | 328 | n/a |
| A: nu ~ Gamma(2, 0.1) | full | 10000 | 1.00009 | 39707.6 | 1.00017 | 39689.3 | 1.00011, 1.00009, 0.999995, 0.999966, 0.999967 | 39559.7, 39765.1, 40049, 39752, 39636.3 | 1419 | n/a |
| A: nu ~ Gamma(2, 0.1) | first_500 | 500 | 0.99967 | 1421.35 | 0.999933 | 1473.91 | 1.00044, 1.00015, 1.00054, 1.00011, 0.999912 | 1806.47, 1827.4, 1910.59, 2014.45, 2014.3 | 960 | n/a |
| A: nu ~ Gamma(3, 0.05) | full | 10000 | 1.00001 | 38264.5 | 1.00006 | 38323.7 | 1.00014, 1.0001, 1.00002, 1, 0.999981 | 38895.3, 39266.3, 39366.8, 39931.7, 40222 | 3157 | n/a |
| A: nu ~ Gamma(3, 0.05) | first_500 | 500 | 1.00157 | 1602.09 | 0.999918 | 1874.16 | 0.999693, 0.999461, 0.998846, 0.99867, 0.998557 | 1927.92, 1962.32, 2004.73, 2029.88, 1996.97 | 2663 | n/a |
| B: nu ~ Gamma(2, 0.9) | full | 10000 | 0.999984 | 39730.5 | 1.00004 | 37208.8 | 0.999987, 0.99997, 0.999949, 0.999937, 0.999957 | 39954.9, 39285.8, 39563, 39918.9, 38801.1 | 335 | n/a |
| B: nu ~ Gamma(2, 0.9) | first_500 | 500 | 0.999624 | 1893.34 | 0.999043 | 1834.19 | 0.999021, 0.999425, 0.999141, 0.998927, 0.998963 | 2014.43, 1999.24, 1917.86, 1819.8, 1784.42 | 296 | n/a |

The `first_500` block truncates each chain to 500 stored draws and recomputes the truncation level from the slice, so its `J_common` is smaller than the full trace's; `J_prior` belongs to the prior band, which is row-sliced. `R-hat S` and `ESS S` are the five monitored times in the order `nu` reports them.

## 4. §5.1 achieved values

Full draws (40 000 per fit); the note's first-500 probe values are in §7.
Figures 1–3 are `5.1/figures/figure_{1,2,3}.png`.

*Figure 1*

| prior | nu prior mean | nu posterior mean | nu 95% CI | nu prior width | nu posterior width | nu 99th pct | n* mean | Pr(3 <= n* <= 15) | n* range | learning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nu ~ Gamma(2, 0.9) | 2.22222 | 1.52275 | 0.36338, 3.64005 | 5.92159 | 3.27667 | 4.29029 | 7.15623 | 0.983 | 2, 27 | yes |
| nu ~ Gamma(2, 0.1) | 20 | 2.67805 | 0.582327, 7.01866 | 53.2943 | 6.43633 | 8.51611 | 9.92747 | 0.8848 | 2, 46 | yes |
| nu ~ Gamma(3, 0.05) | 60 | 4.0573 | 1.016, 10.1665 | 132.12 | 9.1505 | 12.2494 | 12.9846 | 0.71745 | 2, 63 | yes |

*Figure 2*

| prior | mean S(1.1727) | int f dt | f(3.26) | f(2.4) | prior band width | posterior band width | prior band wider |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nu ~ Gamma(2, 0.9) | 0.495203 | 0.975512 | 0.147825 | 0.128715 | 0.858727 | 0.0935809 | yes |
| nu ~ Gamma(2, 0.1) | 0.495337 | 0.974013 | 0.147962 | 0.128313 | n/a | n/a | n/a |
| nu ~ Gamma(3, 0.05) | 0.494974 | 0.972189 | 0.147691 | 0.128965 | n/a | n/a | n/a |

Band widths are the reference prior's (`nu ~ Gamma(2, 0.9)`) on (0, 4].

*Figure 3*

| prior | change points | band width t=2.5 | band width t=4.5 | far / near |
| --- | --- | --- | --- | --- |
| nu ~ Gamma(2, 0.9) | 1.01452, 2.1374, 4.46194 | 0.38667 | 4.69412 | 12.1399 |
| nu ~ Gamma(2, 0.1) | 1.01452, 2.1374, 4.24524 | 0.384343 | 4.51741 | 11.7536 |
| nu ~ Gamma(3, 0.05) | 1.03422, 2.1374, 4.14675 | 0.38407 | 4.16908 | 10.855 |

Gaussian smoothing with a bandwidth of 5 grid points; change points are the sign changes of the smoothed mean on (0, 5).

The per-prior n\* floors in the acceptance table below are amendment A1 (the
original "≥ 0.85 under all three priors" reading was replaced by
0.85 / 0.85 / 0.70), and the change-point tolerances are amendment A2.

Verbatim from `5.1/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.1 --write`.

28 / 28 clauses pass (PASS).

| # | clause | description | target | observed | result |
| --- | --- | --- | --- | --- | --- |
| 1 | figure_1.nu_posterior_mean | posterior mean of nu increases with the prior mean | strictly increasing | 1.52275, 2.67805, 4.0573 | pass |
| 2 | figure_1.n_star_mean_increasing | posterior mean of n* increases with the prior mean of nu | strictly increasing | 7.15623, 9.92747, 12.9846 | pass |
| 3 | figure_1.n_star_posterior_mean[0] | nu ~ Gamma(2, 0.9): posterior mean of n* in [3, 15] | 3, 15 | 7.15623 | pass |
| 4 | figure_1.n_star_probability[0] | nu ~ Gamma(2, 0.9): Pr(3 <= n* <= 15) >= 0.85 | 0.85 | 0.983 | pass |
| 5 | figure_1.learning[0] | nu ~ Gamma(2, 0.9): posterior 95% width below the prior's (learning) | yes | yes | pass |
| 6 | figure_1.n_star_posterior_mean[1] | nu ~ Gamma(2, 0.1): posterior mean of n* in [3, 15] | 3, 15 | 9.92747 | pass |
| 7 | figure_1.n_star_probability[1] | nu ~ Gamma(2, 0.1): Pr(3 <= n* <= 15) >= 0.85 | 0.85 | 0.8848 | pass |
| 8 | figure_1.learning[1] | nu ~ Gamma(2, 0.1): posterior 95% width below the prior's (learning) | yes | yes | pass |
| 9 | figure_1.n_star_posterior_mean[2] | nu ~ Gamma(3, 0.05): posterior mean of n* in [3, 15] | 3, 15 | 12.9846 | pass |
| 10 | figure_1.n_star_probability[2] | nu ~ Gamma(3, 0.05): Pr(3 <= n* <= 15) >= 0.7 | 0.7 | 0.71745 | pass |
| 11 | figure_1.learning[2] | nu ~ Gamma(3, 0.05): posterior 95% width below the prior's (learning) | yes | yes | pass |
| 12 | figure_1.nu_posterior_99 | the 99th percentile of nu follows the prior scale | increasing and >= 2x the narrowest prior's | 4.29029, 8.51611, 12.2494 | pass |
| 13 | figure_2.survival_at_median[0] | nu ~ Gamma(2, 0.9): mean S(1.1727) in [0.42, 0.58] | 0.42, 0.58 | 0.495203 | pass |
| 14 | figure_2.density_mass[0] | nu ~ Gamma(2, 0.9): int f dt in [0.95, 1.05] | 0.95, 1.05 | 0.975512 | pass |
| 15 | figure_2.second_mode_density[0] | nu ~ Gamma(2, 0.9): f(3.26) > f(2.4) | > f(2.4) = 0.128715 | 0.147825 | pass |
| 16 | figure_2.survival_at_median[1] | nu ~ Gamma(2, 0.1): mean S(1.1727) in [0.42, 0.58] | 0.42, 0.58 | 0.495337 | pass |
| 17 | figure_2.density_mass[1] | nu ~ Gamma(2, 0.1): int f dt in [0.95, 1.05] | 0.95, 1.05 | 0.974013 | pass |
| 18 | figure_2.second_mode_density[1] | nu ~ Gamma(2, 0.1): f(3.26) > f(2.4) | > f(2.4) = 0.128313 | 0.147962 | pass |
| 19 | figure_2.survival_at_median[2] | nu ~ Gamma(3, 0.05): mean S(1.1727) in [0.42, 0.58] | 0.42, 0.58 | 0.494974 | pass |
| 20 | figure_2.density_mass[2] | nu ~ Gamma(3, 0.05): int f dt in [0.95, 1.05] | 0.95, 1.05 | 0.972189 | pass |
| 21 | figure_2.second_mode_density[2] | nu ~ Gamma(3, 0.05): f(3.26) > f(2.4) | > f(2.4) = 0.128965 | 0.147691 | pass |
| 22 | figure_3.change_points[0] | nu ~ Gamma(2, 0.9): 3 smoothed-hazard change points within tolerance | points=1.022, 2.366, 4.454, tolerances=0.25, 0.25, 0.35 | 1.01452, 2.1374, 4.46194 | pass |
| 23 | figure_3.band_ratio[0] | nu ~ Gamma(2, 0.9): band at t=4.5 is >= 1.5x the band at t=2.5 | 1.5 | 12.1399 | pass |
| 24 | figure_3.change_points[1] | nu ~ Gamma(2, 0.1): 3 smoothed-hazard change points within tolerance | points=1.022, 2.366, 4.454, tolerances=0.25, 0.25, 0.35 | 1.01452, 2.1374, 4.24524 | pass |
| 25 | figure_3.band_ratio[1] | nu ~ Gamma(2, 0.1): band at t=4.5 is >= 1.5x the band at t=2.5 | 1.5 | 11.7536 | pass |
| 26 | figure_3.change_points[2] | nu ~ Gamma(3, 0.05): 3 smoothed-hazard change points within tolerance | points=1.022, 2.366, 4.454, tolerances=0.25, 0.25, 0.35 | 1.03422, 2.1374, 4.14675 | pass |
| 27 | figure_3.band_ratio[2] | nu ~ Gamma(3, 0.05): band at t=4.5 is >= 1.5x the band at t=2.5 | 1.5 | 10.855 | pass |
| 28 | figure_2.prior_band_is_wider | prior 95% band is at least as wide as the posterior's on (0, 4] | yes | yes | pass |

## 5. §5.2 achieved values

Full draws (40 000 per fit); arm A is the Gamma(2, 0.9) fit, as in the paper.
Figures 4–5 are `5.2/figures/figure_{4,5}.png`, and the protocol is "arm B
versus the first arm-A fit".

*Figures 4–5*

| quantity | value |
| --- | --- |
| prior agreement minimum | 1 |
| survival difference minimum | 0.0136748 |
| arm-B peak time | 11.8335 |
| arm-B peak value | 0.0737799 |
| arm-B maximum above the peak window | 0.0587112 |
| remission window | 0, 20 |

| prior | nu ~ Gamma(2, 0.9) | nu ~ Gamma(2, 0.1) | nu ~ Gamma(3, 0.05) |
| --- | --- | --- | --- |
| nu ~ Gamma(2, 0.9) | 1 | 1 | 1 |
| nu ~ Gamma(2, 0.1) | 1 | 1 | 1 |
| nu ~ Gamma(3, 0.05) | 1 | 1 | 1 |

| quantity | value |
| --- | --- |
| Pr(eta_A > eta_B \| D) | 0.901375 |
| minimum mean F_B - F_A | 0.0136748 |
| Pr(F_B > F_A \| D) range | 0.565575, 0.9121 |
| mean Pr(F_B > F_A \| D) | 0.83625 |
| mean hazard ratio at t=5 | 2.32754 |
| mean hazard ratio at t=25 | 1.15915 |
| hazard-ratio decline (near / far) | 2.00797 |
| Pr(ratio(near) > ratio(far) \| D) | 0.82185 |
| 95% ratio band at t=5 | 0.712012, 5.59736 |
| 95% ratio band at t=25 | 0.280626, 3.08888 |
| n_eta_above_grid | fit_a=1, fit_b=0 |

The Pr(F_B > F_A) floor of 0.5 is amendment A3; `n_eta_above_grid` is a count,
recorded but never gated.

Verbatim from `5.2/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.2 --write`.

10 / 10 clauses pass (PASS).

| # | clause | description | target | observed | result |
| --- | --- | --- | --- | --- | --- |
| 1 | figure_4.prior_agreement_min | each pair of prior curves agrees within the other's 95% band | 0.95 | 1 | pass |
| 2 | figure_4.survival_difference_min | mean S_A - S_B is positive on (0.0, 20.0) | > 0 | 0.0136748 | pass |
| 3 | figure_4.arm_b_peak_time | the arm-B smoothed hazard peaks in [10, 20] | 10, 20 | 11.8335 | pass |
| 4 | figure_4.arm_b_above_peak_max | the arm-B hazard falls after its peak | < peak value 0.0737799 | 0.0587112 | pass |
| 5 | figure_5.prob_eta_a_gt_eta_b | Pr(eta_A > eta_B \| D) in [0.85, 0.96] | 0.85, 0.96 | 0.901375 | pass |
| 6 | figure_5.difference_min | mean F_B - F_A is non-negative on (0.0, 20.0) | >= 0 | 0.0136748 | pass |
| 7 | figure_5.difference_probability_range | Pr(F_B > F_A \| D) never falls below the amendment-A3 floor | minimum >= 0.5 | 0.565575, 0.9121 | pass |
| 8 | figure_5.difference_probability_mean | mean Pr(F_B > F_A \| D) >= 0.75 | 0.75 | 0.83625 | pass |
| 9 | figure_5.hazard_ratio_decline | mean hazard ratio at t=5 is >= 1.3x the one at t=25 | 1.3 | 2.00797 | pass |
| 10 | figure_5.hazard_ratio_order_probability | Pr(ratio(t=5) > ratio(t=25) \| D) >= 0.75 | 0.75 | 0.82185 | pass |

## 6. Acceptance replay

`pytest -m paper` in artifact mode loads the saved draws instead of refitting:

| section | command | result | log |
| --- | --- | --- | --- |
| §5.1 | `DP_PAPER_ARTIFACTS=…/5.1 DP_PAPER_PRESET=paper pytest -m paper tests/test_paper_5_1.py` | **12 passed** | `logs/run-5.1.log` |
| §5.2 | `DP_PAPER_ARTIFACTS=…/5.2 DP_PAPER_PRESET=paper pytest -m paper tests/test_paper_5_2.py` | **8 passed** | `logs/run-5.2.log` |

Artifact mode gives 12/12 and 8/8 rather than the 11/11 and 7/7 of the old
note: the acceptance assertions are unchanged, and one extra test per module
compares the manifest's `code_sha` with `HEAD` (skippable with
`DP_PAPER_ARTIFACTS_ALLOW_SHA_MISMATCH=1` for an older tree). The generated
clause tables of §4–§5 agree clause-for-clause with those tests: 28/28 for
§5.1 and 10/10 for §5.2, from the sections' `summary/acceptance.json` — the
same thresholds, mirrored into `ACCEPTANCE_TARGETS` and guarded by a unit test.

## 7. Agreement with the previous note

The gate of plan §7 Step 2 is the *first-500* column against the note, because
the note's probe read 500 stored draws per chain. Full-draw columns are a
sharper estimate of the same posteriors, not a second gate.

**Identical (the chains).** Every fit's first-500 R̂ and ESS reproduce §3 of the
note: ESS(ν) 1 569.04 / 1 626.92 / 1 793.53 (5.1) and 1 822.29 / 1 421.35 /
1 602.09 / 1 893.34 (5.2) against the note's 1 569 / 1 627 / 1 794 and
1 822 / 1 421 / 1 602 / 1 893, ESS(n\*) likewise to four digits. The sampler
replay is exact; no numpy, preset or code drift is involved.

**Sharpened (the summaries).** With 40 000 draws instead of 2 000, the full-draw
values of §4–§5 are the ones to quote: ν posterior means 1.523 / 2.678 / 4.057
against the probe's 1.50 / 2.71 / 3.98, Pr(η_A > η_B) 0.9014 against 0.89, the
hazard-ratio decline 2.008 against 1.95, and the §5.1 ESS values an order of
magnitude larger.

**Moved (one estimator fix).** The first-500 column initially sliced the stored
functional arrays, which kept the 10 000-draw trace's `J_common` while the
chains were truncated. That left every Figure 4–5 row 0.5–1 % away from the note
(peak time 12.02 vs 11.83; Pr(η_A > η_B) 0.887 vs 0.89) with identical chains.
`slice_result` now recomputes the functionals from the slice, so the column uses
the slice's own truncation level (`21caf16`), and every comparable row matches
the note to its displayed precision.

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

`n_eta_above_grid` is also flagged: it is a count (1 / 0 here), not a summary.
Everything else agrees; the generated delta tables are:

Verbatim from `5.1/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.1 --write`.

Signed `observed - baseline` against `docs/research/dp_gibbs_results.md`. Rows marked `not comparable` cannot match by construction and are not part of the gate.

| row | observed | baseline | delta | not comparable |
| --- | --- | --- | --- | --- |
| figure_1.nu_prior_mean | 2.22222, 20, 60 | 2.22, 20, 60 | 0.00222222, 0, 0 | no |
| figure_1.nu_posterior_mean | 1.50008, 2.71085, 3.98197 | 1.5, 2.71, 3.98 | 8.46853e-05, 0.000848278, 0.0019702 | no |
| figure_1.nu_posterior_95 | 0.375665, 3.64662, 0.591727, 7.07617, 1.0321, 9.99062 | 0.38, 3.65, 0.59, 7.08, 1.03, 9.99 | -0.00433526, -0.00338447, 0.00172708, -0.00383161, 0.00210143, 0.000615433 | no |
| figure_1.nu_prior_95_width | 5.92159, 53.2943, 132.12 | 6.1, 61, 117 | -0.178407, -7.70566, 15.1203 | yes |
| figure_1.nu_posterior_width | 3.27095, 6.48444, 8.95851 | 3.27, 6.49, 8.96 | 0.000950786, -0.0055587, -0.001486 | no |
| figure_1.nu_posterior_99 | 4.2456, 8.34733, 11.6779 | 4.25, 8.35, 11.68 | -0.0043953, -0.00267323, -0.00205611 | no |
| figure_1.n_star_posterior_mean | 7.0205, 9.8775, 12.775 | 7.02, 9.88, 12.78 | 0.0005, -0.0025, -0.005 | no |
| figure_1.n_star_probability | 0.9865, 0.885, 0.7295 | 0.987, 0.885, 0.73 | -0.0005, 0, -0.0005 | no |
| figure_1.n_star_observed_range | 2, 21, 2, 34, 3, 47 | 2, 21, 2, 34, 3, 47 | 0, 0, 0, 0, 0, 0 | no |
| figure_2.survival_at_median | 0.494269, 0.494046, 0.495184 | 0.494, 0.494, 0.495 | 0.000269343, 4.63415e-05, 0.000184164 | no |
| figure_2.density_mass | 0.975517, 0.97386, 0.972549 | 0.976, 0.974, 0.973 | -0.000483393, -0.000140474, -0.000451003 | no |
| figure_2.prior_band_width | 0.867628 | 0.833 | 0.0346276 | yes |
| figure_2.posterior_band_width | 0.0924039 | 0.092 | 0.000403903 | no |
| figure_2.second_mode_density | 0.147417, 0.147794, 0.147174 | 0.147, 0.147, 0.147 | 0.000417266, 0.000793924, 0.000174031 | no |
| figure_2.first_mode_density | 0.129096, 0.128792, 0.12831 | 0.129, 0.129, 0.129 | 9.63999e-05, -0.000208213, -0.000689875 | no |
| figure_3.change_points | 1.01452, 2.1374, 4.46194, 1.01452, 2.1374, 4.24524, 1.03422, 2.1374, 4.16645 | 1.015, 2.137, 4.462, 1.015, 2.137, 4.245, 1.034, 2.137, 4.166 | -0.000475298, 0.000396703, -6.12615e-05, -0.000475298, 0.000396703, 0.000244142, 0.000224211, 0.000396703, 0.000446107 | no |
| figure_3.band_width_near | 0.385825, 0.371638, 0.367466 | 0.386, 0.372, 0.368 | -0.000174628, -0.000362147, -0.000533562 | no |
| figure_3.band_width_far | 4.65224, 4.46276, 4.31872 | 4.65, 4.46, 4.32 | 0.00223937, 0.00275831, -0.00128442 | no |
| figure_3.band_ratio | 12.0579, 12.0084, 11.7527 | 12.1, 12, 11.8 | -0.0421114, 0.00835241, -0.0473197 | no |

Verbatim from `5.2/report/tables.md`, generated by `python -m src.paper_report --from-artifacts 5.2 --write`.

Signed `observed - baseline` against `docs/research/dp_gibbs_results.md`. Rows marked `not comparable` cannot match by construction and are not part of the gate.

| row | observed | baseline | delta | not comparable |
| --- | --- | --- | --- | --- |
| figure_4.prior_agreement | 1, 1, 1, 1, 1, 1, 1, 1, 1 | 1, 1, 1, 1, 1, 1, 1, 1, 1 | 0, 0, 0, 0, 0, 0, 0, 0, 0 | no |
| figure_4.prior_agreement_min | 1 | 1 | 0 | no |
| figure_4.survival_difference_min | 0.0100944 | 0.0101 | -5.57773e-06 | no |
| figure_4.arm_b_peak_time | 11.8335 | 11.83 | 0.0035 | no |
| figure_4.arm_b_peak_value | 0.073475 | 0.0735 | -2.49749e-05 | no |
| figure_4.arm_b_above_peak_max | 0.0592722 | 0.0593 | -2.7762e-05 | no |
| figure_5.prob_eta_a_gt_eta_b | 0.89 | 0.89 | 0 | no |
| figure_5.difference_min | 0.0100944 | 0.0101 | -5.57773e-06 | no |
| figure_5.difference_probability_range | 0.56, 0.9065 | 0.56, 0.91 | 0, -0.0035 | no |
| figure_5.difference_probability_mean | 0.825745 | 0.83 | -0.00425472 | no |
| figure_5.hazard_ratio_mean | 2.2984, 1.17635 | 2.298, 1.176 | 0.000401703, 0.0003492 | no |
| figure_5.hazard_ratio_decline | 1.95384 | 1.95 | 0.00384305 | no |
| figure_5.hazard_ratio_order_probability | 0.8105 | 0.81 | 0.0005 | no |
| figure_5.hazard_ratio_band | 0.663031, 5.4546, 0.282212, 3.10349 | 0.663, 5.455, 0.282, 3.103 | 3.11317e-05, -0.000404303, 0.000211771, 0.000485695 | no |
| figure_5.n_eta_above_grid.fit_a | 0 | n/a | n/a | yes |
| figure_5.n_eta_above_grid.fit_b | 0 | n/a | n/a | yes |

## 8. Artifact inventory

| directory | files | bytes | contents |
| --- | --- | --- | --- |
| `5.1/draws/` | 3 fits (4 chains + functionals each, plus the reference prior band) | 1 132 240 244 | chains, functionals, `fit-meta.json`, `prior-band.npz` |
| `5.2/draws/` | 4 fits (4 chains + functionals each) | 1 476 652 860 | chains, functionals, `fit-meta.json` |
| `5.1/` + `5.2/` summaries | 28 + 32 files in the two section trees | — | `stats.json`, `rhat_ess.csv`, `quantiles.csv`, `acceptance.json`, `report/tables.md`, `figures/*.png` |
| `logs/` | 7 | 8 862 | `env.txt`, `events-{5.1,5.2}.jsonl`, `run-{5.1,5.2}.log`, `pytest-paper-{5.1,5.2}.log` |
| run root | 1 | 11 887 | `MANIFEST.json` |

Array shapes, validated on load (`DPChains` and `Functionals` re-validate on
construction):

| artifact | keys | shape | dtype |
| --- | --- | --- | --- |
| `chain-b.npz` (§5.1) | `theta`, `nu`, `gamma`, `phi`, `n_star`, `log_post`, `time`, `event`, `t_max`, `t_grid`, `meta` | `theta (10000, 200, 2)`; traces `(10000,)` | float64, `n_star` int64 |
| `chain-b.npz` (§5.2) | same | `theta (10000, 20, 2)` | float64 |
| `functionals.npz` | `t_grid`, `S`, `F`, `f`, `h`, `meta` | `(40000, 300)` each | float64 |
| `prior-band.npz` | `t_grid`, `S`, `F`, `f`, `h`, `meta` | `(1000, 300)` each | float64 |

`MANIFEST.json` lists all 67 files with `sha256` and `size`, plus the code SHA,
dirty flag, pixi lock hash, package versions and the per-section settings. It
never lists itself, and `finalize_run` re-stamps `created_at`, so the durable
identity of a tree is its `code_sha` + `file_count` + `total_bytes` (the
SHA-256 above is this finalisation's).

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
  criteria and visible in the clause tables of §4–§5.
- **`log_post` is a diagnostic, not a marginal likelihood.** No model
  comparison, no Bayes factor, no claim beyond Kottas (2002) §5.1/§5.2.
- **The b_phi constant.** The code's prior dataclass uses `b_phi = 1.0` while
  plan decision 1 records `b_phi = 1.77522`. The difference is pre-existing and
  is not changed by this run (changing it would invalidate the comparison with
  the note); it is listed so a reader does not mistake the reproduction for the
  paper's exact prior.
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
per-chain cost). §5.2 totals 3 h 54 m end to end (4 fits) and §5.1 9 h 35 m
(3 fits): 13 h 29 m for the 7-fit run. The n=200 fits cost about 4.5× the n=20
fits per sweep, and the Gamma(3, 0.05) fits about 2× the Gamma(2, 0.9) fits,
because the truncation level J grows with the prior's ν scale (328 → 3 157 in
§5.2, 1 921 → 2 198 in §5.1).

The functionals share is not separately logged. It is visible in the
single-core phase after the four sampling workers exit: for §5.1's
Gamma(3, 0.05) fit that phase ran about an hour of the 3 h 19 m, matching the
old note's "≈45 min of functionals per §5.1 fit". For planning the next run,
budget `per-sweep × 1 505 000` per fit plus ~15 % for functionals and saving,
and rely on per-fit saves to bound any loss.
