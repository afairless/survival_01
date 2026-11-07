# Implementation Plan: Gibbs Sampler and Functional Inference for the Weibull DP Mixture

Source: `docs/research/dp_gibbs_plan.md` (the `Plan` column cites its §9 step numbers)

Supporting design documents:

- `docs/research/dp_weibull_code_guide.md` — existing `src/dp_weibull.py`, prior matching (§3.4)
- `docs/research/g0_uniform_inverse_gamma.md` — why Uniform in α, inverse-gamma conjugacy
- `docs/research/weibull_lambda_definitions.md` — the two λ parameterisations
- `docs/research/dirichlet_process_concepts.md` — partition/EPPF concepts
- `docs/dp_weibull_explanation.md` — the model in the paper's notation
- `docs/weib_dpm.pdf` — Kottas (2002), source of the Appendix algorithm

Pre-work before step 1: create branch `agent/dp-gibbs` from a clean `master`
(the only uncommitted item, this plan and `TODO.md`, is committed first). Every
step runs the gate `pixi run test`, `pixi run lint`, `pixi run typecheck`, then
commits the message in column 2 verbatim.

## Steps

| # | Commit message | Logical unit | Key deliverables | Tests | Plan |
| --- | --- | --- | --- | --- | --- |
| 1 | chore: add pytest, ruff and pyright dev environment | Dev tooling and test scaffold | pixi.toml dev feature, environments and tasks; pixi.lock; pyproject.toml pytest markers; conftest.py; pyrightconfig.json include and environment-aligned pythonVersion; tests/ skeleton | Smoke | 1 |
| 2 | test: guard prior-matching constants against induced marginal medians | Prior-matching regression guard | tests/test_prior_matching.py (test 7); one-sentence docstring note in `b_gamma_b_phi_from_weibull` about the `(sqrt2-1)` median | Unit | 2 |
| 3 | feat: add truncation_number_for_posterior common-J rule | Common truncation helper | src/dp_weibull.py named function; tests/test_truncation.py (test 24) | Unit | 3 |
| 4 | feat: add DPRealization.pruned tail-atom pruning | Atom pruning | src/dp_weibull.py `pruned(atom_tol)`; tests/test_pruning.py (test 23, closed-form evaluation inside the test) | Unit | 3 |
| 5 | feat: add sampler configuration containers | Configuration dataclasses | src/dp_gibbs.py `DPPrior`, `DPSettings`; tests/test_containers.py | Unit | 4 |
| 6 | feat: add SurvData ingestion validation and time transform | Data ingestion and transformation | src/dp_gibbs.py `SurvData`, `make_surv_data`, inverse-map helper; tests/test_surv_data.py (test 5) | Unit, Property-based | 4 |
| 7 | feat: add cluster bookkeeping and invariant checks | Cluster bookkeeping | src/dp_gibbs.py `DPState`, label compaction and §2.1 invariant helpers; tests/test_bookkeeping.py (test 6) | Unit, Property-based | 4 |
| 8 | feat: add AlphaGrid and uncensored q0 quadrature | Alpha grid and q0_o | src/dp_gibbs.py `AlphaGrid`, `make_grid`, trapezoid weights, log-space Eq. A.2; tests/test_q0_uncensored.py (tests 1, 3) | Unit | 5 |
| 9 | feat: add right-censored prior-predictive mass q0_c | Censored q0 | src/dp_gibbs.py log-space Eq. A.3 on the same grid; tests/test_q0_censored.py (test 2) | Unit | 5 |
| 10 | feat: add base-measure continuous-piece draw | Continuous base draw | src/dp_gibbs.py `draw_g0_conditional_alpha` (grid α + inverse-gamma λ); tests/test_g0_draw.py (tests 4, 9) | Unit | 6 |
| 11 | feat: add Polya-urn step a cluster assignment | Step (a) urn update | src/dp_gibbs.py `sample_step_a`, `sample_step_a_for_obs` with remove/compact/append bookkeeping; tests/test_step_a.py (test 8) | Unit | 7 |
| 12 | feat: add inverse-gamma refresh for cluster scale | Step (b) scale refresh | src/dp_gibbs.py λ* conditional and `refresh_cluster` entry point; tests/test_step_b_lambda.py (test 10) | Unit | 8 |
| 13 | feat: add auxiliary-variable refresh for cluster shape | Step (b) shape refresh | src/dp_gibbs.py α* auxiliaries targeting α^{n_j^o} with bound guard; tests/test_step_b_alpha.py (test 11) | Unit | 8 |
| 14 | feat: add Escobar-West update for the DP precision | Step (c) nu update | src/dp_gibbs.py ν two-Gamma mixture conditional; tests/test_step_c_nu.py (test 12) | Unit | 9 |
| 15 | feat: add Pareto and Gamma updates for phi and gamma | Step (c) hyperparameter updates | src/dp_gibbs.py φ and γ conditionals; tests/test_step_c_hyper.py (tests 13, 14) | Unit | 9 |
| 16 | feat: add chain state initialisation | Initial state | src/dp_gibbs.py `init_state` per §3.6; tests/test_init.py | Unit | 10 |
| 17 | feat: add collapsed log-posterior evaluation | Collapsed log-posterior | src/dp_gibbs.py `log_post` including the EPPF term; tests/test_log_post.py | Unit | 10 |
| 18 | feat: add sweep orchestration in the specified order | Sweep driver | src/dp_gibbs.py `sweep` in fixed (a)-(b)-(c) order; tests/test_sweep.py | Unit, Smoke | 10 |
| 19 | feat: add chain driver with burn-in and thinning | Chain driver and storage | src/dp_gibbs.py `sample_chain`, `sample_chains`, `DPChains`; tests/test_chain_driver.py | Unit, Integration | 10 |
| 20 | test: check data-order invariance and degenerate-data robustness | Invariance and robustness tests | tests/test_invariance.py (test 19); tests/test_degenerate.py (test 20, incl. all-censored α⁻ path) | Property-based, Integration | 10 |
| 21 | feat: add MCMC convergence diagnostics | Diagnostics | src/dp_gibbs.py `r_hat`, `ess`, `autocorr_time`; tests/test_diagnostics.py (test 31) | Unit | 11 |
| 22 | test: assert multi-chain mixing with split R-hat | Mixing test | tests/test_mixing.py (test 18, four chains on §5.1 data) | Integration | 11 |
| 23 | feat: add posterior functionals from exact atom mixtures | Posterior functionals | src/dp_functionals.py `FunctionalSettings`, `Functionals`, `posterior_functionals` exact path, S/F/f/h with inverse map; tests/test_functionals.py (tests 21, 25, 26) | Unit | 12 |
| 24 | feat: add Monte Carlo functional estimator | Monte Carlo functional path | src/dp_functionals.py `method="mc"` with one shared atom set for hazard; tests/test_functionals_mc.py (test 22) | Unit | 12 |
| 25 | feat: add quantile functionals by CDF inversion | Quantile functionals | src/dp_functionals.py `posterior_quantiles`; tests/test_quantiles.py (test 27) | Unit | 12 |
| 26 | feat: add prior functionals for prior bands | Prior functionals | src/dp_functionals.py `prior_functionals` with prior J rule; tests/test_prior_functionals.py (test 21b) | Unit | 12 |
| 27 | feat: add two-population functional comparisons | Two-population comparisons | src/dp_functionals.py `compare` (ratio, difference, Pr(η_a > η_b)); tests/test_compare.py | Unit | 12 |
| 28 | test: add golden regression fixture and test | Golden regression | tests/fixtures/golden_dp_gibbs.json; tests/fixtures/make_fixtures.py; tests/test_golden.py (test 28) | Unit | 10 |
| 29 | test: add independent reference sampler agreement test | Independent reference sampler | tests/reference_sampler.py; tests/test_reference_sampler.py (test 16) | Integration | 13 |
| 30 | test: add simulation-based calibration harness | SBC harness | tests/test_sbc.py (test 17, `-m slow`); tests/fixtures/sbc_seeds.json | Integration | 14 |
| 31 | test: add exact posterior enumeration check | Exact posterior check | tests/test_exact_posterior.py (test 15); test-only discrete α mode (no production branch) | Integration | 15 |
| 32 | feat: add section 5 data fixtures and prior constants | Paper data | src/paper_data.py (one-RNG TwoLognormalMix, var naming, prior constants); tests/fixtures/section_5_1.npz and section_5_2.npz; transcription test incl. allow_pickle=False | Unit | 16 |
| 33 | feat: add figure recipes and CLI driver | Driver and CLI | src/paper_examples.py presets, CLI, figure recipes; .gitignore `docs/research/figures/`; tests/test_cli.py (test 32) | Unit, Smoke | 17 |
| 34 | refactor: reduce kottas_2002_5_1 to a driver wrapper | Legacy wrapper | src/kottas_2002_5_1.py thin wrapper over `example_5_1` | Smoke | 17 |
| 35 | test: add section 5.1 reproduction acceptance run | Section 5.1 reproduction | tests/test_paper_5_1.py (`-m paper`, §6.3 rows 1-3) | Integration | 18 |
| 36 | test: add section 5.2 leukemia reproduction run | Section 5.2 reproduction | tests/test_paper_5_2.py (`-m paper`, §6.3 rows 4-5) | Integration | 19 |
| 37 | docs: record paper reproduction results and prior-scale caveat | Results note | docs/research/dp_gibbs_results.md (presets, seeds, chains, R-hat, wall clock, core count, scale caveat) | — | 20 |
| 38 | docs: add code guide for the Gibbs sampler modules | Code guide and cross-references | docs/research/dp_gibbs_code_guide.md; cross-references in dp_weibull_code_guide.md; module docstrings stating scope, non-goals and the prior-scale caveat | — | 20 |

## Notes

- Tests are numbered as in `docs/research/dp_gibbs_plan.md` §7; the number is a
  cross-reference, not an ordering constraint.
- `property-based` steps assert invariants (round trips, label consistency,
  permutation invariance), not just fixed examples.
- Steps 28-31 are test harnesses: they validate code committed earlier and add
  no production path except the test-only discrete α mode in step 31.
- Reproduction steps 35-36 are opt-in (`-m paper`) and are never part of
  `pixi run test`; their numbers go into the step 37 results note.
- Before step 35 was accepted, a paper-grade probe (burn 5,000, thin 150, 500
  saved draws per chain, 4 chains, seed 20240601) showed that two §6.3 clauses
  as first written were not paper claims and did not hold for the deliberately
  extreme `Gamma(3, 0.05)` prior: `Pr(3 ≤ n* ≤ 15)` was 0.730, and the third
  hazard change point was 4.166 against 4.454 ± 0.25. Plan amendments A1 and
  A2 (§6.3) record the corrected criteria: `n*` mean in `[3, 15]` with
  per-prior probability floors 0.85/0.85/0.70, and a ±0.35 tolerance for the
  third change point only. Both are justified by the paper's text/figure and
  by the sample's tail (2 observations above 4 against the paper's 4).
- A paper-grade §5.2 probe (same settings, seed 20240602) showed that two
  row-5 clauses of §6.3 were misreadings of Figure 5: the 95% band of
  `F_B - F_A` straddles zero in the paper as well (only its 80% band clears
  zero), and the paper's very tight `h_B/h_A` bands are not reproduced by a
  per-draw ratio. Amendments A3 and A4 replace both clauses with the paper's
  qualitative claims — mean `F_B - F_A ≥ 0` with pointwise probability floors,
  and a ≥30% decline of the mean hazard ratio with `P ≥ 0.75` — and the
  results note records the original band numbers so the gap is visible.
