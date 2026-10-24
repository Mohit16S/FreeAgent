# Implementation status — 2026-10-25

## Outcome

**Core software pipeline passed on real historical inputs.** See `reports/acceptance.json`: all 10 acceptance checks passed. This establishes end-to-end functionality, not causal transfer validity or calibrated financial accuracy.

**32 tests passed, 0 failed**, latest full run 3.32 seconds (`reports/tests.txt`). Synthetic mathematical fixtures are used only in tests. The actual-app submission test uses the real saved dataset and explicitly hypothetical financial assumptions.

## Completed

- [x] Repository inspection: initially empty workspace, no starter code or data to preserve.
- [x] Documented StatsBomb ingestion with cached original files, URLs, retrieval dates, hashes and agreement.
- [x] Strict schemas, ID mappings, temporal validation, quality reporting, explicit appearance quarantine.
- [x] Genuine match-level signed-minutes Ridge matrix, team/opponent controls, home intercept, chronological alpha selection, match bootstrap and sparse-player safeguard.
- [x] Identifiable Poisson and Dixon–Coles likelihood fitting, convergence checks, corrected score probabilities and expected points.
- [x] Vectorised full/remaining-season simulation, initial standings, common random numbers, rank/objective outputs.
- [x] Incumbent displacement and minutes conservation; explicit impact-to-rate mapping; exact EPA and paired simulation cross-check.
- [x] Player and match parameter uncertainty, scenario availability and separate match-outcome variability.
- [x] Horizon-consistent financial costs, Net Value distributions, configurable verdict thresholds, financial/minutes sensitivity.
- [x] Chronological match backtests, baselines, calibration, per-fold records, failures, error analysis and standalone HTML charts.
- [x] Versioned model artifacts, training data hashes, stale-artifact rejection, seeded CLI and shared YAML/environment configuration.
- [x] Eight Streamlit screens connected to the service; actual real-data submission and result pages tested.
- [x] Documentation, locked tested dependency versions and reproducible setup commands.

## Real data and training evidence

Source: StatsBomb Open Data, Premier League 2015/16 (competition 2, season 27).

- 380 matches; 20 teams; complete double round-robin confirmed.
- 10,302 retained player-match rows; 549 distinct players across the dataset.
- Six match appearance sets quarantined for inconsistent minute totals. All 380 match results retained. See `data/processed/quality.json` and `quarantined_appearances.csv`.
- Model cutoff: 2016-02-01; latest training match: 2016-01-24.
- Score model: 230 training matches; optimiser converged in 23 iterations.
- Player model: 225 training matches with valid appearances; 483 player features plus 20 team controls; design rank 225. Rank deficiency/collinearity is disclosed.
- Selected Ridge alpha: 100. Twenty player bootstrap and twenty score-model parametric bootstrap fits saved.
- Data retrieval records are in `data/raw/manifest.jsonl`; hashes and source timestamps are also in the quality report.

## Actual chronological evaluation

240 held-out matches across three folds:

| Cutoff | Evaluation end | Training matches | Test matches |
|---|---|---:|---:|
| 2015-12-01 | 2016-02-01 | 140 | 90 |
| 2016-02-01 | 2016-04-01 | 230 | 73 |
| 2016-04-01 | 2016-06-01 | 303 | 77 |

| Model | Log loss | RPS | Brier |
|---|---:|---:|---:|
| Dixon–Coles | 1.060002 | 0.214133 | 0.635157 |
| Independent Poisson | 1.059453 | 0.214186 | 0.635370 |
| Training result-frequency baseline | 1.085681 | 0.226943 | 0.657530 |

Lower is better. Dixon–Coles did **not** improve log loss over independent Poisson in this run. Differences are not established as statistically significant.

Player-model held-out outcome reconstruction, **conditional on observed minutes**, covers 239 matches: MAE 1.267548 goals and RMSE 1.668062 goals. It is not a pre-match or causal forecast. One held-out match lacks usable appearances and is excluded from that diagnostic. Match metrics still include it. No failed forecast folds.

## End-to-end scenario evidence

`config/research_scenario.json` is explicitly hypothetical: Harry Kane replacing Roberto Firmino at Liverpool using real pre-February-2016 football estimates; fee, wage, point-price and minutes are arbitrary editable research assumptions, not historical transaction facts.

- Exact expected EPA: 0.398043 points.
- Paired simulation estimate: 0.387400; estimated simulation standard error 0.012581.
- Bootstrap/scenario EPA range: [0.166556, 0.638956], **uncalibrated**.
- No scoring-rate floors activated. Minutes conserved.
- The financial calculation completed and is saved in the scenario report; it must not be cited as an actual player market appraisal.

## Extensions after core integration passed

1. Wage/fee amortisation and explicit accounting horizon: complete for ≤1 year; no DCF or multi-year sporting forecast.
2. Explainability: implemented only for computed components; attack/defence split disclosed as assumed.
3. Alternatives: tested same-role standardised similarity, exposed on player screen; prices unavailable.
4. Availability: scenario-based minutes uncertainty implemented; medical injury model blocked by absent data.
5. Sell-side: tested accounting comparison utility; requires explicit replacement scenario and book value.
6. Budget: tested binary MILP with cash, role and incumbent constraints; additive approximation, not joint multi-signing simulation.

## Commands executed

```bash
python3 -m pytest -q  # initial baseline failed: pytest not installed
uv venv --python /Users/mohit/.local/bin/python3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
export PYTHONPATH="$PWD/src"
.venv/bin/freeagent ingest --competition 2 --season 27
.venv/bin/freeagent validate
.venv/bin/freeagent train --cutoff 2016-02-01 --bootstrap 20
.venv/bin/freeagent backtest --cutoffs 2015-12-01 2016-02-01 2016-04-01 2016-06-01
.venv/bin/freeagent evaluate --scenario config/research_scenario.json
.venv/bin/python -m pytest -q
.venv/bin/python scripts/acceptance.py
uv pip install --python .venv/bin/python --reinstall --no-deps .
.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Network/package setup and localhost binding required sandbox approvals. Dependencies are recorded in `requirements.lock`. The final package was installed normally to avoid an environment-specific editable .pth registration issue.

## Defects encountered and resolved

- Empty environment lacked pytest: installed isolated Python 3.12 dependencies.
- Network blocked in sandbox: resumed public downloads after approval, using cache.
- Nullable lineup endpoint period: handled explicitly; regulation clipping documented.
- Selected Bundesliga release covered only one club: retained raw inspection files, selected complete EPL source.
- Source lineup totals inconsistent: quarantined six EPL appearance sets, with audit records.
- AppTest relative path resolution: switched to verified absolute app path.
- NumPy synthetic-fixture timedelta warnings: replaced with standard-library date arithmetic; final run clean.
- Stale development module in live Streamlit process: restarted with final package. App tests run in clean processes.

## Remaining blockers and supported limits

- No date-stamped historical transfer contracts, reliable fees/wages, injury records or causal EPA truth labels. Transfer walk-forward callback exists, but historical transfer/financial validation is a scaffold, not a completed empirical study.
- Impact-to-rate split and bootstrap coverage are not externally calibrated; defaults are research assumptions.
- Full squad/tactical optimisation, selling-club effects and exact competition-specific head-to-head/playoff tie rules are not implemented. Supported minutes replacement and tie approximations are explicit.
- StatsBomb agreement restricts commercial exploitation and redistribution; commercial deployment requires suitable rights. football-data.co.uk was not ingested because its current terms exclude automated AI training. TransferIQ could not be fetched; no code was copied.

Next highest-priority action: obtain permissioned multi-season appearances/rosters and dated transfer contracts, then validate the impact-to-rate mapping and interval coverage on held-out seasons before interpreting financial verdicts as decision-grade.

Final live-browser verification: restarted dashboard successfully displayed all 380 matches, 549 players, 20 teams, saved cutoff and benchmark table at http://127.0.0.1:8501. Narrow-panel header spacing adjusted after visual inspection.
