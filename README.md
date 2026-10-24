# FreeAgent — Buying Points, Not Names

A local football transfer **research application**: player adjusted plus-minus → score probabilities → paired season scenarios → displacement-aware expected points → financial scenarios. It does not establish causal transfer effects or independently verify fees and wages.

## Run locally

Python 3.11+ (tested with 3.12). From this directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export PYTHONPATH="$PWD/src"
.venv/bin/freeagent ingest --competition 2 --season 27
.venv/bin/freeagent validate
.venv/bin/freeagent train --cutoff 2016-02-01 --bootstrap 20
.venv/bin/freeagent backtest --cutoffs 2015-12-01 2016-02-01 2016-04-01 2016-06-01
.venv/bin/python -m pytest -q
.venv/bin/streamlit run app.py
```

`PYTHONPATH` explicitly selects current source during development, including environments where editable installation registration is disabled. With pip instead of uv, use `python -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.

The dashboard never trains on interaction and has no synthetic demo fallback. In the evaluator enter your own fee, wage, contract, point price with justification, and minutes allocation. These are scenario assumptions, not observed transfer facts. Expensive models are cached; evaluations run only on explicit submission. Dataset downloads are cached and resumable.

For CLI scenarios: `.venv/bin/freeagent evaluate --scenario config/research_scenario.json`. The example is created only after real-data validation, uses real player IDs, and labels financial assumptions explicitly. Global `--data` and `--config` flags precede the subcommand. `FREEAGENT_CONFIG` selects an alternate YAML configuration; no API keys are required.

## Architecture

- `src/freeagent/data`: StatsBomb connector, provenance, strict contracts, appearance quarantine, minutes features.
- `impact/ridge_model.py`: chronological alpha selection, signed player minutes, team controls, match bootstrap.
- `match`: independent Poisson and Dixon–Coles likelihood models.
- `simulation/monte_carlo.py`: vectorised simulations, paired uniforms, initial standings, configurable objectives.
- `valuation`: minutes conservation, financial distributions, similarity, sell-side accounting, binary budget optimisation.
- `services/analysis_service.py`: training, versioned artifacts, propagation, integrated evaluation.
- `evaluation`: expanding historical forecasts, proper scores, calibration, transfer-run interface.
- `app.py`: eight Streamlit screens backed by these services.
- `tests`: labelled synthetic mathematical/integration fixtures plus actual-app checks.
- `reports/backtest`: real-data forecast records and metrics, when generated.

See [methodology](docs/METHODOLOGY.md), [data dictionary](docs/DATA_DICTIONARY.md), [model card](docs/MODEL_CARD.md), [limitations](docs/LIMITATIONS.md), and [execution status](docs/IMPLEMENTATION_STATUS.md).

## Data and permitted use

Connected source: [StatsBomb Open Data](https://github.com/statsbomb/open-data), Premier League 2015/16 (competition 2, season 27). Matches and lineup intervals supply scores, IDs, roles, and player involvement. Raw requests, retrieval timestamps and SHA-256 hashes are stored in `data/raw/manifest.jsonl`; the agreement is retained at `data/raw/LICENSE.pdf`. The source is research-oriented. Its agreement restricts commercial exploitation and redistribution; published analysis requires StatsBomb attribution and brand logo. Review the source agreement before publishing or deploying commercially. Raw and processed source data are excluded from version control.

[football-data.co.uk](https://football-data.co.uk/data.php) was investigated but not ingested: its current terms explicitly exclude automated AI/data-training products. The initially inspected Bundesliga 2023/24 release covers only Leverkusen matches, so it is not used for league-wide results. TransferIQ's reference URL could not be fetched; no code or results were copied.

No reliable fees, wages, injury records, contractual data or historical transfer labels are bundled. CSV imports are supported by placing files matching `docs/DATA_DICTIONARY.md` in `data/processed`, then running `validate`. Use stable provider IDs and explicit team/player mapping tables; never merge by display name.

## Interpretation

Expected EPA integrates match expected points exactly for each parameter draw; paired simulations independently estimate the same difference and provide league-rank outcomes. Bootstrap ranges are **uncalibrated**. Predictive match-outcome ranges are reported separately and depend on a counterfactual coupling assumption. Financial cost is amortised over the specified sporting horizon; a universal value per point is never assumed.

A successful software run is not evidence of real-world causal or commercial validity. The historical report distinguishes match forecasting from player diagnostics conditional on observed minutes. Historical causal EPA and financial verdict accuracy remain unvalidated without defensible labels.

## Verified handoff

The completed local run passed **32 tests** and all real-data software acceptance checks.
Run `.venv/bin/python scripts/acceptance.py` after ingesting, training, backtesting and evaluating the supplied research scenario. This verifies the saved artifacts and paired EPA agreement; it does not validate causal or commercial claims. Full execution evidence and limitations are in `docs/IMPLEMENTATION_STATUS.md`.

For the exact tested dependency versions, install `requirements.lock` instead of `requirements.txt`.
Static, standalone interactive evaluation charts are exported to `reports/backtest/calibration.html` and `prediction_vs_outcome.html`.
