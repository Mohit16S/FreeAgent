# FreeAgent — Buying Points, Not Names

A local football transfer **research application**: player adjusted plus-minus → score probabilities → paired season scenarios → displacement-aware expected points → financial scenarios. It does not establish causal transfer effects or independently verify fees and wages.

## Local setup and portable commands

Use Python 3.12 locally and select Python 3.12 in Streamlit Community Cloud. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt

# Download real data, validate, train all models and calculate backtest metrics:
python -m freeagent.cli setup --evaluate
python -m freeagent.cli status
python -m pytest -q
python -m streamlit run app.py
```

The virtual environment is optional isolation for local development. Application code and deployment commands never invoke its path. Community Cloud installs `requirements.txt`, which explicitly lists runtime dependencies and installs the local FreeAgent package. `requirements-dev.txt` adds pytest. No `PYTHONPATH` setting is required after installation.

Individual reproducible stages are also available:

```bash
python -m freeagent.cli ingest --competition 2 --season 27
python -m freeagent.cli validate
python -m freeagent.cli train --cutoff 2016-02-01 --bootstrap 20
python -m freeagent.cli backtest --cutoffs 2015-12-01 2016-02-01 2016-04-01 2016-06-01
python -m freeagent.cli evaluate --scenario config/research_scenario.json
python scripts/acceptance.py
```

`config/default.yaml` contains seed, bootstrap count, minimum minutes and objective/verdict defaults. `FREEAGENT_CONFIG` optionally selects another YAML file. `FREEAGENT_HOME` optionally sets the writable runtime root for data, models and backtests; otherwise use the project directory. Set environment variables explicitly; `.env.example` is a template, not an automatically loaded secret store. No API key is required for this source.

`setup` defaults to StatsBomb Premier League 2015/16, cutoff 2016-02-01 and 20 bootstrap draws. It preserves validated downloads and reuses a compatible trained model. Add `--evaluate` to calculate metrics; metrics are never populated from sample constants. For another competition/season, supply `--competition`, `--season`, `--cutoff` and compatible `--cutoffs` when evaluating. Global `--data`/`--config` flags precede individual subcommands. For `setup` and `status`, use `FREEAGENT_HOME` rather than `--data` to keep all runtime outputs together.

## Streamlit Community Cloud

The original deployment showed no data because `data/raw`, `data/processed`, `models/artifacts` and `reports` are intentionally ignored by Git. Your Mac's files are not transferred to the cloud. The app now has an explicit runtime setup workflow:

1. Deploy the GitHub repository with `app.py` as the entrypoint and Python 3.12. Keep `requirements.txt`, `pyproject.toml`, `src/`, `config/`, and `.streamlit/config.toml` in the repository.
2. Open **Data & models** (also shown automatically when no validated dataset exists).
3. Optionally select **Also calculate chronological backtest metrics**.
4. Click **Download real data and train models**. This downloads real matches and lineup intervals, validates them and trains the existing player and score models. Progress and stage errors appear in the app.
5. When models show **trained**, open **Transfer evaluator**. Fees, wages, price per point and expected minutes must be your explicit scenario assumptions; no actual transfer prices are invented.

First setup requires outbound HTTPS to `raw.githubusercontent.com` and may take several minutes. A session lock prevents overlapping setup jobs within the app process. Interrupted downloads resume from validated cached files. Runtime files may disappear after redeploy/restart; run the setup button again. Rebuilding runtime files is supported deliberately—do not force-add datasets, virtual environments or pickled models to Git.

The setup process saves source URL/retrieval/hash metadata, data-quality diagnostics and setup-stage status. Invalid JSON is rejected before caching; temporary network errors have bounded retries; unavailable seasons produce actionable errors. Missing or malformed player lineups preserve usable match results and are reported. Fewer than 20 pre-cutoff matches with compatible player appearances blocks player/transfer training. Never treat a match-only import as a complete transfer model.

This deployment is a research application. Review [StatsBomb's source agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf) and attribution requirements before public or commercial use; the app does not grant redistribution/commercial rights. Configure access appropriately for your intended research use. Secrets belong in Community Cloud's Secrets settings, never in the repository. There are no required secrets for the supplied pipeline.

See the official [Community Cloud dependency guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). The cloud requires runtime packages in repository dependency files and runs on Linux; no Mac-specific paths or external system packages are required here.

### Deploy these changes through GitHub

Review your changes, then run these commands yourself from your existing checkout:

```bash
git diff --check
git status --short
git add .gitignore .env.example app.py README.md requirements.txt requirements-dev.txt src tests docs data/README.md
git diff --cached --stat
git commit -m "Add reproducible cloud data and model setup"
git push
```

Push to the branch configured for your Streamlit application. Community Cloud should rebuild from that branch; check its build logs and use Manage app → Reboot if needed. After deployment, run **Data & models** setup on the cloud instance. The implementation work does not commit, push, or change your remote automatically.

If a dependency import fails, verify the selected Python version and that the new requirements/source files reached the deployed branch. If data are unavailable, inspect the Data & models diagnostics and retry; there is no synthetic fallback. If training fails, verify date coverage, appearance coverage and cutoff before retrying. Financial outputs remain disabled until a compatible full model bundle has trained successfully.

## Architecture

- `src/freeagent/data`: StatsBomb connector, provenance, strict contracts, appearance quarantine, minutes features.
- `impact/ridge_model.py`: chronological alpha selection, signed player minutes, team controls, match bootstrap.
- `match`: independent Poisson and Dixon–Coles likelihood models.
- `simulation/monte_carlo.py`: vectorised simulations, paired uniforms, initial standings, configurable objectives.
- `valuation`: minutes conservation, financial distributions, similarity, sell-side accounting, binary budget optimisation.
- `services/analysis_service.py`: training, versioned artifacts, propagation, integrated evaluation.
- `evaluation`: expanding historical forecasts, proper scores, calibration, transfer-run interface.
- `app.py`: eight research screens plus Data & models setup and diagnostics.
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

The updated build passes 49 tests. A clean source-only deployment simulation downloaded the real 380-match season, trained models, calculated backtests and passed all ten software acceptance checks. Details and hosted-deployment limits are recorded in `docs/IMPLEMENTATION_STATUS.md`.
Run `python scripts/acceptance.py` after ingesting, training, backtesting and evaluating the supplied research scenario. This verifies the saved artifacts and paired EPA agreement; it does not validate causal or commercial claims. Full execution evidence and limitations are in `docs/IMPLEMENTATION_STATUS.md`.

`requirements.lock` records the original Mac research environment; Community Cloud uses the portable `requirements.txt`. Use `requirements-dev.txt` for current development and tests.
Static, standalone interactive evaluation charts are exported to `reports/backtest/calibration.html` and `prediction_vs_outcome.html`.
