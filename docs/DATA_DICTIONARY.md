# Data dictionary and CSV contracts

## `data/processed/matches.csv`

| Field | Type / units | Required | Meaning / availability |
|---|---|---|---|
| match_id | string | yes | Provider-native stable match ID |
| date | ISO YYYY-MM-DD | yes | Actual fixture date; outcomes only usable after this date |
| competition | string | yes | Single competition per model |
| season | YYYY or YYYY/YYYY | yes | Provider season label; date year must belong to label |
| home_id, away_id | string | yes | Provider-native team IDs, distinct |
| home_goals, away_goals | nonnegative integer goals | yes | Regulation full-time scores, post-match |
| home_name, away_name | string | no | Display labels; never identity keys |
| source | string | recommended | StatsBomb Open Data or explicit user provenance |

Do not import fixtures with missing scores into training. Simulation fixtures require only `match_id,date,home_id,away_id`; score columns are deliberately removed before evaluation. Extra-time matches are excluded by the StatsBomb connector with an audit reason. Do not mix competitions, IDs from different providers, or season labels without explicit mapping.

## `data/processed/appearances.csv`

| Field | Type / units | Required | Meaning / availability |
|---|---|---|---|
| match_id | string | yes | Foreign key to match |
| player_id | string | yes | Provider-native ID, unique with match_id |
| player_name | string | yes | Display label; duplicate names across IDs allowed |
| team_id | string | yes | Must equal match home_id or away_id |
| minutes | float, regulation-equivalent minutes | yes | 0–90, derived after match |
| position | string | yes | First listed role in match; not a tactical-fit estimate |
| started | boolean | yes | Initial involvement starts at 0 |

StatsBomb lineup positions are clock intervals with nullable final-whistle endpoints. Ingestion clips first-period timestamps at 45 and second-period timestamps at 90, using 90 for a null final-whistle endpoint. Stoppage-time involvement is not modelled. It sums position intervals; it does not infer event-level scoring exposure. Implausible individual minutes or team totals outside [720,990] quarantine the **entire match's appearance set**, preserving the match result and recording the reason. Lower totals caused by dismissals are allowed within that bound. This is a safeguard, not proof that every retained interval is perfect.

Match-only rows may be retained without appearances; they train the score model but never the player model. One-sided appearance coverage fails validation. `quarantined_appearances.csv` preserves rejected appearance records for inspection. `quality.json` reports retained row counts, missingness, duplicate count, coverage, errors, warnings and exclusions. Nothing is silently dropped.

## Identity and provenance

`teams.csv` and `players.csv` are explicit provider-ID → display-name mapping tables. Renames may map to one stable ID. Duplicate names are not merged. Combining providers requires a reviewed ID crosswalk. `raw/manifest.jsonl` records each original URL, UTC retrieval time, SHA-256, source and agreement URL. Cached files retain original retrieval metadata.

Historical records downloaded today can contain provider corrections; this is a retrospective cutoff test, not a replay of archived as-of snapshots. Prediction dates use a strict before-date boundary: training on the same calendar day is disallowed.

## Model fields

Player involvement is signed `minutes / 90` (+ home, − away). Signed team indicator columns control team/opponent associations; Ridge intercept represents home advantage. Target is home minus away goals. Features already have interpretable dimensionless scales; player columns are deliberately not standardised because equal per-90 coefficient penalty is intended. Prior-team-form helper only updates after an entire calendar date.

## Financial scenario JSON

Required: `season`, `club` (team ID), `incoming`, `incumbent` (player IDs), `minutes`, `incumbent_minutes`, `price_per_point`, and `costs`.

`costs` contains `fee`, `annual_wage`, `contract_years`, optional `bonus`, `transaction_cost`, and `horizon_years` (0 < h ≤ 1). Monetary values use one consistent currency, displayed as EUR in the app. Optional `availability: [lower,upper]` is a uniform scenario fraction in [0,1], not a medical probability. `simulations` controls draws per parameter replicate. Document a point-price rationale and the financial-input source. No inferred market prices are supplied.
