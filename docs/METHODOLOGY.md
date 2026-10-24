# Methodology

## 1. Temporal boundaries and player impact

For cutoff c, fit only matches whose date < c. Player vocabularies, rosters inferred from latest appearances, team IDs, minutes safeguards and all coefficients use this training subset. Regularisation alpha is selected from {1,10,100} using the earliest 75% of training dates to predict the final 25%, then refit on the complete pre-cutoff set. Tied dates never straddle the inner split.

Ridge minimises ||y − Xβ − intercept||² + α||β||². y is home goal difference. X uses signed player minutes/90 and signed team indicators. Opponent strength is represented by the opposing team and players. The intercept captures home advantage. Identical teammate exposure is highly collinear with team indicators; Ridge yields a stable numerical solution, not unique causal attribution. Rank and feature counts are reported. Players below 270 historical minutes cannot be evaluated by default.

Bootstrap whole match rows with replacement and refit Ridge at the selected alpha. Report 5th/95th empirical coefficient quantiles. Conditional alpha selection, serial dependence, selection bias and teammate correlation limit these ranges. Separate event-level on-pitch outcomes are not constructed: lineup clocks alone do not provide interval goals. This is explicitly a match-level approximation.

## 2. Score model

log λ_home = intercept + home advantage + attack_home + defence_away.
log λ_away = intercept + attack_away + defence_home.

Positive defence parameters mean more goals conceded. Each attack and defence vector sums to zero for identifiability. SciPy L-BFGS-B minimises weighted negative log likelihood plus a small strength penalty. Recency half-life defaults to 365 days; zero disables weighting. This decay is configurable, not empirically claimed optimal. Optimisation failure raises an error.

Dixon–Coles correction: τ00=1−λh λa ρ; τ01=1+λhρ; τ10=1+λaρ; τ11=1−ρ; otherwise 1. Fit feasibility checks all potential team pairs. Poisson baseline fixes ρ=0. Scores 0–12 are normalised after correction; lost mass >0.1% or negative probabilities fail rather than silently producing invalid forecasts. Unseen teams require retraining; no hidden rating prior.

For parameter uncertainty, simulate joint scores on the original training schedule from the fitted model and refit once per replicate. These conditional parametric bootstrap fits are paired with separately resampled Ridge fits. Cross-model dependence is approximated as independent and is a limitation.

## 3. Season simulation and objectives

For each fixture, inverse-CDF sample the joint score matrix with one uniform per simulated season. Award 3/1/0 points and accumulate goals scored and goal difference. Optional initial standings add already observed points/GD/goals; each remaining fixture is processed once. Shared seeds and identical uniforms couple with/without scenarios. Vectorisation is over simulated seasons; no parallel fitting is required.

Rank by points, GD, goals scored, then shared seeded lots for exact ties. This captures the principal Premier League ranking criteria but approximates unresolved ties that require special rules/playoffs. Head-to-head leagues are not fully supported. Qualification and relegation place counts are configurable. Title/qualification/relegation changes stay separate from expected points and are not monetised without an explicit utility model.

## 4. Displacement and EPA

User selects incoming player and an incumbent last observed at the destination club. Existing squad strength is represented by the fitted historical team model. One-role baseline allocation is explicit; other squad contributions are held fixed. This is a documented minutes-based approximation, not a full lineup/tactics optimiser.

0 ≤ incoming minutes ≤ incumbent baseline minutes ≤ 90 × remaining club fixtures. Incoming minutes remove the same amount from the incumbent. With optional availability, both incoming and displaced minutes shrink together.

ΔGD per match = (βincoming − βincumbent) × incoming minutes / (90 × club fixtures).

Half of ΔGD raises own scoring; half lowers conceding. This changes net GD once, not twice. It is an assumed 50/50 split, not identifiable attack/defence decomposition. Rates have a 0.05 goal floor; floor activations are counted. Invalid Dixon–Coles probabilities after a large shift fail explicitly. The mapping has not been historically calibrated.

EPA is the sum of per-fixture expected-point differences (3P(win)+P(draw)), exactly integrated for each parameter replicate. Paired simulations also calculate EPA as a numerical cross-check and yield rank distributions. This removes inner simulation noise from the main expected-EPA range. Paired realised point differences retain match-outcome variability and depend on the coupling; do not interpret them as identified causal outcomes.

## 5. Financial horizon

NetValue = EPA × PricePerPoint − (Fee_allocated + Wages_allocated).

Fee_allocated = (transfer fee + signing bonus + transaction costs) × horizon_years / contract_years.
Wages_allocated = annual wage × horizon_years.

All amounts share one currency. This is amortised accounting, not total cash commitment or discounted cash flow. The full commitment is reported separately and is never subtracted twice. Horizon must be ≤1 year and ≤contract term. In remaining-season analysis the app defaults to the remaining-fixture fraction of a season; users must justify the matching benefit/cost horizon. Fee and wages of the retained incumbent are common costs; no incumbent wage savings are assumed unless explicitly evaluating a sale.

PricePerPoint is a user-supplied scenario with rationale. Bootstrap/scenario expected Net Value and coupled realised Net Value are separate. Report mean, median, 90% range, positive/loss probability and category probabilities. Defaults: Best Buy if P(NetValue > €1m) ≥ .75; Overpaid if P(NetValue < −€1m) ≥ .75; otherwise Fair. Fair includes inconclusive uncertainty. These predeclared research thresholds are configurable and not validated labels.

## 6. Walk-forward evaluation

Outer folds use training date < cutoff and evaluation cutoff ≤ date < next boundary. Refit each fold. Compare Dixon–Coles, independent Poisson and smoothed training result frequencies using log loss, ordinal RPS and multiclass Brier. Save individual predictions, boundaries, training maximum dates, failures, model version, calibration bins and fold scores.

Player diagnostics predict held-out goal difference **conditional on realised minutes**. They assess outcome reconstruction, not pre-match forecasting; unseen players contribute zero learned coefficient with known team controls. They are not causal transfer validation. No historical transfer-price, causal EPA or financial truth labels exist in the selected dataset. The historical-transfer runner strips realised outcome fields, checks transfer observation date and records callback predictions/failures, but is a scaffold until defensible inputs are supplied.

## 7. Optional utilities

Implemented only after core integration tests passed. Similarity standardises impact, historical minutes and uncertainty width within exact observed role; it is a candidate search, not a cost-adjusted recommendation. Availability is an explicit uniform minutes scenario, never medical injury prediction. Sell-side accounting adds sale fee less book value plus saved wages to an already evaluated replacement Net Value. Binary MILP maximises supplied additive value under cash budget, role capacities and one replacement per incumbent. It compares to no-signing value zero and does not model signing interactions; joint reevaluation remains necessary. No continuous KKT claim is made for a discrete problem.
