# Limitations and unresolved research requirements

- StatsBomb licensing permits research subject to its agreement, not unrestricted commercial use. Data and derived results require appropriate attribution; do not publish source data. Files are local and ignored by Git.
- One historical Premier League season is a demonstration dataset, not broad multi-league, multi-era external validation. Provider data are retrospectively corrected; original availability dates are not archived.
- The inspected Bundesliga release contains only one club's fixtures. It cannot support complete-league conclusions and is not used for the demonstration.
- Lineup clocks omit exact final-whistle timestamps. Regulation clipping excludes added time. Inconsistent source appearances are quarantined, reducing player coverage. Source position labels can change within a game; first role is only a display/similarity label.
- Match-level goals assigned to minutes-share involvement are noisy. Ridge coefficients confound teammates, team tactics, selection, opponent strength and unobserved context. Team effects regularise collinearity but do not establish causal identification.
- Impact-to-rate translation assumes a 50/50 attack/defence split and time-homogeneous minutes allocation. It has not been calibrated on historical transfers. Destination league adaptation, full squad formation, tactical fit and substitution timing are absent.
- Latest observed appearances approximate rosters. Transfers, registration changes or injuries between last observation and cutoff may be missing. Incoming selection from the same competition models destination benefit without automatically reducing the selling club's strength.
- Match bootstrap ignores serial dependence; independent player/match uncertainty resampling ignores covariance. Default 20 draws are a smoke-test research setting, not stable tail calibration. Increase draws and validate coverage before decisions.
- Points/GD/goals ranking uses seeded lots for unresolved ties. Competition-specific head-to-head, playoffs and deduction rules require explicit additions. Partial fixture sets must not be advertised as a complete season.
- Historical future schedule dates may reflect postponements unknown at cutoff. Results are retrospective forecast tests, not fully archived live forecasts.
- Realised player minutes in impact evaluation are labelled conditional covariates, never pre-match knowledge. No causal EPA, financial verdict or injury-prediction performance is asserted.
- Fees, wages, contract duration, bonuses, availability and point prices are supplied scenarios. No market-value or fee benchmark is fabricated. Accounting is amortised for ≤1 sporting year, not DCF or multi-year performance prediction.
- Sell-side and budget utilities are explicit approximations; additive candidate values ignore interactions. Similarity is not an endorsement to buy.
- Pickled/joblib artifacts must be local and trusted. Version metadata prevents accidental version mismatch but is not a security signature.

Highest-priority research next step: obtain permissioned, date-stamped multi-season rosters/appearances and transfer contracts, validate the impact-to-rate mapping, jointly resample team/player estimates, and evaluate uncertainty coverage on truly held-out seasons. Commercial use also needs suitable source rights.
