# Model card — FreeAgent 0.1.0

**Intended use:** reproducible local football analytics research and explicit transfer scenario comparison. Not a causal effect estimator, medical tool, betting guarantee, or independently validated financial valuation.

**Models:** Ridge minutes-share plus-minus with team controls; identifiable Poisson/Dixon–Coles; conditional parametric/bootstrap ensembles; paired Monte Carlo ranking; scenario accounting. No pretrained language model enters sporting calculations.

**Data:** StatsBomb Open Data Premier League 2015/16, when successfully ingested. Full provider schedule contains 380 matches. Actual retained player coverage, exclusions, retrieval hashes and dates are in `data/processed/quality.json` and `data/raw/manifest.jsonl`. No synthetic data enter application reports.

**Training:** pre-cutoff rows only. Intended saved cutoff 2016-02-01. Artifact JSON records actual cutoff, latest training date, row count, versions, data hash and bootstrap count. Player fit uses only matches with valid appearance coverage.

**Evaluation:** expanding chronological folds; real metrics saved in `reports/backtest/metrics.json` and documented in `docs/IMPLEMENTATION_STATUS.md` only after successful execution. Proper scores use actual held-out match outcomes. Player diagnostics are conditional on observed minutes. No causal transfer or financial verdict validation is available.

**Uncertainty:** bootstrap/scenario ranges, not calibrated confidence intervals. Expected-value and coupled realised-outcome distributions are separate. Empirical interval coverage has not been established.

**Safeguards:** strict schemas, temporal cutoffs, minimum historical minutes, minutes conservation, invalid probability rejection, convergence checks, explicit failure reports, no unseen-team defaults, no synthetic fallback, source provenance and matching financial horizon.

**Known failures:** correlated player identities, sparse minutes, selection bias, missing roster changes, unmodelled tactical fit, extrapolative goal-rate shifts, unresolved league tie rules, omitted selling-club impact and unvalidated point-price assumptions. Read `LIMITATIONS.md` before interpretation.
