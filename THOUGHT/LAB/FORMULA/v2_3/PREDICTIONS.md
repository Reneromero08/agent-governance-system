# PREDICTIONS.md - Preregistration Ledger

Append-only ledger of quantitative predictions for Living Formula v2.3.

Format, one row per prediction:

| Column | Meaning |
|---|---|
| P-NNN | Prediction ID, zero-padded 3-digit, monotonically increasing |
| date | YYYY-MM-DD the prediction was registered |
| question | Q<N> the prediction belongs to |
| registry IDs | VARIABLES.md IDs of the quantities involved |
| predicted quantity | What is being predicted (named quantity, units) |
| threshold | Numeric success/failure criterion, stated before the run |
| linked verdict | Path to the VERDICT.md that consumed this prediction, or - |

RULE: entries are APPEND-ONLY and must be written BEFORE the experiment
runs. Never edit or delete a row; supersede with a new row.

## Ledger

| P-NNN | date | question | registry IDs | predicted quantity | threshold | linked verdict |
|-------|------|----------|--------------|--------------------|-----------|----------------|
| P-001 | 2026-06-12 | Q25 | SIGMA-FEISTEL-01, DF-FEISTEL-01 | Single-round Feistel byte match rate vs sigma = 2^(-h) across 10 masks (h = popcount) | C1: identity fit R2 >= 0.99, slope in [0.98, 1.02], abs(intercept) <= 0.01. C2: at most 1 of 10 masks deviates from 2^(-h) by more than 3 standard errors (200 trials per mask) | - |
| P-002 | 2026-06-12 | Q49 | DF-EMB-01 | Df * alpha product of embedding covariance spectra vs the constant 8e = 21.746, across models (MiniLM, MPNet), vocabulary sizes N in [20, 150], and matched random baselines | C1: abs(product - 8e)/8e <= 0.05 at N = 75 for both models. C2: no systematic N-dependence (linear fit R2 < 0.9 or relative range <= 10 percent). C3: Monte Carlo over constants in [15, 30]: fraction fitting as well as 8e is < 0.05. C4: matched i.i.d. Gaussian baseline differs from real product by > 10 percent at N = 75 AND real-minus-random delta stable within 30 percent over N in [30, 150] | - |
