# Non-Abelian Holonomy Report

Candidate M: Wilczek-Zee shared phase-frame holonomy.

Mechanism survived: `True`
Classification: `SOURCE_WILCZEK_ZEE_PACKAGE_REPRODUCED_STRICT_SCOPE`

V3 reconstruction actually performed:

- Recomputed the primary-vs-reordered boundary Frobenius separation from reproduced 2×2 matrices: `1.9817209383199668`.
- Recomputed primary/reuse unitarity defects: `3.236828524569469e-16`, `4.660953740672398e-16`.
- Recomputed discrete formula parity errors: `1.8041963683411463e-13`, `3.419029244888734e-13`.
- Confirmed mpmath oracle did not import the production backend and agreed on noncommutation.

Finding:

The source Wilczek-Zee package reproduced in strict scope, and the branch-local toy harness separately demonstrates a noncommuting reversible hidden-frame transaction. V3 did not independently rebuild the Wilczek-Zee geometry, so this report no longer claims independently verified Wilczek-Zee transfer.

Baseline discipline:

The strongest compact classical baseline is the identical 2×2 matrix recurrence, with closed-form fixed-loop modules available. Therefore this is not a non-collapse resource separation, not a Small Wall result, and not physical Family 10h evidence.
