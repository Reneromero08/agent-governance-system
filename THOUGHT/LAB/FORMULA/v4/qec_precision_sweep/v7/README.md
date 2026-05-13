# v7: Information-Theoretic Sigma — NEGATIVE

Date: 2026-05-13
Status: Complete. Verdict: NEGATIVE — I(S:F) is fundamentally incompatible with the formula.

## Design

Tested the Light Cone axiom-level definition of sigma = I(S:F) — mutual information between logical state and syndrome fragment. Operationalized as:

```
sigma = (H2(p) - H2(p_L)) / d
```

With E=1.0, grad_S=p, Df=d.

## Result

| Metric | DEPOL | MEAS |
|--------|-------|------|
| Direct MAE | 102.59 | 76.77 |
| Direct R2 | -1920 | -1310 |
| Alpha | 0.05 | 0.05 |

Complete failure. Sigma is bounded in [0,1/d], so sigma^Df always produces tiny numbers and never grows with distance — even below threshold where distance actually helps.

## Why It Failed

The formula requires sigma to be a **multiplicative gain factor** that can exceed 1 below threshold. I(S:F) is an **additive information measure** bounded in [0,1]. These are fundamentally incompatible.

The DOMAIN_MAPPINGS.md gives two options for sigma in QEC: "code compression/fidelity factor OR logical information per resource." The fidelity-factor interpretation is multiplicative and works (v2-v6). The information-theoretic branch of the OR does not work in this formula position.

## Takeaway

This is a domain-level constraint: for QEC, sigma must be the per-distance gain exp(delta_logR / delta_Df), not an information-theoretic entropy. The axiom-level I(S:F) definition does not apply to QEC when using R = (E/grad_S) * sigma^Df — it would require a different formula structure (additive, not multiplicative).

## Files

- `v7/code/info_theoretic.py`
- `v7/results/v7_depol/`
- `v7/results/v7_meas/`
