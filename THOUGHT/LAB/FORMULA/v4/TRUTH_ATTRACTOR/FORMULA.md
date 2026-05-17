# The Formula for Truth-Tracking

## R = (E / grad_S) * sigma^Df

The same functional form. A different domain mapping.

---

## Domain Mapping: Truth-Tracking

| Symbol | Values Alignment | Truth-Tracking |
|--------|-----------------|----------------|
| E | value core / task intent | truth-attractor strength. Fixed at 1.0. The attractor is the invariant -- zero path difference across independent fragments. |
| grad_S | goal drift, adversarial pressure, contradiction entropy | verification entropy -- the log of the number of independent fragments that disagree. Higher disagreement = higher grad_S = harder to achieve R. |
| sigma | constitutional compression ratio (tokens of constitution / tokens of equivalent rules) | verification compression ratio -- how many claims the system can verify per unit of computation. Not semantic density. Inference density. |
| Df | scale-nesting depth (number of scales the constitution spans) | independent verification fragment count -- the number of mutually non-interfering channels that must agree for a claim to reach R > 0.7. Minimum: 2. Standard: 3. |
| R | alignment retention / jailbreak resistance | truth-tracking accuracy -- agreement between the system's outputs and verified ground truth across multiple verification channels. Measured as: R_truth = Tr(rho * C_epistemic) |

---

## The Limit Cases

### Case 1: Single Fragment (Df = 1)
R_truth = E / grad_S * sigma

When only one fragment verifies a claim, R is capped. The system cannot reach CONFIRMED status. This prevents echo chambers -- single-source "truth" is always PROVISIONAL by construction.

### Case 2: No Verification (sigma = 0)
R_truth = 0

When the system cannot verify against any fragment, R is identically zero. The silence protocol activates. The system says "I don't know."

### Case 3: Full Agreement (Df >= 3, sigma high)
R_truth >> 1

When three or more independent fragments agree with high compression (efficient verification), R is maximized. The system outputs CONFIRMED with high confidence.

---

## Relationship to the Cybernetic Truth Control Law

The control law does not change:

T = 1 / (R + epsilon)

What changes is what R measures:

- Old (Phase 4a): R = Tr(rho * C_values) -- proximity to values attractor
- New: R = Tr(rho * C_epistemic) -- proximity to cross-fragment agreement truth

---

## Key Difference

Values alignment R can be high with zero accuracy (Phase 4 result).

Truth-tracking R can only be high when independent fragments agree. High R implies accuracy by construction.

This is the tradeoff: lower maximum R, but R that actually means something.
