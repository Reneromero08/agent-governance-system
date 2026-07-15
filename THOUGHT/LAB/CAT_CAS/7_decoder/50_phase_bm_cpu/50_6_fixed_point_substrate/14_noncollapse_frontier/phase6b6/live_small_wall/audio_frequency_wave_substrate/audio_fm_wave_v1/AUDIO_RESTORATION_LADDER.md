# Audio Restoration Ladder

Status: `DIGITAL_R0_LIKE_ONLY_PHYSICAL_THRESHOLDS_UNFROZEN`

## Tiers

```text
R0 = file/sample or byte return
R1 = measured-output return
R2 = accepted observable-state equivalence
R3 = multi-instrument carrier return
R4 = closure up to predeclared invariant
```

The reference engine establishes an algebraic inverse and float-domain sample return.
This is R0-like digital restoration only. It does not establish physical relaxation,
R1, R2, R3, or R4.

## Future Physical Sequence

Every candidate must capture:

```text
baseline state
prepared state
post-source pre-query state
post-query state
active-restoration state
post-restoration state
time-matched natural-relaxation state
no-restoration state
wrong-inverse state
reordered-inverse state
carrier-off state
```

Classification must compare distributions, not one sample, and must freeze thresholds
from baseline/control data before candidate scoring.

## R1

R1 asks only whether the declared receiver output returns within a frozen tolerance.
It can miss hidden modal residue and is insufficient for a persistent carrier claim.

## R2

R2 requires equivalence over the complete accepted observable vector, for example:

```text
complex transfer-function bins
impulse-response coordinates
modal amplitudes and phases
ring-down constants
intermodulation coordinates
source-off and query-off floors
```

The equivalence metric must be multivariate and calibrated on repeated baseline
captures. No numeric R2 threshold is frozen offline because no physical baseline
distribution exists.

## R3 And R4

R3 requires independent instruments or modalities, such as voltage plus current, or
pickup voltage plus displacement. R4 additionally requires the declared invariant to
close under the complete preparation-query-restoration path. A future holonomy claim
cannot substitute byte equality or one sensor for R4.

## Killing Controls

- Natural relaxation distinguishes active inverse from waiting.
- No restoration tests whether the system would look restored anyway.
- Wrong inverse and reordered inverse test operator specificity.
- Carrier-off tests instrument and software baseline.
- Query-off tests whether measurement itself manufactures displacement.
- Time matching prevents longer rest intervals from masquerading as a better inverse.

## Overclaim Stop

Any report that maps digital sample equality to physical R2 must be classified:

```text
RESTORATION_OVERCLAIM
```

and cannot freeze a physical prototype.
