# Public Transducer Null-Law Audit

## Scope

This audit closes the public-calibration loop for the retained V3 independent
window Change-to-Dirty transducer. It does not rewrite the historical V3
adjudication and does not promote any private OrbitState claim.

Preserved historical classifications:

- `BALANCED_PHYSICAL_TRANSDUCER_PARTIAL`
- `V1_PARTIAL_V2_TRANSFER_CANDIDATE`
- `PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE`

## Retained V3 Facts

The V3 aggregate `change_to_dirty` means were:

```text
q=-1536  -1901.3125
q=-1024  -1274.625
q=-512    -633.9375
q=0          4.4375
q=512      633.8125
q=1024    1276.5
q=1536    1913.6875
```

Both fresh processes had the same direct sign convention.

The retained V3 evidence passed sign, oddness, gain, monotonicity,
source-order invariance, independent-subcapture-order invariance,
gain-normalized factor laws, strata, restoration, and fresh-process
consistency. Every nonzero logical mapping and physical reversal error was
below `0.25`.

The official retained failures were q0-law failures:

- aggregate `heldout_q0_null_law`;
- aggregate `paired_logical_mapping_invariance_law`;
- replicate q0/logical mapping failures;
- replicate 0 paired physical reversal at q0.

## Null-Law Diagnosis

The retained `null_ceiling()` used the maximum from the build-null sample rather
than a calibrated prospective quantile or fixed bound. The build and held-out
mapping samples had equal size, so under exchangeability the held-out maximum is
below the build maximum with probability approximately `0.5`. Repeating the
unchanged public protocol would therefore chase an unstable maximum gate rather
than answer a new physical question.

The retained `q0_null_semantics()` used:

```text
pair_bound = C
physical_bound = 2C
```

even though the logical pair residual and physical pair sum have the same
magnitude under the mapping convention. The retained
`max_heldout_f0_abs` duplicated `max_heldout_mapping_abs`.

This explains why the public carrier can be treated as an independently
reproduced odd public Change-to-Dirty transducer without retroactively changing
the official V3 class.

## Engineering State

Non-official engineering state recorded for the next private package:

```text
PUBLIC_CHANGE_TO_DIRTY_ODD_TRANSDUCER_REPRODUCED
PUBLIC_MAX_NULL_GATE_NOT_A_VALID_REASON_TO_DELAY_PRIVATE_COUPLING
```

The official scientific state remains:

```text
PUBLIC_INDEPENDENT_WINDOW_TRANSDUCER_CANDIDATE
```
