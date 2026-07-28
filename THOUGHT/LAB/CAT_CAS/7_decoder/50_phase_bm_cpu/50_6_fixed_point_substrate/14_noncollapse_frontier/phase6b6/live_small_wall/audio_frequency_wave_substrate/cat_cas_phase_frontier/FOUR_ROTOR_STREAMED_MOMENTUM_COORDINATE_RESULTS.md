# Streamed Total-Momentum Coordinate Phase Closure

## Result

Accepted bounded claim:

```text
BOUNDED_TOPOLOGY_STREAMED_TOTAL_MOMENTUM_COORDINATE_PHASE_CLOSURE_ELIMINATES_DENSE_QUOTIENT_FREE_PLAN_WITH_ACTUAL_RESTORATION_AND_REUSE
```

Claim ceiling:

```text
FOUR_OPEN_CHAIN_ROTATION_INVARIANT_ROTORS_GRID17_DEPTHS1_TO64_SOFTWARE_COMPLEX128_SCIPY_POCKETFFT_NO_FFT_INTERNAL_WORKSPACE_BOUND
```

Evidence:

```text
/tmp/four-rotor-streamed-momentum-coordinate-final.oba4A3
```

The exact global-rotation quotient still stores the same `17^3 = 4,913`
complex carrier cells. Its free update no longer retains a `17^3` complex
phase table. After the quotient FFT, public total-momentum conservation
derives

```text
n0 = -(n1 + n2 + n3) mod 17
```

one 17-cell `n3` slice at a time. The explicitly materialized `n0` index
slice is a topology-derived classical coordinate. It is not projected, but it
is also not a hidden or unresolved phase-resident port.

## Exactness and catalytic lifecycle

The lifted quotient agrees with the independent dense four-rotor execution
through depths `1,2,4,8,16,32,64`. At depth 64 the full-state error is
`2.048e-14`, boundary error is `8.438e-15`, and restoration error is
`1.742e-14`.

The depth-32 primary restores the actual borrowed carrier within
`8.614e-15`. An unrelated second program consumes that same restored carrier,
reaches restoration generation two, restores within `2.310e-15`, and agrees
with fresh execution at the boundary within `7.994e-15`. Missing, wrong, and
reordered inverse controls separate by `0.3851`, `0.03912`, and `0.5748`.
No verification baseline is reloaded for restoration.

## Resource accounting

For the depth-32 primary:

| Resource | Retained-table quotient | Streamed coordinate |
|---|---:|---:|
| quotient carrier | 78,608 B | 78,608 B |
| retained public plan | 78,880 B | 408 B |
| plan compilation payload | 157,760 B | 408 B |
| maximum explicit engine arrays | 236,368 B | 118,592 B |
| maximum explicit wrapper arrays | 314,976 B | 197,200 B |

The retained-plan reduction is `193.333x`. The streamed engine uses at most a
289-cell pair factor and a 17-cell free slice, performs 18,496
total-momentum-coordinate closures, retains zero inverse-history bytes, and
has a 2,837,368-byte verification peak. PocketFFT internal workspace remains
outside the explicit-array bound.

This repair trades retained operator memory for repeated coordinate
rematerialization. Its measured warm execution is not reduced. The best
matched classical streamed quotient is identical.

## Scientific boundary

This result removes the dense compiled free-plan defect. It does not reduce
the 4,913-cell carrier, change the remaining `N^(rotors-1)` width law, create
an unresolved internal phase port, establish a distinct phase resource, show
computational advantage, cross the Small Wall, or establish unbounded
computation.

The surviving obstruction is:

```text
RELATIVE_COORDINATE_EXPONENTIAL_GROWTH_AND_MATCHED_CLASSICAL_QUOTIENT_IDENTITY
```

The next phase-owned mechanism must attack resident quotient-carrier growth
or supply a phase-native closure/resource not inherited immediately by the
matched compact classical recurrence.
