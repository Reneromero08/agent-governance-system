# M141 independent review

Decision: `PASS` at strict bounded scope.

M141 tests one exact representation change on the M140 matrix-free
anisotropic radial carrier.  Instead of storing the same power-of-17
denominator with every cyclotomic coefficient, it stores 272 integral
coefficients, one shared base-17 denominator exponent, and one greatest-common
`pi = 1-zeta_17` exponent.  Every represented 17-cell state is reconstructed
exactly before final projection and the actual inverse.

Across the three declared public program families and depths
`1,2,4,8,16,32,64`, the shared denominator ledger reduces the repeated-
denominator final-state accounting by more than half.  This is a storage-
representation result, not a phase advantage.  Common-pi extraction does not
provide the missing height closure: depth-1 states have common-pi exponent 8
but larger residual coefficient payloads after extraction; every tested state
from depth 2 onward has common-pi exponent 0.  From depth 1 to 64, shared-
denominator payload grows from 957 to 35,425 bits for `PRIMARY`, 951 to 35,369
for `REUSE`, and 981 to 35,380 for `ALTERNATE`.  Maximum residual signed width
reaches 133, 134, and 133 bits respectively.  The denominator exponent reaches
32 in all three families.

The independent oracle imports neither M140 nor M141 production code.  It
independently compiles the public schedules, runs the older retained exact
17-by-17 Fourier recurrence, implements integral division by `1-zeta_17`
directly, and remeasures the final states.  All 21 final commitments,
boundaries, denominator powers, pi valuations, coefficient widths, and payload
records agree in 252 field comparisons.  All separate exact inverses restore
the seed, and a one-bit payload mutation is detected.  The oracle's retained
289-cell kernel is verification-only and is not charged to the accepted M140
matrix-free path.

The production transactions preserve exact algebraic restoration, same-
backing unrelated reuse, zero retained inverse history, and no snapshot
reload.  The normalization buffers and ledgers carry
`NO_RESTORATION_CLAIM`; they are exact transient measurements, not a second
catalytic carrier.  The strongest compact classical comparison remains the
identical shared-scale-normalized 17-coordinate recurrence.

This review establishes neither a universal or asymptotic height lower bound,
an optimal exact representation, CATVM machine custody, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, nor unbounded catalytic
computation.  It isolates the next obstruction: exact unit or multi-embedding
balancing must reduce growing residual height without merely moving the same
ledger into another classical representation.
