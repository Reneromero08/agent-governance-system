# Exact Cyclotomic Cubic Phase Tensor-Train Format

## Phase-wave law

The carrier is an exact matrix-product wave over five local phase labels.
Every scalar lies in `Q(zeta_5)` with canonical basis
`1,zeta,zeta^2,zeta^3` and `zeta^4=-(1+zeta+zeta^2+zeta^3)`.

Public brickwork rounds interleave:

```text
normalized local F5 Fourier interference
two-site cubic controlled phase
    zeta^(gamma * (x^2*y + x*y^2))
```

The cubic gate is nonseparable and outside the quadratic Gauss/Clifford
closure. Roots of unity alone are not treated as a resource: the operative
mechanism is unresolved cyclotomic amplitude, Fourier addition, and
nonseparable phase coupling.

## Exact local closure

Each two-site update merges only adjacent tensors. A deterministic exact
skeleton factorization over `Q(zeta_5)` reconstructs the merged matrix
exactly and supplies its bond rank. No truncation is allowed. A forced rank
cap fails at the first nonzero discarded pivot.

The final boundary is one fixed public product-chirp transition amplitude,
represented by four rational cyclotomic coefficients. Tensor entries, bond
states, pivots, and intermediate amplitudes are not projected.

## Restoration and verification

The inverse walks the actual public gates in reverse, conjugating Fourier
and cubic factors. No inverse matrices are retained. When the state returns
to rank one, deterministic gauge canonicalization is applied and exact
machine equality is checked against the public sealed product state. The
same carrier then executes an unrelated circuit.

An independent implementation reduces the circuit into `F11` and `F31`,
using primitive fifth roots and separate modular TT elimination. Matching
bond vectors and final boundary residues provide rank lower bounds; exact
cyclotomic skeleton factors provide matching upper bounds.

## Claim ceiling

Evidence is bounded to widths `2,4,6`, central-gate crossings `1,2,3`, and
software exact arithmetic. The result establishes sequential TT rank growth,
not unbounded growth. The accepted representation is itself the best matched
exact classical TT, so it does not establish a distinct phase resource,
advantage, Small Wall crossing, or physical waveform execution.
