# F17 Gaussian multi-port phase quotient review

## Decision

Classification:
`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:
`SEPARATE_REFERENCE_PARITY`

Restoration class:
`EXACT_ALGEBRAIC_RESTORATION`

The supported result is a bounded symbolic F17 Gaussian phase-chart
calculation. It is not physical phase-state restoration.

## Independently checked

The focused source review checked the quadratic convention
`q^T A q / 2 + b^T q + c`, symmetric phase shears, positive and negative
single-port Fourier kernels, the Schur-complement coefficient update, the
quadratic-character sign, the nonsingular Gaussian overlap boundary, reverse
inverse order, exact coefficient restoration, backing-array identity, and
unrelated restored-carrier reuse.

The separate oracle does not import the production backend or call its
compiler or projection. It consumes hashed public module descriptors and
reexecutes the exact coefficient recurrence, boundary, forward/inverse path,
and reuse path with Python integer lists at 2, 4, 8, 16, and 32 ports. At 2
and 4 ports it also materializes the explicit complex F17 phase vector and
checks the forward boundary and both restorations within the predeclared
`2e-11` tolerance.

## Repairs made before qualification

- Fourier scratch now includes the simultaneously live updated linear vector.
- Projection reports a conservative `5*n^2+n` temporary F17-cell upper bound.
- Public compiler simulation cost is separated from determinant-search
  elimination, inversion, and scratch cost.
- Compilation cost is reported for both the primary and unrelated reuse
  programs.
- Algebra payload, machine metadata, two-carrier accepted-path residency, and
  three-carrier verification-only residency are distinguished.
- Dense-oracle cells are labelled logical vector cells rather than a process
  peak; Python, NumPy, native-library, and allocator peaks remain unbounded.

## Strict ceiling

The evidence covers Linux/Python/NumPy, F17 quadratic Gaussian coefficient
state, ports `{2,4,8,16,32}`, two public algorithmic program families,
nonzero Fourier pivots, and nonsingular final overlap boundaries. The public
compiler simulates the declared initial coefficient law and inspects public
coefficient determinants to select a final shift. The separate oracle checks
the emitted descriptors; it does not independently regenerate the compiler.
Missing, wrong, and reordered controls are production-package controls.

The accepted path uses `n^2+n+1` F17 cells and one sign bit, retains no inverse
history, reloads no baseline, restores the same coefficient arrays exactly,
and reuses the same logical carrier. The identical exact Gaussian coefficient
recurrence is also a compact classical implementation. No strongest-classical
optimality result is established.

This result does not establish CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, inference, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.
