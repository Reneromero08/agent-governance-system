# Determinant Sensor Resource Audit

## Target Dimension

For `n` public Boolean variables, the exact assignment-space phase oracle has matrix
dimension:

```text
N = 2^n.
```

Its clean-sector determinant winding equals `#SAT(F)`. The relevant question is whether
a determinant sensor is polynomial in `n`, not merely polynomial in `N`.

## Quantum Determinant Estimation

Agerskov and Splittorff, *Quantum Determinant Estimation*, arXiv:2504.07497, construct a
unitary determinant-phase algorithm based on a completely antisymmetric state. Their
resource laws include:

```text
O(N log^2 N + t^2) operations
N log N qubits for the antisymmetric register
t*N controlled applications for t phase bits
```

The antisymmetric state contains one occupied register for each matrix basis dimension.
Applied directly to the clean SAT oracle, `N = 2^n`, so carrier size and controlled
applications are exponential in public input length.

Source: https://arxiv.org/abs/2504.07497

## Spectral-Sampling Log Determinant

Giovannetti, Lloyd, and Maccone, *A quantum algorithm for estimating the determinant*,
arXiv:2504.11049, obtain logarithmic matrix-dimension dependence for log-determinant
estimation under conditions including a positive-definite sparse matrix with spectrum
bounded away from zero.

The SAT detector needs exactly the excluded boundary:

```text
satisfiable   -> at least one exact zero eigenvalue
unsatisfiable -> no zero eigenvalue
```

For a unique zero mode, uniform spectral sampling encounters the relevant eigenspace
with probability `1/N = 2^-n`. Regularizing the zero eigenvalue restores a condition or
precision dependence tied to that rare spectral mass. The positive-definite
log-determinant result therefore does not supply an exact zero-mode latch.

Source: https://arxiv.org/abs/2504.11049

## Normalized Trace

DQC1-style trace estimation returns a normalized quantity. For the phase oracle at
`theta = pi`:

```text
Tr(U_F(pi)) / N = 1 - 2*#SAT(F)/N.
```

The unique-witness displacement from UNSAT is `2/N = 2^(1-n)`. Constant-additive-error
trace estimation cannot resolve that boundary. Plain sampling requires inverse-square
gap resources; amplitude-style improvements still retain inverse-gap dependence.

Recent formal work classifies constant-additive normalized trace estimation for broad
functions of log-local Hamiltonians as DQC1-complete, but this does not change the
exponentially small SAT signal.

Source: https://arxiv.org/abs/2604.01519

## Black-Box Determinant and Rank

Dorn and Thierauf prove tight `Theta(N^2)` quantum query bounds for determinant and rank
verification in their matrix-entry query model.

Source: https://doi.org/10.1016/j.ipl.2008.11.006

This is not an unconditional lower bound for the succinct, highly structured
`ConstraintHolo` access model. It is a control against claiming that an arbitrary
black-box determinant sensor is sublinear in the assignment-space dimension.

## Current Conclusion

No reviewed determinant method supplies all of:

```text
polynomial carrier size in n
exact clean-sector isolation
constant unique-witness margin
polynomial controlled applications
no positive-definite or condition-number exclusion
native inverse restoration
standard-model polynomial transfer
```

The topological index is complete. The polynomial-resource sensor remains open.
