# M140 independent review

Decision: `PASS` at strict bounded scope.

The reviewed mechanism is limited to radial functions on the anisotropic
plane `F17^2` with norm `Q(x,y)=x^2-3y^2`, the declared quartic norm-phase
gates, and the normalized anisotropic phase Fourier transform.  It replaces
M139's retained 289-cell public Fourier kernel with the exact factorization

```text
s(t) = sum_q zeta17^(-q t) f(q),                    t != 0
F(f)(r) = delta[r=0] sum_q f(q)
          - 17^-1 sum_(t != 0) zeta17^(-r/(4t)) s(t).
```

The accepted Fourier path neither calls the per-entry generator nor enumerates
coordinates.  It retains no 17-by-17 kernel.  Its counted update uses 272
source contractions plus 272 target contractions, or 544 character products
per Fourier.  Maximum update scratch is 38 exact field cells: 16 spectral
cells, up to 17 output cells, and the live state sum, accumulator, character,
product, and updated value.  This is 55 cells with the 17-cell resident carrier
and 56 including the retained `17^-1` generator scalar.  Verification-only
coordinate kernels and dense coordinate transforms are excluded from the
accepted resource path and reported separately.

The independent oracle imports no M140 production module.  It reconstructs
the public geometry, schedules, factorized forward and inverse recurrence,
final-state commitments, and exact resident-payload measurement.  It matches
all seven exact transactions, the exact payload tuple
`[2203, 2243, 4364, 8709, 17552, 35394, 70992]`, and all 48 finite-field
structural transactions.  It also checks every kernel entry over `F103` and
`F137`, five exact `Q(zeta17)` entries, two dense 289-coordinate transforms,
and mutations omitting one parameter or reversing the reciprocal sign.  The
qualifier regenerated both result files byte-for-byte.

Exact inverse order, exact algebraic restoration, same-backing unrelated
reuse, zero retained inverse history, and absence of snapshot reload are
confirmed.  The strongest executed compact classical comparison is the
identical matrix-free 17-coordinate recurrence.  The M139 289-kernel
recurrence is retained in the comparison frontier because it uses less warm
arithmetic and scratch at the cost of its retained kernel.

This review does not establish CATVM custody, general nonlinear relation
quotients, asymptotic work reduction, a distinct phase resource,
computational advantage, a Small Wall crossing, physical execution,
replacement of physical bits with pi, or unbounded catalytic computation.
