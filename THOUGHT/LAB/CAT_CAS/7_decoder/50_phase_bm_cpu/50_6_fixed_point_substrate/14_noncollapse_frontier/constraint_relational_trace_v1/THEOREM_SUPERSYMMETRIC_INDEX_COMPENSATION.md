# Finite Supersymmetric Index Compensation

## Statement

Let the bosonic and fermionic sectors each contain one state per Boolean assignment and
let

```text
Q_F |x,B> = sqrt(E_F(x)) |x,F>,
```

where `E_F(x)` is the number of violated public clauses. Then

```text
H_F = {Q_F, Q_F^dagger}
```

has the same energy `E_F(x)` in both graded sectors.

Every satisfying assignment contributes one bosonic and one fermionic zero mode. Hence

```text
W_F = Tr((-1)^grading exp(-beta H_F))
    = dim ker(H_B) - dim ker(H_F)
    = #SAT(F) - #SAT(F)
    = 0.
```

The same compensation follows abstractly from finite-dimensional rank-nullity. For a
finite complex with fixed graded dimensions, its Euler/Witten index is determined by
those dimensions and is independent of the differential.

## Consequence

Topological supersymmetry and instanton dynamics may organize the route to an
attractor, but the finite Witten index of the square assignment pairing is not a SAT
presence bit.

A formula-dependent index requires at least one of:

```text
an open or infinite Fredholm boundary
a formula-dependent graded carrier dimension
a non-Fredholm boundary contribution
an explicitly unpaired solution sector
```

The first and third remain legitimate CAT_CAS directions. The second risks compiling the
answer into carrier size. The fourth is equivalent to the unresolved clean-sector access
problem unless produced natively from public clause geometry.

## Claim boundary

```text
FINITE_SUPERSYMMETRIC_INDEX_SHORTCUT_REJECTED
DYNAMICAL_INSTANTON_PREPARATION_REMAINS_OPEN
P_EQUALS_NP_NOT_PROVEN
```
