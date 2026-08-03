# M125 Independent Review: Paired Phase-Basis Holographic Matchgate Closure

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident basis/weight/scalar carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Transient Kasteleyn and elimination buffers: `NO_RESTORATION_CLAIM`

The verified family is even open square grids only.  Exact `Q(zeta_17)`
boundaries agree at `n=2,4,6` for both public descriptor families, and
independent `F103/F137` structure agrees at `n=2,4,6,8,10,12`.

## Independent reconstruction

The production basis pair is

```text
T = [[1,1],[p,-p]]
S = [[1/2,1/2],[p^-1/2,-p^-1/2]]
```

and exact reexecution confirms `T S^T = I`.  Consequently each shared native
edge closes by identity substitution.  The transformed degree-four exact-one
generators have nonzero components in both fermion parities, so their compact
resident representation is not a parity-preserving matchgate signature in
that basis.  After the public paired bases cancel, however, the remaining core
is precisely a weighted planar perfect-matching problem.

The oracle imports neither production nor predecessor code.  It uses a custom
Fraction power basis for `Q(zeta_17)`, a memoized exact weighted-matching
recursion at `n=2,4,6`, and a separate modular row-profile matching recurrence
for every structural case.  It reconstructs perfect-matching counts `2`, `36`,
and `6728`.  Direct native Holant enumeration is restricted to the bounded
`n=2` control and agrees with the holographic boundary.

The public Kasteleyn signing assigns horizontal sign `+1` and vertical sign
`(-1)^column`.  Face checks pass, while a public horizontal reference matching
calibrates the determinant sign without inspecting the target answer.

## Restoration, controls, and resources

Only the final scalar boundary is projected.  Basis, edge-weight, and scalar
loads are reversed on the actual carrier, generation advances exactly, and an
unrelated `n=6` program reuses the same backing with fresh/restored agreement.
No inverse history or snapshot reload is used.

Controls cover basis mutation, missing and reordered inverse, wrong ownership,
premature projection, null carrier, absence of a snapshot command, face-signing
validity, mixed-parity witnesses, and the bounded direct native-Holant parity.

For even `N`, resident state is `2N(N-1)+8` field cells.  The determinant
matrix has `N^2/2` rows and `N^4/4` cells; caller plus elimination copy and five
named scalars use `N^4/2+5` named transient cells.  The accepted determinant
uses `O(N^6)` field operations, with exact bit complexity reported separately.
Payload is an upper bound over named logical cells.  Python containers,
allocator/native-library storage, bigint internals, and whole-process peak are
excluded explicitly.

## Claim ceiling

M125 establishes polynomial closure for this growing-treewidth matchgate
family while keeping compact mixed-parity native generators and avoiding
native signature tables or edge-assignment enumeration.  The strongest
compact classical method is the identical paired-basis compilation and
Kasteleyn determinant.  It does not establish arbitrary planar Holant closure,
CATVM custody, a distinct phase resource, computational advantage, Small Wall
crossing, physical waveform execution, replacement of physical bits with pi,
or unbounded catalytic computation.
