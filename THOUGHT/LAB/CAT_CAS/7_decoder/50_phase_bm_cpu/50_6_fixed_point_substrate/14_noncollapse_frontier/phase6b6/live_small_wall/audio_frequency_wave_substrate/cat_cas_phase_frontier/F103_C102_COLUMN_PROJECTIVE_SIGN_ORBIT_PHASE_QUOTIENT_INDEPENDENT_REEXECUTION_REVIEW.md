# F103 C102 column-projective sign quotient independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`  
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Mechanism

The M158 A-register linear operations are homogeneous and act independently on
matrix columns, apart from public column permutations. The nonlinear B shear
uses only coefficientwise A squares. Therefore changing the sign of any whole
A column at any of the nine nodes leaves every declared B boundary invariant.
M160 canonicalizes each of these 45 sign orbits after every native operation
and transports one private orientation bit per column.

Production executes both public families at depths 1, 4, 16, and 64. All eight
quotient boundaries equal the identical raw coefficient recurrence. The
canonical carrier returns exactly to its sealed state under the actual inverse,
then the orientation ledger unseals the exact original raw A payload on the
same NumPy backing. An unrelated depth-16 transaction after depth 1 matches a
fresh carrier at restoration generation two without snapshot reload.

## Independent reconstruction

The scalar oracle imports neither M160 production, M158 production, nor NumPy.
It reuses the separately qualified scalar M158 reference for the raw recurrence
and independently implements sign canonicalization, orientation transport,
quotient forward/inverse order, and unsealing. It matches 32 production fields
across all eight cases.

The oracle separately flips each of the 45 column-sign generators. Every
generator maps to the same canonical representative as the unflipped seed,
preserves the depth-4 public B boundary, and unseals to the exact distinct raw
input. Because all seed columns are nonzero, the action is free and the orbit
has order `2^45`. Restoring an arbitrary member therefore requires at least 45
bits of distinguishing information.

## Resource result

The quotient is algebraically lawful but not materially compact. Its canonical
representative still occupies all 45,900 F103 cells. The ideal orbit cardinality
removes 45 bits, while the exact restoration ledger returns exactly 45 bits;
the net lossless information reduction is zero. The ledger needs at least six
packed bytes, and the strongest matched classical method remains the identical
45,900-cell coefficient recurrence. Canonicalization adds work and no runtime
or memory advantage is claimed.

## Claim ceiling

The result is limited to C5, two declared rotating-hub families, depths through
64, and independent whole-column signs. It does not cover other scalar groups,
singular quotients unrelated to sign, other interfaces, or other phase state
laws. It establishes no CATVM custody, distinct phase resource, computational
advantage, Small Wall crossing, physical execution, physical bit replacement,
or unbounded computation.
