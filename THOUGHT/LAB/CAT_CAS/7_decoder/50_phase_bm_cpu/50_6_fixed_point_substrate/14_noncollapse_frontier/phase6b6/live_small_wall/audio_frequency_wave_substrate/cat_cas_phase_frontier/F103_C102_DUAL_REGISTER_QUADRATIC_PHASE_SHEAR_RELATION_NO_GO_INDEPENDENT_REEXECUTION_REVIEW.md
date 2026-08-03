# F103 C102 dual-register quadratic shear independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`  
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Reexecution boundary

The production package was reexecuted for 12 public programs: interfaces C5
and C7, both declared families, and depths 1, 2, and 4. The scalar oracle
imports neither the production module nor NumPy. It reuses the already
independently qualified M157 scalar group-algebra reference only for the
unchanged rotation, rank-one composition, intersection, and exact inverse
kernels. It independently reconstructs the M158 program descriptors, second
register seed, coefficientwise quadratic shear, inverse order, collision,
character-coupling support, final boundary, and coordinate law.

The oracle made 96 exact comparisons. All 12 forward register commitments and
boundary commitments matched, and all 12 scalar inverse executions restored
the exact two-register coefficient state.

## Preserved strict result

The reversible update

```text
B[e] <- B[e] + gamma[e] A[e]^2  (mod 103)
```

breaks the M157 single-character evaluation quotient. The explicit pair
`A=1` and `A=1+(t-5)` has the same value at `t=5` before the shear and a
different projected B value afterward. For the declared public shear, the
quadratic boundary observable has 102 nonzero diagonal Hessian entries, so an
arbitrary-input linear sketch supporting that observable needs dimension at
least 102 within this exact scope.

The multiplier transform has support 51 rather than 102. The stronger dense-
kernel interpretation is therefore rejected. The exact quadratic dependency
law nevertheless lets each of the 102 input characters influence every output
character through at least one companion character.

The two-register payload is 45,900 F103 cells at C5 and 89,964 at C7, exactly
`204 n^2`. The strongest matched compact classical method is the identical
dual-register coefficient recurrence and has the same coordinate count. This
is a useful nonlinear phase mechanism and a no-go for the old one-character
quotient, not evidence of a distinct phase resource or advantage.

## Controls

Independent scalar controls passed for missing inverse, mutated inverse,
reordered inverse, disabled shear, the single-character collision, the rank-102
Hessian, and all-character causal coupling. Production-local controls also
reject null carriers, wrong owners, wrong types, premature final projection,
and resident coefficient projection. Exact restored-carrier reuse ran on the
same NumPy backing for an unrelated second program and for eight repeated
cycles without snapshot reload.

## Package-local ceiling

The independent oracle does not verify NumPy backing addresses, production
operation counters, the production full-character implementation path, the
direct-process custody state machine, or excluded allocator/native-library
workspace. Physical NumPy transient peak bytes were not measured, and no
memory or runtime advantage is claimed. There is no CATVM process boundary in
this package and no shared latent port with multiple consumers.

Nothing here establishes compact growing-depth closure, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
or silicon execution, replacement of physical bits with pi, or unbounded
computation.
