# M179 Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`  
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Exact scope

The production package replaces the preceding full-state projective-orbit
search with a typed non-Gaussian generating relation.  Its open phase is

```text
f(X,t) = det(X)/t
```

for a symmetric `3 x 3` matrix over `Fq` and nonzero `t`.  The public
singular completion family has seven matrix congruence classes, classified by
rank and discriminant square class.  On `det(X)=0, t!=0` it permits one scalar
coefficient per singular class and multiplicative `t` character; on `t=0`
it permits one coefficient per matrix class.  It therefore has
`5*(q-1)+7` coordinates, not fixed width in `q`.

Over `q=5` with amplitudes in `F41`, an exhaustive search of all 256 source
and target character-signature pairs finds exactly five closures.  Each is
checked at all 78,125 boundary points.  The primary `(0,0)->(2,2)` closure
uses 17 nonzero source-completion coordinates and three nonzero target
coordinates.  The joint transformed-source/target-boundary space has rank 35.

The same seven-stratum family does not transfer to the next tested safe prime.
Over `q=7` with amplitudes in `F43`, every one of the 1,296 source/target
character pairs fails on 576 evenly spread open-boundary points.  Any global
closure in the declared family would have to satisfy those equations, so the
sample is sufficient to reject that family; it is not a full enumeration of
all `7^7` output points.

## Independent reconstruction

The no-import oracle independently rebuilds the finite fields, symmetric
matrix rank and congruence classes, determinant phase, singular basis, and
modular column-space tests.  Its seven-axis transform uses flattened
matrix multiplication rather than the production tensor contraction.

For `q=5`, it independently searches all character pairs, recovers the same
five signature pairs, and rechecks each sealed completion vector on all
78,125 points.  For `q=7`, it does not call the production DFT.  It evaluates
the sampled transform as direct character sums grouped by determinant, trace
pairing, and congruence class, and again finds zero survivors among all 1,296
pairs.

The oracle also executes the exact forward transform and normalized inverse
on the primary residue carrier.  Equality is elementwise modular equality,
not hash equality.  The production package separately confirms that the same
array backing is restored, then consumed by an unrelated linear-coordinate
phase shear at restoration generation two, agrees with a fresh execution,
and restores again without a snapshot.

## Controls and accounting

The open stratum without completion, a wrong target character, a one-cell
completion-region perturbation, and a missing inverse axis all fail.  A
reordered-axis inverse is not a valid negative control because the separable
DFT axis operators commute.  Null carrier rejection is explicit.

The result counts the `q^7` residue carrier, `5*(q-1)+7` completion
coordinates, dense q=5 singular-basis materialization, seven separable
transform stages, modular column-space solving, full-boundary verification,
inverse work, verification copies, and restored-carrier reuse.  Python object
allocation and native-library private workspace are excluded and declared.
The strongest matched classical baseline is the identical seven-axis
separable DFT with the identical public stratum descriptor and character-sum
recurrence.

## Claim ceiling

```text
Q5_F41_FULL78125_POINT_AND_Q7_F43_576_POINT_TRANSFER_ATTACK_SYMMETRIC3_DETERMINANT_OVER_SCALE_PUBLIC_SEVEN_CONGRUENCE_STRATA_MULTIPLICATIVE_CHARACTER_COMPLETION_DIRECT_PROCESS_SOFTWARE
```

The result establishes one bounded arithmetic-local closure and a transfer
failure for the declared completion family.  It does not establish a
transferable growing-prime closure, fixed-rank growth, CATVM custody, a
machine-hidden intermediate, a distinct phase resource, computational
advantage, a Small Wall crossing, physical waveform execution, physical-bit
replacement, or unbounded computation.

The next obstruction is to replace scalar stratum weights with a genuinely
transferable bounded-conductor or critical-locus object and to test whether
that object closes without moving growth into conductor rank, carrier size,
or an equivalent compact classical trace recurrence.

## Durable identities before the science commit

- Production source SHA-256:
  `a7fe434d16d7686d26a683769d859cd5a8448a560c1e85a3286cd81e4ab293a9`
- Sealed production result SHA-256:
  `1ee51d47cbf7bc93ed81ccee59cdf60eebe20e36664ffa5efd09573803094f45`
- Independent oracle source SHA-256:
  `6fc5870cfb690a906d288774887e772334164231494493382719c7bbd5aa359b`
- Sealed independent result SHA-256:
  `fb71c8172ea7e81784ffc69cd93125da788969252a8315b610e1687de93de195`
- Qualifier SHA-256:
  `3b32e4698e3964be5bb4e166479318777fdb4c936fba3d590d8f021063700b05`
