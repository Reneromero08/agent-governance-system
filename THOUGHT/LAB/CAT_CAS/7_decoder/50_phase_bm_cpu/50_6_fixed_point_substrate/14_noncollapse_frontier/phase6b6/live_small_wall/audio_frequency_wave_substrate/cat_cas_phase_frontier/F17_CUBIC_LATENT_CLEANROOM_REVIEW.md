# Focused independent review: exact F17 single-cubic-latent quotient

**Scientific source commit:** `251d89cf9ffbf48f769fe8b34b9a5c4103c67d76`
**Decision:** `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`
**Verification level:** `SEPARATE_REFERENCE_PARITY`
**Restoration class:** `EXACT_ALGEBRAIC_RESTORATION`

## Strict result

The reviewed source establishes this bounded software claim:

```text
BOUNDED_EXACT_F17_SINGLE_CUBIC_LATENT_CHARACTER_SUM_MULTI_PORT_PHASE_QUOTIENT_WITH_NATIVE_FOURIER_SHEAR_CLOSURE_RESTORATION_AND_REUSE
```

Its ceiling is Linux/Python/NumPy, exact arithmetic over F17, one unresolved
cubic latent coordinate, public-port widths 2, 4, 8, 16, and 32, two generated
program families, nonzero Gaussian Fourier pivots, a nonsingular final public
Gaussian block, and one fixed 17-value final character trace.

The accepted carrier stores `(A,b,l,c,a,d,e,sign)`. Public shears and Fourier
modules update that tuple while retaining the shared latent coordinate in the
character-sum expression. The third finite difference of the latent cubic is
`6 mod 17`, so the represented phase law is outside the predecessor's
quadratic Gaussian chart. This does not establish closure for multiple cubic
latents or arbitrary non-Gaussian phase laws.

## Independent reconstruction

The separate oracle does not import the production backend, compiler, or
projection. It consumes the hashed public module descriptors and reimplements
the exact coefficient recurrence, cyclotomic boundary reduction, reverse
execution, and unrelated reuse with Python integer lists at every tested
width.

At ports 2 and 4 it also constructs the explicit complex public-port state.
At port 2 a second dense reconstruction retains the latent coordinate as an
independent 17-cell axis throughout all public forward modules and contracts
it only at final projection. The production boundary, inverse restoration,
and reuse boundary agree within the predeclared `3e-11` dense-oracle
tolerance. Exact list-state comparisons use equality.

No source file was copied into the oracle. The oracle independently
reimplements the equations; its only production inputs are serialized public
module descriptors and their hashes.

## Adversarial findings and repairs

The first reviewed working version exposed four reduced latent-polynomial
coefficients, contracted the latent before the dense oracle's module
execution, retained compiler candidate/witness lists, undercounted simultaneous
projection output, and asserted only primary-transaction array backing.

The committed source repairs those package defects:

- the boundary serializes only the final 16 cyclotomic coefficients and
  declared public metadata;
- the dense oracle retains an explicit latent axis at port 2;
- compiler selection uses a scalar count and fixed-width port mask without
  candidate or witness lists;
- projection scratch counts the 17-cell histogram and 16-cell canonical
  output simultaneously;
- both transactions assert all coefficient-array backing, generation 2, and
  lease 3.

The no-smuggle statement remains package-serialization-local. There is no
isolated service or machine-enforced CATVM custody in this package.

## Controls and resource law

Missing, wrong, and reordered inverses fail in the production package. These
controls and compiler generation remain source-local; the independent oracle
reexecutes the emitted descriptors rather than regenerating the compiler or
its controls.

Resident F17 cells at ports 2, 4, 8, 16, and 32 are:

```text
12  28  84  292  1,092
```

The corresponding semantic fixed-width payloads, including the sign bit, are:

```text
61  141  421  1,461  5,461 bits
```

The accepted path counts baseline plus carrier, compiled descriptors,
determinant-search compilation, execution scratch, final projection scratch,
boundary serialization, restoration, and reuse. Python object overhead,
NumPy/native-library internals, allocator state, OS state, and whole-process
peak remain outside the bound. Dense arrays are verification-only.

## Claim ceiling

Preserved subclaims:

- exact one-cubic-latent F17 coefficient closure across the declared widths;
- multiple public Fourier consumers of the same unresolved latent coupling;
- fixed 17-value final trace without a `17**ports` accepted-path tensor;
- exact algebraic restoration, same-backing reuse, and no baseline reload;
- a phase algebra strictly broader than the quadratic Gaussian predecessor.

Rejected interpretations:

- multiple-latent or arbitrary non-Gaussian closure;
- non-enumerative latent closure;
- machine-enforced no-smuggle custody or CATVM execution;
- necklace-carrier integration;
- strongest compact classical method;
- a distinct phase resource, computational advantage, Small Wall crossing,
  catalytic inference, physical waveform execution, physical bit replacement,
  or unbounded computation.

The identical exact coefficient recurrence remains a compact classical
implementation. The next obstruction is growth in the interaction signature
or the final `17**k` trace when the number of independently interacting cubic
latents grows.
