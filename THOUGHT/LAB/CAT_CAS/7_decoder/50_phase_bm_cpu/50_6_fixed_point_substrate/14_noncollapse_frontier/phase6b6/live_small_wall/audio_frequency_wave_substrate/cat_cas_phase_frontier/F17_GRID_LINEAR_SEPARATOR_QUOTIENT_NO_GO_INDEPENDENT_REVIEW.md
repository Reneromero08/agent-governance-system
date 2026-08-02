# Independent review: exact F17 grid linear-separator quotient obstruction

## Decision

- Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`
- Verification level: `INDEPENDENT_ORACLE_REEXECUTION`
- Factor-carrier restoration: `EXACT_ALGEBRAIC_RESTORATION`
- Transient projection buffers: `NO_RESTORATION_CLAIM`
- Claim-blocking findings: none

## Reexecution and proof review

The review inspected the production source, the independent oracle, the M119
factor carrier, and the M120 Kronecker-butterfly repair. It independently
reexecuted the six public descriptor cases at `n=2,3,4`. Production boundaries
agree coefficient-for-coefficient with both the oracle butterfly recurrence
and its separate Gray-histogram reconstruction.

The certified separator is the penultimate-to-final-row interface. This avoids
an assumption that a longer unresolved lower tail is everywhere nonzero. The
legal continuation-function matrix factors as `C D V`: `C` and `V` are
invertible Kronecker products, while `D` is an invertible diagonal of phase
roots. Consequently, a fixed `Q(zeta17)`-linear encoder through which all such
functionals factor must be injective and must retain at least `2^n` field
coordinates.

The determinant norm exponents are correct. Each tensor family contributes
`n * 2^(n-1)` powers of 17 and the combined exponent is `n * 2^n`.
Independent exact resultants also give `Norm(zeta17^j - 1) = 17` for every
`j=1,...,16`.

The independent oracle imports neither production code nor its phase backend.
It reconstructs explicit continuation, vertical, and combined matrices over
both `F103` and `F137`. All have ranks `4,8,16`. Setting one vertical weight to
zero and duplicating one local continuation choice each reduce rank to
`2,4,8`. A coordinate-drop encoder has a nonzero kernel vector that a valid
continuation detects.

The accepted production certificate uses only the analytic local determinant
law. It neither materializes dense rank matrices nor enumerates continuation
families. Dense matrices are limited to the bounded independent oracle. The
descriptor interface accepts arbitrary public nonzero F17 unary and edge
weights; executed restoration and reuse cover two distinct generated
descriptors per size.

Exact reverse operations and seed unload restore the borrowed factor carrier
on its original backing before unrelated reuse. The accepted path uses no
snapshot reload and retains no inverse history. Transient projection buffers
are not borrowed carrier state and carry no restoration claim.

## Strict ceiling

The result rejects only a uniform fixed linear separator encoder that must
support arbitrary field messages and every legal nonzero continuation. The
analytic law applies at arbitrary width; formula certificates are emitted for
`n=1,...,16`, while computational phase and rank reexecution remains bounded to
`n=2,3,4`.

It does not adjudicate nonlinear or program-dependent encodings, restricted
descriptor families, ADD/MTBDD, MPS/MPO, matchgate/Pfaffian or other
holographic algorithms, global contractions, rematerialization, approximation,
or total time, memory, and bit complexity. It establishes no distinct phase
resource, computational advantage, Small Wall crossing, CATVM custody,
catalytic inference, physical waveform execution, unbounded computation, or
replacement of physical bits with pi.
