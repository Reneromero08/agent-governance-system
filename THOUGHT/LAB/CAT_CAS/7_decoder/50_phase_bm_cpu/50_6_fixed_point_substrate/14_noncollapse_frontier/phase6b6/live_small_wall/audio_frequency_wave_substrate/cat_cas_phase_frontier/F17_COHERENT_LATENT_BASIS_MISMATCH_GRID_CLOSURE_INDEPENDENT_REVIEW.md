# M126 Independent Review: Coherent Latent Basis-Mismatch Grid Closure

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident weight and latent phase-carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Transient Kasteleyn and elimination buffers: `NO_RESTORATION_CLAIM`

The verified family is restricted to two public descriptor families on even
open square grids. Exact `Q(zeta_17)` boundaries agree at `n=2,4,6`, and
independent `F103/F137` structural executions agree at `n=2,4,6,8,10`.

## Independent reconstruction

A fixed 17-coordinate unresolved phase port is represented by two resident
vectors. Every latent coordinate controls all edges of each of two grid-wide
basis-mismatch modules through

```text
T(zeta^(control*latent)) S(1)^T = diag(1,zeta^(control*latent)).
```

The modules are separated by exact Fourier and quadratic-chirp updates. A
mutation of either basis control or chirp changes the final boundary, and an
independent finite-field control confirms that Fourier and chirp order changes
the latent state. Only the final `w[0]` scalar is projected.

The oracle imports neither M126 production nor the M125 matchgate package. It
shares the established generic exact-arithmetic backend, but independently
implements row-profile weighted matching, the complete 34-coordinate latent
recurrence, bounded native-Holant enumeration at `n=2`, and the paired-basis
identity. All six exact cases, twenty dual-field structural cases, twelve
bounded native-Holant checks, and four basis identities agree.

## Restoration, controls, and resources

The accepted path streams public loads and each of the 17 sector closures
directly into resident state. It materializes neither a latent diagonal
boundary vector nor an edge-local signature table or edge assignment table.
The actual reverse order rematerializes both module closures and reverses every
Fourier, chirp, shear, and load operation. The same exact carrier backing is
restored, generation advances once per transaction, and an unrelated program
reuses it with fresh/restored boundary and resource-signature agreement. No
inverse history or snapshot reload is used.

Controls cover missing inverse, wrong ownership, premature projection, null
carrier, reordered inverse, basis and chirp mutations, Fourier/chirp order,
intermediate serialization, and snapshot-command absence. Control carriers
are executed sequentially and are excluded from the accepted-path peak.

For even `N`, resident state is `4N(N-1)+34` field cells. Each determinant has
dimension `N^2/2`; caller matrix, elimination copy, and five named scalars use
`N^4/2+5` field cells. Fourier uses 34 named input/output cells, so the stated
transaction transient ceiling is `max(N^4/2+5,34)`. Final boundary payload
bits and JSON bytes are recorded per transaction. Python containers,
allocator/native-library storage, bigint internals, and whole-process peak are
excluded. Full exact bit complexity is not established.

## Claim ceiling

M126 establishes a bounded coherent shared latent port with two grid-wide
consumers, final-only projection, exact restoration, and reuse. It repairs the
single public cancellation in M125, but the fixed 17-state port is exactly the
same classical direct sum: the strongest matched baseline is the identical
34-coordinate recurrence plus 17 compact Kasteleyn closures per module.

It does not establish arbitrary latent geometry or planar Holant closure,
CATVM custody, a distinct phase resource, computational advantage, Small Wall
crossing, physical waveform execution, replacement of physical bits with pi,
or unbounded catalytic computation.
