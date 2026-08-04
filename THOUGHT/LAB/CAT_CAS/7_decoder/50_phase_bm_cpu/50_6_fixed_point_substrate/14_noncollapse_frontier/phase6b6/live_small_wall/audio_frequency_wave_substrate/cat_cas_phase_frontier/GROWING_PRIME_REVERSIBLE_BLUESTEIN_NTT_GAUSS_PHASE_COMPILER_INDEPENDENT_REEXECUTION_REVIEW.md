# M183 independent reexecution review

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Verified mechanism

M183 changes the M181 update law rather than adding more streamed prime
fixtures. For `h=q-1`, it recognizes the complete Gauss family as the length-h
Fourier transform of the public additive phase orbit. Bluestein's chirp
identity converts that transform into one linear convolution of width

```text
M = least power of two at least 2h-1.
```

The exact auxiliary residue fields contain roots of order q, 2h, and M. The
power-of-two convolution is evaluated by radix-two NTT butterflies. Every
declared q=5 through q=53 descriptor matches the direct h-by-h Gauss transform
exactly.

The carrier is one backing with `h+3M` exact field cells: h descriptor cells
and three M-cell left-chirp, kernel, and product segments. Forward compilation
uses compute-copy-uncompute. It initializes the two chirps, transforms them,
accumulates their pointwise product into the third segment, inverse-transforms
that segment, shears the h results into the descriptor, and reverses all
scratch work. Thus all 3M scratch cells are zero while the compiled descriptor
is resident.

One compiler invocation executes six length-M transforms and exactly
`3M log2(M)` butterflies. The inverse rematerializes the same compiler and
subtracts the resident descriptor, for twelve transforms over the full
lifecycle. This changes the asymptotic compiler law from theta-q-squared
direct character-sum work to theta-M-log-M exact field work, with M linear in
q. No finite runtime speed claim is made.

## Projection repair

The first projection draft would have scanned the multiplicative orbit once
per Mellin channel and quietly preserved quadratic work. The accepted path
does not do that. It scans public topology at most three times to obtain the
determinant and scale character steps, then advances both character values by
one field multiplication per channel. Projection uses h channels, at most 3h
character-orbit visits, and no retained discrete-log table. Accepted forward
compiler plus projection work is theta-M-log-M plus q.

The projected final scalar is retained outside inverse history. Every primary
transaction restores the exact zero carrier on the same backing. An unrelated
program then consumes that restored backing, matches a fresh carrier in final
boundary and resource signature, and restores it again. Missing inverse,
wrong additive-phase inverse, omitted-frequency inverse, and null-carrier
controls fail. No snapshot is used, and direct-process bookkeeping is not
generation or lease enforcement.

## Independent reconstruction

The oracle imports neither production nor M180/M181. It independently finds
all field roots, constructs direct Gauss tables, and implements a recursive
out-of-place radix-two transform instead of production's iterative in-place
NTT. Its independently derived Bluestein descriptors and boundary scalars
match production in all 14 fields. It separately rematerializes inverse state,
checks same-backing restoration and fresh/restored reuse, and repeats the
missing, wrong-phase, and omitted-frequency attacks.

At q=5 and q=7, the oracle also directly evaluates the original
seven-dimensional open relation over 49,600 and 603,288 nonzero source terms.
Those final scalars match the compiled descriptor boundaries exactly.

## Resource accounting and baseline

Declared M ranges from 8 through 128. The one-backing carrier grows from 28
through 436 field cells, compared with M180's 17 through 209 counted
materialized-table cells and M181's ten-cell streamed workspace. After
forward, only h descriptor cells are nonzero; nevertheless the allocated 3M
scratch capacity is counted because it is used by the accepted path.

Twiddles are generated from public roots and no twiddle table is retained.
The carrier bit capacity grows with both cell count and auxiliary-prime width.
Python objects, allocator behavior, modular exponentiation internals, and
whole-process memory are not claimed. M180's direct table, M181's fixed-cell
quadratic stream, and M183's linear-state subquadratic transform are all kept
as distinct state/work points.

The strongest compact classical implementation is the identical reversible
Bluestein/NTT compiler. It has the same cells, roots, butterflies, projection,
inverse, and reuse law. M183 therefore establishes no state, work, or
computational advantage.

## Claim ceiling and next obstruction

The result is limited to the 14 declared prime and new auxiliary-root fields,
one rank-three nonzero-scale transaction per field, unrelated same-field
reuse, and direct-process exact residue software. It establishes no sublinear
state, fixed exact bit width, CATVM custody, machine-enforced hidden
intermediate, distinct phase resource, computational advantage, Small Wall
crossing, physical waveform execution, replacement of physical bits with pi,
or unbounded computation.

The quadratic-work obstruction is repaired, but only by accepting a linear
descriptor plus linear transform scratch and an identical classical NTT. The
next phase-owned repair must fuse final-boundary projection with the additive
transform without materializing the h descriptor, or introduce a phase-native
operation whose useful law is not the same classical NTT recurrence. It must
retain exact restoration and count any adjoint weights, chirps, scratch, and
rematerialization.
