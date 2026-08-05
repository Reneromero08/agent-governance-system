# M193 Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Structural quotient

M192 showed that one scale plus eight multiplicative pair coordinates do not
contain the off-diagonal scattering output from three rotors onward. M193
keeps arbitrary values on distinct pair-signature fibers instead of requiring
one monomial parameterization. Public topology compilation proves exact
equitability separately for every one of the sixteen signed scattering shift
bases. The diagonal two-body phase is constant on each signature fiber, so
the combined declared operation family closes on the quotient.

| rotors | necklace cells | pair-signature fiber cells | ratio |
| ---: | ---: | ---: | ---: |
| 2 | 9 | 9 | 1.000 |
| 3 | 57 | 33 | 1.727 |
| 4 | 285 | 165 | 1.727 |
| 5 | 1,197 | 621 | 1.928 |

This is an invariant fiber-constant subspace, not a quotient for arbitrary
necklace inputs. It does not establish a fixed-rank law: the quotient still
grows from 9 to 621 cells over the declared rotor counts.

## Independent reconstruction

The independent oracle imports no production module. It enumerates bosonic
occupations from multisets and creates shift transitions by ordered particle
labels, rather than production mode multiplicities. It retains all full
necklace shift bases as verification state, derives the quotient separately,
and compares every target row in every fiber for each shift.

For both F103 and F239, quotient and full-necklace execution agree exactly at
primary depth eight. The two implementations agree exactly on state
commitments, public boundaries, source and quotient dimensions, all compiler
work counts, all 16-basis plan nonzero counts, restoration, reuse, and inverse
controls. The quotient plan sizes are `272`, `2,448`, `21,904`, and `131,168`
nonzero field coefficients. At five rotors compilation visits 20,349
occupation histograms, 263,568 nonzero mode-pair-shift terms, and 383,040
ordered-particle-shift terms.

## Restoration and resources

The accepted two-register shear retains twice the quotient fiber count and
uses at most two quotient-sized word buffers. It retains sixteen public shift
basis plans, with no answer-bearing inverse history, assignment table, or
relation table. The inverse recomputes the actual public word and subtracts it
from the same target backing. Primary and unrelated reuse restore every field
cell exactly; restored and fresh reuse boundaries agree; generation reaches
two. Missing, wrong, reordered, and null controls discriminate in every case.

Full necklace vectors and shift bases are verification-only. Public topology
compilation still materializes occupation and necklace objects, and its work
and counts are not erased by the smaller runtime carrier. Python containers,
allocator overhead, big-integer expression temporaries, compiler memory, and
whole-process peaks are excluded. The 16-basis public plan may dominate the
carrier state, so no whole-path memory or timing advantage is claimed.

The strongest compact classical implementation is the identical
pair-signature fiber quotient recurrence. M193 changes the phase machine's
logical carrier, but does not establish a resource unavailable to compact
classical software.

## Claim ceiling

`GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS2_TO5_F103_F239_ALL16_SIGNED_PAIR_SHIFT_BASES_PRIMARY_DEPTH8_REUSE_DEPTH5_DIRECT_PROCESS_SOFTWARE_ONLY`

No CATVM custody, arbitrary necklace quotient, fixed-rank growing closure,
distinct phase resource, computational advantage, Small Wall crossing,
physical waveform execution, physical bit replacement, or unbounded
computation is established.
