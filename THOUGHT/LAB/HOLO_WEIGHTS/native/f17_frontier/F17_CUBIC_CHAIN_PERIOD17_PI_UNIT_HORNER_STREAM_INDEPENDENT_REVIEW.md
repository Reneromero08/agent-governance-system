# Independent review: single-resident Horner phase recurrence

Base head:

`0019d8521159210bc6e09c6351911304c7e6ab0a`

Reviewed production source SHA-256:

`bc9d94b0b90ba62dfab4476ff034b8b2a0e1a2905063d2cb531cf0471d99dc45`

Reviewed oracle source SHA-256:

`2035f73c40c52fdd882a760f317e5fa824606ff62c7e69b1f4649e41f7cf9128`

Reviewed qualifier SHA-256:

`2832b26759c792c91035fdb6eefabe4c003b1b35f497ad7426b488f681dc4403`

Reviewed full production output SHA-256:

`4f6e8f074c3914fe2cee8fa3d6654b4bcc0ba40f098796ea387150b321300b2d`

Reviewed full oracle output SHA-256:

`ae9b1b502aee3f09d76e38a101304c44c24f0c8ffa7ad0b1f45f2c9354fdd538`

Classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration class:

`EXACT_ALGEBRAIC_RESTORATION`

No blocking semantic or accounting defect remains within the bounded package
scope.

The Horner indexing is exact. Low-to-high recurrence coefficients encode the
public polynomial. The schedule initializes with the highest coefficient
times the seed, iterates through the remaining coefficients by operator,
scalar term, and addition, then applies the operator once more. This computes
the declared public recurrence with exactly sixteen operator applications.

The separate oracle advances coefficients sequentially by `x mod q` rather
than using production binary polynomial powering. It independently reproduces
both public-family boundaries at periods 1 and 64, the prior recurrence
boundaries, every named resource tuple, and inverse outputs.

Resource claims are limited to one resident carrier vector, six named phase
vector checkpoints, five named raw vector checkpoints, and the declared named
payload/component formulas. The normalized sixteen-element coefficient
program is counted, and its precursor scaled program is released before
checkpoint accounting. Internal operations, Python objects, allocator state,
simultaneous component peaks, native-library storage, and whole-process peaks
are not bounded.

All four phase named totals remain above the matched raw Horner checkpoint:
99,971 versus 10,005; 107,579 versus 10,097; 3,244,245 versus 2,790,766;
and 3,970,780 versus 2,901,994 bits. The identical normalized Horner
execution remains available to classical software.

Period-64 restoration is independently executed for both families with
canonical zero payload and ledgers, same backing, and generation and lease
equal to one. Period-1 cross-family primary-to-reuse execution agrees with a
fresh carrier and restores the same original backing. Cross-family reuse is
established only at period 1; retained generation and lease bookkeeping means
full carrier-object restoration and bounded repeated-use metadata are not
established.

The strict ceiling is Linux x86-64 repository Python software, two fixed
public F17 period-17 families, periods 1 and 64, the fixed public
49-direction/seven-unit balancing regime, one resident 17-ring-element
carrier vector, named checkpoint payload and component formulas, exact
boundary/resource/inverse/restoration parity, and period-1 cross-family reuse.

This does not establish global unit optimality, fixed or asymptotic width,
fixed total footprint, a whole-process peak, period-64 cross-family reuse,
bounded repeated-use metadata, machine-enforced custody or no-smuggle,
computational advantage, a distinct phase resource, Small Wall crossing,
catalytic inference, physical execution, physical bit replacement, or
unbounded computation.
