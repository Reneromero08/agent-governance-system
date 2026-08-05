# Independent reexecution review: Rotor-6 momentum-channel rank

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `NO_RESTORATION_CLAIM`

Result:
`PASS_RANK8_EXACT_F103_PORT_CELL_AND_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_REJECTED`

For the declared Rotor-6 F103 carrier, three separate eight-coordinate maps
have exact rank eight:

- the M204 first-pass momentum-port map at one valid necklace coordinate;
- the eight reflection-paired channel operators on eight valid off-diagonal
  bracelet coordinates; and
- eight public weight vectors from family zero, steps zero through seven.

The port witness is `6 I_8`, the operator witness is `2 I_8`, and their
determinants are 98 and 50 in F103. The public-weight witness determinant is
80. All are nonzero.

## Meaning of the certificate

The port map is surjective onto `F103^8`, with image cardinality
`103^8 = 12,667,700,813,876,161`. An exact lossless encoding of every port
value into fewer than eight cells of the same F103 alphabet is therefore
impossible. Independently, the channel-operator and public-weight spans reject
a uniform F103-linear operator quotient below rank eight that preserves the
declared families.

This does not reject lossy or program-specialized quotients, or encodings that
use fewer cells from a larger alphabet. It is not a lower bound for arbitrary
nonlinear software representations measured in bits.

## Production construction

The accepted diagnostic retains only three `8 x 8` witness matrices, their
public coordinate codes and program descriptors, and determinant/rank
scratch. It streams 64 port-coordinate comparisons and 896 nonzero direct
two-body candidate terms, of which 16 contribute to the operator witness. It
does not retain a dense `2,277 x 2,277` operator, the 74,613-cell occupation
sector, transition plans, or permanent assignments.

The initial exploratory coefficient used the source-oriented bosonic
multiplicity and produced `60 I_8`. Before evidence was sealed, inspection of
the actual branch operator established that its matrix rows are target-
oriented. The accepted implementation now streams the real direct row law and
produces `2 I_8`; the independent factor derivation agrees exactly.

## Independent construction

The independent oracle imports no CAT_CAS module. It enumerates all 74,613
six-particle occupations, reconstructs 4,389 cyclic necklaces and 2,277
dihedral bracelets, and confirms that every witness coordinate is present.
That full topology is verification-only state.

It then derives the operator witness twice: once from the direct two-body row
law and once from the two one-body factor actions. The witnesses agree. It
reimplements the public program and scattering law, verifies rank eight over
all seven families and all seventeen steps, and evaluates both witness
determinants by the 8! permutation formula rather than production Gaussian
elimination. Drop-row and duplicate-row/column mutations reduce every
applicable rank to seven.

## Resource and claim ceiling

Accepted peak named logical state is 292 slots:

```text
192 retained witness field cells
64 determinant/rank scratch field cells
3 determinant field cells
17 topology code integers
16 public program descriptor integers
```

Python objects, arbitrary-width integer storage, allocator/interpreter memory,
serialization, timing, and whole-process peaks are excluded and are not zero.
The strongest matched classical method is the identical streamed F103 rank
certificate.

The result is limited to grid 17, six exchange-symmetric rotors, the declared
rotation- and reflection-invariant sector, channels one through eight, F103,
public family zero steps zero through seven for the retained minor, and a
direct-process software diagnostic. M204's execution, exact restoration, and
reuse remain separately valid; this diagnostic itself makes no restoration
claim.

It does not establish CATVM custody, general rotor transfer, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, catalytic inference, or
unbounded computation.
