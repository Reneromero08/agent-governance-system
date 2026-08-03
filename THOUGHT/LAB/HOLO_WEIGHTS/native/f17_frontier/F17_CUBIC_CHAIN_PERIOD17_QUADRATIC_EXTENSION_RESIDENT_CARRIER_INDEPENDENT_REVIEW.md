# M116 quadratic-extension resident-carrier review

## Decision

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level   SEPARATE_REFERENCE_PARITY
restoration class    EXACT_ALGEBRAIC_RESTORATION
```

Scientific source parent:

`86fd1c944a139996709b04714be6713fc64726cd`

Reviewed production source SHA-256:

`1eea251c3dee2e6e0673376c0285fed88de31efc7a547418cbe960b1c89bb111`

Reviewed separate oracle source SHA-256:

`552bb44edd11ae6e3c9cf1270cac6a1cabd6c914c9f7f8d228ae37a066110e24`

Reviewed production output SHA-256:

`0db1f3104d9c20ec1d91e50d163522ef2065608c41c4d91c39bd58587869bb94`

Reviewed oracle output SHA-256:

`e855a872ec8ae72d9c24b10de3646f6ddc06e085d6220df6cb5dd3bb85668f89`

Both implementations were executed from the repository Linux virtual
environment. No hosted workflow reexecution is established.

## Exact mechanism

The production carrier represents each element of `Q(zeta_17)` as

```text
x = A + zeta B,
```

where `A` and `B` each use the existing eight-coordinate integral basis of
the real subfield and satisfy

```text
zeta^2 - s1*zeta + 1 = 0.
```

The pair is an integral coordinate isomorphism. The forward coordinate
matrix has determinant `+1`; it is not a quotient and it does not reduce
algebraic dimension. Each element still contains 16 independent integer
coordinates.

Pair addition is componentwise. Pair multiplication uses three real-subfield
products:

```text
ac    = A*C
bd    = B*D
cross = (A+B)*(C+D)

real = ac-bd
zeta = cross-ac-bd+s1*bd
```

Multiplication by `s1` is the fixed linear action in the declared real basis,
not a fourth real-subfield product. Conjugation is
`(A+s1*B,-B)`. The 256 ordered basis products, 16 basis conjugations, 16
basis round trips, and the defining quadratic relation all match the
predecessor full-cyclotomic arithmetic exactly.

## Accepted execution path

The existing full-cyclotomic Horner construction remains in the forward
path. Its output is streamed into the two-lane representation, after which
the full vector is not retained. The resident projection and public
unit-ledger action operate on the actual two-lane carrier. Projection performs
one pair-to-full lift of the selected public scalar boundary; it does not
materialize a full projected vector.

Restoration independently rematerializes the public full Horner output,
converts it to the pair representation, checks it before mutation, and
subtracts it from the actual resident carrier. It does not lift the resident
pair back to a full carrier, retain inverse history, or reload a baseline.
The full forward construction and full inverse rematerialization are both
declared and counted. Therefore `boundary-only full lift` describes only the
pair-to-full lift operation; it does not mean that the whole transaction is
pair-native.

## Independent reconstruction

The separate oracle does not import the production M116 module. Its semantic
pair arithmetic uses the power basis of the real subfield, while its resource
reexecution separately converts to the production integral `S` basis. This
separation checks both the extension law and the declared implementation
schedule instead of copying production coordinates as its only oracle.

The oracle independently reexecutes both public program families at periods
1 and 64. It reproduces all four public boundaries, raw-Horner boundaries,
forward and inverse resource tuples, checkpoint payloads, retained public
tables, named totals, exact restoration, and the period-1 unrelated reuse
sentinel. Its power-basis pair result equals its independent integral-basis
schedule for every case. All algebra, topology, and table checks pass.

## Resource result

```text
period family   M115 total  M116 total  increase  raw Horner  M116 - raw
1      PRIMARY       93,790      94,616       826      10,005      84,611
1      REUSE        101,475     102,400       925      10,097      92,303
64     PRIMARY    3,324,441   3,325,225       784   2,790,766     534,459
64     REUSE      3,695,435   3,696,278       843   2,901,994     794,284
```

The resident pair is also larger than the predecessor full representation in
all cases:

```text
period family   pair resident  full resident  pair - full
1      PRIMARY          3,036          2,937             99
1      REUSE            5,452          5,254            198
64     PRIMARY        222,134        222,077             57
64     REUSE          211,318        211,202            116
```

The pair conversion live maxima are 6,530, 11,503, 464,047, and 441,411
payload bits; they include the empty borrowed-carrier backing while the full
output and partial pair output coexist. Pair projection counts its
accumulator, scale, product, normalized scalar state, and three-real-product
multiplication schedule. Its conservative live maxima are 11,238, 10,181,
713,834, and 721,563 bits. Full forward and inverse Horner checkpoints,
search temporaries, retained public tables, conversion input and output,
resident state, the one final scalar lift, and raw Horner are all reported.

The named total is a conservative sum of component maxima that need not be
simultaneous. It is not a measured whole-process peak. Python object headers,
reference and container storage, allocator state, native-library memory, and
bigint implementation scratch remain excluded. Verification-only algebra and
carrier attacks are reported separately from accepted execution work.

The strongest matched classical implementation can execute the identical
two-by-eight recurrence with the same pair laws, public coefficients,
operator, and boundary. Raw Horner and M115 remain additional baselines. No
comparison establishes an advantage.

## Restoration and controls

All four cases reproduce the raw-Horner boundary and restore the exact
canonical state on the original backing. Period-1 reuse runs the unrelated
second public program on that restored backing and matches a fresh carrier in
both boundary and declared phase-resource signature. Restoration generation
and lease both advance to two. These are observed direct-process metadata,
not machine-enforced CATVM custody.

The production package retains controls for reordered inverse, missing
inverse, wrong inverse before mutation, mutation before inverse mutation,
null carrier, absent snapshot command, exact restoration, no baseline reload,
and the single permitted final-boundary lift. The restoration class is
`EXACT_ALGEBRAIC_RESTORATION`. The separate oracle rehosts the algebra,
boundaries, resource schedule, restoration arithmetic, and reuse boundary;
object-backing identity and the negative-control exception paths remain
source-local checks rather than an independently rehosted service.

## Strict ceiling

This package establishes only:

```text
LINUX_X86_64_PYTHON
TWO_PUBLIC_F17_PERIOD17_FAMILIES
PERIODS_1_AND_64
EXACT_POST_FORWARD_TWO_BY_EIGHT_REAL_SUBFIELD_STORED PAIR
PAIR_NATIVE_PROJECTION_AND_LEDGER MATERIALIZATION
ONE_SPLIT_TO_FULL_FINAL_SCALAR_BOUNDARY_LIFT
FULL_CYCLOTOMIC_FORWARD_AND_INVERSE_REMATERIALIZATION
EXACT_ORIGINAL_BACKING_RESTORATION_AND PERIOD1 CROSS_FAMILY_REUSE
SEPARATE_REFERENCE_PARITY
SOFTWARE_ONLY
```

It does not establish dimension or rank reduction, a fully pair-native
recurrence, lower cost than matched raw Horner, a distinct phase resource,
computational advantage, Small Wall crossing, CATVM custody, catalytic
inference, physical waveform execution, replacement of physical bits with
pi, or unbounded computation.

## Next obstruction

Replacing resident full-cyclotomic coordinates with two real-subfield lanes
changes representation but not rank. It slightly worsens measured resident
payload and total named cost, while full forward and inverse construction
remain and classical software has the identical pair recurrence. Repeating
this representation at larger periods would not remove the obstruction. A
successor must either eliminate the full construction through a lawful
pair-native recurrence and still survive the identical classical comparison,
or introduce a phase-owned coupling whose useful state law is not immediately
the same compact classical recurrence.
