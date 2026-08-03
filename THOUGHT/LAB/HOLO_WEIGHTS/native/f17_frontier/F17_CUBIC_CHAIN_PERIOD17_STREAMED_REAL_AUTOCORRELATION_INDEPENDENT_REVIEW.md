# M114 streamed real-autocorrelation review

## Decision

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level   SEPARATE_REFERENCE_PARITY
restoration class    EXACT_ALGEBRAIC_RESTORATION
```

Scientific source parent:

`319118cc4958304ee5c24d7a6dc9602adb277b1d`

Reviewed production source SHA-256:

`c872e78bf42d94ab5c3f84989faa3dbfae8d4232dec970fd8a5bbb7b0ccefe1c`

Reviewed separate oracle source SHA-256:

`52aa4cdfb8d53dd866220a9e4636ee980444cbd4d85e8af389486b12a52a045f`

Reviewed full production output SHA-256:

`ab00e77c96dceef82c0d24e2537305ea5425dcb201bd20cd670eee63dda70d43`

Reviewed full oracle output SHA-256:

`165e15f898c97a5124d53bfd24ef398bab013e42d662b92bddbab74261acd44c`

Both implementations were executed from the repository Linux virtual
environment. No hosted workflow reexecution is established.

## Independent reconstruction

Production extends the M113 integral basis
`(1,s_1,...,s_7)`. The separate oracle does not import the production M114
module and instead extends the M113 degree-eight power basis

```text
Q[y] / (
    y^8 + y^7 - 7 y^6 - 6 y^5
    + 15 y^4 + 10 y^3 - 10 y^2 - 4 y + 1
).
```

For every nonzero balance call, both implementations independently perform:

```text
full carrier element
-> full cyclotomic x * conjugate(x) temporary
-> exact full-to-real conversion
-> degree-eight real accumulator
```

The temporary full product is released before the next carrier cell. Neither
accepted implementation constructs the predecessor's summed degree-sixteen
norm. Both retain the full carrier and the one final certified full
cyclotomic unit action.

The oracle independently reconstructed the two public program families,
operators, Horner recurrences, boundaries, inverse rematerialization,
restoration, and period-1 cross-family reuse. It matched every production
forward and inverse resource tuple and every named subtotal.

## Exact shapes and work

The first attempted gate incorrectly assumed every norm call had 17 cells.
A focused diagnostic established that final scalar unit materialization also
uses a lawful singleton norm. The repaired gate declares both shapes and
rejects all others.

```text
period family   calls  17-cell  singleton  terms  unexpected
1      PRIMARY      5        4          1     69           0
1      REUSE        5        4          1     69           0
64     PRIMARY     93       63         30   1101           0
64     REUSE       98       66         32   1154           0
```

Every term is one full cyclotomic multiplication, one exact conversion, and
one real-subfield addition. Term counts equal the exact sum of input cells.

## Resource result

```text
period family   streamed named total  raw Horner  streamed - raw
1      PRIMARY                 96,088      10,005          86,083
1      REUSE                  103,778      10,097          93,681
64     PRIMARY              3,466,223   2,790,766         675,457
64     REUSE                3,840,024   2,901,994         938,030
```

The maximum named streamed-norm live payload is respectively 10,270,
10,334, 465,870, and 475,093 bits. It includes the persistent real
accumulator together with conversion or addition temporaries. The named
component total is a conservative sum of component maxima, not a measured
simultaneous process peak.

The full aggregate norm is eliminated, but this does not improve the accepted
path below either matched baseline. All four named phase totals remain above
the raw Horner recurrence. More importantly, ordinary compact classical
software can execute the identical term order, exact conversion, real
accumulator, and search. No computational separation follows.

The M114 totals are 2,609 bits above M113 at period 1 and 8 bits above M113 at
period 64 for each corresponding family. This is not a regression in
algebraic state: M114 removes the full aggregate. It is the consequence of
counting the real accumulator simultaneously with the transient full product
and converted term, which M113's aggregate conversion-pair metric did not
express.

## Restoration and controls

The independently reconstructed paths preserve:

- exact public boundaries against raw Horner and the prior recurrence;
- exact inverse rematerialization from public topology;
- exact payload, pi-ledger, and unit-ledger restoration;
- the original carrier backing;
- unrelated period-1 reuse on the restored carrier;
- wrong, missing, and reordered inverse controls inherited and reexecuted
  through the M113 carrier controls;
- no retained inverse history or baseline reload.

The restoration class is `EXACT_ALGEBRAIC_RESTORATION`. Generation and lease
values remain observed, unenforced bookkeeping. This package does not claim
CATVM custody or machine-enforced no-smuggle behavior.

## Strict ceiling

This package establishes only:

```text
LINUX_X86_64_PYTHON
TWO_PUBLIC_F17_PERIOD17_FAMILIES
PERIODS_1_AND_64
FULL_CYCLOTOMIC_PER_ELEMENT_HERMITIAN_PRODUCTS
IMMEDIATE_EXACT_DEGREE8_REAL_CONVERSION_AND_ACCUMULATION
NO_SUMMED_DEGREE16_NORM
ONE_FULL_CERTIFIED_ACTION
ONE_RESIDENT_HORNER_CARRIER
EXACT_BOUNDARY_RESOURCE_RESTORATION_AND_PERIOD1_REUSE_PARITY
SOFTWARE_ONLY
```

It does not establish a full real-subfield carrier, elimination of
per-element full products, elimination of the final full action, a distinct
phase resource, computational advantage, Small Wall crossing, CATVM custody,
physical waveform execution, replacement of physical bits with pi, catalytic
inference, or unbounded computation.

## Next obstruction

The summed full norm was removable without semantic loss, but each Hermitian
term, the resident carrier, and the certified action remain in the full
cyclotomic representation. The identical compact classical stream remains
available. A successor must either derive the Hermitian term natively in the
real field without materializing the full product, move a larger lawful
carrier operation into a genuinely phase-owned representation, or test a
nonclassical coupling that survives the strongest compact classical
comparison.
