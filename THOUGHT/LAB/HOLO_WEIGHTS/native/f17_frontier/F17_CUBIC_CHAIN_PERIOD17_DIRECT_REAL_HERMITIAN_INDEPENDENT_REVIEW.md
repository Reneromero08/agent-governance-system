# M115 direct real-Hermitian generator review

## Decision

```text
classification       INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification level   SEPARATE_REFERENCE_PARITY
restoration class    EXACT_ALGEBRAIC_RESTORATION
```

Scientific source parent:

`2ed6ae2a906a6873b58a77cabaf886377a8234bb`

Reviewed production source SHA-256:

`6d60cd4eb7853c208d5b6a51d3ee710a85c0144719997efd329d583b57377e28`

Reviewed separate oracle source SHA-256:

`e1165eef8a9936103c8cc0238534e998be171f1af7c148233498b774ba204e00`

Reviewed production output SHA-256:

`7ff624c3471b244ad99eff0f2c5e444ca8c7da5af973a3f995ca8c8a56d35fea`

Reviewed oracle output SHA-256:

`2e20b0c1cafe406adc482fe2463291cd8ac77579237188fc15e5e90dcfe991a4`

Both implementations were executed from the repository Linux virtual
environment. No hosted workflow reexecution is established.

## Exact mechanism

For a canonical full-field element

```text
x = sum(a_j zeta^j, j=0..15),  a_16 = 0,
P_k = sum_j a_j a_(j+k) with indices modulo 17,
```

the production implementation directly returns the eight integral
real-basis coordinates

```text
(P_0-P_8, P_1-P_8, ..., P_7-P_8).
```

This follows from `P_k=P_(17-k)` and
`1+s_1+...+s_8=0`. Production retains `P_8`, streams `P_0` through `P_7`,
and releases each integer product immediately. Per Hermitian term it performs
exactly 136 integer products, 127 accumulation additions, and eight final
subtractions. It does not materialize a conjugate tuple, a degree-16 product,
a degree-16 reduction buffer, or a compiled answer-bearing table.

The 136 production controls comprise all 16 basis inputs and all 120 pair
sums. They span every monomial of the quadratic map and match M114's full
product followed by exact full-to-real conversion. Those full products are
verification baselines outside the accepted path and are reported separately.

## Independent reconstruction

The separate oracle does not import the production M115 module and does not
use cyclic autocorrelation for its semantic result. It works in

```text
Q[y] / (
    y^8 + y^7 - 7 y^6 - 6 y^5
    + 15 y^4 + 10 y^3 - 10 y^2 - 4 y + 1
),
```

writes each public basis power as `zeta^n=A_n(y)+zeta*B_n(y)`, and computes

```text
x*conjugate(x) = A^2 + y*A*B + B^2.
```

That degree-eight quadratic-extension result supplies the oracle norm. A
second independent schedule reexecution reproduces production resource
events and checks semantic parity on every encountered carrier element. The
oracle made 10,811 such per-element schedule checks in its full case and
control campaign. Its 136 signed basis/pair controls also match the prior
full product, and all 136 homogeneity checks pass.

The oracle independently reexecutes the two public program families,
periods 1 and 64, all four forward and inverse resource tuples, boundaries,
restoration, mutations, and period-1 cross-family reuse. Every tuple matches.

## Exact shapes and work

The accepted norm-call shapes remain those established by M114:

```text
period family   calls  17-cell  singleton  terms  integer products
1      PRIMARY      5        4          1     69             9,384
1      REUSE        5        4          1     69             9,384
64     PRIMARY     93       63         30   1101           149,736
64     REUSE       98       66         32   1154           156,944
```

All accepted counters for full cyclotomic norm products, materialized
conjugates, full-to-real conversions, and degree-16 norm scratch are zero.
The full carrier and the one certified unit action remain degree 16.

## Resource result

```text
period family   M114 named total  M115 named total  reduction  raw Horner  M115 - raw
1      PRIMARY             96,088            93,790      2,298      10,005      83,785
1      REUSE              103,778           101,475      2,303      10,097      91,378
64     PRIMARY          3,466,223         3,324,441    141,782   2,790,766     533,675
64     REUSE            3,840,024         3,695,435    144,589   2,901,994     793,441
```

The direct norm event maxima are respectively 7,972, 8,031, 324,088, and
330,504 payload bits. They reduce the corresponding M114 norm maxima by the
same amounts shown in the table. This is a real reduction in named algebraic
state for the declared schedule, not a whole-process memory measurement.

The reported component total remains a conservative sum of maxima that need
not be simultaneous. Python object headers, list/reference storage, allocator
state, native-library memory, and bigint implementation scratch are excluded.
The resident carrier payload is counted once; production accesses its element
tuples by reference. The public loop schedule has zero retained table bits.

The strongest matched classical implementation is the identical direct
degree-eight bilinear recurrence with the same 136 products, 127 additions,
eight subtractions, term order, accumulator, unit search, public programs,
and boundaries. M114's full-product stream and raw Horner are retained as
additional baselines, not treated as the strongest comparison. M115 remains
above raw Horner in all four cases and establishes no advantage.

## Restoration and controls

The independent reexecution preserves:

- exact public boundaries against raw Horner and the prior recurrence;
- exact inverse rematerialization from public topology;
- exact payload, pi-ledger, and unit-ledger restoration;
- zero pending operations and zero retained inverse history;
- the original carrier backing without snapshot reload;
- correct observed generation and lease values;
- unrelated period-1 reuse on the restored carrier;
- inherited wrong, missing, reordered, and mutation controls.

The restoration class is `EXACT_ALGEBRAIC_RESTORATION`. Generation and lease
remain observed metadata rather than enforced CATVM custody.

## Strict ceiling

This package establishes only:

```text
LINUX_X86_64_PYTHON
TWO_PUBLIC_F17_PERIOD17_FAMILIES
PERIODS_1_AND_64
DIRECT_EXACT_8_COORDINATE_HERMITIAN_TERM_GENERATION
NO_ACCEPTED_DEGREE16_HERMITIAN_NORM_PRODUCT
ONE_FULL_CYCLOTOMIC_HORNER_CARRIER
ONE_FULL_CERTIFIED_ACTION
EXACT_BOUNDARY_RESOURCE_RESTORATION_AND_PERIOD1_REUSE_PARITY
SOFTWARE_ONLY
```

It does not establish a full real-subfield carrier, a real-subfield certified
action, a distinct phase resource, computational advantage, Small Wall
crossing, CATVM custody, physical waveform execution, replacement of physical
bits with pi, catalytic inference, or unbounded computation.

## Next obstruction

The unit-search norm no longer needs any degree-16 Hermitian product. The
resident phase carrier and the certified unit action still require the full
cyclotomic representation, and compact classical software executes the same
direct bilinear map. The next experiment must either introduce a phase-native
nonclassical trace coupling that survives the strongest compact classical
comparison or prove a lawful full-carrier quotient with exact boundary lift;
enlarging this fixture would not remove the obstruction.
