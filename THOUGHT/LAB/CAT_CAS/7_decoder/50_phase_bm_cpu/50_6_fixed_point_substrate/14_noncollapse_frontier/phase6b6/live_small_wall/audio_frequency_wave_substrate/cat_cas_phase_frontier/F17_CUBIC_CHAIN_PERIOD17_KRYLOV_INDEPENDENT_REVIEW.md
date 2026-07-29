# F17 period-17 Krylov diagnostic independent review

Scientific source commit:
`a1d96fcc8d705c059a595f8899cf208be1cd8b74`

## Decision

```text
classification:
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification_level:
    SEPARATE_REFERENCE_PARITY

restoration_class:
    EXACT_ALGEBRAIC_RESTORATION
```

No algebraic source defect was found in the sealed scope. The production
diagnostic compiles two fixed public period-17 F17 cubic-chain families and
applies the period block as a black-box map on a 272-coordinate fixed
cyclotomic message. The separate oracle does not import or call the
production compiler, transfer, gauge selector, projection, or inverse.

## Preserved strict subclaims

The PRIMARY seed Krylov image has exact dimension 241 over each of `F41` and
`F73`. The REUSE image has exact dimension 256 over each field. These modular
ranks imply rational-rank lower bounds 241 and 256, respectively. They do
not establish the exact rational Krylov dimensions, exact minimal
polynomials, or an exact `Z[zeta17]` quotient.

The independent reverse-pivot oracle recompiles the public descriptors,
reconstructs fixed-basis integer transfer, extracts maximal power-of-17
content, reconstructs the adaptive omitted-root gauge, and matches every
exact projective tuple at periods 1, 2, 4, and 8.

PRIMARY adaptive payload is:

```text
3,462  6,263  13,205  26,164 bits
```

REUSE adaptive payload is:

```text
3,629  6,567  13,949  27,689 bits
```

Maximum residual quotient-coefficient signed width grows from 15 to 99 bits
for PRIMARY and 16 to 104 bits for REUSE. Every residual coefficient gcd is
one after exact power-of-17 content extraction.

## Restoration and reuse

At period 4, the sealed adaptive dependency performs subtractive inverse
restoration on the actual resident messages. The original backing identity
is retained, canonical restored state contains only zero messages,
generation and lease reach two, and unrelated REUSE execution matches a
fresh carrier. No inverse history or baseline reload is retained.

This supports `EXACT_ALGEBRAIC_RESTORATION` only for the declared period-4
transaction. It is not CATVM custody evidence.

## Resource scope and baseline

Production accounting explicitly includes logical carrier messages, pivot
and scale metadata, public descriptors, metric-verification buffers, the
fresh verification carrier, and aggregate work for all three restoration
transactions. Separate-oracle accounting includes the 69,632-field-cell
Krylov basis, concurrent modular work for a 70,720-field-cell explicit peak,
and the 816-integer-cell final fixed/adaptive/semantic coexistence peak.
Gauge candidate and encoding buffers are recorded.

Python object overhead, allocator and native-library peaks, bit-operation
cost, and whole-process state remain unbounded. These figures are
component-level logical accounting, not total process memory.

The identical exact 272-coordinate period-block recurrence is the matched
compact classical method and inherits the same projective reductions. The
strongest family-specific compact method is not established.

## Rejected interpretations

The evidence does not establish an exact rational Krylov order below 272, an
exact `Z[zeta17]` Krylov quotient, fixed integer width, constant reversible
storage, transferable arbitrary-family behavior, arbitrary graph topology,
CATVM custody, a distinct phase resource, computational advantage, Small
Wall crossing, physical waveform execution, replacement of physical bits
with pi, catalytic inference, or unbounded computation.
