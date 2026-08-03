# Independent Review: Period-17 Pi-Content Recurrence

## Decision

```text
classification
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification
    SEPARATE_REFERENCE_PARITY

restoration
    EXACT_ALGEBRAIC_RESTORATION
```

This decision is bound to:

```text
production source
    870009ad4e668f81db3c30b8bf15f693232a664e93bcb7098286e58a975ab4f9

oracle source
    c0a7617edff7faea96ed43520ab7b8de3ede7a4ad729725ead0d635121f6e634

production full result
    24700376f9f96534a1d20a221f1709091b12a2f811be6352d7d94684f6cfe1f4

oracle result
    a6064ed9223e600863590a819117eaaed0f36ebaae9e3d314a3837d6acbefacc
```

## Verified Scope

The separate exact-integer oracle imports no production implementation. It
recompiles both public period operators, checks both complete annihilator
identities, and advances recurrence coefficients by sequential
multiplication by `x mod q`. Production instead uses binary polynomial
powering. The oracle has its own residual-plus-pi-exponent arithmetic.

The oracle exactly reproduces every coefficient decomposition, boundary,
boundary pi valuation, raw recurrence baseline width and payload, and all
nine resident-carrier resource fields for both public families at periods
`1`, `4`, `16`, `64`, and `256`. Both stderr streams are empty.

Factoring common powers of `pi = 1 - zeta_17` leaves a small exponent ledger,
but increases the exact residual height. All ten declared cases use more
resident signed-integer payload than the identical raw recurrence. The same
conclusion holds after expressing the normalized residuals in the tested
integral pi basis. At period 256:

```text
family   raw payload   normalized zeta payload   normalized pi payload
PRIMARY    5,850,470                 27,417,447              27,460,355
REUSE      6,112,042                 28,372,693              28,415,308
```

The valid inverse subtracts the output, recurrence coefficients, basis
messages in reverse order, and seed from the actual carrier. It restores all
payload and pi ledgers to zero on the original backing, then reuses that
backing for the unrelated public family. A nonzero coefficient-ledger
mutation changes the boundary.

## Claim Ceiling

```text
LINUX_X86_64_PYTHON_EXACT_TWO_PUBLIC_F17_PERIOD17_CUBIC_PATH_FAMILIES_Q_ZETA17_PI_CONTENT_LEDGER_NORMALIZED_RECURRENCE_PERIODS1_4_16_64_256_ZETA_AND_PI_INTEGRAL_BASIS_RESIDENT_PAYLOAD_DIAGNOSTIC_EXACT_SUBTRACTIVE_PAYLOAD_AND_LEDGER_RESTORATION_SOFTWARE_ONLY
```

The accepted comparison counts resident signed-integer carrier payload and
declared ledgers. It does not establish a process-memory bound. The raw
baseline is retained and is the identical strongest compact recurrence.

## Required Limitations

- The two coordinate checks do not establish an intrinsic lower bound over
  every integral basis or every cyclotomic-unit balancing strategy.
- The normalized pi-basis payload is compared with the available raw
  recurrence in zeta coordinates; this is enough to reject this candidate as
  a smaller machine, not to establish basis-optimality.
- Named transient counters are partial. Python object overhead, allocator
  peak, internal multiplication work, and whole-process storage are not
  bounded.
- The finite periods do not prove an asymptotic residual-height lower bound.
- Generation and lease metadata advance, so full carrier-object equality and
  bounded repeated-use metadata are not claimed.
- Wrong and reordered inverses are detected, but rejected attempts do not
  provide failure-atomic rollback.
- No CATVM custody, distinct phase resource, computational advantage, Small
  Wall crossing, physical waveform execution, physical replacement of bits,
  or unbounded computation is established.

## Next Obstruction

Compulsory pi content has a compact ledger, but dividing it out amplifies the
residual in both tested integral coordinate systems. The next experiment
should test exact multi-embedding cyclotomic-unit balancing after pi
factorization. Repeating the same periods or families does not address this
obstruction.
