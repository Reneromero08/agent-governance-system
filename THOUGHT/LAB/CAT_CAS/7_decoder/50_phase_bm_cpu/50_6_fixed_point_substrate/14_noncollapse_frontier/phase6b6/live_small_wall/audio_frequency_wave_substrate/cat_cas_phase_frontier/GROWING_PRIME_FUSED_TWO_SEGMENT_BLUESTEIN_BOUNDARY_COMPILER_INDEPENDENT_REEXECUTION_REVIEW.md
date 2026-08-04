# Fused two-segment Bluestein compiler independent reexecution review

## Decision

```text
INDEPENDENTLY_VERIFIED_STRICT_SCOPE
INDEPENDENT_ORACLE_REEXECUTION
EXACT_ALGEBRAIC_RESTORATION
```

The bounded 14-field result is accepted at its declared direct-process exact
residue-software ceiling. The source commit containing this report is the
scientific source head; its exact Git SHA is recorded by the subsequent
provenance and claim-authority reconciliation.

## Independent reconstruction

The no-import oracle does not import the production compiler or its
predecessor. It independently reconstructs:

- primitive roots and the declared auxiliary fields;
- the multiplicative-character log map and direct Gauss values;
- an out-of-place recursive radix-two NTT rather than the production iterative
  in-place NTT;
- the Bluestein chirp and public kernel spectrum;
- the fused two-segment multiply, boundary projection, exact reverse, and
  unrelated restored-carrier reuse;
- the original seven-dimensional relation sum for q=5 and q=7.

All 14 production and oracle cases agree on field, convolution width, logical
carrier capacity, kernel-spectrum commitment, primary boundary, unrelated
reuse boundary, NTT calls, and butterfly count.

## Observed bounded law

For `h=q-1` and the least power of two `M >= 2h-1`, the production transaction
uses one scalar plus two `M`-cell exact field segments. The declared cases use
17 through 257 logical carrier cells, versus M183's 28 through 436 cells. Each
compiler executes six NTTs, `3M log2(M)` butterflies, `2M` in-place spectral
multiplications, `M` public-spectrum inversions, and an O(q) final projection.
The scalar is the only resident cell after forward cleanup. The actual reverse
restores the same list backing exactly, and an unrelated program reuses it with
the same boundary and resource signature as fresh execution.

This is an abstract field-operation and logical-carrier law. It is not a peak
Python-process-memory claim and does not count allocator/native-library
internals or modular-exponentiation bit complexity. The independent recursive
oracle deliberately uses out-of-place buffers and derives the production
logical schedule; it does not claim the oracle's own peak memory is `1+2M`.

Package verification separately materializes one `q-1` direct-Gauss table and
performs `(q-1)^2` generation terms plus two O(q) boundary projections. That
verification is executed and released before carrier allocation, is reported
per case, and is not hidden inside the accepted fused transaction.

## Controls and corrections

Every declared case passes missing inverse, wrong-program inverse,
frequency-zero spectral omission, forced singular kernel spectrum, and null
carrier controls. Frequency zero is the only omitted frequency claimed; other
frequencies are not asserted to affect every boundary. Singular-spectrum
rejection is not represented as atomic rollback. No snapshot or lease metadata
is used.

Production commitments stream cell encodings and do not create a full spectrum
slice or joined O(M) digest string. The qualifier reexecutes both tracked
programs in the assigned worktree using process substitution and
`PYTHONDONTWRITEBYTECODE=1`; it creates no temp directory, worktree, clone,
bytecode cache, or RAM-backed artifact.

## Claim ceiling

The result establishes a bounded exact logical-carrier reduction for the 14
declared prime/auxiliary-field pairs. The strongest matched classical baseline
is the identical fused recurrence. It does not establish CATVM custody,
sublinear allocated state, fixed exact bit width, a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

## Pre-seal source hashes

```text
4fdf8fb8f1f77d353554dc982dd9f6311adc5f7116acd647a6ce50c8575f7098  growing_prime_fused_two_segment_bluestein_boundary_compiler.py
2a803ffe3a8e477286da453854a02511bd5cfee22f83613dba2ca66a6c7e569a  growing_prime_fused_two_segment_bluestein_boundary_compiler_independent_oracle.py
208a4731c0f0db0218970b6e030f858e739fcafa3502963ac5a6ac951d959691  GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_RESULTS.json
8671f50cc89e248018908d42022278e96a1d296bf97ee7fd5ff8d40adb4704e7  GROWING_PRIME_FUSED_TWO_SEGMENT_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_ORACLE.json
```
