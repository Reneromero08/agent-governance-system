# Independent reexecution review: SU(2)_8 fusion-path braid diagnostic

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`  
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Exact scope

The production mechanism is the radical-free source-weight path model of the
`SU(2)_8` Temperley-Lieb representation over exact `Q(zeta_40)`. It retains
one coefficient for every bounded `A_9` vacuum fusion path. Adjacent braid
generators consume overlapping internal fusion channels without projecting
them. The only projected value is the final coefficient of the public vacuum
pairing path, interpreted as the bounded vacuum plat-closure amplitude.

The exact ceiling is:

```text
FORMAL_SU2_LEVEL8_TEMPERLEY_LIEB_A9_VACUUM_FUSION_PATH_REPRESENTATION_QZETA40_EVEN_STRANDS4_6_8_10_12_14_16_EIGHT_SWEEP_FAMILY_PRIMARY16_FIVE_SWEEP_FAMILY_REUSE16_VACUUM_PLAT_BOUNDARY_DIRECT_PROCESS_ONLY
```

## Independent construction

The oracle imports no CAT_CAS module. It independently implements
`Q(zeta_40)`, enumerates bounded vacuum paths explicitly, and evolves the
target-weight Temperley-Lieb gauge. Production instead retains only the
public `A_9` completion-count table and streams path rank/unrank in the
source-weight gauge. The oracle maps its final vector back through the
independently reconstructed diagonal path-weight similarity before hashing.

The two implementations agree exactly for every executed strand count
`4,6,8,10,12,14,16` on dimensions, nonzero support, coefficient payload,
maximum coordinate widths, full-state commitments, final-boundary
commitments, boundary payload, public topology payload and topology
commitments. They also agree on the primary and unrelated reuse states and
boundaries.

The oracle's explicit path list and index map are a verification baseline,
not part of the accepted streamed production path.

## Algebra and custody controls

Exact controls verify the Yang-Baxter relation, far-generator commutation,
adjacent-generator noncommutation, and one-generator inverse. The production
also rejects wrong owner, wrong operation type, premature projection,
reordered inverse, and null carrier; detects missing inverse; and changes
under a semantic generator perturbation. No snapshot command exists.

The primary and unrelated reuse transactions reverse the public braid word
on the same coefficient-list backing. Both restore the exact source and
canonical port metadata; restored reuse agrees with fresh reuse at generation
two. No baseline reload or retained inverse history is used.

## Resource result

The fixed local alphabet does not yield fixed global state. Executed carrier
cells grow `2,5,14,42,132,429,1430` from 4 through 16 strands, and every cell
is nonzero after the declared eight sweeps. At 18 strands, the first
Jones-Wenzl truncation removes only one untruncated Catalan path, leaving
4,861 cells.

The 16-strand forward carrier carries 256,269 exact payload bits. Its public
topology uses 153 integer cells and 516 bits; category constants use 20 field
cells and 754 bits. Production retains no path list, local action plan,
operation list or inverse history, but the full 1,430-cell coefficient vector
is material. The primary lifecycle streams two 17-label paths at a time (34
label cells), eight logical local field scratch cells, 343,200 path unrank
calls, 80,081 path rank calls, and 7,877,584 topology-count reads.

The strongest compact classical baseline is the identical exact
Temperley-Lieb fusion-path recurrence with the same `A_9` rank/unrank law.

## Claim boundary

This package establishes exact bounded braid action, unresolved internal
fusion-channel residency, final-only boundary projection, restoration, and
reuse. It does not establish a compact global fusion-tree carrier, bounded
coefficient width, CATVM custody, a distinct phase resource, computational
advantage, Small Wall crossing, physical waveform execution, replacement of
physical bits with pi, catalytic inference, or unbounded computation.
