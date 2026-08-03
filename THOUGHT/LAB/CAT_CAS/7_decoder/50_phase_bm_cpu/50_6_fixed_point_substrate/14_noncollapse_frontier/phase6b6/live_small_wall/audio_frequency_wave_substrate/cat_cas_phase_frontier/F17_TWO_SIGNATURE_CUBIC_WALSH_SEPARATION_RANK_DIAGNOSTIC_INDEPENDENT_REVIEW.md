# M138 independent review: two-signature separation-rank diagnostic

Scientific source commit:
`d556c65e6f9410713700db474d4a7d90af44d244`

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class:

- resident `C17 x C17` group-algebra carrier:
  `EXACT_ALGEBRAIC_RESTORATION`
- compiler, projection, rank, commitment, classical-baseline, and oracle
  buffers: `NO_RESTORATION_CLAIM`

## Decision

PASS after repair. The separate oracle reconstructs all 62 exact and
dual-field transactions without importing the M138 production module. It
matches public program fingerprints, coefficient-surface ranks, final scalar
boundaries, independent 289-residue boundaries, hidden state commitments,
resident exact payload tuples, and exact inverse seeds. Six bounded direct
Boolean-assignment checks over `F103` and `F137` agree with both recurrences.

The declared `PRIMARY`, `REUSE`, and `ALTERNATE` families reach exact matrix
separation rank 17 at their tested ceilings. This rejects only uniform matrix
factor charts capped below rank 17 for those families. It does not establish
a universal 578-cell minimum or a general rank-r lower bound.

## Repairs required by adversarial review

The pre-seal review rejected four defects, all repaired before the scientific
source commit:

1. Both 289-cell Walsh output surfaces are simultaneously live. The accepted
   path now reports 578 scratch cells, their combined exact payload, and 1,156
   resident-plus-scratch logical cells.
2. Projection storage is narrowed to 20 persistently named field cells (17
   moments plus observation, boundary, and component), explicitly excluding
   separately reported rank work and expression temporaries.
3. The classical comparison is an explicit resource frontier: an executed
   retain-both full diagnostic, an executed eight-dynamic-field plus 17-public-
   integer streamed final-scalar recurrence, and an unexecuted conservative
   320-field-cell rematerialized full-diagnostic construction. No streamed
   payload tuple is claimed.
4. A real wrong-order inverse is applied to the post-projection resident
   carrier and fails exact seed restoration. Forward descriptor grammar
   rejection remains a separate control.

## Resource and claim ceiling

The canonical carrier contains 578 exact field cells. Exact resident payload
for the eight `Q(zeta17)` cases is
`[18504,18528,18744,19369,21392,25842,40028,110637]` bits. The measured
resident-plus-update-scratch payload is
`[37004,37040,37364,38323,41501,49067,76295,216450]` bits.

Python containers, allocator state, native-library storage, bigint internal
workspace, hashlib internal state, bit-operation complexity, and whole-process
peak memory are excluded. The streamed/rematerialized classical points trade
memory for work, so no single classical point is declared dominant.

This result does not establish CATVM custody, a distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded catalytic computation.

## Replay

The qualifier returned:

```text
QUALIFIED_F17_TWO_SIGNATURE_CUBIC_WALSH_SEPARATION_RANK_DIAGNOSTIC_STRICT_SCOPE
```

Byte-identical seal hashes:

- production:
  `94c6703d984dd0cd7228f863ea351c8851d99334a9e1693478a194425a63a3f4`
- oracle:
  `ab05adb3540d1a5117813219596ac0176c2d5a8b8d44cb4cdf9ae9bc010967b3`
