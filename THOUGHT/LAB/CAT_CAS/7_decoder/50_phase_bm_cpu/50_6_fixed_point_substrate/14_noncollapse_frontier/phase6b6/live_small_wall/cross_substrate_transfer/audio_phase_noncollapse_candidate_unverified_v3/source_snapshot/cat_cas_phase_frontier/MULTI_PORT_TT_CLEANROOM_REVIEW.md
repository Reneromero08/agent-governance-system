# Multi-port tensor-train focused review

Classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration class:

`INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT`

## Scope

The reviewed package is limited to Linux x86-64 Python/NumPy/SciPy
`complex128`, the 285-cell exchange-symmetric rotation-invariant
four-rotor necklace sector, binary port counts 2 through 6, the two
deterministically generated local-chain/ring-joint program families, and the
predeclared relative SVD cutoff `2e-12`.

The independent oracle does not import the tensor-train backend, program
compiler, or projection implementation. It reconstructs the public programs,
executes the direct `285 * 2**p` recurrence as an explicitly counted
verification-only baseline, and compares forward boundaries, inverse
restoration, reuse boundaries, and final numerical matricization ranks.

## Review findings and repairs

The first review found four claim/accounting defects:

1. Elementwise peak TT bond ranks alone did not certify simultaneous canonical
   matricization ranks. The repaired oracle now measures the final canonical
   dense state at every cut. All cuts have full tolerance-defined numerical
   rank, and the smallest retained singular value is more than `1.3e10` times
   the predeclared cutoff in the largest tested case.
2. Reuse initially identified only the outer carrier object. The evidence now
   calls this logical-carrier-container reuse and explicitly records that core
   array backing is not preserved.
3. Resource accounting omitted concurrent restoration baselines and fresh
   parity carriers. Those resident carrier cells are now counted separately;
   temporary TT/SVD/generator counts are explicitly component maxima rather
   than a whole-process RSS peak.
4. The package initially called the identical TT recurrence the strongest
   compact classical baseline. That wording was removed. The package records
   an identical TT reference and a smaller direct dense reference for these
   cases, while leaving the strongest compact classical method unestablished.

The repaired qualifier reruns the production path twice, requires byte
identity, runs the separate dense oracle, compares both durable summaries, and
rejects the stale strongest-baseline interpretation.

## Accepted strict result

For the tested cases, final canonical ranks are:

```text
p=2: [4, 2]
p=3: [8, 4, 2]
p=4: [16, 8, 4, 2]
p=5: [32, 16, 8, 4, 2]
p=6: [64, 32, 16, 8, 4, 2]
```

The first rank equals `2**p`; all later cuts are also full numerical rank.
Peak TT core storage is `[1160, 2364, 4900, 10484, 23700]` complex cells,
which exceeds the matched explicit dense state sizes
`[1140, 2280, 4560, 9120, 18240]`.

Correct reverse-topology execution restores the logical carrier within the
declared numerical quotient and permits unrelated logical-container reuse.
Missing, wrong, and reordered inverse controls separate. No baseline reload
or retained inverse history is used.

## Claim ceiling

This establishes a bounded no-compaction obstruction for the tested program
families. It does not establish a universal rank lower bound, fixed-rank
multi-port closure, fixed backing storage, CATVM custody, a strongest compact
classical method, a distinct phase resource, computational advantage, Small
Wall crossing, catalytic inference, physical waveform execution, or
replacement of physical bits with pi.

Qualifier evidence:

`LAW/CONTRACTS/_runs/cat_cas_multi_port_tt/qualifier2/qualification.json`
