# Family 10h Affinity Fix and Local Paired Discovery Summary

Status: discovery checkpoint only.

This checkpoint emits no scientific claim, does not promote `SMALL_WALL_CROSSED`, and does not rewrite retained evidence.

## Runtime failure

The official local paired differential attempt reached physical preflight, consumed the attempt, and then stalled before the first raw row was flushed. The sealed result remains:

`FAMILY10H_LOCAL_PAIRED_DIFFERENTIAL_CUSTODY_INVALID`

The reproduced failure was a scheduler/affinity deadlock, not a negative science result. After row 0, the receiver parent process stayed pinned to CPU 5. On row 1, the source child inherited CPU 5 affinity and equal `SCHED_FIFO` priority. The parent then busy-waited on `source_ready` without yielding, so the child could not run and pin itself to CPU 4.

## Repair

Patched source:

`family10h_relation_spatial_pair_readout_v1_1_segmented/relation_spatial_runtime.c`

Patched source SHA-256:

`1edd1f03f7e1cfb53089cb90c58a6c7ff5870c370cc75ecddb9c24c8d015ec26`

The patch pins the parent to the source/receiver CPU pair before forking each physical row, yields while waiting for child progress, and keeps a pause in the child release wait loop.

## Discovery evidence

Original runtime one-row diagnostic:

- remote root: `/root/catcas_live_small_wall/discovery_runtime_stall_diag_20260723_130420`
- archive SHA-256: `60aba8de42137cb3309dd5a62366a2217137f149c196a60072a5e38da4498c7a`
- raw records: 1
- pair observations: 256
- source-death receipts: 1
- return code: 0

Original runtime 64-row diagnostic:

- remote root: `/root/catcas_live_small_wall/discovery_runtime_scale_64_20260723_130637`
- return code: 124
- raw records: 0
- pair observations: 250
- source-death receipts: 0
- observed parent CPU: 5
- observed child CPU: 5

Patched runtime 64-row diagnostic:

- remote root: `/root/catcas_live_small_wall/discovery_affinity_fix_scale_64_20260723_130935`
- return code: 0
- raw records: 64
- pair observations: 16,384
- source-death receipts: 64

Patched runtime full primary segment:

- remote root: `/root/catcas_live_small_wall/discovery_affinity_fix_full_primary_20260723_131047`
- archive SHA-256: `a938ae1842b1321a1eebd7a65cbef898eb4599d5fcf490233e44933f9009e8c4`
- archive size: 10,962,585 bytes
- raw records: 2,048
- pair observations: 524,288
- source-death receipts: 2,048
- return code: 0

Patched runtime full ten-variant local paired sweep:

- remote root: `/root/catcas_live_small_wall/discovery_affinity_fix_local_paired_sweep_20260723_131247`
- archive SHA-256: `703818e1db1733b5e2b89717c5e92b09eae00b187db730de598b4a45543de762`
- archive size: 109,930,316 bytes
- variants: 10
- raw records: 20,480
- pair observations: 5,242,880
- source-death receipts: 20,480
- return code: 0
- recomputed analysis SHA-256: `aa20cc557b2cabe7d6b299aa1e9faada6ca5d0b14b7cf8114a59bbf057f1ec82`
- recomputed compact SHA-256: `0f2cbd116d21c97bbfdbed25f44152f56fc9a012de541538c3c51c7a65d44aea`

## Recomputed local paired result

| Round | R_primary | R_sham | D_local | max_abs_generic | generic_to_D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.06642166194704319 | 0.0014210011699119507 | 0.06500066077713124 | 0.006068271746737386 | 0.09335707782331264 |
| 1 | 0.06654767749486046 | -0.006516615135940624 | 0.07306429263080108 | 0.012692203654615958 | 0.1737128109725299 |

Gate-like discovery summary:

- round gates pass: true
- one-factor strata pass: true
- stratum count: 28
- negative D strata: none
- negative primary strata: none
- variant failures: none
- generic controls pass the `0.25 * D_local` envelope in both rounds
- absolute `R_sham` sign remains diagnostic only and is not stable

## Next action

Regenerate and freeze the segmented runtime/package from the affinity repair commit, then continue the local paired differential confirmation path without restoring the absolute sham-sign gate.
