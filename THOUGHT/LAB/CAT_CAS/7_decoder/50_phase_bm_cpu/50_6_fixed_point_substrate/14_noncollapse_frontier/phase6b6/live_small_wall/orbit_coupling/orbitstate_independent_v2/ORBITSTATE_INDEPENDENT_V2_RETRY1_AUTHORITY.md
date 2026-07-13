# OrbitState Independent V2 Retry-One Authority

This authority overlay repairs only the deployment-layout defect observed in the
first operational transaction. It does not rewrite the frozen scientific
contract, public schedule, private source map, thresholds, result vocabulary, or
Small Wall promotion law.

## Attempt-Zero Provenance

- science package: `orbitstate_independent_v2_0`
- attempt-zero transaction: `orbitstate_independent_v2_0`
- controller status: `ORBITSTATE_CONTROLLER_TARGET_FAILED_NO_COPYBACK`
- target return code: `1`
- exception: `IndexError: 9`
- controller-result SHA-256: `5a5b82542d1c9268bbfaba051c4528d9859446b26c781c6c0a7a0e86e835f669`
- local attempt-zero root retained: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/runs/orbitstate_independent_v2_0`
- remote attempt-zero root retained: `/root/catcas_live_small_wall/orbitstate_independent_v2_0`
- target output root created: `false`
- hardware execution began: `false`
- PMU preflight executed: `false`
- feature freeze executed: `false`
- unblinding executed: `false`
- scientific classification emitted: `false`

No target measurements, PMU custody, source execution, raw capture, feature
extraction, unblinding, or adjudication occurred in attempt zero.

## Retry-One Transaction

- science package ID: `orbitstate_independent_v2_0`
- transaction run ID: `orbitstate_independent_v2_1`
- live authority value: `orbitstate_independent_v2_1`
- future local root: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/orbit_coupling/runs/orbitstate_independent_v2_1`
- future remote root: `/root/catcas_live_small_wall/orbitstate_independent_v2_1`

The frozen public schedule and private source map remain the science payload for
`orbitstate_independent_v2_0`. They are not rerandomized because attempt zero
produced no measurement and exposed no scientific observation.

## Unchanged Science Hashes

- contract SHA-256: `1f586b4648a516723f5a77cfc381d0e4a8c305dd9446a28289975a3ad3c49507`
- public schedule JSON SHA-256: `709063e1d789971f8ac36d2fc94094738015150baae8e75065909c774a079b7b`
- public schedule TSV SHA-256: `57aaf5635e0ea1bcecd17f6efc0383f6ce08a893751d9203d1c87b0e4c7a7876`
- private source-map canonical SHA-256: `b952f2a161e782dfe41e9dfca21ba4f6bf2902bc69392d9ad52915daa3955464`
- private source-map file SHA-256: `619189f66d32610053c3899616d656d16a21814a45074191f37f95acdbf58325`
- public null-law audit SHA-256: `fc6321f2b898a3f97766e90d68267b67bea79aa32b26ad110cf085deda09c01e`

## Authorization Boundary

Future live authorization must bind:

```powershell
$env:ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING = "<new-final-commit>"
$env:ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256 = "<new-manifest-file-sha>"
$env:ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY = "orbitstate_independent_v2_1"
```

The retry-one controller may create, transfer to, execute under, copy back from,
and clean only `/root/catcas_live_small_wall/orbitstate_independent_v2_1`.
The attempt-zero local and remote roots are provenance only.
