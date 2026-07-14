# Final Gain-Covariant Confirmation Implementation Audit

This package freezes the offline, live-capable implementation for
`orbitstate_gain_covariant_confirmation_v1_0`. It does not authorize or consume a
live authority and does not contact the target.

## Boundary

- Starting commit: `4ff588cea11343bf38d4c96c1281d34cbf1961ed`
- Science package ID: `orbitstate_gain_covariant_confirmation_v1_0`
- Transaction/run ID: `orbitstate_gain_covariant_confirmation_v1_0`
- Public seed: `orbitstate-gain-covariant-final-confirmation-public-seed-5b7b1338-a7daa611`
- Frozen contract SHA-256: `31af6869bdf4e25634b1e408830015af2b2c4f20202b2df28492b2e1a9a90860`
- Expected local evidence root: `D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\runs\orbitstate_gain_covariant_confirmation_v1_0`
- Expected remote root: `/root/catcas_live_small_wall/orbitstate_gain_covariant_confirmation_v1_0`

## Frozen Artifacts

- Public schedule JSON SHA-256: `98142e30d26a4322315c897e2c6053a5d16408ca78991f77a6503d8a8a1eabcf`
- Public schedule TSV SHA-256: `a2588ab1621a0b518b14205613b9b075a23ea56ed51fb20fc1c808872926efb6`
- Private map file SHA-256: `d1c34fb5cf4b032a0a83d704188b2b24ec3d15a508247226a2c3501782f3c699`
- Private map canonical SHA-256: `bd62da5ad12df997fa6ffcd8e13f4d6d0a244f5fdf73b97e38d71b2403bb6842`

The current implementation manifest binds the generated source bundle, runtime
binary, runtime self-test, disassembly, public/target/controller self-tests,
transport simulation, deployment layout self-test, feature-boundary self-test,
and Sol implementation audit hashes. Those generated hash values are intentionally
read from `GAIN_COVARIANT_IMPLEMENTATION_MANIFEST.json` rather than duplicated in
this prose file.

## Implemented Law

The adjudicator uses only `post_projection` and `equal_orbit_odd_zero` to compute
per-replicate control gain:

```text
g_control = mean(g_post_projection, g_equal_orbit)
```

It rejects any control-gain input containing `pre_projection_d`,
`pre_projection_fold`, or `source_polarity_inversion_d`. Both replicates must pass
independently; aggregate geometry is diagnostic only.

The gain-normalized target, fold, and polarity checks use the frozen OrbitState
vectors:

```text
Q_d = 1536 * exp(+i*2*pi*23/256)
Q_fold = 1536 * exp(-i*2*pi*23/256)
Q_polarity = -1536 * exp(+i*2*pi*23/256)
```

Strong and near-zero phase controls are partitioned by sealed source receipts using
`abs(q_theta) >= 256`. The near-zero absolute bound is fixed at `152` and is applied
to raw map legs, logical pair averages, physical reversal averages, and decoded
first-harmonic nulls as separate statistics.

## Offline Validation

Validated offline modes include syntax checks, strict C compilation, runtime
self-test, schedule validation, PMU-preflight static proof, public blindness,
private-map sealing, feature-boundary proof, gain-estimator no-smuggle mocks,
gain-normalized geometry mocks, strong/near-zero partition mocks, target self-test,
controller self-test, prepare-only, validate-only, fake success transport, fake
target failure transport, fake no-output failure, copy-back size/SHA corruption,
cleanup failure, historical-root mutation rejection, deployment-layout test, source
bundle reconstruction, JSON parsing, and disassembly generation.

## Future Authority

The future live invocation must use exactly these environment variables:

```powershell
$env:ORBITSTATE_GAIN_COVARIANT_V1_COMMIT_BINDING = "<final-commit>"
$env:ORBITSTATE_GAIN_COVARIANT_V1_MANIFEST_SHA256 = "<manifest-file-sha>"
$env:ORBITSTATE_GAIN_COVARIANT_V1_LIVE_AUTHORITY = "orbitstate_gain_covariant_confirmation_v1_0"
.\.venv\Scripts\python.exe "THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\orbit_coupling\orbitstate_gain_covariant_confirmation_v1\run_gain_covariant_confirmation_v1.py" --execute-authorized
```

No live command is authorized by this audit.
