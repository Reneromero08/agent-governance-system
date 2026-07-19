# P0 resonance and load-law repair contract

**Status:** `P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED`
**Parent audit:** `P0_POST_QUALIFICATION_AUDIT.md`  
**Preserved result:** `P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED`  
**Restored decision:** `P0_BUILD_READINESS_PACKET_FROZEN`
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`

This is an offline architecture and synthetic-reference result. It authorizes no vendor contact, cart action, purchase, fabrication, assembly, wiring, probing, power, instrument command, playback, recording, acquisition, physical calibration, target contact, or physical claim.

## Selected route

The repair selects a calibration-derived carrier frequency. It does not add a load-matching network and does not freeze a physical operating frequency before an as-built assembly exists.

The prospective one-pass calibration law is frozen in `P0_RESONANCE_LOAD_SANITY_MODEL.json`:

```text
coarse search: 32760..32820 Hz at 1 Hz
fine search: +/-1 Hz at 0.025 Hz
calibration excitation: 0.100 Vpp
fit: bounded complex BVD magnitude and phase
accepted f_carrier_hz: 32768..32820 Hz
maximum f_carrier_u95_hz: 0.050 Hz
accepted Q: 4000..60000
selection: maximum fitted motional response in the accepted interval
retry law: one pass; reject on failure
```

The prospective BVD binary-corner sanity model bounds expected resonance, Q, decay, and terminal amplitude. It is not a continuous enclosure, physical calibration, or permission to operate anything.

## Strict frequency custody

Every future record must bind:

```text
f_carrier_hz
f_carrier_u95_hz
f_witness_hz
f_witness_hz == 2 * f_carrier_hz
calibration artifact SHA-256
calibration raw-data SHA-256
calibration analyzer SHA-256
source and instrument queryback SHA-256
```

The calibration artifact must also bind the exact assembly, carrier population, Q or decay, completion time, and `primary_observed=false`. Calibration must complete before the assignment commitment. The source-preparation receipt must begin at least 3.000 s before primary acquisition.

The analyzer propagates the bound tuple through the source queryback, drive/reference fit, C2 transfer, I/Q projection, reconstruction, cycle counting, off-resonance controls, and matched-arm comparison. Nominal-frequency constants are restricted to synthetic fixture generation; an AST regression test rejects their use by operational analysis functions.

The calibration receipt alone is not sufficient. The bound raw file must be a strict coarse/fine complex sweep whose selected frequency, Q, decay, and uncertainty are recomputed by the offline analyzer. Each role also carries a raw off-resonance response at exactly `f_carrier + max(20 Hz, 20 calibrated linewidths)`. Self-consistently rehashed invalid bytes, a too-close probe, or a response ratio above 0.020 are hard rejections.

## Corrected claim laws

The 131072-point result is called a `complete binary-corner sweep`. It is not called a complete continuous uncertainty envelope.

The analyzer retains the directly observable differential clipping gate. It no longer derives or claims true input common mode from differential bytes. Any future powered operating envelope must separately define and validate common-mode observability before authority.

The earlier four research-correction reviews are described as four role-separated root-bound review declarations. Their structured booleans do not establish externally reproducible independence. This repair requires one focused final read-only review of the exact repaired candidate root.

## Required deterministic controls

The package must pass:

```text
positive bound-frequency reproduction
unbound frequency rejection
wrong calibration hash rejection
calibration-after-assignment rejection
source-queryback frequency mismatch rejection
witness 2:1 relation mismatch rejection
hard-coded operational-frequency regression scan
existing signal-path model, analyzer, controls, ordering proof, and validator
```

## Adjudication

```text
P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED
P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED
P0_BUILD_READINESS_PACKET_FROZEN
NON_EXECUTING_P0_BUILD_READINESS_ONLY
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

Nothing here grants procurement, build, or execution authority.
