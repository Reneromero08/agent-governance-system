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
fine search: 81 unique half-step-offset points spanning +/-1.0125 Hz at 0.025 Hz
calibration excitation: 0.100 Vpp
canonical capture: 143 frequency blocks, 4096 samples/channel/block, 1 MS/s, signed int16 little-endian CH0/CH1 interleaved
complex extraction: independent sine/cosine/DC fits for CH0 source monitor and CH1 carrier response; H=Z1/Z0 with propagated point uncertainty
fit: H(f)=B+C/(1+i*2Q*(f-f0)/f0), with complex B and C
solver: deterministic bounded NumPy variable projection and pattern refinement; no random initialization
optimizer f0 bounds: 32760..32821 Hz; a solution on either optimizer boundary is rejected
accepted f_carrier_hz: 32768..32820 Hz
maximum f_carrier_u95_hz: 0.050 Hz
accepted Q: 4000..60000
maximum Q U95 fraction: 0.10
maximum reduced chi-square: 5.0
maximum condition number: 1e8
minimum source SNR: 50
minimum resonance SNR: 25
minimum resonance-to-background ratio: 0.20
selection: converged optimizer-interior single-pole fit in the accepted interval; both accepted-frequency endpoints are interior to the optimizer domain
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
native SDK-export calibration bytes and SHA-256
canonical calibration payload bytes and SHA-256
canonical adapter source SHA-256
calibration analyzer SHA-256
source and instrument queryback SHA-256
frequency-grid SHA-256
```

The calibration artifact must also bind the exact calibration assembly and carrier population, frequency list, sample rate, block size, channel order, scales and offsets, source amplitude and offset, settling time, fitted complex background and gain, Q and U95, residual metrics, fit condition, start/completion UTC, and `primary_observed=false`. This is one global pre-assignment DUT-A/FC135 reference; all five later records bind that same calibration identity and frequency tuple while retaining their own separate primary assembly/population identities. Removed-resonator B and 1-pF-dummy C are not falsely represented as having produced a resonance calibration. The implemented native input is the frozen SDK-export int16 representation. No proprietary container parser is claimed; any original proprietary container must be preserved separately in a future authorized acquisition. Calibration must complete before the assignment commitment. The source-preparation receipt must begin at least 3.000 s before primary acquisition.

The analyzer propagates the bound tuple through the source queryback, drive/reference fit, C2 transfer, I/Q projection, reconstruction, cycle counting, off-resonance controls, and matched-arm comparison. Nominal-frequency constants are restricted to synthetic fixture generation; an AST regression test rejects their use by operational analysis functions.

The calibration receipt alone is not sufficient. The analyzer must derive both complex channel points directly from the bound raw samples; it may not normalize inverse transfer, rotate or scale the response, or subtract a background before fitting. The complex background and gain are fitted parameters. The final raw block is a separately bound off-resonance response at `f_carrier + max(20 Hz, 20 calibrated linewidths)` and is excluded from the resonance fit. Its background-subtracted response ratio plus U95 must not exceed 0.030. The 0.030 gate replaces the contradictory 0.020 literal because an ideal single pole at exactly twenty linewidths has magnitude about 0.025. Self-consistently rehashed invalid bytes, a too-close probe, or an excessive measured response remain hard rejections.

This calibration-realism disposition preserves the BVD circuit/load sanity model as a separate prospective architecture calculation. It replaces only the synthetic-perfect calibration parser; it does not reopen the carrier, sense topology, relays, netlist, BOM, fabrication design, or signal-path witness result.

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
12 raw physical-realism positive calibration fixtures
15 raw model/measurement negative calibration fixtures
10 calibration custody and chronology adversaries
two fresh-process byte-identical reproductions
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
