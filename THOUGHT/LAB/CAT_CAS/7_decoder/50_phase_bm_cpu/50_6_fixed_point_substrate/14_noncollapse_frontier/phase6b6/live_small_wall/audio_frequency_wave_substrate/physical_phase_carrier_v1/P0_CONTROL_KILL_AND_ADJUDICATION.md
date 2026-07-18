# P0 Control, Kill, and Adjudication Law

**Status:** `FROZEN_ARCHITECTURE__NOT_EXECUTED`<br>
**Decision ceiling:** architecture only<br>
**Current physical authority consumed:** 0

## 1. Control matrix

Every physical arm below requires a later explicit authority packet. `Yes` in
the last column means it would consume that later authority; it does not mean
the present architecture task performed the arm.

| Arm | Purpose and mechanical distinction | Prospectively expected observation | Kill condition | Required evidence | Consumes future physical authority? |
|---|---|---|---|---|---|
| Source remains on | Establish sustained-source signature, not ringdown | CH1 remains phase coherent and nondecaying with CH0 | If it is presented as post-source persistence | Raw CH0-CH2 and switch trace | Yes |
| Source off, resonator removed | Bound source, switch, cable, voltage detector, and digitizer response without mechanical carrier | No admitted 256-cycle response above SNR 10 | `epsilon_feed > T_feed` or coherent decay survives | Empty-socket topology image, raw channels, metrics | Yes |
| Dummy electrical load | Match C0 and ESR-scale electrical behavior without quartz motion | Switching/RC transient ends before `t_admit` | Dummy produces admitted ringdown or exceeds feedthrough cap | Dummy identity and impedance calibration | Yes |
| Off-resonance drive | Separate broadband/switch response from stored resonant mode | No long response at `f_ref`; any transient follows electronics | Admitted ringdown meets primary carrier law | Exact drive offset and raw evidence | Yes |
| Zero-drive acquisition | Measure detector and environmental noise | Noise-only I/Q under frozen estimate | SNR 10 carrier-like response or threshold inflation required | Raw channels and noise estimate | Yes |
| Detector-only replay | Test high-impedance voltage detector/digitizer memory using a calibrated non-carrier-side test-port impulse with carrier absent | Detector impulse response obeys its <=10 us settling budget and coherent f_ref output is below T_feed | Detector reproduces admitted post-source lifetime or exceeds backaction bound | Detector input-admittance calibration, injection identity, raw response | Yes |
| Controller-buffer replay | Test command queues, AWG memory, and interface buffering with resonator absent | No CH1 carrier after guard | Buffer history produces admitted trace | Controller logs, empty-socket raw bytes | Yes |
| Reference-only leakage | Provide only continuous C2 into the passive CH0 monitor with zero C1 physical drive and no trigger cable | CH1 remains at noise | C2 alone produces admitted carrier | Reference wiring, raw channels, I/Q | Yes |
| Phase-reference inversion | Rotate the offline gauge without changing raw CH1 | Both arms rotate together; negation metric is gauge invariant | Raw bytes change, one arm alone is favored, or result exists only under one gauge | Original and inverted offline results from same hashes | No additional contact; analysis only |
| Channel or cable swap | Detect label, channel, and cable-path dependence | Rebound topology changes roles exactly as declared; no label survives without signal | Result follows filename/channel label rather than physical path | Wiring record, channel map, raw hashes | Yes |
| Wrong termination | Expose source-side and midpoint termination dependence | Feedthrough/transient grows; arm is rejected | Wrong termination still qualifies as source-off primary | Measured termination and switch trace | Yes |
| Wrong guard interval | Prove early apparent persistence is switching/electronics compatible | Pre-guard region differs from admitted region | Primary claim uses any pre-guard sample or shortened guard passes when 10 ms fails | Raw trace with both windows marked | No additional contact; alternate analysis of same raw bytes |
| Switch transient | Source gate/relays actuate with no resonant preparation | Short bounded transient ending before guard | Transient persists into 256 usable cycles | Witness trace, empty/dummy raw channels | Yes |
| Matched 0/pi arms | Test the physical antipodal relation | `z_pi approximately -z_0` with matched envelopes, f, tau, and controls | Any required metric exceeds its sealed threshold | Sealed assignment, both raw packets, calibration binding | Yes |
| Unmatched-amplitude arm | Prove envelope matching is necessary and metric-sensitive | `epsilon_A` deliberately fails while phase processing remains defined | It accidentally enters matched adjudication or metric is insensitive | 0.800 amplitude command and raw evidence | Yes |
| Wrong-frequency arm | Reject nearby electrical or analysis artifacts | Frequency metric and/or matched complex relation fails | It passes the primary frequency and negation laws | Frozen offset and raw evidence | Yes |
| Random-phase arms | Show phase result follows prepared phase continuously, not a binary label | Fixed offsets map to corresponding circular means | Results collapse to file labels 0/pi or all appear antipodal | Three fixed offsets and sealed mapping | Yes |
| Environmental disturbance | Bound mount, temperature, vibration, and acoustic sensitivity | f, tau, or phase shifts consistently with recorded disturbance | Unrecorded disturbance, primary pair outside limits, or disturbance indistinguishable from source-off result | Temperature, acceleration, RH, topology | Yes |
| Offline synthetic replay | Prove the analysis accepts known synthetic truths but cannot authorize a physical result | Numerical metrics reproduce injected synthetic phase/decay | Synthetic bytes enter physical manifest or contact count | Separate synthetic manifest and source | No |
| Source-left-on negative | Prove a software mute or label cannot masquerade as isolation | Witness remains DRIVE and arm is categorically rejected | Any source-left-on arm is admitted as post-source | CH0-CH2 and source command log | Yes |
| Gate-only isolation | Ablate the relay barrier while exercising the analog gate on a dummy/removed carrier | Loaded leakage/transient is bounded but never qualifies as primary source-off | Gate-only data enter primary evidence or exceed feedthrough cap | Frozen bypass topology, CH0/CH2, raw trace | Yes |
| Relay-only isolation | Bypass the analog gate while exercising K1/K2/K3 on a dummy/removed carrier | Relay transition is bounded and never qualifies as primary source-off | Relay-only data enter primary evidence or exceed feedthrough cap | Frozen bypass topology, CH0/CH2, raw trace | Yes |
| K1 stuck closed | Inject the frozen continuity stimulus with K1 forced closed and K2/K3 in OFF defaults | Witness and injection scan identify the single failure | Failure is not detected or is admitted | Contact-state and injection evidence | Yes |
| K2 stuck closed | Inject the frozen continuity stimulus with K2 forced closed and K1/K3 in OFF defaults | Witness and injection scan identify the single failure | Failure is not detected or is admitted | Contact-state and injection evidence | Yes |
| K3 guard failed open | Inject the frozen continuity stimulus with guard open and K1/K2 OFF | Witness and feedthrough scan identify the missing guard | Failure is not detected or is admitted | Contact-state and injection evidence | Yes |
| Timing mismatch | Test dependence on the source-off epoch | Quarter-carrier-cycle delay remains CH0-gauge neutral ideally; any change quantifies switching sensitivity, and the arm stays categorically unmatched | Mismatched arm is treated as matched or its schedule is normalized away | Gate witness, CH0 dual-tone gauge, raw evidence | Yes |
| Wrong-phase pi/2 | Test continuous phase chart | Circular mean follows pi/2, not pi | It passes the antipodal half-turn law | Sealed phase command and raw evidence | Yes |

No control outcome can be imported from simulation as physical evidence.

## 2. Artifact discrimination

| Apparent persistence source | Mechanical discriminator |
|---|---|
| Residual source drive | Upstream CH0 plus ordered auxiliary relay witnesses; removed-resonator and dummy arms; guarded termination; separate per-event actual signal-pole evidence required |
| Direct electrical feedthrough | Dummy matched to C0; source kept active upstream; `epsilon_feed` upper bound |
| Driver energy storage | No driver component downstream of K2; CH0 plus CH2 auxiliary-contact timing; sealed continuity/injection evidence; wrong-termination arm; actual signal-pole evidence cannot be inferred from CH2 |
| Voltage detector or input-network ringdown | >=10*f0 bandwidth, <=10 us settling budget, detector-only replay, sealed input admittance, no analysis before 10 ms guard |
| Digitizer memory or software-start correlation | Continuous pre/post capture, raw native bytes, zero-drive and controller-buffer controls; no external trigger exists |
| Reference leakage | Reference-only arm, physical disconnection, common-gauge inversion, channel swap |
| DSP filter memory | Windowed least-squares projection whose every sample is post-guard; no recursive filter state |
| File or metadata replay | Raw-native hash first, opaque arm IDs, strict manifest, synthetic packet segregated |
| Acoustic room reflection | Localized hermetic carrier, enclosure/environment trace, no speaker/microphone chain |
| Relay shock or switching EMI | Switch-transient arm, mechanical separation, acceleration trace, 10 ms guard |

## 3. Global hard kills

Any one of these conditions stops P0 adjudication:

1. The actual carrier is not a mechanically defined displacement or strain mode.
2. The topology differs from the frozen source-to-carrier diagram.
3. A drive-capable component exists downstream of the final series-open relay.
4. Any software timestamp substitutes for the physical switch/contact witnesses.
5. A source-off witness is missing, contradictory, late, or changes after
   `t_admit`.
6. K1/K2 do not reach 1,000 stable code-0 samples before K3 is allowed to
   release, or no separately reviewed per-event actual-signal-path witness or
   exact force-guided-contact guarantee covers the signal poles.
7. Any primary raw byte is altered after its first SHA-256 seal.
8. Any analysis window touches pre-guard data.
9. Any primary signal clips, contains nonfinite data, loses samples, or violates
   the frozen channel map.
10. The resonator-removed or dummy control exceeds `T_feed`.
11. Fewer than 256 consecutive usable cycles satisfy SNR and uncertainty.
12. Calibration requires any threshold above its hard cap.
13. A threshold, filter, frequency, window, guard, or arm mapping changes after
    primary data are inspected.
14. The source-left-on, wrong-termination, wrong-frequency, pi/2, timing, or
    amplitude control is admitted as matched primary evidence.
15. A result depends on filenames, metadata, expected labels, reference-only
    data, controller buffers, or synthetic replay.
16. The primary pair exceeds 0.20 C temperature difference or lacks the frozen
    environmental record.
17. A scientific failure is retried or an integrity replacement exceeds the
    one-replacement law.
18. Any claimed contact count is inconsistent with the immutable run ledger.

No averaging over a failed arm repairs a kill.

## 4. Prospective arm-order law

The calibration packet freezes one deterministic order generated without
runtime randomness:

```text
offline synthetic conformance vectors
zero-drive
resonator-removed
dummy C0 within +/-1% of measured carrier C0
switch-transient
detector-only replay: eight-cycle Hann burst at f_ref through detector test port
controller-buffer replay
reference-only
gate-only isolation
relay-only isolation
K1 stuck-closed continuity/injection test
K2 stuck-closed continuity/injection test
K3 failed-open continuity/injection test
wrong termination
off-resonance at f_ref + max(20 Hz, 20 measured linewidths)
source-remains-on
source-left-on
mandatory channel/cable swap
opaque matched arm A
opaque matched arm B
amplitude mismatch at 0.800 calibrated amplitude
wrong frequency at f_ref + max(20 Hz, 20 measured linewidths)
timing mismatch at +0.250 nominal carrier cycle
wrong phase at pi/2
random phases in [pi/7, 3*pi/7, 5*pi/7]
environmental disturbance: +1.0 +/-0.1 C within the 20-30 C envelope
offline wrong-guard analysis
offline phase-reference inversion
```

The matched A/B mapping remains sealed until byte integrity, topology, switch,
environment, and control validity close. Every physical item waits at least
`max(10*tau_A,10 s)` and five consecutive nonoverlapping windows below
`3*sigma_A` before the next item. The channel/cable swap is mandatory; if its
frozen substitute topology cannot be built without changing the carrier load,
P0 stops rather than waiving it.

The list is exhaustive and each item occurs exactly once. Any control-integrity
failure invalidates the entire physical sequence. One complete replacement
sequence is allowed only under the integrity-only gate; a second integrity
failure or any scientific failure stops. Control values, order, reset, and
evidence cannot change on replacement. The invalid predecessor's complete
acquisition packet, inner manifest, and signed raw-root receipt remain bound in
the final packet; receipt-only preservation is insufficient. Its six frozen
identity/configuration/calibration/assignment records must be byte-identical to
the selected attempt, its own contact record must contain exactly 27 physical
acquisitions, and each signed acquisition interval must lie strictly between
the calibration-root receipt and its corresponding raw-root receipt.

The machine-readable authority is the 30-element `ordered_control_ids` const in
`P0_EVIDENCE_SCHEMAS.json`. `outcomes` is an exact-key object: one and only one
entry per ordered ID. Each entry binds
`control_evidence/<control_id>.json`, whose strict submanifest is reconstructed
against final packet bytes by `p0_packet_validator.py`. A PASS adjudication
requires all 30 outcomes PASS; a missing, extra, duplicated, misordered,
unbound, or INVALID/FAIL control rejects PASS.
This is structural coherence only: `p0_packet_validator.py` always returns
`scientific_authority=false` and cannot emit P0A-P0C. A future raw-derived
analyzer, malformed-payload suite, and independent review must recompute every
scientific and control-semantic result before any physical adjudication.

## 5. P0 claim ladder

### P0A

```text
PHYSICAL_PHASE_RESONANCE_CHARACTERIZED
```

Meaning only: one mechanical resonance, linear preparation region, measurement
chain, and bounded Q/lifetime have been measured.

### P0B

```text
PHYSICAL_PHASE_CARRIER_POST_SOURCE_OBSERVED
```

Meaning only: a phase-sensitive signal attributable to the declared mechanical
mode survives the frozen physical isolation boundary and controls.

### P0C

```text
PHYSICAL_PHASE_CARRIER_PI_RELATION_CHARACTERIZED
```

Meaning only: matched 0/pi preparations satisfy the frozen post-source complex
antipodal and matching laws.

This architecture task emits none of P0A-P0C.

## 6. Architecture adjudication

The non-executing packet may emit:

```text
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN
```

only when:

```text
one preferred and one fallback mechanical carrier class are selected
the source-off and phase-readout laws are prospective and mechanical
0/pi arms, metrics, controls, evidence, BOM, safety, and silicon mapping close
four independent architecture reviews with externally checkable provenance pass with zero open material findings
no hardware, audio, target, or instrument has been contacted or operated
```

Claim ceiling:

```text
NON_EXECUTING_PHYSICAL_PHASE_CARRIER_ARCHITECTURE_ONLY
```

The architecture token does not establish resonance, post-source persistence,
the pi relation, physical computation, restoration, Ising behavior, silicon
operation, bit replacement, or a Wall crossing.

## 7. Next authority boundary

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

The current packet must stop before that boundary.
