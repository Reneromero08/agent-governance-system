# P0 post-qualification audit

**Audit status:** `P0_POST_QUALIFICATION_AUDIT_COMPLETE`  
**Audited commit:** `62b941446cb074c0effb454662b898b140aa1ffb`  
**Preserved result:** `P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED`  
**Corrected build-readiness decision:** `P0_BUILD_READINESS_BLOCKED`  
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`  
**Next engineering boundary:** `P0_RESONANCE_LOAD_LAW_REPAIR_REQUIRED`  
**Physical, procurement, assembly, and instrument authority:** none

This audit inspected the committed circuit model, final netlist, analyzer, AST/data-flow proof, mutation suite, packet, results, findings, and claim law rather than accepting the completion report. It performs no hardware, vendor, cart, purchasing, instrument, playback, recording, or target operation.

## 1. What remains established

The actual-path repair is mechanically coherent at its stated level:

```text
C2_REF_IN
-> exact 1.00 Mohm injection at N_GATE_OUT
-> K1 signal contact
-> N_MIDPOINT
-> K2 signal contact
-> N_ELECTRODE_A
-> OPA810
-> CH1
```

The analyzer evaluates the complete-path transfer before K3 guard acceptance. A passing future record may support only:

```text
ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT
```

This means at least one actual series contact interrupted the end-to-end source path. It does not identify either contact and does not prove that both opened. The repair's model, custody, ordering, and claim narrowing remain useful and are not withdrawn.

## 2. Audit finding A-01: stale hashes in the completion report

**Severity:** documentation error  
**Status:** closed by this audit

The user-visible completion report repeated three hashes from the earlier blocked packet. The committed final package carries these identities:

```text
P0_FINAL_NETLIST.json
10dc13d96668731e50db02a4c22e49b4398e90d1ddb3c887167da6491c8c5603

P0_NONPURCHASING_BOM.json
9b83eb29cd2d94e0eee3ff7fcab532ce8e1a862df63167abba420289bb6bd47c

P0_PCB_FABRICATION_RELEASE.json
c137482d45ebdc62c832cac612cfc365795014258e940907a23b0f5fa7204438
```

The repository packet and result artifact already contain the correct hashes. No committed-byte repair is required for this finding.

## 3. Audit finding A-02: resonance/load law is not closed

**Severity:** blocker  
**Status:** open

The selected carrier is Epson `Q13FC1350000401`, specified in the packet as a 32.768 kHz, 12.5 pF-load FC-135 crystal. The final netlist does not implement a 12.5 pF load. Its carrier-side planning law instead caps detector and layout capacitance at 4.00 pF, with a 3.20-4.00 pF model range.

The witness model derives series resonance from the 12.5 pF nominal loaded-frequency identity:

```text
fs = 32768 / sqrt(1 + Cm / (C0 + 12.5 pF))
```

but the ringdown carrier is loaded by the materially smaller detector-side capacitance. Applying the same committed BVD relation to the model's own endpoint ranges gives:

```text
modeled series resonance:
32749.537420 .. 32764.412852 Hz

modeled loaded resonance at 3.2 .. 4.0 pF:
32773.862762 .. 32810.885343 Hz

shift relative to hard-coded 32768 Hz:
+5.862762 .. +42.885343 Hz

modeled series-resonance linewidth:
0.607050 .. 7.076097 Hz

hard-coded-drive detuning:
approximately 4.14 .. 14.15 modeled linewidths
```

The exact values are prospective model consequences, not physical measurements. They establish the missing mechanism: the packet hard-codes C1 at 32768 Hz and the analyzer at `F_REF = 32768`, while the selected part and frozen load do not prospectively bind the actual assembled resonance to that frequency.

The contract itself requires:

```text
characterize one resonance
then freeze drive frequency
```

The current packet reverses that order. It freezes the nominal catalog frequency before the as-built loaded resonance is known.

The steady-state 65.536 kHz signal-path witness model does not repair this. It proves prospective end-to-end isolation discrimination, not that the selected quartz/load/source system will prepare a sufficiently observable mechanical ringdown at the hard-coded carrier frequency.

Therefore `P0_BUILD_READINESS_PACKET_FROZEN` is withdrawn pending a separately qualified resonance/load repair.

## 4. Audit finding A-03: continuous uncertainty envelope is overstated

**Severity:** material  
**Status:** open

The circuit model evaluates all `2^17 = 131072` endpoint combinations for each candidate. That is a complete binary-corner sweep, not by itself a proof that every continuous parameter value inside the hyperrectangle is bounded by the corner extrema.

The packet may retain the exact phrase:

```text
complete binary-corner sweep
```

It may not treat corner extrema as a complete continuous uncertainty envelope unless it adds one of:

```text
an analytic monotonicity/extremum proof;
a validated interval-arithmetic enclosure;
or another rigorous global-bound method.
```

Dense sampling can be diagnostic but is not a substitute for an enclosure proof.

## 5. Audit finding A-04: common-mode gate is conditional, not measured

**Severity:** material execution gate  
**Status:** open

The analyzer computes:

```text
common_mode_peak = 0.5 * max(abs(CH0), abs(CH1))
```

from differential payload channels. A differential record does not generally reveal each digitizer input's true common-mode voltage. The metadata and topology assume calibrated negative legs at analog ground, but the instrument is explicitly not galvanically isolated and the package requires positive-to-ground, negative-to-ground, and differential admittance calibration.

Before physical execution, the packet must either:

```text
bind an independently measured per-leg/common-mode acquisition path;
prove the negative-leg potential and return network make the derived gate conservative;
or remove the derived quantity and replace it with an observable operating-envelope gate.
```

The synthetic gate remains a software check only.

## 6. Corrected scientific state

```text
P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED
P0_BUILD_READINESS_BLOCKED
PHYSICAL_PHASE_CARRIER_NOT_YET_OBSERVED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```

No procurement or unpowered-build authority should be consumed from the prior frozen token. The current authorized work remains non-executing repair, analysis, documentation, review, commit, and push under the existing build-readiness authority.

## 7. Required repair outcome

The next packet must emit exactly one:

```text
P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED
P0_RESONANCE_LOAD_LAW_REPAIR_BLOCKED
P0_RESONANCE_LOAD_LAW_REPAIR_INCONCLUSIVE
```

Only `ESTABLISHED`, plus closure of A-03 and A-04, may restore:

```text
P0_BUILD_READINESS_PACKET_FROZEN
```

The later authority boundary remains:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

It is not active while this audit hold is open.
