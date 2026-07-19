# P0 Build-Readiness Authority

**Authority:** `AUTHORIZE P0 BUILD-READINESS ONLY`  
**Authority date:** `2026-07-16`  
**Package:** `physical_phase_carrier_v1`  
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`  
**Current status:** `P0_BUILD_READINESS_PACKET_FROZEN`
**Physical authority:** none

## Authorized result

The package now contains an exact non-purchasing architecture, deterministic offline analyzer, synthetic fixtures and adversaries, calibration-derived resonance/load law, bounded binary-corner modeling, and reviewed source/path custody.

The resonance/load repair establishes only:

```text
P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED
P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED
P0_BUILD_READINESS_PACKET_FROZEN
```

It establishes no physical resonance, persistence, phase relation, computation, restoration, silicon behavior, bit replacement, or Wall crossing.

## Forbidden work

This authority does not permit human vendor communication or quote request, inventory or cart action, purchase, fabrication, assembly, soldering, wiring, probing, physical inspection, power, waveform generation, instrument operation, playback, recording, acquisition, physical calibration, SSH/SCP, target contact, or physical claims. It also does not permit commit or push without a separate explicit user instruction; that separate instruction governs repository publication only and grants no physical authority.

The committed common-mode law is prospective: current differential bytes support only differential clipping. A future powered operating envelope must separately define and validate common-mode observability.

## Review statement

The research correction has four role-separated root-bound PASS declarations at root `97441363687e8d8de2daeffb1fbad157cf94f01b30e1feeb05bdeff718aa33b4`. They are not described as externally reproducible independence. The resonance/load repair adds one focused final read-only review bound to its exact candidate root.

## Stop boundary

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

That boundary is not consumed by this packet. Procurement, unpowered build, and powered execution remain separate future authority decisions.
