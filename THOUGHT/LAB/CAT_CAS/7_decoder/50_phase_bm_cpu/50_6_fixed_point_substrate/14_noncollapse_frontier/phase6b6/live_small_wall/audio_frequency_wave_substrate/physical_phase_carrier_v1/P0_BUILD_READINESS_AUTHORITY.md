# P0 Build-Readiness Authority

**Authority:** `AUTHORIZE P0 BUILD-READINESS ONLY`  
**Authority date:** `2026-07-16`  
**Authority parent:** `2cfa87a7f10aae224768a0c3283b3e035a10123b`  
**Package:** `physical_phase_carrier_v1`  
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`  
**Current status:** `P0_BUILD_READINESS_BLOCKED`<br>
**Physical authority:** none

## 1. Authorized work

This authority permits only the final non-executing preparation required before a later
procurement or physical-execution decision:

```text
select exact manufacturer parts and ordering identities
freeze the exact four-channel acquisition and source topology
freeze exact gate, relay, detector, environment, limiting, and witness components
draft the final wiring/netlist and assembly drawings
implement and qualify the raw-to-derived scientific analyzer
implement malformed-payload, calibration, custody, and synthetic-control fixtures
bind final manufacturer datasheets and manuals by SHA-256 when locally materialized
produce an exact non-purchasing BOM
produce a procurement-readiness packet
produce an unpowered-build packet
produce a separate bounded physical-execution contract
run read-only reviews and repository checks
```

The package may replace a reference-class candidate with an exact component only when the
replacement remains inside the already frozen carrier/access class and the substitution is
explicitly documented. It may not silently change the physical state, source-off law,
observable, metric law, claim ceiling, or silicon-transposition target.

## 2. Forbidden work

This authority does not permit:

```text
human vendor communication or quote request
inventory reservation or sample request
cart action or purchase
commit or push without a separate explicit user instruction
fabrication order
unpowered physical assembly
soldering, wiring, probing, continuity measurement, or inspection of actual parts
power application
waveform generation
ADC, DAC, oscilloscope, digitizer, lock-in, or instrument operation
piezo, quartz, relay, switch, accelerometer, or transducer operation
audio playback or recording
live hardware, target, SSH, or SCP contact
physical data generation
physical claim generation
retry or no-retry execution authority
```

No document produced under this authority may imply that a part was acquired, inspected,
calibrated, connected, or operated.

## 3. Required build-readiness deliverables

The build-readiness packet must close all of the following before promotion:

```text
exact carrier ordering specification and permitted procurement substitutions
exact source, source monitor, current limiter, gate, relay, guard, and termination parts
exact high-impedance sensing implementation and input-admittance budget
exact four-simultaneous-channel 16-bit acquisition instrument and raw export law
exact environment sensors and trigger/timebase integration
final channel map, star-ground/shield law, relay witness code, and source-off netlist
raw scientific analyzer with build/self-test/verify modes
strict raw/evidence/calibration schemas and malformed-payload negative suites
synthetic ideal, feedthrough, detector-memory, switch-transient, phase, decay, and null fixtures
analysis portability and numeric-reproduction law
final conservative limits derived from the selected parts
non-purchasing BOM with quantities, alternates, and disallowed substitutions
unpowered assembly procedure that remains separately unauthorized
bounded execution contract that remains separately unauthorized
four independent read-only reviews with externally checkable task/receipt
provenance and zero open material findings
```

## 4. Required decision token

The build-readiness package may emit exactly one:

```text
P0_BUILD_READINESS_PACKET_FROZEN
P0_BUILD_READINESS_BLOCKED
P0_BUILD_READINESS_INCONCLUSIVE
```

`P0_BUILD_READINESS_PACKET_FROZEN` means only that exact parts, source code, schemas,
fixtures, topology, limits, and later execution law are ready for a separate user decision.
It establishes no physical resonance, post-source persistence, pi relation, computation,
restoration, silicon behavior, bit replacement, or Wall crossing.

The package-local validator may prove candidate-byte structure and tamper
detection, but it may not mint this token from self-asserted local reviewer
strings. Automatic `build`/`verify` freeze emission remains disabled until
externally checkable review provenance and zero open material findings are
separately supplied and reviewed.

## 5. Stop boundary

After build readiness closes, stop at:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

A later procurement authority does not automatically authorize assembly. A later unpowered
build authority does not automatically authorize powered execution. A later powered
execution authority must bind the actual acquired part identities, calibration records,
wiring inspection, operating limits, retry law, and safety boundary.

## 6. Current decision

```text
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN
P0_BUILD_READINESS_AUTHORIZED
P0_BUILD_READINESS_BLOCKED
P0_BUILD_READINESS_PACKET_NOT_FROZEN
PHYSICAL_PHASE_CARRIER_NOT_YET_OBSERVED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```
