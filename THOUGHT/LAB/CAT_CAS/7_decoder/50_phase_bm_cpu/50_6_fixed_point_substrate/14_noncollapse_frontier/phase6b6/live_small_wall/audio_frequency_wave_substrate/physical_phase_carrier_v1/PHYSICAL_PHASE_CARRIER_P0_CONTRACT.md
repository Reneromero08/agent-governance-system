# Physical Phase Carrier P0 Contract

**Status:** `P0_BUILD_READINESS_BLOCKED__NO_HARDWARE_AUTHORITY`<br>
**Package:** `physical_phase_carrier_v1`<br>
**Architecture decision:** `PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN`<br>
**Authority:** `AUTHORIZE P0 BUILD-READINESS ONLY`<br>
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`<br>
**Selected next boundary:** `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD`<br>
**Parent result:** `AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED`<br>
**Root directive:** `REPLACE THE BIT WITH PI`<br>
**Operation:** physical-carrier architecture plus non-executing build readiness only<br>
**Hardware authority:** none

## 1. Selection

The next CAT_CAS audio-lane boundary is not a larger software instance, a scaling study,
a phase-language implementation, or a full physical Ising machine.

It is the smallest physical translation that can establish the carrier itself:

```text
one deliberately addressable mechanical resonant mode
-> prepare a phase state
-> remove the sustaining source
-> observe post-source ringdown phase
-> compare matched 0 and pi preparations
-> prove the transducer/controller is not the retained state
```

This contract selects one hermetic, electrode-addressed 32.768 kHz quartz tuning-fork
mechanical mode as the preferred development carrier. A localized external-drive
PZT/brass diaphragm mode near 6.3 kHz is the one fallback carrier class. Open-air speaker
to room to microphone propagation is not the preferred P0 implementation.

The eventual dense target remains a deliberately engineered silicon mechanical or phononic
resonator. P0 is the slow, observable mechanical chart used to close the access model.

## 2. Why This Boundary Comes Next

The bounded software ladder is now established:

```text
recursive phase representation
complete-tree temporal recurrence
software catalytic carrier loop
continuous S1 Ising sector with final 0/pi projection
```

Repeating those results at larger numerical scale would not close the current missing
mechanism. The missing mechanism is a physical phase state whose custody can be separated
from its source, detector, controller, and serialized record.

A single resonator cannot yet implement arbitrary coupling or a physical Ising problem.
It can establish the physical phase coordinate, source-off persistence, antipodal action,
readout, and kill controls required before a coupled network is meaningful.

## 3. Declared Physical Carrier

The declared carrier is one mechanical normal mode of a resonant body.

```text
carrier:
    mechanical displacement / strain mode

transducer:
    electrical, piezoelectric, capacitive, optical, or other bounded interface

controller:
    waveform generation, switching, timing, acquisition, and evidence custody

observable:
    phase-sensitive I/Q estimate of the declared mechanical mode
```

The drive voltage is not automatically the carrier. The ADC record is not the carrier.
The reference oscillator is not the carrier. The carrier claim attaches only to the
mechanical resonant mode after transduction and source-separation controls close.

The bounded analytic chart is:

```text
z(t) = I(t) + i Q(t) = A(t) * exp(i * theta(t))
```

where `theta(t)` is the physical mode phase relative to a prospectively frozen reference.

## 4. Selected Development Class

Frozen carrier classes:

```text
preferred:
    hermetic electrode-addressed 32.768 kHz quartz tuning-fork mechanical mode
    exact prospective identity: Epson Q13FC1350000401 (FC-135, 12.5 pF; any-quantity tape-cut suffix)

fallback:
    localized external-drive PZT/brass diaphragm mechanical mode near 6.3 kHz
    reference datasheet class: Murata 7BB-20-6
```

The preferred access path uses phase-synchronous electrical preparation, an
always-connected high-impedance differential electrode-voltage readout across
the complete two-terminal loaded BVD network, and a two-stage source-off boundary:
fast analog route-to-termination followed by a fail-safe, independently witnessed guarded
relay barrier. The mechanical state is the quartz tine displacement/strain mode.

The Epson ordering identity is frozen prospectively but is not acquired, reserved, or
authorized for purchase. Its official document bytes and all incoming lot identities remain
mandatory gates under a later explicit authority.

## 5. Silicon Transposition

The development and target charts are:

```text
P0 bench chart                     silicon target chart
--------------                     --------------------
mechanical resonant mode           silicon elastic eigenmode
piezo/electrical preparation       piezo, electrostatic, optical, or hybrid preparation
I/Q phase readout                  phase-sensitive displacement/strain readout
ringdown envelope                  silicon-mode ringdown / quality factor
0 versus pi preparation            antipodal silicon-mode orientation
source-off switch boundary         isolated post-drive silicon state
```

The carrier-independent contract is:

```text
same phase state space
same antipodal pi relation
same source-separation law
same observable definition
same kill controls
same claim ceiling
```

P0 success does not transfer a physical claim to silicon. It freezes the access law that a
later silicon implementation must satisfy.

## 6. P0 Experiment Lifecycle

The prospectively required lifecycle is:

```text
1. characterize one resonance and bounded linear operating region
2. freeze drive frequency, amplitude, phase reference, acquisition timing, and source-off law
3. prepare theta = 0 arm
4. remove or isolate the sustaining source at the declared boundary
5. acquire post-source ringdown I/Q
6. prepare theta = pi arm under matched amplitude and timing
7. remove or isolate the source under the same boundary
8. acquire post-source ringdown I/Q
9. compare antipodal relation, amplitude envelope, frequency, and decay law
10. run feedthrough, controller-replay, off-resonance, dummy-load, and detector controls
```

No adaptive threshold, phase choice, time window, filter, or winning arm may be selected
after observing the primary result.

## 7. Source-Off Boundary

The source-off boundary is explicit and observable:

```text
stage A:
    ADG1419BRMZ MSOP SPDT at +/-5 V, controlled through IN, routes the
    downstream drive node away from C1 to 50 Ohm with a 560 ns
    full-temperature transition bound

stage B:
    K1 and K2 normally-open series relay contacts de-energize first
    their auxiliary code must remain 0 for 1,000 samples while K3 stays energized
    only then may K3 de-energize its midpoint guard contact to 50 Ohm
    final auxiliary code 8 must remain stable for 1,000 samples

witnesses:
    source-side CH0, resonator-side CH1, auxiliary-contact-state CH2
    actual signal-pole state requires a separately reviewed per-event witness
    or exact force-guided-contact guarantee; CH2 alone is insufficient

first admissible raw sample:
    10.000 ms after the ordered auxiliary transition reaches stable code 8,
    and never admissible for a physical source-disconnect claim until the
    actual-signal-path evidence gate above is closed
```

The frozen prospective source is one SDG1032X in `HIGH_Z` load mode with 50 Ohm
physical outputs. C1 is a continuous 32,768 Hz, 0.400 Vpp, 0 V-offset sine at
phase 0 or pi. C2 is a continuous 65,536 Hz, 0.100 Vpp, 0 V-offset sine at fixed
zero phase that enters only the passive CH0 monitor. Both outputs remain on for
the complete software-prearmed free-running record. No burst or external trigger
is part of P0. CH2 locates the isolation event and the C2/C1 fit supplies the
record-local phase gauge.

The contract must freeze:

```text
switch topology or gating mechanism
command time
measured isolation delay
residual electrical feedthrough envelope
acquisition guard interval
first admitted post-source sample
maximum accepted controller or buffer latency
```

A software mute command alone is not sufficient evidence of source removal. Qualification
must show what electrical or physical path is disconnected, gated, terminated, or bounded.

The post-source interval must begin after the maximum admitted source, buffer, switching,
and detector transient interval.

## 8. Antipodal Pi Law

Two preparations differ only by a phase half-turn:

```text
theta_pi = theta_0 + pi mod 2*pi
```

Required relation over the declared post-source observation window:

```text
z_pi(t) approximately -z_0(t)
```

The final contract must prospectively freeze:

```text
complex negation error metric
amplitude-envelope matching metric
frequency matching metric
decay-constant matching metric
minimum usable post-source cycles
phase uncertainty law
```

A sign reversal in one detector channel is insufficient. The complete complex relation and
matched carrier envelopes must close.

## 9. Required P0 Observables

At minimum:

```text
resonant frequency f0
quality factor or decay constant
post-source usable cycle count
I/Q trajectory
phase estimate and uncertainty
amplitude envelope
noise floor
source-feedthrough estimate
0/pi complex-negation error
0/pi amplitude-envelope mismatch
0/pi frequency mismatch
0/pi decay-law mismatch
```

Raw acquisition, calibration, reference, switch-state, and environmental records must be
preserved before derived metrics are trusted.

## 10. Required Controls

The P0 package must include at minimum:

```text
source remains on
source-off with resonator removed or replaced by dummy load
off-resonance drive
zero-drive acquisition
detector-only replay
controller or buffer replay
reference-phase inversion
mandatory cable or channel swap, or stop if no load-preserving substitute closes
matched 0/pi preparations
unmatched-amplitude negative control
wrong-frequency negative control
wrong source-off timing
post-window selected before acquisition
```

The package must distinguish:

```text
mechanical ringdown
from
residual drive, electrical feedthrough, digitizer memory, filtering, software-start correlation,
reference leakage, and offline waveform replay
```

## 11. Claim Ladder

### P0A: Resonance characterized

```text
PHYSICAL_PHASE_RESONANCE_CHARACTERIZED
```

Meaning only that one bounded resonance and acquisition chain are identified.

### P0B: Post-source phase carrier observed

```text
PHYSICAL_PHASE_CARRIER_POST_SOURCE_OBSERVED
```

Meaning only that phase-sensitive evidence attributable to the declared carrier survives
the admitted source-off boundary for a prospectively frozen interval.

### P0C: Antipodal pi relation characterized

```text
PHYSICAL_PHASE_CARRIER_PI_RELATION_CHARACTERIZED
```

Meaning only that matched 0 and pi preparations produce the declared post-source complex
antipodal relation under controls.

No single P0 token establishes computation, memory beyond the measured ringdown, physical
catalysis, restoration, Ising solving, silicon operation, bit replacement, or a Wall
crossing.

## 12. Forbidden Promotion

P0 does not establish:

```text
arbitrary programmable coupling
phase-bistable self-locking
physical recursive phase trees
physical catalytic restoration
physical Ising computation
silicon phononic computation
general phase-native memory
speed, energy, capacity, or scaling advantage
hardware bit replacement
Small Wall or Big Wall crossing
```

Natural ringdown persistence is not automatically computational memory. A prepared pi arm
is not automatically a binary computer. A transducer response is not automatically a
carrier state.

## 13. Next Rungs After P0

Only after P0 closes may the lane select among:

```text
P1  two physical resonators with controlled phase-sensitive coupling
P2  second-harmonic or equivalent 0/pi phase-bistable locking
P3  bounded physical recursive modulation / complete-object operator
P4  physical query, latch, reverse schedule, and restoration
P5  bounded physical phase-native Ising instance
S2  one deliberately addressable silicon mechanical mode under the same P0 law
```

The preferred geodesic is:

```text
P0 mechanical phase access
-> P1 controlled two-mode relation
-> P2 antipodal locking
-> P3/P4 physical recursive catalytic lifecycle
-> silicon transposition
-> bounded physical Ising projection
```

Silicon feasibility and bench development may proceed in parallel as architecture work,
but no silicon physical result is inherited from P0.

## 14. Current Authorization

Authorized now:

```text
read-only research
architecture comparison
contract drafting
measurement-plan drafting
control and kill-matrix drafting
BOM and access-model analysis without purchase or operation
```

Not authorized by this file:

```text
purchasing
powered circuit operation
audio playback or recording
ADC/DAC use
speaker, microphone, piezo, resonator, or transducer operation
live hardware contact
remote target contact
physical claim generation
```

A later explicit execution authority must identify the actual hardware, limits, wiring,
source-off mechanism, measurement chain, no-retry or retry law, and safety boundary.

## 15. Frozen architecture packet

The bounded non-executing architecture is:

```text
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_AND_MEASUREMENT_PACKET
```

It selects the preferred and fallback carrier classes, access and source-off model,
measurement/metric law, preparation arms, controls, evidence formats, non-purchasing BOM,
safety limits, silicon translation, and independent review requirements.

The next exact boundary is:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

This packet stops before that boundary.

## 16. Current Decision

```text
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
PHYSICAL_PHASE_CARRIER_P0_BOUNDARY_SELECTED
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN
P0_BUILD_READINESS_BLOCKED
PHYSICAL_PHASE_CARRIER_NOT_YET_OBSERVED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```
