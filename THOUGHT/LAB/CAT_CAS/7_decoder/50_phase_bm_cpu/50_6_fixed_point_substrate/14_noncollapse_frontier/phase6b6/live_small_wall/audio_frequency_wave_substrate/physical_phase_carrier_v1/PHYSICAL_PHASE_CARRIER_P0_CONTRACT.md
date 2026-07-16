# Physical Phase Carrier P0 Contract

**Status:** `BOUNDARY_SELECTED__CONTRACT_CANDIDATE__NO_HARDWARE_AUTHORITY`  
**Package:** `physical_phase_carrier_v1`  
**Selected next boundary:** `PHYSICAL_PHASE_CARRIER_P0_POST_SOURCE_PI_CHARACTERIZATION`  
**Parent result:** `AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED`  
**Root directive:** `REPLACE THE BIT WITH PI`  
**Operation:** physical-carrier architecture and experiment contract only  
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

This contract selects an electrode-addressed piezoelectric or electromechanical resonator
at audio or low-ultrasonic frequency as the first development carrier. Open-air speaker to
room to microphone propagation is not the preferred P0 implementation.

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

Preferred P0 carrier order:

```text
1. electrode-addressed piezoelectric mechanical resonator
2. electromechanical resonator with direct displacement-sensitive readout
3. low-frequency electronic resonator only if mechanical access is unavailable
```

The first implementation should avoid an uncontrolled open-air propagation path. A sealed
or mechanically localized resonator is preferred because the carrier boundary is easier to
identify and later transpose into silicon.

This document does not select a commercial part number, authorize a purchase, or authorize
powered execution.

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

The source-off boundary must be explicit and observable.

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
cable or channel swap where mechanically possible
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
residual drive, electrical feedthrough, digitizer memory, filtering, trigger leakage,
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

## 15. Immediate Next Work

The next bounded package is:

```text
PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_AND_MEASUREMENT_PACKET
```

It must select the actual carrier class and access model, define the measurement chain,
freeze the source-off law, define all metrics and controls, identify required equipment,
and return a buildable but non-executed packet.

The packet must not contact hardware.

## 16. Current Decision

```text
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
PHYSICAL_PHASE_CARRIER_P0_BOUNDARY_SELECTED
PHYSICAL_PHASE_CARRIER_NOT_YET_OBSERVED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```
