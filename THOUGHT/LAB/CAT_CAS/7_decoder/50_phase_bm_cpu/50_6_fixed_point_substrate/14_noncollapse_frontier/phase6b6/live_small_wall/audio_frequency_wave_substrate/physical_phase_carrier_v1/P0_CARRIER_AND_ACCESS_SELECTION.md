# P0 Carrier and Access Selection

**Status:** `FROZEN_ARCHITECTURE__NOT_EXECUTED`<br>
**Decision:** `PHYSICAL_PHASE_CARRIER_P0_ARCHITECTURE_PACKET_FROZEN`<br>
**Root directive:** `REPLACE THE BIT WITH PI`<br>
**Research cutoff:** 2026-07-16<br>
**Hardware authority:** none

## 1. Exact selection

Preferred P0 carrier class:

```text
one hermetically packaged, electrode-addressed 32.768 kHz quartz tuning-fork
mechanical mode, using a Micro Crystal CC7V-T1A-class unit as the reference
datasheet candidate
```

Fallback P0 carrier class:

```text
one mechanically localized external-drive PZT/brass diaphragm mode near
6.3 kHz, using a Murata 7BB-20-6-class element as the reference candidate
```

The preferred carrier is the displacement and strain field of the quartz tines.
It is not the drive voltage, source oscillator, switch state, voltage-sense
buffer, digitizer record, reference clock, or reconstructed complex trace.

The preferred access model is time-multiplexed electrical preparation and
continuous high-impedance piezoelectric electrode-voltage sensing. Source removal uses a fast,
phase-synchronous analog gate followed by a fail-safe guarded relay barrier.
The readout remains connected and unchanged across source-off.

No commercial ordering suffix is frozen by this packet. Before any later
purchase or execution, the final part number and revision must be bound to a
downloaded manufacturer datasheet by SHA-256.

## 2. Why this class

The reference CC7V-T1A datasheet identifies a vacuum-sealed quartz tuning-fork
resonator at 32.768 kHz with 50 kOhm typical / 70 kOhm maximum series
resistance, 3.7 fF typical motional capacitance, 1.2 pF typical static
capacitance, and 1.0 uW maximum drive level. Its hermetic, localized mode makes
the mechanical boundary more identifiable than a room-scale acoustic path.

For the series Butterworth-Van Dyke branch,

```text
Q_series = 1 / (2*pi*f_s*R_1*C_1)
tau_A    = Q_series / (pi*f_s)
N_e      = Q_series / pi
```

where `tau_A` is the amplitude e-folding time and `N_e` is the amplitude
e-folding cycle count. Using the datasheet's typical `C_1`:

| Planning case | R1 | Q_series | tau_A | amplitude e-fold cycles |
|---|---:|---:|---:|---:|
| Typical | 50 kOhm | 26,254 | 0.255 s | 8,357 |
| 70 kOhm resistance case | 70 kOhm | 18,753 | 0.182 s | 5,969 |

These are architecture estimates, not measured values or guaranteed minima,
because the datasheet gives `C_1` as typical and the final electrical loading
changes the observed Q. They justify a conservative 10 ms isolation guard and
leave a large prospective ringdown budget. P0 must still measure its own
frequency, decay, Q, loading, and usable cycles.

The peer-reviewed ringdown literature describes the same bounded lifecycle:
excite a quartz tuning fork, turn the excitation source off, and estimate
resonance and Q from the decaying response. It also warns that demodulator time
constants can distort the transient. P0 therefore preserves raw channels and
uses a frozen offline projection with no acquisition-time smoothing.

## 3. Carrier comparison

| Candidate | Declared stored state | Accessible band and expected lifetime | Source/readout boundary | Repeatability and disturbance | Equipment burden | Silicon similarity | P0 decision |
|---|---|---|---|---|---|---|---|
| Electrode-addressed quartz tuning-fork crystal | Localized quartz tine displacement/strain mode | 32.768 kHz; datasheet-derived planning Q about 19k-26k before final loading | Same two electrodes transduce the mode; the drive lead opens while a fixed high-impedance, low-capacitance differential voltage input remains across the electrodes | Hermetic package is repeatable; temperature, board strain, soldering, and electrical loading remain material | Low-voltage generator, guarded switch, high-impedance differential buffer, simultaneous digitizer | Strong access-model match to hybrid piezoelectric-on-silicon resonators | **Preferred** |
| Localized ceramic PZT/brass diaphragm | Flexural diaphragm displacement/strain mode | Reference 7BB-20-6 class at 6.3 kHz; Q not guaranteed by the product sheet and likely more loading-sensitive | Direct electrode drive and sense are simple, but high static capacitance and acoustic radiation make feedthrough separation harder | Mounting, air loading, temperature, and clamp geometry strongly affect the mode | Simple electronics; mechanical fixture and shielding required | Moderate match to piezo-on-silicon, weaker confinement match | **Fallback** |
| Macroscopic metal tuning fork or beam with bonded transducers | Metal tine/beam displacement mode | Audio to ultrasonic depending geometry; lifetime depends strongly on mount and air damping | Separate actuator and optical or second-transducer readout can make custody excellent | Clamp, added mass, adhesive, air damping, and nearby vibration dominate repeatability | Mechanical fixture plus vibrometer or independent displacement sensor | Good mechanical semantics, weak packaging and scaling match | Compared; not selected |
| Direct DAC-to-ADC loopback | Interface and buffer voltage/record state, not a retained mechanical mode | Response lasts only through interface/filter/buffer memory | Source and observable are the same electrical chain | Highly repeatable but cannot separate physical carrier custody | Lowest | No mechanical similarity | Rejected; replay/control only |
| Passive electrical LC resonator | Electric and magnetic field energy | Audio/ultrasonic values are accessible; Q depends on passive loss | Source can be physically opened and ringdown measured, but state is electrical | Repeatable and useful for source-off calibration | Low | Access-law analogue only | Rejected as P0 carrier; retained as passive electronics control |
| Active oscillator or feedback/delay loop | Controller-maintained electrical/acoustic state | Apparent persistence may be arbitrarily long while powered | Source, gain, delay, and retained state cannot be separated under the P0 mechanical claim | Repeatable but persistence is actively manufactured | Low to moderate | Control-plane analogue only | Rejected; buffer/feedback adversary |
| Sealed acoustic cavity or tube | Distributed air-pressure/velocity mode plus wall motion | Audio/ultrasonic; lifetime depends on ports, wall loss, and transducer loading | Speaker/microphone boundaries and cavity ports require separate custody | More stable than a room but sensitive to geometry, temperature, and sealing | Moderate enclosure and acoustic instrumentation | Mechanical but weak localization match | Lower-priority future control; not selected |
| Open-air speaker-room-microphone path | Distributed cone, air-pressure, reflection, microphone, and room state | Audio band; lifetime and modes are room dependent | Source, transducers, room reflections, acoustic feedback, and interface buffers are hard to separate | Highly sensitive to geometry, temperature, humidity, people, and reflections | Easy to excite but hard to bound | Poor match to localized silicon mode | Rejected as the first path |

The fallback remains mechanical. The LC/active-oscillator path is not eligible
to satisfy P0 merely because it is convenient.

## 4. Frozen physical and access model

```text
base phase clock / source waveform
        |
        v
source monitor CH0 and continuous master phase clock
        |
        v
phase-synchronous analog gate -> 50 ohm source termination
        |
        v
fail-safe guarded relay barrier (series-open / midpoint guard / series-open)
        |
        v
electrode A of two-terminal quartz tuning fork
        |  BVD motional branch in parallel with C0
        v
electrode B bonded to the carrier-side analog reference

electrodes A/B -> always-connected >=100 MOhm, <=3.0 pF differential
                 voltage input and fixed >=100 MOhm bias return
        |
        v
mechanical-sense CH1

gate plus relay contact witnesses -> calibrated 4-bit resistor code on CH2
enclosure vibration -> CH3
temperature / humidity -> timestamped environmental record
all raw channels -> immutable native capture -> offline I/Q and metrics
```

The controller commands preparation and isolation. It is not inside the
carrier boundary. The source remains observable upstream after isolation. The
mechanical-sense input is downstream of both isolation stages. Its measured
input resistance, capacitance, protection network, bias return, cable
capacitance, and carrier-side ground/shield admittance form the complete
post-source electrical load. They are fit with the loaded BVD model and may
reduce measured Q; none may change between preparation and ringdown.

## 5. Element custody table

| Element | Stores energy? | Can retain waveform history? | Can inject after source-off? | Can leak phase reference? | Can mimic ringdown? | Independent kill/substitution |
|---|---|---|---|---|---|---|
| Quartz mechanical mode | Yes, elastic and kinetic | Yes, over measured ringdown | No independent source; it returns piezoelectric current | Only through its prepared phase | This is the claimed P0 state if controls close | Remove/replace resonator; off-resonance; disturb mount |
| Drive source and buffer | Yes, electrically | Yes, output stages and filters | Yes | Yes | Yes | Keep running upstream while isolate; terminate; substitute dummy |
| Analog phase gate | Parasitic charge only | Brief switching transient | Yes, through leakage/charge injection | Yes | Yes | Bypass/disable only in declared negative controls; dummy-load trace |
| Guarded relay barrier | Coil and parasitic electrical energy | Contact bounce/transient | Yes during transition; no lawful post-guard source path | Through stray capacitance | Yes | Independent contact witnesses; source-side observation; continuity/injection tests of the midpoint guard |
| 50 ohm termination | Negligible reactive energy if verified | No intended history | No | No intended reference | Only through parasitics | Substitute open/wrong termination negative arm |
| High-impedance differential voltage detector | Yes, input/protection and amplifier energy | Yes, finite impulse response and saturation recovery | Its passive input admittance loads the carrier; coherent input-referred drive is forbidden above the detector-only bound | Yes through supply/coupling | Yes | Detector-only replay; dummy capacitance; power/reference ablation; input-admittance calibration |
| Digitizer | Buffer memory only | Yes, serialized samples and trigger buffers | Cannot drive if input-only topology closes | Can correlate through shared clock | Yes in records | Zero-drive, replay, channel swap, native-byte inspection |
| Reference clock/model | No carrier energy | Yes, numerical phase history | Not physically connected to carrier after preparation | It is the declared gauge | Can manufacture apparent phase offline | Reference inversion, omission, and substitution controls |
| Controller | Electrical state | Yes, queues and buffers | Can command source/switches | Yes | Yes | Independent contact trace; controller-buffer replay control |
| Evidence recorder | File storage only | Yes | No physical drive path | Metadata can leak labels | Can fake persistence only in records | Hashes, blind arm IDs, synthetic-replay control |
| Mount/enclosure/environment | Mechanical and thermal energy | Yes | Yes, by vibration or thermal drift | No intended phase reference | Yes | Accelerometer plus temperature/RH trace; disturbance arm |

Any unlisted post-source energy store or signal path is a topology change and
invalidates the frozen packet.

## 6. Mechanically defined P0 object

```text
physical state:
    one antisymmetric quartz tuning-fork displacement/strain normal mode

preparation:
    exactly 32,768 phase-controlled cycles at calibrated f_ref

source disconnect:
    phase-synchronous analog route-to-termination, followed by witnessed
    de-energized relay series opens and guarded midpoint termination

query:
    fixed-electrode piezoelectric voltage projected onto a frozen complex
    reference by weighted least squares

observable:
    z(t) = I(t) + iQ(t), amplitude, unwrapped phase, frequency, tau_A, Q,
    usable cycles, and source-feedthrough bound

lifetime:
    measured tau_A and the consecutive post-source interval satisfying the
    frozen amplitude, phase-uncertainty, and feedthrough laws

disturbance:
    electrical loading, switch impulse, relay shock/EMI, mounting strain,
    temperature, enclosure vibration, and acoustic loading

reset operation:
    wait at least max(10 measured tau_A, 10 s) and until five consecutive
    analysis windows lie below 3 sigma of the frozen noise estimate
```

The reset operation is inter-arm hygiene only. Natural decay is not catalytic
restoration and authorizes no R2 or Wall claim.

## 7. Primary and official sources

Sources were read for architecture only; no vendor was contacted and no order
or cart action occurred.

- [Micro Crystal CC7V-T1A datasheet](https://www.microcrystal.com/fileadmin/Media/Products/32kHz/Datasheet/CC7V-T1A.pdf) — preferred reference carrier specifications.
- [Epson FC-135 application datasheet](https://download.epsondevice.com/td/pdf/app/FC-135_en.pdf) — independent 32.768 kHz quartz candidate with 70 kOhm typical ESR, 3.4 fF typical motional capacitance, and 1.0 uW absolute maximum drive.
- [Quartz tuning-fork resonance tracking and ringdown](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960650/) — primary literature for source-off transient characterization and demodulator cautions.
- [Murata 7BB-20-6 product page](https://www.murata.com/en-us/products/productdetail?partno=7BB-20-6) — fallback localized PZT/brass diaphragm class.

## 8. Selection boundary

This document selects a buildable carrier class and access model. It establishes
no resonance, post-source state, pi relation, computation, restoration, silicon
operation, bit replacement, or Wall crossing.
