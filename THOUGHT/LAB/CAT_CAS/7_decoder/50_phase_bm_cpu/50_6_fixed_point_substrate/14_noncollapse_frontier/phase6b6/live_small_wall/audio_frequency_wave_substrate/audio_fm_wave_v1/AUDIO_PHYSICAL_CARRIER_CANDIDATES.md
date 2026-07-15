# Audio Physical Carrier Candidates

Status: `REVIEWED_CANDIDATE_SET_NO_PHYSICAL_FREEZE`

No carrier is frozen for construction or operation in this offline lane. A candidate
survives only when its physical state, source disconnect, query, observable, lifetime,
disturbance, and restoration operation can be mechanically stated.

## Comparison

| Candidate | Post-source energy state | Disconnect quality | Query quality | Restoration burden | Decision |
| --- | --- | --- | --- | --- | --- |
| Direct DAC-to-ADC loopback | None beyond filters/buffers | Poorly interpretable | Calibration stimulus only | Buffer drain | Reject as carrier; retain as calibration |
| Passive electrical resonator | L/C electric and magnetic energy | Strong with analog switch/relay | Weak gated probe or coupled pickup | Discharge/damp plus baseline tomography | Primary survivor |
| Electromechanical resonator | Modal displacement and velocity | Strong with open-circuit drive gate | Piezo/coil probe or weak impulse | Active damping plus modal baseline | Secondary survivor |
| Sealed acoustic cavity/tube | Acoustic pressure and particle-velocity modes | Strong with sealed source port gate | Weak speaker/piezo query, microphone readout | Absorption/counterdrive plus modal baseline | Later survivor |
| Feedback/delay loop | Samples/charge in active loop and delay elements | Source separable, active energy remains | Injection at summing node | Loop mute, drain, reset, baseline | Control/platform survivor only |

## Uniform Candidate Field Matrix

`UNFROZEN` means the field is explicitly a blocker rather than an omitted assumption.

| Required field | Direct loopback | Passive electrical | Electromechanical | Sealed cavity/tube | Feedback/delay |
| --- | --- | --- | --- | --- | --- |
| Physical state | filter/cable/buffer transient | capacitor voltage and inductor current | modal displacement and velocity | pressure and particle-velocity modes | loop samples/charge and active gain state |
| Source operation | DAC frames | bounded waveform through fixed impedance | bounded coil/piezo force | bounded internal transducer burst | gated initial summing-node injection |
| Mechanical disconnect | drain queue and open DAC path | break-before-make conductor isolation | open drive port, pickup remains | open electrical drive plus frozen termination | isolate source injection, loop remains powered |
| Query | calibration stimulus only | weak separate-port impulse/burst | weak separate pickup-port force | weak separate transducer burst | bounded receiver-node injection |
| Observable | ADC samples | differential voltage plus current/pickup | pickup voltage plus displacement/velocity | two pressure/velocity locations | two loop taps plus energy |
| Lifetime | filter/queue impulse response | measured ring-down to SNR floor | mode-specific ring-down | mode-specific environmental ring-down | gain/delay/quantization lifetime |
| Disturbance/loading | converter filters and ADC load | probe impedance changes loaded Q | pickup and air/mount damping | microphone/query acoustic impedance | tap loading and query injection |
| Nonlinearity | converter clipping/filter response | ESR, switch charge, saturation, buffer clipping | hysteresis and transducer distortion | transducer distortion and acoustic nonlinearity | saturation, quantization, active gain |
| Restoration | drain/flush only | damping/counterdrive plus natural/wrong controls | active damping plus modal tomography | absorber/counterdrive plus termination controls | mute/open/drain/reset plus baseline |
| Hardware definition | interface/queue inventory UNFROZEN | schematic/components/switch truth table UNFROZEN | element/mount/ports UNFROZEN | geometry/transducers/terminations UNFROZEN | loop topology/delay/gain/reset UNFROZEN |
| Ordinary explanation | direct transduction and buffering | linear loaded RLC plus weak nonlinearity | damped coupled oscillator | linear modes, multipath, distortion | ordinary powered active memory |
| Capacity observable | queue depth/filter order | modal covariance/effective rank/precision | persistent modal effective rank | persistent acoustic modal rank | delay length/effective rank/quantization |
| Noise risks | clock/quantization/buffer timing | Johnson, switch, ADC, supply, component drift | thermal/mechanical, mount, pickup noise | microphone, environmental, boundary drift | active noise, clock jitter, quantization |
| Safety freeze | no carrier use | max voltage/current/energy and timeout UNFROZEN | max force/displacement/temperature UNFROZEN | max acoustic exposure/pressure UNFROZEN | loop gain/stability/saturation timeout UNFROZEN |
| Side information | timestamps/queue state | route, switch state, component calibration | mount, pickup, temperature | geometry, termination, temperature/humidity | delay/gain/reset/clock state |

No physical freeze is lawful while any selected-candidate `UNFROZEN` entry remains.

## A. Direct DAC-to-ADC Loopback

Physical state: converter reconstruction filters, anti-alias filters, cable capacitance,
and interface buffers. Source operation: DAC output. Disconnect: gate or unplug the DAC
path and drain every software/hardware queue. Query: a later receiver stimulus cannot
meaningfully interrogate an independent persistent loopback state. Observable: ADC
samples. Lifetime: ordinarily only filter impulse response or queued frames.

Measurement disturbance: ADC loading is small but the pipeline timing dominates.
Restoration: drain/flush buffers and verify impulse response baseline. Ordinary
explanation: direct transduction, filter memory, and interface buffering.

Decision: instrumentation calibration only. It is not a carrier unless a separately
identified resonant or delay element is inserted and isolated.

## B. Passive Electrical Resonator - Primary Survivor

Physical state: capacitor voltage and inductor current, generalized to coupled modal
coordinates `(v_Ck, i_Lk)`. Source operation: a bounded prepared waveform drives the
resonator through a fixed source impedance. Source disconnect: a break-before-make
analog switch or relay opens the drive conductor and independently exposes the query
port. Merely setting DAC samples to zero is not disconnect.

Query operation: a low-energy, precommitted impulse, phase-coded burst, or weak coupled
probe enters through a separate resistor/coupler. Observable: differential voltage and,
if available, a second current or pickup coordinate. Lifetime: measured ring-down to a
predeclared SNR floor, not inferred from nominal Q.

Measurement disturbance: probe impedance and ADC loading alter Q and phase; loaded and
unloaded transfer functions must be compared. Nonlinearity: component ESR, switch
charge injection, inductor saturation, and amplifier clipping if an active buffer is
used. Restoration: controlled damping resistor or phase-inverted drive, followed by
time-matched natural relaxation, no-restoration, wrong-inverse, and carrier-off arms.

Minimum hardware definition before freeze:

```text
schematic and component tolerances
separate source/query/read ports
break-before-make switch truth table
maximum voltage/current/energy
loaded transfer-function baseline
ring-down floor and safe timeout
two-observable restoration plan
buffer-drain and source-off proof
```

Ordinary explanation: linear RLC impulse response or weak component nonlinearity. A
future result must first be interpreted in that ordinary model.

## C. Electromechanical Resonator - Secondary Survivor

Physical state: modal displacement and velocity of a piezo, beam, plate, diaphragm, or
spring-mass element. Source operation: bounded coil or piezo drive. Disconnect: open the
drive port with a mechanical/electronic gate while leaving an independent pickup path.
Query: weak impulse, phase-coded force, or electrical probe through a separate electrode
or pickup coil. Observable: pickup voltage plus displacement/velocity when available.

Lifetime: modal ring-down measured over temperature and mounting conditions.
Measurement disturbance: pickup loading, air damping, fixture coupling, and query force.
Restoration: active damping/counterdrive followed by both pickup and mechanical modal
tomography. Ordinary explanation: damped coupled oscillators and transducer hysteresis.

Blocker before freeze: exact element, mounting geometry, independent source/query
ports, safety energy, and second observable are not selected.

## D. Sealed Acoustic Cavity Or Tube - Later Survivor

Physical state: pressure and particle-velocity modal coordinates fixed by sealed
geometry. Source operation: bounded internal transducer burst. Disconnect: electrical
source gate plus mechanically sealed port; software mute is insufficient. The source
transducer termination must be frozen as open, shorted, or damped because its residual
electromechanical impedance changes cavity modes and Q. Query: a separate weak
transducer or piezo injects a precommitted burst. Observable: microphone pressure and
preferably a second pressure/velocity location.

Lifetime: mode-specific ring-down with temperature/humidity logged. Measurement
disturbance: microphone and query-transducer acoustic impedance. Restoration: absorber
gate or counterdrive, compared with natural relaxation and wrong inverse. Open, short,
and damped source-termination arms must be included in baseline, ring-down, carrier-off,
and restoration campaigns. Ordinary explanation: linear room/tube modes, transducer
distortion, termination-dependent loading, and multipath.

Decision: survives architecture review but is not first prototype. Environmental drift
and uncontrolled multipath make causal separation harder than an electrical resonator.

## E. Feedback Or Delay Loop - Active Control Survivor

Physical state: charge or samples circulating in a declared physical/mixed-signal loop.
Source operation: initial injection through a gated summing node. Disconnect: isolate
the source injection path while the loop remains powered. Query: separate bounded
injection at a receiver node. Observable: two tapped loop coordinates and loop energy.

Lifetime: determined by loop gain, delay, quantization, and saturation. Measurement
disturbance: tap loading and active gain noise. Restoration: mute injection, open loop,
drain/reset delay elements, verify zero-state impulse response, and compare carrier-off
and wrong-reset controls. Ordinary explanation: active memory powered by the loop.

Decision: useful as a controllable active-memory comparison, not evidence that a
passive carrier persisted or borrowed computation.

## Surviving Order

```text
1. passive electrical resonator
2. electromechanical resonator
3. sealed acoustic cavity or tube
4. feedback or delay loop as an active-memory control
```

Direct loopback is rejected as a carrier. None is authorized for hardware operation.
