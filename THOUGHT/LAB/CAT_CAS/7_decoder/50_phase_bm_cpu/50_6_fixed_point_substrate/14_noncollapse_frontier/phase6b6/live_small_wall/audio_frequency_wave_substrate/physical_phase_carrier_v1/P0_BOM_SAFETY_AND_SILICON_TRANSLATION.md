# P0 BOM, Safety, and Silicon Translation

**Status:** `FROZEN_ARCHITECTURE__NOT_EXECUTED`<br>
**Purchases:** 0<br>
**Cart actions:** 0<br>
**Hardware contacts:** 0

## 1. Non-purchasing BOM

This is an architecture bill of materials, not an order list. Availability and
specifications were checked from manufacturer material on 2026-07-16. Every
final part, suffix, revision, limit, and datasheet hash must be confirmed again
under a later explicit authority.

| Function | Frozen class / reference candidate | Current specification used by architecture | Requirement before execution |
|---|---|---|---|
| Preferred carrier | Micro Crystal CC7V-T1A-class hermetic 32.768 kHz quartz tuning-fork crystal | 32.768 kHz; 50/70 kOhm typical/max series resistance; 3.7 fF typical C1; 1.2 pF typical C0; 1.0 uW max drive; vacuum-sealed ceramic package | **CONFIRM:** exact ordering suffix, load capacitance, tolerance, temperature grade, max drive, ESR, C0/C1, revision, SHA-256 |
| Preferred-class alternate | Epson FC-135-class 32.768 kHz crystal | 32.768 kHz; 70 kOhm typical ESR; 3.4 fF typical C1; 1.0 pF typical C0; 0.01-0.5 uW operating drive and 1.0 uW absolute max | **CONFIRM:** exact suffix and all limits; alternate does not silently replace the selected reference |
| Fallback carrier | Murata 7BB-20-6-class external-drive PZT/brass diaphragm | 6.3 kHz nominal, plus/minus 0.6 kHz; 350 Ohm resonant impedance; 10 nF at 1 kHz; 20 mm brass plate | **CONFIRM:** current approval sheet, drive limit, Q, mount, acoustic enclosure; requires a new frozen calibration envelope |
| Phase gate | ADG1419BRMZ, 8-lead MSOP, +/-5 V, IN-controlled SPDT | 4.5 Ohm typical on resistance; break-before-make; full-temperature transition max 560 ns; MSOP has no EN; nonzero off capacitance and charge injection | **CONFIRM:** exact revision, rails, IN truth table, loaded transition, charge injection, isolation at 32.768 kHz, leakage, SHA-256 |
| Physical barrier | Three Omron G6K-2F-Y-class DPDT low-signal relays | 3 ms max operate; 3 ms max release; 3 ms max bounce; 1,000 MOhm min insulation at 500 VDC | **CONFIRM:** exact coil/contact form, break-before-make behavior, contact witness wiring, revision |
| Terminations | 50.00 Ohm, 0.1%, low-inductance resistors; guarded midpoint network | Source route and source-side midpoint guard terminate at the single frozen star reference | Measure each resistor before topology seal; bind lot and calibration |
| Current limiting | 100 kOhm minimum series resistance, 0.1% | Fault/current limiting only; it does not by itself guarantee motional current or power | Measure actual value and require calibrated drive to satisfy the lowest voltage, current, and power cap |
| Sense stage | Guarded ADA4530-1-class electrometer voltage buffer, used as an always-connected differential-to-reference input | Effective `Rin>=100 MOhm`, total carrier-side `Cin<=3.0 pF`; ADA4530-1 reference has 2 MHz unity-gain crossover, 6 MHz typical closed-loop bandwidth, and <=20 fA input bias through 85 C; a manufacturer application note reports about 2 pF on its guarded buffer board | Freeze exact dual/single-ended implementation, input/bias/protection admittance, cable capacitance, transfer function, coherent input-referred backaction, settling <=10 us, noise, revision, SHA-256 |
| Source | Phase-coherent low-voltage sine generator | Exact 32,768-cycle preparation, 0/pi phase command, <=1.000 us marker uncertainty | Freeze output impedance, clock accuracy, phase-command latency, waveform-memory behavior |
| Acquisition | Four-channel simultaneous differential digitizer or oscilloscope | Exactly 1 MSa/s/channel, >=16 bits, DC coupled, 3,101,000 samples/channel, nominal command index 1,101,000, native raw export, no averaging | Freeze ranges, skew <=0.100 us and corrected residual <=0.020 us, trigger uncertainty <=1 sample, differential/common-mode and ground limits, native parser/hash |
| Reference/calibration instrument | Zurich Instruments MFLI-class lock-in / impedance analyzer is an optional reference, not the sole raw recorder | DC-500 kHz base range; 60 MSa/s, 16-bit input conversion; differential voltage/current inputs; optional impedance analysis | If used, bind options and disable demodulator output as primary evidence; full four-channel raw custody still required |
| Mount | Low-strain carrier PCB in a closed conductive enclosure | Resonator package stationary; relay board physically separate by >=0.50 m of cable path | Freeze solder process, board strain relief, enclosure geometry, cable identity |
| Shielding | Conductive enclosure, guarded cabling, single-point analog ground | No speaker, microphone, acoustic drive, wireless link, or floating reference | Photograph and continuity-test topology |
| Environment | Calibrated temperature sensor, calibrated RH sensor, and enclosure accelerometer are all required | Temperature 20.00-30.00 C with matched mean delta <=0.20 C; RH 20.0-60.0 % with matched mean delta <=2.0 percentage points; demeaned raw CH3 RMS <=0.050 m/s^2, peak <=0.500 m/s^2, matched RMS delta <=0.010 m/s^2; temperature/RH exactly 10 samples/s | Bind every sensor calibration, master-trigger alignment, placement, units, clipping limit, and cadence |

### Manufacturer and instrumentation links

- [Micro Crystal CC7V-T1A](https://www.microcrystal.com/en/products/khz-quartz-crystals/cc7v-t1a)
- [Micro Crystal CC7V-T1A datasheet](https://www.microcrystal.com/fileadmin/Media/Products/32kHz/Datasheet/CC7V-T1A.pdf)
- [Epson FC-135](https://www.epsondevice.com/crystal/en/products/crystal-unit/fc135.html)
- [Epson FC-135 application datasheet](https://download.epsondevice.com/td/pdf/app/FC-135_en.pdf)
- [Murata 7BB-20-6](https://www.murata.com/en-us/products/productdetail?partno=7BB-20-6)
- [Analog Devices ADG1419](https://www.analog.com/en/products/adg1419.html)
- [Analog Devices ADA4530-1](https://www.analog.com/en/products/ada4530-1.html)
- [Analog Devices AN-1373 guarded-buffer capacitance note](https://www.analog.com/en/resources/app-notes/an-1373.html)
- [Omron G6K relay family](https://components.omron.com/eu-en/products/relays/G6K)
- [Zurich Instruments MFLI](https://www.zhinst.com/ch/en/products/mfli-lock-in-amplifier/)
- [MFLI June 2025 product leaflet](https://www.zhinst.com/sites/default/files/documents/2025-07/zi_mfli_leaflet_v2.pdf)

No vendor contact, quote request, inventory reservation, sample request, or
purchase occurred.

## 2. Conservative electrical and mechanical limits

These limits are architecture caps. A later final datasheet may require lower
values and can never be overridden by this packet.

| Quantity | P0 architecture cap | Reason / confirmation law |
|---|---:|---|
| Quartz terminal amplitude | <=0.200 Vpp | Conservative relative to reference 1.0 uW max drive; **CONFIRM** from final BVD calibration |
| Estimated motional power | <=0.100 uW | Tenfold below reference max; use the lower of voltage and power caps |
| Estimated motional current | <=2.000 uA rms | Motional-branch cap; **CONFIRM** from loaded BVD fit |
| Generator output | <=1.000 Vpp | Includes current-limiting network; no offset |
| Source current under any normal state | <=5.000 uA rms | 100 kOhm minimum series limit |
| Analog signal rails | <=plus/minus 5.0 V | Low-voltage bench envelope even if switch supports more |
| Relay coil supply | exactly rated 5 V class, current-limited | **CONFIRM** exact coil suffix and suppression |
| Detector output and digitizer input | remain within 80% of selected full scale | Any clipping kills the arm |
| Preparation on-time | exactly `32768/f_ref` s | Exactly 32,768 calibrated carrier cycles; near 1 s, not asserted exactly 1 s |
| Inter-arm wait | >=max(10*tau_A, 10 s) | Return-to-noise hygiene, not restoration |
| Duty cycle over any 60 s | <=10% drive-on | Conservative heating and aging bound |
| Ambient temperature | 20-30 C | Within ordinary lab range |
| Matched-pair temperature delta | <=0.20 C | Quartz frequency is temperature sensitive |
| Ambient RH | 20.0-60.0 %RH | Required calibrated environment channel; equality passes |
| Matched-pair mean RH delta | <=2.0 percentage points | Rejects unmatched humidity exposure |
| Enclosure acceleration RMS | <=0.050 m/s^2 after raw-sample demeaning | No filter or spectral selection; equality passes |
| Enclosure acceleration peak | <=0.500 m/s^2 absolute demeaned sample | Rejects shocks and handling |
| Matched-pair acceleration RMS delta | <=0.010 m/s^2 | Rejects unmatched mechanical disturbance |
| Enclosure motion | no handling or cable movement from calibration through pair | Numeric acceleration gates remain mandatory |
| Mechanical package stress | no clamp, bend, lid contact, or manual load on crystal package | **CONFIRM** final mounting and solder rules |

### Electrical safety

1. No exposed mains, high voltage, floating earth, or defeated protective earth.
2. One documented analog-ground star; shields terminate as frozen in topology.
3. The source path contains the measured current limiter before the carrier.
4. Detector input protection must not add a drive path and must remain below carrier
   leakage/settling limits.
5. Relay coils use bounded suppression; suppression current cannot share the
   mechanical-sense return.
6. The fail-safe state with controller power removed is source-off.
7. Instrument input ratings, common-mode limits, and ground references must be
   confirmed from final manuals before connection.
8. Any topology continuity or insulation failure blocks power.

### ESD and handling

ESD handling is permitted only under later authority with all instruments and
the source unpowered. Use grounded ESD controls compatible with the final
device datasheet. Do not use relay or crystal dielectric-test voltages as
operating limits. No shock, vibration, or stress qualification value authorizes
intentional mechanical loading.

### Fallback safety

The PZT/brass fallback is not executable under the quartz packet. Its
capacitance, drive limit, mounting stress, acoustic radiation, and stored
energy require a new final-datasheet and calibration freeze. It may not inherit
the quartz voltage, current, power, guard, or uncertainty thresholds.

## 3. Silicon transposition

| Target class | Mechanical state | Preparation interface | Readout interface | Source-off equivalent | Phase reference | Ringdown observable | Pi preparation | Likely P1 coupling | Fabrication/access barrier |
|---|---|---|---|---|---|---|---|---|---|
| Silicon MEMS resonator | Flexural, extensional, disk, beam, or contour elastic eigenmode | Electrostatic, thermal, or bonded/thin-film piezo drive | Capacitive, piezoresistive, or optical displacement | Open/terminate drive electrode; remove bias if required and separately bounded | Electrical clock or optical local oscillator | Displacement/current decay and phase | Invert the phase of the resonant preparation | Electrostatic spring, mechanical link, or shared substrate mode | Micron gaps, parasitics, packaging, vacuum, bias feedthrough |
| Silicon phononic-crystal cavity | Localized elastic defect mode inside a phononic band gap | Thin-film piezoelectric electrode or optical force | Piezoelectric charge or optical cavity shift | Isolate electrode or extinguish optical pump with independent witness | Electrical or optical phase reference | Localized-mode I/Q and amplitude decay | Half-turn of electrical or optical drive | Evanescent elastic coupling or engineered waveguide | Nanofabrication, release, anchor loss, electrode loss, mode identification |
| Silicon optomechanical resonator | Silicon mechanical mode coupled to an optical cavity | Radiation pressure, photothermal force, or external piezo | Homodyne/heterodyne optical displacement | Independent optical shutter/AOM plus pump monitor; bound cavity photon decay | Optical local oscillator | Optical phase/displacement ringdown | Optical-drive phase inversion | Optical bus, mechanical bridge, or radiation-pressure interaction | Laser noise, optical cavity access, alignment, heating, source/cavity separation |
| Hybrid piezoelectric-on-silicon resonator | Silicon or composite elastic mode with localized piezoelectric transducer | Direct electrode drive through AlN, LiNbO3, quartz, or related film | Direct charge/current or optional optical cross-check | Guarded electrode isolation closely matching P0 | Electrical phase clock with independent switch witness | Piezoelectric current/displacement I/Q and decay | Voltage half-turn at the electrode | Elastic waveguide, mechanical bridge, tunable electrode coupling | Thin-film integration, electrode loss, foundry access, package parasitics |

## 4. Selected future silicon target

```text
hybrid piezoelectric-on-silicon phononic-crystal cavity
```

It is the most plausible future transposition because it preserves P0's direct
electrical phase preparation, source-isolation, and phase-sensitive electrical readout while moving
the stored state into a deliberately localized silicon elastic mode. It avoids
making an optical cavity mandatory for the first silicon translation, while an
optical readout can remain an independent custody cross-check.

Primary literature demonstrates the relevant architecture rather than this
project's result:

- [Direct piezoelectric excitation of a silicon phononic-crystal slab resonator](https://doi.org/10.1016/j.sna.2011.03.014) reports silicon modes at 120 and 129 MHz with measured Q of 3,600 and 10,000.
- [High-Q silicon optomechanical resonators in air](https://arxiv.org/abs/1109.4705) demonstrates optical transduction of silicon mechanical modes up to 1.35 GHz with mechanical Q around 4,000.
- [Thin-film quartz piezoelectric phononic-crystal resonators](https://arxiv.org/abs/2406.14660) demonstrates direct piezoelectric phononic access and ringdown at a much more demanding scale.

These sources establish feasibility of carrier classes only. They do not
transfer any physical, computational, restoration, or silicon claim to P0.

## 5. Boundary

This packet defines what a later build would require. It does not authorize
purchase, fabrication, assembly, wiring, powering, measurement, playback,
recording, ADC/DAC use, or target contact.
