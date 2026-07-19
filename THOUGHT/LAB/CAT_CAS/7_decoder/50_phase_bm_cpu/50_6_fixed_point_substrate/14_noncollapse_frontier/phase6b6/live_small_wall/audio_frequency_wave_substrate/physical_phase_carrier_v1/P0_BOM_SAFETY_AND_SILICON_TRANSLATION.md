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
| Preferred carrier | Epson `Q13FC1350000401` FC-135 hermetic 32.768 kHz quartz tuning-fork crystal | 32.768 kHz; 12.5 pF load; 70 kOhm maximum ESR; 0.5 uW maximum drive; suffix `01` is the any-quantity tape-cut packing code; the private local capture matches the retained legacy byte count/SHA-256, while the PDF itself remains uncommitted | **CONFIRM:** acquired marking, packing code, lot, revision, SHA-256, measured resonance and linearity under later authority |
| Preferred-class alternate | Micro Crystal CC7V-T1A-class 32.768 kHz crystal | Architecture reference only; not a permitted substitution in P0 netlist rev C | Requires a new reviewed netlist/BOM root and cannot silently replace the selected Epson identity |
| Fallback carrier | Murata 7BB-20-6-class external-drive PZT/brass diaphragm | 6.3 kHz nominal, plus/minus 0.6 kHz; 350 Ohm resonant impedance; 10 nF at 1 kHz; 20 mm brass plate | **CONFIRM:** current approval sheet, drive limit, Q, mount, acoustic enclosure; requires a new frozen calibration envelope |
| Phase gate | ADG1419BRMZ, 8-lead MSOP, +/-5 V, IN-controlled SPDT | 4.5 Ohm typical on resistance; break-before-make; full-temperature transition max 560 ns; MSOP has no EN; nonzero off capacitance and charge injection | **CONFIRM:** exact revision, rails, IN truth table, loaded transition, charge injection, isolation at 32.768 kHz, leakage, SHA-256 |
| Physical barrier | Three Omron G6K-2F-Y-class DPDT low-signal relays | 3 ms max operate; 3 ms max release; 3 ms max bounce; 1,000 MOhm min insulation at 500 VDC | **CONFIRM:** exact coil/contact form, break-before-make behavior, contact witness wiring, revision |
| Terminations | 50.00 Ohm, 0.1%, low-inductance resistors; guarded midpoint network | Source route and source-side midpoint guard terminate at the single frozen star reference | Measure each resistor before topology seal; bind lot and calibration |
| Current limiting | 100 kOhm minimum series resistance, 0.1% | Fault/current limiting only; it does not by itself guarantee motional current or power | Measure actual value and require calibrated drive to satisfy the lowest voltage, current, and power cap |
| Sense stage | Guarded Texas Instruments `OPA810IDT` unity-gain buffer | 12 GOhm || 2 pF common-mode plus 0.5 pF differential typical; hard as-built gates `Rin,U95>=100 MOhm` and `Cin,U95<=4.00 pF` | Bind SBOS799E SHA-256, exact SOIC pin map, guarded land, complete coupon admittance, transfer function, recovery and coherent backaction |
| Sense input protection | No external clamp or series component on `N_ELECTRODE_A` | External protection is intentionally omitted because added leakage/capacitance would alter the carrier; normal drive is bounded by the upstream 100 kOhm limiter and K1/K2 barriers; OPA810 absolute/ESD ratings are not operating permissions | Grounded ESD handling with source, digitizer and cables absent; reject any added electrode component or failed as-built admittance gate |
| Source | SIGLENT `SDG1032X` phase-coherent dual-channel generator | `HIGH_Z` load mode with 50 Ohm physical outputs; continuous C1 32.768 kHz, 0.400 Vpp, 0 V, 0/pi command; continuous C2 65.536 kHz, 0.100 Vpp, 0 V, fixed zero phase; both remain on for the whole record | Query back model, serial, firmware, load, impedance, waveform, frequency, amplitude, offset, phase, mode and output state per channel; calibrate accuracy, common-reference behavior, phase-command latency and CH0 persistence |
| Acquisition | Spectrum `DN2.592-04` four-simultaneous-channel true-differential digitizer | Exactly 1 MSa/s/channel, 16 bits, DC coupled, 3,101,000 samples/channel, software-prearmed free run, native raw export, no averaging and no external trigger | Freeze ranges, input admittance, skew <=0.100 us and corrected residual <=0.020 us, differential/common-mode and ground limits, native parser/hash |
| Reference/calibration instrument | Zurich Instruments MFLI-class lock-in / impedance analyzer is an optional reference, not the sole raw recorder | DC-500 kHz base range; 60 MSa/s, 16-bit input conversion; differential voltage/current inputs; optional impedance analysis | If used, bind options and disable demodulator output as primary evidence; full four-channel raw custody still required |
| Mount | Low-strain carrier PCB in a closed conductive enclosure | Resonator package stationary; control and carrier enclosures are joined by the exact fixed 150 +/-2 mm harness; no contradictory 0.50 m separation claim remains | Freeze solder process, board strain relief, enclosure geometry, cable identity |
| Shielding | Conductive enclosure, guarded cabling, single-point analog ground | No speaker, microphone, acoustic drive, wireless link, or floating reference | Photograph and continuity-test topology |
| Environment | Exact SHT45 raw-word/CRC record and ADXL354 analog Z channel are required | Temperature 20.00-30.00 C with matched mean delta <=0.20 C; RH 20.0-60.0 % with matched mean delta <=2.0 points; raw CH3 RMS <=0.050 m/s^2 and peak <=0.500 m/s^2; temperature/RH exactly 10 samples/s | Bind sensor serial, command, raw words, CRC-8, monotonic/UTC map, calibration, placement, units, clipping and cadence |

### Manufacturer and instrumentation links

- [Micro Crystal CC7V-T1A](https://www.microcrystal.com/en/products/khz-quartz-crystals/cc7v-t1a)
- [Micro Crystal CC7V-T1A datasheet](https://www.microcrystal.com/fileadmin/Media/Products/32kHz/Datasheet/CC7V-T1A.pdf)
- [Epson FC-135](https://www.epsondevice.com/crystal/en/products/crystal-unit/fc135.html)
- [Epson FC-135 application datasheet](https://download.epsondevice.com/td/pdf/app/FC-135_en.pdf)
- [Murata 7BB-20-6](https://www.murata.com/en-us/products/productdetail?partno=7BB-20-6)
- [Analog Devices ADG1419](https://www.analog.com/en/products/adg1419.html)
- [Texas Instruments OPA810](https://www.ti.com/product/OPA810)
- [Texas Instruments OPA810 SBOS799E](https://www.ti.com/lit/ds/symlink/opa810.pdf)
- [Omron G6K relay family](https://components.omron.com/eu-en/products/relays/G6K)
- [Zurich Instruments MFLI](https://www.zhinst.com/ch/en/products/mfli-lock-in-amplifier/)
- [MFLI June 2025 product leaflet](https://www.zhinst.com/sites/default/files/documents/2025-07/zi_mfli_leaflet_v2.pdf)

No human vendor outreach, quote request, inventory reservation, sample request, or
purchase occurred.

## 2. Conservative electrical and mechanical limits

These limits are architecture caps. A later final datasheet may require lower
values and can never be overridden by this packet.

| Quantity | P0 architecture cap | Reason / confirmation law |
|---|---:|---|
| C1 source command | exactly 0.400 Vpp in `HIGH_Z` load mode, 0 V offset | Prospective frozen command; any queryback mismatch stops |
| C2 gauge command | exactly 0.100 Vpp in `HIGH_Z` load mode, 0 V offset | C2 is not an intended carrier drive, but its bounded passive coupling through the CH0 summing network, finite C1 source impedance and limiter must be swept and bounded; any queryback mismatch stops |
| Quartz terminal amplitude | complete-corner prospective <=0.187807 Vpp; hard cap <=0.200 Vpp | The loaded BVD model includes the exact 100 kOhm `N_SRC` drive shunt and both resistor tolerances; **CONFIRM** from final as-built loaded BVD calibration |
| Estimated motional power | complete-corner prospective <=0.002298 uW; hard cap <=0.100 uW | Below the selected carrier's 0.5 uW maximum-drive identity; use the lower of voltage, current and power caps |
| Estimated motional current | complete-corner prospective <=0.186507 uA rms; hard cap <=2.000 uA rms | Motional-branch cap; **CONFIRM** from the as-built loaded BVD fit |
| Generator output | C1 exactly 0.400 Vpp; C2 exactly 0.100 Vpp; both 0 V offset | The general safety ceiling remains 1.000 Vpp, but it does not authorize changing the frozen commands |
| Source current under any normal state | <=5.000 uA rms | 100 kOhm minimum series limit |
| Analog signal rails | <=plus/minus 5.0 V | Low-voltage bench envelope even if switch supports more |
| Relay coil supply | exactly rated 5 V class, current-limited | **CONFIRM** exact coil suffix and suppression |
| Detector output and digitizer input | remain within 80% of selected full scale | Any clipping kills the arm |
| Qualified preparation interval | exactly the final `32768/f_ref` s before `n_gate` | Both source channels are already continuously stable before recording; this is an analysis interval, not a self-ending burst |
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

## 3. Research custody and simulation boundary

The canonical repository-safe research dependency is
`research/P0_research_bundle_2026-07-18`, imported from source commit
`cb53976612cbe83bec82df826a9889418f7e0b89`. Its manifest contains 35 source
records. The bound custody snapshot records 11 private locally hash-verified
captures, 6 URL-plus-legacy-hash records whose bytes are absent, and 18 manual
captures. Metadata and hashes are candidate inputs; third-party PDFs, HTML,
models, download receipts, and generated archives remain ignored and
uncommitted. From the bundle directory, refresh with the repository virtual
environment using `python scripts/download_sources.py --all`, then
`python scripts/verify_downloads.py` and
`python scripts/build_custody_snapshot.py`.

The deterministic waveform generator remains a signal-level analyzer test, not
a physical observation. The separate root-bound circuit model now closes the
prospective source path over 131,072 complete binary corners per candidate: the
100 kOhm source limiter and `N_SRC` shunt, ADG1419 D/SA/SB nodes, relay
states/parasitics, FC-135 loaded-frequency BVD network, OPA810 input/output,
digitizer differential/common-mode loading, cable/PCB/enclosure capacitance,
noise, ADC quantization, and independent 1 MOhm injection tolerance. It also
freezes detector-only and exact-1-pF controls. A future physical result still
requires as-built calibration and raw-byte custody; one favorable synthetic
record is not evidence that the physical circuit produced it.

## 4. Silicon transposition

| Target class | Mechanical state | Preparation interface | Readout interface | Source-off equivalent | Phase reference | Ringdown observable | Pi preparation | Likely P1 coupling | Fabrication/access barrier |
|---|---|---|---|---|---|---|---|---|---|
| Silicon MEMS resonator | Flexural, extensional, disk, beam, or contour elastic eigenmode | Electrostatic, thermal, or bonded/thin-film piezo drive | Capacitive, piezoresistive, or optical displacement | Open/terminate drive electrode; remove bias if required and separately bounded | Electrical clock or optical local oscillator | Displacement/current decay and phase | Invert the phase of the resonant preparation | Electrostatic spring, mechanical link, or shared substrate mode | Micron gaps, parasitics, packaging, vacuum, bias feedthrough |
| Silicon phononic-crystal cavity | Localized elastic defect mode inside a phononic band gap | Thin-film piezoelectric electrode or optical force | Piezoelectric charge or optical cavity shift | Isolate electrode or extinguish optical pump with independent witness | Electrical or optical phase reference | Localized-mode I/Q and amplitude decay | Half-turn of electrical or optical drive | Evanescent elastic coupling or engineered waveguide | Nanofabrication, release, anchor loss, electrode loss, mode identification |
| Silicon optomechanical resonator | Silicon mechanical mode coupled to an optical cavity | Radiation pressure, photothermal force, or external piezo | Homodyne/heterodyne optical displacement | Independent optical shutter/AOM plus pump monitor; bound cavity photon decay | Optical local oscillator | Optical phase/displacement ringdown | Optical-drive phase inversion | Optical bus, mechanical bridge, or radiation-pressure interaction | Laser noise, optical cavity access, alignment, heating, source/cavity separation |
| Hybrid piezoelectric-on-silicon resonator | Silicon or composite elastic mode with localized piezoelectric transducer | Direct electrode drive through AlN, LiNbO3, quartz, or related film | Direct charge/current or optional optical cross-check | Guarded electrode isolation closely matching P0 | Electrical phase clock with independent switch witness | Piezoelectric current/displacement I/Q and decay | Voltage half-turn at the electrode | Elastic waveguide, mechanical bridge, tunable electrode coupling | Thin-film integration, electrode loss, foundry access, package parasitics |

## 5. Selected future silicon target

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

## 6. Boundary

This packet defines what a later build would require. It does not authorize
purchase, fabrication, assembly, wiring, powering, measurement, playback,
recording, ADC/DAC use, or target contact.
