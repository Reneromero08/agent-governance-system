# P0 Build-Readiness Exact Component Seed

**Status:** `P0_BUILD_READINESS_BLOCKED`<br>
**Authority:** `AUTHORIZE P0 BUILD-READINESS ONLY`  
**Operation:** exact-component research and handoff only  
**Purchasing authority:** none  
**Hardware authority:** none

This file records exact candidates that satisfy or nearly satisfy the frozen P0 architecture.
The local build-readiness qualification must independently verify current manufacturer
material, bind downloaded documents by SHA-256, close every stated caveat, and either
freeze or reject each candidate. This file is not an order list.

## 1. Carrier identity

### Exact build carrier candidate

```text
Manufacturer: Epson
Family:       FC-135
Frequency:    32.768 kHz
Load:         12.5 pF
Product code: Q13FC1350000401
Packing code: 01 = any quantity, vinyl bag (tape cut)
```

This is an explicit same-class substitution for build readiness. It remains a hermetic,
electrode-addressed 3.2 mm x 1.5 mm x 0.9 mm tuning-fork quartz unit. It does not change
the declared mechanical carrier, source-off law, phase observable, or silicon translation.
The Micro Crystal CC7V-T1A remains predecessor comparison context because its numeric
`20xxxx` ordering identity is generated per specification and has no public universal order
number. It is not a permitted P0 rev C substitution. The selected Epson exact-product brief,
not the Micro Crystal comparison, supplies the prospective carrier limits.

The exact Micro Crystal comparison specification is:

```text
CC7V-T1A / 32.768 kHz / 12.5 pF / +/-20 ppm / TA / QC
```

The final packet selects Epson `Q13FC1350000401`. No primary execution threshold is
inherited from the Micro Crystal comparison; Epson-specific loaded characterization and
conservative limits are frozen prospectively, and as-built characterization remains a
separate later gate.

Official sources:

- https://www.epsondevice.com/crystal/en/products/crystal-unit/fc135.html
- https://www.epsondevice.com/crystal/en/techinfo/ic-partners/rohm/
- https://www.microcrystal.com/fileadmin/Media/Products/32kHz/Datasheet/CC7V-T1A.pdf

## 2. Source and acquisition

### Phase-coherent source candidate

```text
SIGLENT SDG1032X
```

Frozen build-readiness source contract:

```text
C1 is continuous at calibration-bound f_carrier_hz, exactly 0.400 Vpp, 0 V offset, phase 0 or pi
C2 is continuous at bound f_witness_hz = 2 * f_carrier_hz, exactly 0.100 Vpp, 0 V offset, fixed zero phase
both channels use HIGH_Z load mode while retaining 50 Ohm physical output impedance
at least ceil(3 * f_carrier_hz) C1 cycles before the gate are the qualified preparation interval
0 and pi phase commands are remote-readable and repeatable
no external trigger, sync cable, or self-ending burst exists
both outputs remain on for the complete record and CH0 proves persistence
output impedance and low-amplitude accuracy are calibrated
firmware and programming-guide identities are frozen
```

The manufacturer material establishes the dual-channel source class. The frozen P0 setup
does not use its burst or trigger features. The packet must not trust a GUI display as
execution custody; exact per-channel queryback of model, serial, firmware, load mode,
physical output impedance, waveform, frequency, amplitude, offset, phase, continuous mode,
and output state plus CH0 source-monitor evidence are required.

Official source:

- https://www.siglent.com/my/products-overview/sdg1000x/

### Conforming acquisition candidate

```text
Spectrum Instrumentation digitizerNETBOX DN2.592-04
```

Architecture fit:

```text
4 true differential inputs
separate 16-bit ADC/amplifier per channel
simultaneous sampling
20 MS/s maximum, operated at exactly 1 MS/s/channel
```

Required binding before freeze:

```text
DC-coupled differential mode on all four channels simultaneously
at least 3,101,000 native samples per channel in one continuous record
raw integer-code export with no averaging or hidden DSP
channel skew, software-prearmed free-run start law, and explicit absence of external trigger
input impedance, input capacitance, common-mode range, and ground/isolation law
SDK/driver version and deterministic native parser
```

Official source:

- https://spectrum-instrumentation.com/products/details/DN2592-04.php

### Acquisition fallback requiring explicit topology review

```text
PicoScope 5462E
```

It provides four native 16-bit channels, deep memory, and a built-in AWG, but its BNC
inputs are single-ended. It is not automatically conforming with the frozen differential,
no-extra-ground-bond topology. It may be admitted only if the final packet prospectively
re-freezes a lawful external differential/isolation front end and proves the resulting
star-ground, skew, noise, and input-admittance laws. It cannot silently replace the
DN2.592-04 path.

Official source:

- https://www.picotech.com/oscilloscope/picoscope-5000e-series-16-bit-usb-oscilloscope

## 3. Source-off chain

### Fast phase gate

```text
Analog Devices ADG1419BRMZ
```

Operate at `+/-5 V`. The MSOP variant has no `EN`; `IN` routes between DRIVE and the 50-ohm
termination. The final design must account for the specified full-temperature transition,
charge injection, leakage, and large parasitic capacitances. The analog switch is upstream
of the mechanical relay barrier and cannot establish source removal by itself.

Official sources:

- https://www.analog.com/en/products/adg1419.html
- https://www.analog.com/media/en/technical-documentation/data-sheets/adg1419.pdf

### Guarded physical barrier

```text
3 x Omron G6K-2F-Y DC5
```

This is the highly insulated DPDT, single-side-stable, 5 V coil configuration. The final
packet must freeze coil drivers, suppression, contact/witness allocation, PCB footprint,
contact orientation, relay separation, and the exact fail-safe netlist. Manufacturer data
lists 21.1 mA nominal coil current, 237 ohm coil resistance, 3 ms maximum operate/release,
and 1,000 megohm minimum insulation resistance at 500 VDC.

Official sources:

- https://components.omron.com/us-en/products/relays/G6K
- https://components.omron.com/sites/default/files/datasheet_pdf/K106-E1.pdf

## 4. Carrier sense and environment

### High-impedance sense selection

```text
Texas Instruments OPA810IDT
```

The exact SOIC-D buffer is frozen prospectively in unity gain on a guarded carrier board.
The sensitive input is adjacent to the carrier. Because the official 2 pF common-mode and
0.5 pF differential capacitances are typical rather than guaranteed, every populated
coupon must independently close `Rin,U95 >= 100 Mohm` and `Cin,U95 <= 4.00 pF` before a
full unpowered build can proceed.

Official sources:

- https://www.ti.com/product/OPA810
- https://www.ti.com/lit/ds/symlink/opa810.pdf

### Analog enclosure-vibration candidate

```text
Analog Devices ADXL354CEZ
```

Use one prospectively selected analog axis for CH3 and retain the other axes only as
non-authoritative auxiliary records unless the channel map is re-frozen. The final packet
must freeze range, sensitivity, analog bandwidth, antialias network, mounting orientation,
self-test, supply/reference, calibration, and clipping law.

Official source:

- https://www.analog.com/en/products/adxl354.html

### Temperature/humidity candidate

```text
Sensirion SHT45-AD1B-R2
```

The final packet must freeze the 10 Hz polling/record law, timestamp alignment, CRC checks,
calibration identity, placement, self-heating restrictions, and missing/cadence rejection.
It is auxiliary environmental custody, not a phase carrier or primary acquisition channel.

Official source:

- https://sensirion.com/products/catalog/SHT45?show_inventory=SHT45-AD1B-R2

## 5. Exact items still requiring local build-readiness closure

The local packet must choose exact identities for:

```text
Epson packing suffix or accepted distributor ordering identity
carrier footprint/adapter and low-strain PCB
50.00-ohm 0.1% low-inductance terminations
100-kohm 0.1% current limiter
fixed >=100-Mohm bias return
relay MOSFET drivers and flyback/suppression parts
isolated gate/relay control supply and digital isolator
OPA810 supply, guarded carrier land, bias return, local bypass and output isolation
ADXL354 board or custom mounting PCB and antialias components
SHT45 carrier board and timestamping controller
four-bit CH2 witness resistor network
connectors, guarded/triax/coax cable identities, enclosure, and star-ground hardware
continuous C1/C2 passive source-monitor distribution with software-prearmed free-run acquisition and no trigger cable
calibration references and required adapters
ESD, current limiting, fusing, and test points
```

No final BOM is closed until every component has a manufacturer identity, quantity,
permitted substitution law, truthful source-custody state, conservative operating limit,
and netlist role. A historical hash without the corresponding bytes is retained as a
historical expectation, not relabeled as a local capture.

## 6. Canonical research custody dependency

The authoritative repository-safe bundle is
`research/P0_research_bundle_2026-07-18`, sourced from commit
`cb53976612cbe83bec82df826a9889418f7e0b89`. Its canonical manifest contains
35 records: 24 core component/instrument sources, one controller design-resource
record, eight scientific-background sources, and two vendor-model resources.
The current repository-safe custody snapshot records 11 private locally
hash-verified captures, 6 URL-plus-legacy-hash records without local bytes, and
18 manual captures. Source metadata and hashes are committed candidate inputs;
third-party document/model bytes, HTML, receipts, and generated archives are
ignored and remain private.

From the bundle directory, a private refresh uses the repository virtual
environment:

```text
python scripts/download_sources.py --all
python scripts/verify_downloads.py
python scripts/build_custody_snapshot.py
```

Revision custody explicitly uses ADR45xx Rev. G, ADuM140D Rev. K, ST UM2591
Rev. 2 (April 2026), and SHT4x PDF Version 7.1 (March 2025) while preserving
the product-page 04/2025 label and every historical byte/hash expectation.

## 7. Build-readiness decision boundary

This seed authorizes no purchase or hardware contact. It exists to prevent another agent
from repeating broad carrier research and to force explicit resolution of the remaining
exact-part and topology gaps.

The final packet must stop at:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```
