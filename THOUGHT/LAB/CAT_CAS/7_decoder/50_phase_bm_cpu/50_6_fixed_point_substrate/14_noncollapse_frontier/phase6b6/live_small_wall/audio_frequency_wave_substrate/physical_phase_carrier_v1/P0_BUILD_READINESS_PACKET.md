# P0 non-executing build-readiness packet

## Frozen scope

- authority: `AUTHORIZE P0 BUILD-READINESS ONLY`
- claim ceiling: `NON_EXECUTING_P0_BUILD_READINESS_ONLY`
- next authority boundary: `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD`
- status: `P0_BUILD_READINESS_BLOCKED`; the actual signal-pole witness is unresolved, and no procurement, fabrication, assembly, connection, power, playback, acquisition, or instrument command is authorized
- exact design hashes: netlist `0ceb9096be95ba917ff0d11d8ebd57e09bfd2c445ca6dddf5f873ef741721225`, BOM `bbffd15125c0a71e16a79621fff55f86802a8e5839dfb5777c27b259d24bcdd4`, fabrication `1e526904624090f575822d3cc1412807667f22a223ca7b3edd1e607d9611094e`

This packet describes a prospective physical experiment. It reports no physical result. No human vendor outreach, cart, stock, procurement, instrument command, audio interface, target, or hardware contact occurred while producing it. Automated public-source HTTP retrieval attempts are separately disclosed in the research custody snapshot and private ignored download receipt; they are not described as zero network or zero server contact.

## Canonical research custody dependency

The repository-safe research dependency is `research/P0_research_bundle_2026-07-18`, imported from commit `cb53976612cbe83bec82df826a9889418f7e0b89`. Its canonical `p0.research-bundle-manifest.v1` manifest contains exactly 35 records and has SHA-256 `4dd4df0d806b083d9a9e7fe3c0f255fc56e729d3be83de1ba567fade42166fad`. Current repository-safe custody metadata records 11 locally hash-verified private-cache captures, 0 current captures that differ from retained legacy hashes, 6 URL-plus-legacy-hash records without local bytes, 18 manual captures, and 0 prospective identities. These counts describe the bound metadata snapshot; downloaded third-party document/model bytes, receipts, HTML captures and generated archives remain ignored, private and uncommitted.

Private refresh command, from the research-bundle directory, is `D:\CCC 2.0\AI\agent-governance-system\.venv\Scripts\python.exe scripts/download_sources.py --all`, followed by the same interpreter with `scripts/verify_downloads.py` and `scripts/build_custody_snapshot.py`. A URL or historical hash alone is never described as local byte custody. Revision alerts retain ADR45xx Rev. G rather than Rev. F, ADuM140D Rev. K, ST UM2591 Rev. 2 (April 2026), SHT4x PDF Version 7.1 (March 2025) alongside the product-page 04/2025 label, and the current Omron/Nexperia identities without discarding historical hashes.

## What is being tested

The bounded question is whether the phase of a mechanically stored 32.768 kHz quartz state remains distinguishable after the source path is physically isolated. The software analyzer is an ordinary deterministic reference and cannot establish physical persistence. The maximum later claim remains a bounded physical-carrier observation under the separately frozen execution contract.

## Three complete matched assemblies

`P0-DUT-A` contains Epson `Q13FC1350000401`; suffix `01` is the manufacturer's any-quantity tape-cut packing code and matches the one-piece build quantity. `P0-DETECTOR-B` has the carrier position deliberately open; `P0-DUMMY-C0-C` substitutes exactly one Murata `GJM1555C1H1R0BB01D` 1.0 pF C0G part. Each fixture owns its control board, carrier board, sensor board, controller, two enclosures, fixed harness, and six labeled coaxes. Only the source and digitizer are shared, and only one complete assembly may be connected in a future record.

## Sense admittance and the OPA810 correction

The selected sense part is Texas Instruments `OPA810IDT`, official `SBOS799E`, SHA-256 `74c61ac238989c94c1cf0d70da41bff6e167a590f86e06bae7cdd734d8fd26fa`. The data sheet gives 12 Gohm in parallel with 2 pF common-mode input and 0.5 pF differential input as typical values. Those are planning values, not guarantees. At 32,768 Hz, 4.00 pF has reactance 1,214,255.852 ohm; the FC-135 70 kohm maximum ESR is 0.057648 of that value. The prospective budget is 2.50 pF typical amplifier + 0.30 pF guarded land/routing + 0.20 pF K2/carrier land + 0.15 pF bias body/pads + 0.15 pF contamination + 0.30 pF reserve = 3.60 pF, leaving 0.40 pF margin. Every future populated coupon must separately prove `Cin,U95 <= 4.00 pF` and `Rin,U95 >= 100 Mohm`; otherwise assembly stops. External electrode clamps are deliberately absent because their capacitance or leakage would change the carrier load. Normal drive is bounded upstream by the 100 kohm limiter and two relay barriers; OPA810 absolute input/current and on-die ESD ratings are damage limits, not operating permission. Handling therefore requires grounded ESD controls with every source, instrument and external cable absent.

## Continuous dual-tone source and phase gauge

The assumed source is one SIGLENT `SDG1032X` in `HIGH_Z` load mode; its physical output impedance remains 50 ohm. C1 is a continuous 32,768 Hz sine at exactly 0.400 Vpp and 0 V offset with phase command 0 or pi. C2 is a continuous 65,536 Hz sine at exactly 0.100 Vpp and 0 V offset with fixed zero phase. Both outputs share the source's internal reference; no burst and no external trigger is used. The C1 passive monitor tap is at `C1_IN`, upstream of the exact 100 kohm limiter and ADG1419, so routing the downstream node to 50 ohm cannot erase the source witness. C1 and C2 are each passively presented through 100 kohm to CH0, with 1 Mohm return, so CH0 carries a source-continuity witness and a 2x phase gauge. C2 is not an intended carrier drive, but the passive monitor network creates a bounded linear C2-to-C1 coupling path through `R_MON_C2`, the monitor node, `R_MON_C1`, the finite C1 source impedance and `R_LIMIT`; that path must be included in the circuit model, feedthrough sweep and empty/dummy controls rather than asserted to be zero. Its fixture-end BNC shell has no internal conductor, preventing a second low-impedance source-return bond.

The C1 ceiling is prospective and conservative: 0.400 Vpp is 0.141421356 Vrms; with 50 ohm source impedance, the exact 100 kohm limiter and the FC-135 70 kohm maximum ESR, the resistive bound is 0.831646 microampere rms, 0.164658 Vpp across the ESR and 0.048415 microwatt motional dissipation. These are planning bounds only; a future as-built path must meet the separately frozen terminal-voltage, current and power caps by calibrated measurement.

For a future record the source must be queried back for model, serial, firmware, output impedance, waveform, frequency, amplitude, offset, phase, continuous mode and output state on both channels. The record metadata also binds the sealed one-standard-uncertainty fields `phase_skew_standard_uncertainty_rad` and `phase_drive_cal_standard_uncertainty_rad`; the analyzer combines them with lag-7 Newey-West drive-fit covariance for the individual and matched-arm phase gates. Any mismatch stops before accepting bytes.

## Software-prearmed acquisition

The assumed Spectrum `DN2.592-04` uses four simultaneous true-differential ADC paths at exactly 1,000,000 samples/s and 3,101,000 signed 16-bit samples per channel. It free-runs after software pre-arm. CH0 is the dual-tone source monitor, CH1 the OPA810 sense output, CH2 the four-bit source-off witness and CH3 the ADXL354 Z axis. There is no trigger cable. CH2 locates the gate event; half the fitted C2 phase supplies the continuous phase gauge, with the pi branch resolved by the commanded C1 arm.

The digitizer inputs are true differential but not galvanically isolated. Their positive-to-ground, negative-to-ground and differential admittances must be surveyed across 32.768 and 65.536 kHz and included in calibration. No document calls them an isolation barrier.

No proprietary DN2 container parser is claimed. The frozen acquisition boundary is an exact SDK export mode producing one headerless, sample-major, little-endian signed-int16 payload with four channels and exactly 3,101,000 samples per channel (24,808,000 bytes), plus canonical strict JSON metadata. The adapter is an export configuration and receipt, not a guessed parser: it binds SDK/driver identity, adapter source SHA-256, SDK-export hash/count, the identical analyzer-payload hash/count, and explicit false assertions for sample loss, reordering, averaging, filtering, resampling, clipping concealment and unit ambiguity. Any additional proprietary container is preserved separately but never parsed by this analyzer. The reference fixtures exercise the canonical export contract. Actual SDK/export bytes and adapter source cannot be frozen before acquisition-equipment authority, so their exact identities remain a procurement-time gate; physical `analyze` must reject until they are supplied and hash-bound.

## Physical source-off sequence

At the decoded gate event, ADG1419 routes C1 away from the drive path to an exact 50.00 ohm termination while C1 and C2 remain continuously energized. After 250 microseconds K1 and K2 release while K3 remains energized and therefore cannot clamp the carrier midpoint. CH2 must first decode code 0 for 1,000 consecutive samples. Only then may K3 deenergize to the guard state; CH2 must then decode code 8 for 1,000 consecutive samples. The complete ordered transition must settle within 14,500 samples. The first admissible post-isolation sample is `max(t_gate + 0.560 us, final stable-code-run end) + 10.000 ms`. CH0 must match the sample-level reconstructed C1+C2 waveform throughout the complete one-second pre-gate preparation and from the gate through record end, and both tones must also remain within 2 percent amplitude and 0.010 rad phase in the frozen contiguous segments. A muted source, absent reference tone, wrong ordering, late contact, or hidden trigger is a hard rejection.

CH2 observes auxiliary poles, not the actual signal poles. The ordered sequence prevents the candidate's earlier K2/K3 clamp hazard, but it does not by itself prove that K1/K2 signal contacts opened during a particular event. Physical execution and every source-disconnect claim remain blocked until a separately reviewed per-event actual-signal-path witness or an exact force-guided-contact guarantee is mechanically bound to the selected relay and failure modes.

The ADuM gate-secondary logic supply and the three relay witness contacts use the same exact `ADR_REF_3V3` rail. The gate bit is actively driven low when inactive, so its 80.6 kohm branch remains a shunt in every b0=0 code; inactive relay-contact branches float. The corrected nominal 80.6/40.2/20.0/10.0 kohm ladder into 1.00 kohm has ordered code centroids from 0 to 520.543716 mV, stable-OFF code 8 at 296.654026 mV, and a minimum adjacent gap of 24.996066 mV. Those nominal values are not calibration: a future unpowered build must measure every resistor, verify low-drive impedance, recompute all sixteen prospective centroids, and a later powered calibration must establish unique ten-sigma-separated measured centroids before any record can be admissible.

## Complete wiring and access custody

`P0_FINAL_NETLIST.json` binds every relay contact, relay driver, switch truth row, source-off state, failure response, connector, external cable, inter-enclosure conductor, internal pigtail, power domain and temporary continuity access. The Nucleo-to-control link is nine fixed 60 +/-2 mm PTFE conductors. Control-panel pigtails are exact 55 +/-2 mm cuts; carrier-panel pigtails are 45 +/-2 mm; the vibration-board power/reference link is 35 +/-2 mm. The C2 fixture shell has no internal conductor. No permanent test-point hardware exists, and no electrode test point is allowed.

## Ground and shield law

C1 provides the sole intentional low-impedance source return. The inter-enclosure RG-178 shield is the sole intentional low-impedance `AGND_EXPORT` to `AGND_STAR` bond. All BNC bodies and the M4 star stud are insulated from enclosure metal. Controller ground, relay ground and analog ground remain galvanically separate. C2 has no rig-end shell connection. Because the digitizer inputs are not galvanically isolated, their positive-to-ground, negative-to-ground and differential admittances are additional calibrated parasitic return paths and must be included in the loaded model; they are never described as zero or as an isolation barrier. The exact forbidden-pair list and connector returns are in `P0_FINAL_NETLIST.json`.

## Environmental custody

Each SHT45 row contains monotonic nanoseconds, UTC, sensor serial, command `0xfd`, raw temperature and RH words, both CRC-8 bytes, and converted values. CRC uses polynomial 0x31 and initial value 0xff. UTC is linked to the monotonic clock by a frozen mapping. CH3 raw ADXL354 data must satisfy RMS <=0.050 m/s2 and peak <=0.500 m/s2, with axis, gain, offset and filter calibration bound to the record.

## Simulation boundary and unresolved physics layer

The current reference is a valid signal-level analyzer test: it directly constructs deterministic four-channel synthetic ringdown records and proves that the unchanged analyzer recovers or rejects them. It is not a first-principles prediction that the complete proposed circuit produces those records. The separate unresolved device/circuit layer is `source -> 100 kohm limiter -> ADG1419 vendor model -> relay state/parasitic model -> FC-135 Butterworth-Van Dyke model -> OPA810 vendor model -> digitizer differential/common-mode loading -> cable/PCB/enclosure parasitics and resistor/amplifier noise -> ADC quantization -> canonical four-channel raw payload -> existing unchanged scientific analyzer`.

That future layer must sweep motional R/L/C, shunt capacitance, loaded Q and resonance, OPA810 input capacitance and leakage, switch off-capacitance/charge-injection/leakage, relay bounce and release timing, cable and board capacitance, source-monitor feedthrough including the bounded C2 path, digitizer differential/common-mode admittance, clock/phase-reference error, amplifier/resistor noise, ADC quantization, environmental perturbation, and matched empty/1 pF dummy controls. Its output is the region of parameter space that survives the fixed analyzer, not one favorable inserted ringdown. Until that separate test exists, complete-circuit physical plausibility remains unresolved.

## Staged restoration ladder

1. The current authority permits only authored bytes and offline deterministic tests.
2. `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD` may later authorize either procurement planning or an unpowered build step; the user must choose the extent explicitly.
3. A separately authorized unpowered coupon stage captures official documents, incoming identities, board bytes, continuity, isolation and OPA810 admittance.
4. A separately authorized unpowered full-assembly stage may proceed only if every coupon gate passes.
5. Powered calibration and physical execution remain outside all current authority and require another explicit instruction after the complete assembled receipt is reviewed.

No stage bootstraps authority for the next one.

## Claim law

Offline PASS means only that the deterministic analyzer distinguishes the frozen synthetic positive fixtures from the frozen adversaries. A future physical claim requires committed raw bytes, exact identities, complete source-off witnesses, matched controls and independent adjudication. No optimization, Ising, catalytic-loop, capacity, restoration, Wall, or computation-advantage claim is authorized.
