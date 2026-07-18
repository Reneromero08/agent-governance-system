# P0 unpowered assembly packet

Status: **NOT AUTHORIZED**. This is a prospective, non-purchasing procedure under `AUTHORIZE P0 BUILD-READINESS ONLY`. Every physical verb below is conditional on a later explicit user instruction.

## Stage U0 — incoming custody

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: capture manufacturer, complete order code, quantity, lot/date/packing code, marking photographs, official-document bytes and SHA-256 for every BOM line. Reject any substitution. Bind PCB fabrication bytes and enclosure machining report to the reviewed qualification root.

Before opening any ESD-sensitive package, record the grounded mat, wrist-strap tester, earth point, room temperature/RH and operator identifier. Source, digitizer, controller USB, external coaxes and all power leads remain physically absent. The ESD receipt is evidence only and cannot authorize the next stage.

## Stage U1 — bare-board coupons

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: inspect dimensions, drill, finish, layer stack and net isolation for all nine PCBs. Use a current-limited LCR fixture with no source or digitizer attached to characterize the guarded carrier input land. Reject any undeclared continuity, no-connect copper or fabrication-byte mismatch.

## Stage U2 — sense coupons

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: populate OPA810, K2, bias return and local bypass on each carrier/control coupon, leaving FC-135/dummy positions open. Measure `Cin,U95` and `Rin,U95` over the frozen environment envelope. Require <=4.00 pF and >=100 Mohm independently for A, B and C. This gate is mandatory because the amplifier data-sheet capacitances are typical, not guaranteed.

## Stage U3 — remaining unpowered board assembly

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: use the lower of every exact component document's soldering limits and the qualified board-process limit. Record paste/flux/cleaner identity, profile or iron-tip temperature, dwell, operator and rework count. Populate in this fixed order: fuses, current limiters and terminations; isolated supplies and bypasses; isolators, gate, relay drivers and suppression; K1/K3 and witness ladder; reference and sensors; K2, bias return and OPA810 last. Inspect polarity and pin 1 after each group, then clean and verify guarded surfaces using the qualified process. Confirm each `NC::` land has no copper beyond its pad and every `DNP::` location is open. Do not install FC-135 or dummy until K2 land continuity has been recorded. Any undocumented rework or exceeded document limit rejects the assembly.

## Stage U4 — continuity and isolation

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: execute every entry in `continuity_tests` and every forbidden pair in `P0_FINAL_NETLIST.json`. Record instrument identity, range, uncertainty, probes, raw readings and photographs. K1/K2/K3 signal-pole continuity and auxiliary-pole continuity are separate receipts. Do not infer signal isolation from an auxiliary contact. Measure every CH2 ladder resistor and the 3.3 V-reference path unpowered, compute the prospective 16-code centroid ordering from the measured values, and reject any swapped weight or nonmonotone code. No voltage is applied to a carrier electrode in this stage.

## Stage U5 — exact fixture populations

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: populate A with exact Epson `Q13FC1350000401`, leave B open, and populate C with exact Murata `GJM1555C1H1R0BB01D` 1.0 pF C0G. Apply the lowest-stress attachment process allowed by each exact document and record temperature/dwell. Photograph top, bottom, pin orientation, land wetting and enclosure clearance. Re-run input admittance and all continuity checks.

## Stage U6 — mechanics and harness

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: machine holes from `P0_PCB_FABRICATION_RELEASE.json`; deburr, clean and dimensionally inspect them. Install integral-isolation BNC bodies and the insulated M4 star. Bind every lid screw, standoff, PEEK washer, board screw, Nucleo tray and flush PEEK tray screw to its exact BOM line and fabrication geometry. Inspect the complete tray profile and tolerances, install the Nucleo only by the bounded four-ramp perimeter-force method in the fabrication release, then execute every tray insertion, lateral-motion, upward-retention, keepout, flush-head and visual acceptance check. Build one unspliced 150 +/-2 mm inter-enclosure harness per fixture. Build the exact internal cut sets from `intra_enclosure_harnesses`: nine 60 +/-2 mm Nucleo conductors, control-panel 55 +/-2 mm pigtails, carrier-panel 45 +/-2 mm pigtails, and two 35 +/-2 mm carrier-to-vibration conductors. The C2 panel shell receives no internal conductor. Verify every conductor end-to-end, every shield endpoint, pull retention, strain relief, and the sole inter-enclosure shield bond before mounting boards at the frozen origins and standoff heights.

Label every board, enclosure, fixed conductor and external coax with fixture `A`, `B` or `C` plus its exact connector/net identity. Route the guarded sense node only on its authored board, maintain the declared coax bend radius, keep relay/control conductors outside the carrier keepout and photograph every shield termination and strain relief before closure.

## Stage U7 — closed unpowered assembly receipt

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: with every external cable absent, re-run all forbidden-pair and path tests through the closed enclosures. Attach the six dedicated labeled external coaxes without source, digitizer, controller USB or power. Verify connector-center and shield mapping. Create one canonical as-built manifest containing the reviewed candidate root, board/fabrication hashes, every component and document identity, lot/packing fields, process receipts, measured continuity/isolation/admittance values, instrument calibration identities, photographs and file hashes. Seal it before closure, stop, and return the complete receipt for review.

No step in this packet authorizes power, playback, recording, instrument commands, source output, acquisition, calibration, or experiment execution. The next boundary remains `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD`.
