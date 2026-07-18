# P0 authored assembly drawings

All dimensions are millimetres. Datum for every PCB is its lower-left corner, component side up. Panel holes use the machine-coordinate inner-face origins and signed axes in `enclosure_design.panel_face_datums`; they are never mirrored by an exterior viewing convention. These are reviewed prospective drawings under `AUTHORIZE P0 BUILD-READINESS ONLY`; they authorize no fabrication or assembly.

## D001 complete system

One selected complete assembly connects SDG C1 to `J_SRC_C1`, SDG C2 to `J_SRC_C2`, and DN2 channels 0-3 to `J_CH0` through `J_CH3`. No external-trigger cable exists. The two fixture enclosures are joined only by the fixed 150 +/-2 mm harness. Never mix boards or cables across A/B/C.

## D002 control board

`P0-SOURCE-OFF-CONTROL-REV-B-*` is 84.0 x 64.0 x 1.6, four layer, with 3.2 finished non-plated holes at (4,4), (80,4), (4,60), (80,60). It contains C1 limiting/gating, passive C1+C2 monitor, K1/K3, relay drivers, isolated supplies, isolators, calibrated witness ladder, SHT45 and all local bypasses. Exact per-reference coordinates are in `P0_PCB_FABRICATION_RELEASE.json`. Nine exact 60 +/-2 mm Alpha 2840/1 conductors bind the published Nucleo CN3/CN4 lands to the named control-board lands; no detachable internal interface or splice is used.

## D003 carrier/sense board

The A/B/C carrier variants are 68.0 x 32.0 x 1.0, two layer, with 3.2 finished non-plated holes at (4,4), (64,4), (4,28), (64,28). OPA810 pin 3 and the electrode land use a <=5 mm guarded route; no test point, clamp, cable or added copper touches that node. Pins 1, 5 and 8 are isolated no-connect lands. K2 package lands are the only pre-population continuity access.

## D004 vibration board

`P0-ENV-SENSOR-REV-B-*` is 24.0 x 24.0 x 1.0 with 2.7 finished non-plated holes at (2.5,2.5), (21.5,2.5), (2.5,21.5), (21.5,21.5). The ADXL354 Z arrow points normal to the board and toward the enclosure lid. All four supply nodes have local 100 nF capacitors and both internal 1.8 V nodes have 100 kohm discharge paths. Exact 35 +/-2 mm Alpha 2840/1 conductors carry `ADR_REF_3V3` and `AGND_STAR` from the carrier board; the Z output goes directly to its 45 +/-2 mm panel pigtail.

## D005 control enclosure

The control enclosure is exact custom design `P0-CUSTOM-ENCLOSURE-124X82X36-REV-A`. Material, tub/lid dimensions, bare-metal seam, ten fasteners, panel datums/counterbores, tolerances, electrical acceptance, PEEK retention systems, four isolated BNC holes, gland hole, board origins and standoff heights are mechanically bound by `enclosure_design` and the exact `ENC-CONTROL-*` rows in the fabrication release. The Nucleo uses exact `P0-NUCLEO-PEEK-EDGE-TRAY-REV-B`; its admitted-board interval, seat, anchor/countersink geometry, in-outline clip contacts, worst-case deflection budget, tolerances, insertion method, keepout and six retention acceptance checks are fabrication fields. Each control-enclosure mount also binds the +z rotation pivot/convention, board/tray local-to-machine equations, resolved tray footprint and all four floor-anchor centers; no exterior-view mirroring or implicit rotation is permitted. All tray/standoff/washer/screw identities and quantities are separate BOM lines. Maintain >=5 mm lid clearance, >=4 mm wall clearance and >=15 mm coax bend radius. Exact 55 +/-2 mm internal pigtails bind C1, CH0 and CH2 by RG-178 center/shield pairs. C2 uses one PTFE center conductor only; its isolated panel shell has no internal conductor.

## D006 carrier enclosure

The carrier enclosure uses the same exact custom design and acceptance law. Its two isolated BNC holes, gland counterbore, insulated star hole, board origins, PEEK retention identities and standoff heights are the exact `ENC-CARRIER-*` entries. The star hardware must measure at least 1000 Mohm to enclosure metal before the harness shield is attached. Exact 45 +/-2 mm RG-178 pigtails bind CH1 and CH3; their shields terminate only at `AGND_STAR`.

## D007 fixed harness

One Belden 83269 RG-178 center carries `N_MIDPOINT`; its shield is the sole `AGND_EXPORT`-to-`AGND_STAR` bond. Six separately identified Alpha 2840/1 conductors carry +5V_RELAY, K2_COIL_LOW, ADR_REF_3V3, N_WIT_K2, +5V_SENSE and -5V_SENSE. Both `1427CG13` glands use 20.42 +/-0.10 holes. There is no connector or splice.

## D008 witness ladder

The ADuM gate-secondary supply is `ADR_REF_3V3`, so `IN_GATE` and the K1/K2/K3 auxiliary contacts select the same nominal 3.3 V reference through 80.6, 40.2, 20.0 and 10.0 kohm respectively. `IN_GATE` is actively driven low when b0=0, so the 80.6 kohm branch remains a shunt; inactive relay-contact branches float. `N_WITNESS` returns through 1.00 kohm. The corrected nominal equation and all sixteen centroids are frozen in `witness_law`; code 8 is 296.654026 mV and minimum nominal adjacent separation is 24.996066 mV. All 16 centroids and the low-driven b0 impedance are measured later; adjacent centroids must differ by at least ten pooled sigma and every sample must have unique +/-3 sigma membership.

## D009 isolation and no-connect drawing

Every `NC::reference.pin` entry is a land with no copper beyond the pad. Every `DNP::` entry is unpopulated and isolated. The controller, relay and analog domains have no undeclared galvanic bond. The C2 rig-end shell is isolated. DN2 input admittance is calibrated and is not described as isolation.

## D010 fixture population matrix

- A: FC-135 populated; dummy open.
- B: FC-135 and dummy both open.
- C: FC-135 open; exact 1.0 pF C0G dummy populated.

All other parts, boards, enclosures, harnesses and dedicated external cables are fixture-local and matched. The complete machine-readable topology is `P0_FINAL_NETLIST.json`; the coordinate release has 9 PCB instances and 6 enclosure instances.
