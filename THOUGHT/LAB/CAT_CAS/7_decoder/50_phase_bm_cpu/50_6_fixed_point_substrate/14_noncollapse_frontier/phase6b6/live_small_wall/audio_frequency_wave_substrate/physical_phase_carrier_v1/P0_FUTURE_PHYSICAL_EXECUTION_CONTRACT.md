# Future P0 physical execution contract

Status: **NOT AUTHORIZED**. Claim ceiling: `NON_EXECUTING_P0_BUILD_READINESS_ONLY`. This contract is a prospective hard-stop checklist and grants no authority.

## Required prior receipts

1. Separate user authority for the exact proposed stage.
2. Reviewed qualification root binding netlist, BOM, fabrication, analyzer, schemas, fixtures and all four independent reviews.
3. Official-document hashes and incoming custody for every acquired item.
4. Passed bare-board, sense-coupon, continuity, isolation, mechanics and closed-assembly receipts for A/B/C.
5. Exact as-built `Cin,U95 <= 4.00 pF` and `Rin,U95 >= 100 Mohm` on all three sense paths.

## Future source contract

One exact SDG1032X in `HIGH_Z` load mode, with 50 ohm physical output impedance, supplies continuous C1=32768 Hz at 0.400 Vpp and 0 V offset and phase-locked C2=65536 Hz at 0.100 Vpp and 0 V offset. C1 carries the 0/pi arm command. C2 remains fixed at zero phase, enters the passive CH0 monitor, and also enters the exact 1.00 Mohm TNPW branch to `N_GATE_OUT`; it never passes through ADG1419. The exact 100 kohm `N_SRC` drive shunt is on the ADG D/SA side during OFF and cannot define the C2 witness. Both source channels stay on through the entire record. No burst and no external trigger is allowed. All source/queryback and uncertainty custody remains mandatory.

## Future acquisition contract

One exact DN2.592-04 captures four simultaneous true-differential channels at 1,000,000 samples/s for 3,101,000 samples/channel using software-prearmed free run. Its input mode is frozen to 1 Mohm in parallel with 30 pF and both negative legs are bound to calibrated AGND so common mode is derived from raw differential bytes. CH2 locates source isolation and C2 supplies the phase gauge. No proprietary container parser is claimed by this packet. A separately authorized acquisition must preserve any original proprietary container, run the frozen SDK lossless-export mode to the exact 24,808,000-byte signed-int16 canonical payload, require SDK-export and analyzer-payload hashes/counts to be identical, and supply actual byte descriptors for adapter source, instrument/source querybacks, native export, assignment commitment/reveal, calibration, chronology, the assembly manifest, the event-specific version-2 topology receipt, a unique four-state C2 topology scan and a unique C1-only nonlinear-control trace. The receipt must bind the assigned role, exact A/B/C assembly and population, native payload, chronology, querybacks, raw scan/control bytes and pre-acquisition scan times. All four times must parse from exact `YYYY-MM-DDTHH:MM:SS.ffffffZ` UTC form. A roles reuse one exact A manifest; B and C use distinct manifests. Cross-assembly, cross-event, duplicate-scan, duplicate-control, noncanonical-time or post-acquisition replay is a hard stop. Missing or mismatched bytes are a hard stop; hash-shaped metadata alone is insufficient.

## Future source-off contract

ADG1419 terminates C1 into 50.00 ohm. The same-ADG-state 192-sample C2 pre-window completes before K1/K2 release at 250 microseconds. K3 stays energized/electrically open while code 0 remains stable for 1,000 samples and the 960-sample C2 isolated-path window passes. Only then may K3 release to guard; code 8 must remain stable for 1,000 samples and the 10 ms guard follows. Auxiliary contacts do not identify either signal pole. A passing end-to-end transfer event supports only the exact actual-path isolation token and at least one-open meaning. Any guard masking, wrong-node injection, missing C2, excessive open feedthrough, inversion, bounce, re-entry, hidden buffer, replay, trigger, muted source, wrong termination, wrong fixture, invalid CRC or ancestry mismatch rejects the record.

## Future controls

Every session requires A DUT, B detector-only and C exact-dummy records with randomized order fixed before connection. Ordinary DSP replay, decoded-spin replay, flat waveforms, spectral leakage, metadata leakage, query preselection, file/buffer persistence and restoration overclaim remain explicit adversaries. A physical observation cannot establish a catalytic, Ising, optimization, capacity, Wall or computation-advantage claim.

## Stop law

The first identity, custody, calibration, environment, topology, witness, raw-byte, control or analyzer failure stops the future run. No retry or parameter adjustment is implicit. Nothing in this file permits hardware contact now. Next boundary: `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD`.
