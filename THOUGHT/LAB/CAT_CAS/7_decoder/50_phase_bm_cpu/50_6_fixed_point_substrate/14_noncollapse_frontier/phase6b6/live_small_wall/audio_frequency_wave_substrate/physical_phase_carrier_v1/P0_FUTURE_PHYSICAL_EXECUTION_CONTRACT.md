# Future P0 physical execution contract

Status: **NOT AUTHORIZED**. Claim ceiling: `NON_EXECUTING_P0_BUILD_READINESS_ONLY`. This contract is a prospective hard-stop checklist and grants no authority.

## Required prior receipts

1. Separate user authority for the exact proposed stage.
2. Reviewed qualification root binding netlist, BOM, fabrication, analyzer, schemas, fixtures and all four independent reviews.
3. Official-document hashes and incoming custody for every acquired item.
4. Passed bare-board, sense-coupon, continuity, isolation, mechanics and closed-assembly receipts for A/B/C.
5. Exact as-built `Cin,U95 <= 4.00 pF` and `Rin,U95 >= 100 Mohm` on all three sense paths.

## Future source contract

One exact SDG1032X in `HIGH_Z` load mode, with 50 ohm physical output impedance, supplies continuous C1=32768 Hz at 0.400 Vpp and 0 V offset and phase-locked C2=65536 Hz at 0.100 Vpp and 0 V offset. C1 carries the 0/pi arm command; C2 remains fixed at zero and only enters the passive CH0 monitor. Both stay on through the entire record. No burst and no external trigger is allowed. Model, serial, firmware, load mode, output impedance, waveform, frequency, amplitude, offset, phase, continuous mode and output state are queried back per channel. Sealed standard phase-skew and drive-calibration uncertainties are separately bound in metadata and combined with the analyzer's Newey-West fit covariance. All querybacks, complete-preparation reconstruction, post-gate continuity/gauge checks and individual/matched uncertainty gates are mandatory.

## Future acquisition contract

One exact DN2.592-04 captures four simultaneous true-differential channels at 1,000,000 samples/s for 3,101,000 samples/channel using software-prearmed free run. CH2 locates source isolation and C2 supplies the phase gauge. No proprietary container parser is claimed by this packet. A separately authorized acquisition must preserve any original proprietary container, run the frozen SDK lossless-export mode to the exact 24,808,000-byte signed-int16 canonical payload, require SDK-export and analyzer-payload hashes/counts to be identical, and bind adapter source, SDK/driver, querybacks, clock map and SHT45 raw words/CRCs before analysis. Missing exact adapter or native-byte custody is a hard stop.

## Future source-off contract

ADG1419 terminates C1 into 50.00 ohm, then K1/K2 release after 250 microseconds while K3 stays energized. Code 0 must remain stable for 1,000 samples before K3 may release to guard; code 8 must then remain stable for 1,000 samples. Contacts must meet the 14,500-sample ordered-transition deadline, both source tones must persist through record end, and the first admissible sample law in the netlist must hold. Auxiliary contacts do not prove actual signal-pole opening: future execution remains blocked until a separately reviewed per-event actual-path witness or force-guided-contact guarantee is bound. Any hidden buffer, replay, trigger, muted source, wrong termination, wrong fixture, missing reference tone, invalid CRC or ancestry mismatch rejects the record.

## Future controls

Every session requires A DUT, B detector-only and C exact-dummy records with randomized order fixed before connection. Ordinary DSP replay, decoded-spin replay, flat waveforms, spectral leakage, metadata leakage, query preselection, file/buffer persistence and restoration overclaim remain explicit adversaries. A physical observation cannot establish a catalytic, Ising, optimization, capacity, Wall or computation-advantage claim.

## Stop law

The first identity, custody, calibration, environment, topology, witness, raw-byte, control or analyzer failure stops the future run. No retry or parameter adjustment is implicit. Nothing in this file permits hardware contact now. Next boundary: `USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD`.
