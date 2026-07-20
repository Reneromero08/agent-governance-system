# P0 post-qualification audit disposition

**Original audited head:** `388e5944834138a435838ddb470714c90b122465`
**Preserved result:** `P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED`  
**Disposition:** `P0_RESONANCE_LOAD_LAW_REPAIR_ESTABLISHED`
**Restored status:** `P0_BUILD_READINESS_PACKET_FROZEN`
**Claim ceiling:** `NON_EXECUTING_P0_BUILD_READINESS_ONLY`

The post-qualification audit correctly identified documentation and analysis-law defects; it did not invalidate the previously established synthetic signal-path witness result.

## Closed findings

- `A-02`: the nominal 32768 Hz operational dependency is replaced by a strict calibration-bound `f_carrier_hz`, `f_carrier_u95_hz`, and `f_witness_hz` tuple. Calibration must precede assignment and primary acquisition.
- `A-03`: the 131072-point result is now named a complete binary-corner sweep, not a continuous uncertainty envelope.
- `A-04`: the unsupported derived common-mode metric is removed. Differential clipping remains; future common-mode observability is a separate powered operating-envelope prerequisite.
- `A-05`: the earlier four reviews are described accurately as role-separated root-bound declarations, not externally reproducible independence. One focused final review covers the exact resonance/load repair root.

## Mechanical evidence

```text
P0_RESONANCE_LOAD_SANITY_MODEL.json
p0_resonance_load_law.py
test_p0_resonance_load_law.py
p0_scientific_analyzer.py
P0_BUILD_READINESS_SCHEMAS.json
P0_SCIENTIFIC_FIXTURES.json
P0_ANALYZER_REFERENCE_RESULTS.json
P0_SIGNAL_PATH_ORDERING_PROOF.json
```

The calibration artifact now binds DUT-A/FC135 as one global pre-assignment calibration identity; native SDK-export, canonical payload, adapter and analyzer hashes; the full raw CH0/CH1 sampling schema and frequency grid; fitted complex background and gain; selected frequency and U95; Q and U95; residual and conditioning metrics; source and instrument querybacks; start/completion times; and `primary_observed=false`. Every later role binds that same reference while separately binding its own primary assembly/population. The removed-resonator and dummy controls are never relabeled as resonance-producing calibrations. The raw-byte analyzer fits both channels itself and then fits `B+C/single-pole`; it no longer accepts a pre-normalized synthetic-perfect inverse response. The analyzer rejects missing or changed frequency custody, flat/background-only data, competing resonances, invalid Q/frequency, insufficient signal, excessive noise/background/residual, clipping, malformed grids, late calibration, queryback mismatch, and a broken witness relation.

This is a narrow calibration-realism correction. It preserves the carrier architecture, OPA810 topology, relay topology, C2 signal-path witness, netlist, BOM, fabrication design, complete binary-corner model, and all previous signal-path conclusions. The recalibrated 0.030 off-resonance gate is a direct consistency correction: a single pole at twenty linewidths has magnitude about 0.025, so the earlier 0.020 literal could not admit its own selected model.

The subsequent narrow settling-law correction preserves that realistic fitter and freezes a 5.000 s minimum dwell per frequency block. A 143-entry monotonic chronology binds every command, acquisition interval, commanded frequency and payload-block hash before fitting. The targeted first-order complex-state fixtures establish that valid 5.000 s and longer dwell passes through the existing fit gates at the accepted Q ceiling, while 0.050 s, 0.500 s, one-nanosecond-short, missing, duplicated, reordered, nonmonotonic, frequency-mismatched, chronology-hash-mismatched and payload-order-mismatched controls reject for their intended mechanisms. This disposition is `P0_CALIBRATION_SETTLING_LAW_ESTABLISHED`; it changes no carrier, fitter, path witness, netlist, BOM or fabrication result.

## Boundary

All evidence remains offline and synthetic. No hardware, target, vendor, cart, purchase, power, playback, recording, acquisition, or physical calibration was contacted or performed.

The next boundary is exactly:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```
