# Phase 6 Navigation

**Role:** ENTRYPOINT. Directory index and chronological story. Tells you what each
sub-phase is, where it lives, and its status. Not a forward roadmap -- see
`PHASE6_ROADMAP.md` for what's next.

The 14 numbered directories are chronological by build order. Each answers a distinct
question. The dependency chain: 02_fold_audit (no-smuggle gate) is the foundation that
made every later negative trustworthy. Every probe from 03 onward imports its gate.

## Status at a Glance

| # | Directory | Question | Status |
|---|-----------|----------|--------|
| 01 | target_generator | A/B baselines for the fixed-point map | DONE |
| 02 | fold_audit | Is the orientation bit absent from public data? | DONE (MI=0 proven) |
| 03 | nonhermitian_sensors | Can non-Hermitian topology read orientation? | DONE (6/6 FAIL_CHANCE) |
| 04 | holo_phase_substrate | Can the flagship phase cavity read orientation? | DONE (reads min(d,N-d) at 1.0, FAIL_CHANCE orientation) |
| 05 | black_hole_eigen | Can eigen-collimation inside the coset space cross? | DONE (period rings in poly, orientation does not) |
| 06 | superradiant_sieve | Can collective superradiance cross? | DONE (loophole real, no gain, subexp Kuperberg) |
| 07 | dram_rowbuffer | DRAM row-buffer lock-in simulation | DONE (sim partial, no phase) |
| 08 | chiral_phase_kickback | Can a pre-projection chiral tape expose orientation? | DONE (FAIL_CHANCE, hidden control live) |
| 09 | transient_fold_probe | Can the public transient of f(x) carry orientation? | DONE (FAIL_CHANCE, smuggle caught) |
| 10 | cross_core_wormhole | Cross-core .holo traversal via cache + PDN | LIVE (PDN lock-in confirmed, trials=300/mode pending) |
| 11 | pdn_catalytic_tape | Post-mortem sim explaining cross_core sim/hardware gap | DONE |
| 12 | chiral_lane_frontier | 6A: Can public execution geometry generate a fold-odd carrier? | CLOSED (all no-smuggle tracks fail orientation) |
| 13 | substrate_frontier | L2/L3/L4 implementation attempt on Phenom II | PARTIAL (L2/L3 pass, L4 blocked by fold symmetry) |
| 14 | noncollapse_frontier | 6B: Can OrbitState evolve without collapsing to scalar verifier? | ACTIVE (L4B.1-L4B.5 open) |

## What is OPEN

1. **14_noncollapse_frontier, L4B.1-L4B.5** -- Complex phase-bearing OrbitState, reversible
   path-history accumulator, expanded .holo transcript, invariant family beyond
   fold_symmetry, physical substrate mapping.

2. **10_cross_core_wormhole, PDN lock-in** -- Second core pair v4:s5 sweep running on the
   Phenom at /root/slot2_pdn/. Trials=300/mode needed for strict all-9-gates witness.

3. **5.10 gate** -- Phase 5.10C (reproducible basin selection) is the hard prerequisite for
   any Phase 6 Mode C run. Currently: `BASIN_SCAN_NOT_COMPLETED`.

## What is CLOSED

- The construct/substrate frontier is measured-closed at the orientation boundary
  (TERMINUS.md). The Phenom is a real/scalar substrate; the orientation quadrature is
  physically absent from public cosines.
- All no-smuggle tracks (02-09, 12) converge on the same boundary.
- 13_substrate_frontier L4 blocked: forward scan + fold-even verify + target collapse.

## Dependencies

```
02_fold_audit (no-smuggle gate)
  |
  +--> 03_nonhermitian_sensors  --> 04_holo_phase_substrate --> 05_black_hole_eigen --> 06_superradiant_sieve
  +--> 08_chiral_phase_kickback
  +--> 09_transient_fold_probe
  +--> 12_chiral_lane_frontier (Track I uses 10_cross_core_wormhole T300 data)
         |
         +--> 14_noncollapse_frontier (paradigm shift after chiral lane frontier closed)

10_cross_core_wormhole (builds on 5.10 driven lock-in, feeds Track I in 12)
  |
  +--> 11_pdn_catalytic_tape (post-mortem)

13_substrate_frontier (L2/L3 pass, L4 blocked before paradigm shift)
  |
  +--> 14_noncollapse_frontier (doctrine downgrades 13's L2/L3 as "mechanical warmup")
```

## Hard Prerequisite Gate

Phase 6 Mode C does not run until Phase 5.10C passes. See:
`../../50_5_10_encoding_wall/PHASE5_10_TO_PHASE6_HANDOFF.md`

5.10 prepares the basin. Phase 6 couples the prepared basin to the fixed-point map.
Without 5.10C, a Phase 6 null is uninterpretable and a positive could be a basin artifact.

## Key Documents

| Document | Purpose |
|----------|---------|
| PHASE6_ROADMAP.md | **Active forward-looking roadmap.** What's next, in what order. |
| SPEC.md | Original Phase 6 design spec (d-invariant fixed-point test) |
| TERMINUS.md | Synthesis verdict: construct/substrate frontier measured-closed |
| REPORT_SESSION_LATTICE_CLIMB.md | Narrative session log of all 7 walls climbed |
| CAT_CAS_LAB_STATE_AUDIT.md | Lab state coherence check |
| 12_chiral_lane_frontier/PHASE6_MASTER_SYNTHESIS.md | Chiral lane frontier synthesis |
| 14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md | Noncollapse active navigation surface |
| 13_substrate_frontier/EXP50_SUBSTRATE_FRONTIER_STATUS.md | L2/L3/L4 status |
| 10_cross_core_wormhole/STATUS.md | Live Phenom sweep status |
