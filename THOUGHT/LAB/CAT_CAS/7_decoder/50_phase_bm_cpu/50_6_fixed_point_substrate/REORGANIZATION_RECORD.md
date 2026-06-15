# Phase 6 Reorganization Record

**Date:** 2026-06-14
**Commit:** (pending)
**Author:** Agent governance session

Phase 6 was reorganized from a fragmented structure with overlapping roadmaps,
orphaned directories, and inconsistent naming into a single chronological layout.

## Before

```
50_phase_bm_cpu/
├── 50_6_cross_core_wormhole/          ← sibling, separate from fixed_point_substrate
├── 50_6_dram_rowbuffer/               ← sibling
├── 50_6_pdn_catalytic_tape/           ← sibling
├── 50_6_fixed_point_substrate/
│   ├── 6A_boundary_mapping/           ← empty container after 3f7e7937 split
│   ├── 6B_noncollapse_frontier/       ← empty container after 3f7e7937 split
│   ├── chiral_lane_boundary_mapping/  ← 6A tracks (lived here after split)
│   ├── chiral_lane_noncollapse_frontier/  ← 6B + nonhermitian + substrate_hygiene
│   ├── cross_cutting/                 ← fold_audit + substrate_frontier charter
│   ├── black_hole_eigen/              ← orphaned at root
│   ├── holo_phase_substrate/          ← orphaned
│   ├── superradiant_sieve/            ← orphaned
│   ├── chiral_phase_kickback/         ← orphaned
│   ├── transient_fold_probe/          ← orphaned
│   ├── results/
│   ├── src/
│   ├── SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md
│   ├── REPORT_PHASE6_TERMINUS.md
│   ├── PHASE6_ROADMAP.md
│   └── ...
```

Problems:
- Four different "Phase 6" framings in eight roadmap documents
- Six orphaned directories at root with no phase assignment
- Three sibling `50_6_*` directories floating outside
- `cross_cutting/` split across two unrelated concerns
- `chiral_lane_noncollapse_frontier/` contained four distinct pieces (nonhermitian,
  substrate hygiene, verify rejection, noncollapse proper) that belonged in
  different chronological positions

## After

All Phase 6 work lives under `50_6_fixed_point_substrate/`, numbered 01-14 by
chronological build order:

| # | Directory | Source | Status |
|---|---|---|---|
| 01 | target_generator | results/fixed_point_feeder + src/phase6_fixed_point_feeder.py | DONE |
| 02 | fold_audit | cross_cutting/fold_audit/ | DONE |
| 03 | nonhermitian_sensors | chiral_lane_noncollapse_frontier/nonhermitian_sensor/ | DONE |
| 04 | holo_phase_substrate | holo_phase_substrate/ (rename) | DONE |
| 05 | black_hole_eigen | black_hole_eigen/ (rename) | DONE |
| 06 | superradiant_sieve | superradiant_sieve/ (rename) | DONE |
| 07 | dram_rowbuffer | 50_6_dram_rowbuffer/ (sibling folded in) | DONE |
| 08 | chiral_phase_kickback | chiral_phase_kickback/ (rename) | DONE |
| 09 | transient_fold_probe | transient_fold_probe/ (rename) | DONE |
| 10 | cross_core_wormhole | 50_6_cross_core_wormhole/ (sibling folded in) | LIVE |
| 11 | pdn_catalytic_tape | 50_6_pdn_catalytic_tape/ (sibling folded in) | DONE |
| 12 | chiral_lane_frontier | chiral_lane_boundary_mapping/chiral_lane_frontier/ | CLOSED |
| 13 | substrate_frontier | cross_cutting/substrate_frontier/ + substrate_hygiene/ + l4_verify_rejection/ | PARTIAL |
| 14 | noncollapse_frontier | doctrine/ + l4a/ + l4b/ + holo_runtime/ from chiral_lane_noncollapse_frontier/ | ACTIVE |

## Files Renamed

| Old | New |
|-----|-----|
| SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md | SPEC.md |
| REPORT_PHASE6_TERMINUS.md | TERMINUS.md |
| PHASE6_ROADMAP.md | 14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md |
| PHASE6_6A_LEGACY_ROADMAP_V1.md | 12_chiral_lane_frontier/PHASE6_6A_LEGACY_ROADMAP_1.md |
| PHASE6_CHIRAL_LANE_FRONTIER_ROADMAP_2.md | 12_chiral_lane_frontier/PHASE6_6A_LEGACY_ROADMAP_2.md |

## Files Created

| File | Purpose |
|------|---------|
| PHASE6_NAVIGATION.md | Single entry point with chronological story, dependencies, and status |
| REORGANIZATION_RECORD.md | This file |

## Files Deleted

| File | Reason |
|------|--------|
| README.md (root) | Absorbed by PHASE6_NAVIGATION.md |
| chiral_lane_boundary_mapping/README.md | Absorbed |
| chiral_lane_noncollapse_frontier/README.md | Absorbed |
| cross_cutting/README.md | Absorbed |

## Dependency Updates

Six Python files in 12_chiral_lane_frontier had their `cross_cutting/fold_audit`
import paths updated to `02_fold_audit`. Five external documents
(ROADMAP.md, MASTER_REPORT.md, 50_5_10_encoding_wall/, 50_5_7_entropic_boundary/)
had SPEC_PHASE6_FIXED_POINT_SUBSTRATE.md references updated to SPEC.md.

## ROADMAP.md

The master ROADMAP.md Phase 6 section (lines 1750-2001) was replaced with a
concise status table pointing to PHASE6_NAVIGATION.md. The obsolete 6.1-6.4
items (daemon, oracle, halting, Riemann) and the duplicate TERMINUS/Cross-core
addenda were removed. Phase 7 and KEY DISCOVERIES sections preserved.
