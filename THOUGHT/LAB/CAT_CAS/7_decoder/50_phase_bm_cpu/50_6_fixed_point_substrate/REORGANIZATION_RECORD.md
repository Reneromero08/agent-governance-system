# Phase 6 Reorganization Record

**Original date:** 2026-06-14
**Status:** `HISTORICAL_LAYOUT_RECORD__SUPERSEDED_STATUS_FIELDS`
**Current navigation:** `PHASE6_NAVIGATION.md`

This file records the structural reorganization that consolidated Phase 6 under `50_6_fixed_point_substrate/`. Its layout history remains valid; its old task statuses and “pending commit” language are not current authority.

---

## Before

```text
50_phase_bm_cpu/
├── 50_6_cross_core_wormhole/
├── 50_6_dram_rowbuffer/
├── 50_6_pdn_catalytic_tape/
└── 50_6_fixed_point_substrate/
    ├── fragmented boundary/noncollapse containers
    ├── orphaned mechanism directories
    ├── overlapping roadmaps
    └── split cross-cutting material
```

Problems included overlapping Phase 6 framings, orphaned mechanism directories, sibling experiment roots, and mixed historical/active roadmaps.

---

## Consolidated layout

All Phase 6 work now lives under `50_6_fixed_point_substrate/`, ordered by build lineage:

| # | Directory | Role | Current authority |
|---|---|---|---|
| 01 | `target_generator` | fixed-point feeder and baselines | historical feeder |
| 02 | `fold_audit` | no-smuggle/fold foundation | canonical evidence |
| 03 | `nonhermitian_sensors` | reference sensors | historical result |
| 04 | `holo_phase_substrate` | early phase model | historical result |
| 05 | `black_hole_eigen` | eigen reference | historical result |
| 06 | `superradiant_sieve` | collective reference | historical result |
| 07 | `dram_rowbuffer` | DRAM simulation | historical result |
| 08 | `chiral_phase_kickback` | chiral probe | historical result |
| 09 | `transient_fold_probe` | transient probe | historical result |
| 10 | `cross_core_wormhole` | independent PDN carrier work | evidence feeder |
| 11 | `pdn_catalytic_tape` | sim/hardware-gap analysis | historical analysis |
| 12 | `chiral_lane_frontier` | Phase 6A boundary mapping | closed evidence layer |
| 13 | `substrate_frontier` | tape warmup and scalar-route rejection | corrected historical layer |
| 14 | `noncollapse_frontier` | Phase 6B `.holo` architecture | active |

---

## Canonical renamed files

| Canonical file | Role |
|---|---|
| `SPEC.md` | historical pre-pivot specification authority map |
| `TERMINUS.md` | Phase 6A/boundary synthesis |
| `PHASE6_NAVIGATION.md` | canonical Phase 6 entry point |
| `PHASE6_ROADMAP.md` | active master task ledger |
| `14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md` | active Phase 6B task ledger |

---

## Current coherence note

The 2026-06-18 Phase 6B repair does not undo the directory reorganization. It corrects the scientific/status layer that evolved afterward:

- old scalar charter/spec marked superseded;
- ambiguous L3 warmup invalidated and source corrected;
- Class B W_B claim invalidated and design repaired;
- legacy L4A `.holo` distinguished from canonical `HoloObject`;
- semantic integrity and review governance added;
- navigation/roadmaps synchronized.

Use Git history for exact move provenance. Use current navigation and roadmaps for scientific status.
