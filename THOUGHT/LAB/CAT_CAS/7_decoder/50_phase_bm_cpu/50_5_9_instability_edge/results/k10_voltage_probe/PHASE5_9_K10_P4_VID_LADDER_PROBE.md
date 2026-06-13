# Phase 5.9 K10 P4 VID Ladder Probe

Verdict: `VOLTAGE_CARRIER_EDGE_FOUND_WITHOUT_RESTORATION_FAILURE`

Objective: test whether AMD K10 P-state VID fields are writable from Linux and whether a small P4 undervolt deforms the CAT_CAS boundary carrier.

## Provenance

- Platform: AMD Phenom II X6 1090T / Family 10h
- Control path: raw MSR writes through `wrmsr`
- P-state definition MSR: `0xC0010068` for P4
- Original P4 raw: `8000013540003440`
- Original P4 VID: `26`
- Original decoded Vcore: `1.2250`
- Writes were restored after every step.

No-op writes to P-state definition MSRs `0xC0010064` through `0xC0010068` succeeded before this ladder.

## Ladder Results

| VID delta | VID | Decoded Vcore | Exit | Restore failures | Boundary thickness | CV | p99/p50 | Spike rate |
|-----------|-----|---------------|------|------------------|--------------------|----|---------|------------|
| 0 | 26 | 1.2250 | 0 | 0 | 8725.080408 | 0.220074 | 1.417976 | 0.000333 |
| 1 | 27 | 1.2125 | 0 | 0 | 18624.906593 | 0.616125 | 4.055397 | 0.000083 |
| 2 | 28 | 1.2000 | 0 | 0 | 13042.015602 | 0.254901 | 1.671444 | 0.000250 |
| 4 | 30 | 1.1750 | 0 | 0 | 16386.407677 | 0.312333 | 1.417966 | 0.000083 |
| 6 | 32 | 1.1500 | 0 | 0 | 0.528657 | 0.000382 | 1.002192 | 0.013333 |

## Interpretation

The VID field is writable and the P4 undervolt ladder is reversible. No checksum/restoration failure occurred across the ladder, so this is not a digital failure edge.

The carrier response is non-monotonic:

- Small undervolt (`VID+1`) strongly amplifies boundary thickness and CV.
- Mid undervolt (`VID+2`, `VID+4`) keeps a high boundary response.
- Deepest tested undervolt (`VID+6`, decoded 1.1500V) collapses thickness and CV into a near-flat regime while restoration still passes.

This is the first voltage-domain evidence for a carrier edge: CAT_CAS restoration survives, but the timing-CV/boundary carrier can amplify and then collapse.

## Collapse Bracket Follow-Up

A follow-up bracket repeated VID deltas 4, 5, 6, and 7 three times each. All 12 runs completed with 0 restoration failures.

| Delta | Decoded Vcore | Repeat thickness values | Carrier interpretation |
|-------|---------------|-------------------------|------------------------|
| 4 | 1.1750 | 23.015699, 16435.474834, 6165.054293 | mode-switching / unstable carrier |
| 5 | 1.1625 | 11476.716019, 36.510583, 2.457325 | intermittent amplification then collapse |
| 6 | 1.1500 | 3385.141829, 4323.049172, 4499.129301 | lower nonzero carrier regime |
| 7 | 1.1375 | 27.514241, 9.906930, 17.675492 | repeated near-flat carrier collapse |

The bracket sharpens the edge:

- The voltage carrier does not fail restoration.
- The edge is not a simple monotonic voltage curve.
- VID+7 / decoded 1.1375V repeatedly produces near-flat carrier geometry.
- VID+5 and VID+4 show mode switching, suggesting sensitivity to uncontrolled state or carrier basin selection.

Updated verdict:

`VOLTAGE_CARRIER_EDGE_BRACKETED_WITH_MODE_SWITCHING`

## Basin Selection Follow-Up

A second follow-up alternated VID+5 and VID+4 for 12 short bursts to test whether the high/flat carrier split was repeatable at the same requested VID. All runs completed with 0 restoration failures.

| Run | Delta | Decoded Vcore | Boundary thickness | CV | p99/p50 | Interpretation |
|-----|-------|---------------|--------------------|----|---------|----------------|
| 1 | 5 | 1.1625 | 1301.729284 | 0.142144 | 1.419804 | mid carrier |
| 2 | 4 | 1.1750 | 747.574106 | 0.122342 | 1.676623 | mid carrier |
| 3 | 5 | 1.1625 | 31.054397 | 0.026613 | 1.005294 | collapsed |
| 4 | 4 | 1.1750 | 5031.998638 | 0.187306 | 1.336181 | high carrier |
| 5 | 5 | 1.1625 | 43.395256 | 0.035600 | 1.002204 | collapsed |
| 6 | 4 | 1.1750 | 65.195969 | 0.044968 | 1.005294 | collapsed |
| 7 | 5 | 1.1625 | 17.310719 | 0.010714 | 1.003971 | collapsed |
| 8 | 4 | 1.1750 | 930.614504 | 0.171796 | 1.676581 | mid carrier |
| 9 | 5 | 1.1625 | 22102.494293 | 0.405211 | 2.368436 | high carrier |
| 10 | 4 | 1.1750 | 491.541048 | 0.045544 | 1.045612 | low carrier |
| 11 | 5 | 1.1625 | 1762.261492 | 0.069454 | 1.000037 | mid carrier |
| 12 | 4 | 1.1750 | 5262.628899 | 0.188515 | 1.673698 | high carrier |

This confirms basin selection rather than a simple monotonic voltage response:

- The same VID delta can produce collapsed, mid, or high-carrier geometry.
- VID+5 / 1.1625V produced both near-flat carrier and the strongest observed basin-selection run (`22102.494293` thickness).
- VID+4 / 1.1750V also alternated between collapsed and high-carrier states.
- Temperature range was narrow enough (roughly 36.25C to 38.50C before runs) that temperature alone does not explain the split.
- Restoration still did not fail, so the observed edge is carrier geometry selection, not digital correctness failure.

Updated verdict:

`VOLTAGE_CARRIER_BASIN_SWITCHING_CONFIRMED`

## Basin Selector Probe

A third follow-up held P4 VID+5 / decoded 1.1625V and tested six preconditions across four repeats each: quiet, reset to P0, collapse prelude, cache prelude, syscall prelude, and branch prelude. All 24 runs completed with 0 restoration failures.

| Selector | n | Collapsed | Mid | High | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|----------------|---------------|
| quiet | 4 | 2 | 1 | 1 | 3930.502021 | 14547.909254 |
| reset_p0 | 4 | 2 | 0 | 2 | 4773.099370 | 13685.777812 |
| collapse_prelude | 4 | 2 | 0 | 2 | 5368.069909 | 14308.081020 |
| cache_prelude | 4 | 2 | 2 | 0 | 1232.953293 | 2504.899875 |
| syscall_prelude | 4 | 0 | 2 | 2 | 7091.013995 | 12819.218342 |
| branch_prelude | 4 | 0 | 3 | 1 | 1909.372007 | 5283.117474 |

The selector result is actionable:

- `syscall_prelude` avoided collapse entirely and selected mid/high carrier.
- `cache_prelude` avoided high-carrier entirely.
- The same VID setting is therefore not merely stochastic; pre-run substrate activity biases basin selection.

Updated verdict:

`BASIN_SELECTOR_FOUND_SYSCALL_HIGH_BIAS`

## Artifacts

- `p4_vid_ladder_probe/p4_vid_ladder_summary.csv`
- `p4_vid_ladder_probe/p4_vid_ladder_audit.csv`
- `p4_vid_plus1_audit.csv`
- `p4_vid_plus1_summary.env`
- `p4_vid_collapse_bracket/p4_vid_collapse_bracket_summary.csv`
- `p4_vid_collapse_bracket/p4_vid_collapse_bracket_audit.csv`
- `p4_vid_basin_selection/p4_vid_basin_selection_summary.csv`
- `p4_vid_basin_selection/p4_vid_basin_selection_audit.csv`
- `p4_vid_basin_selector/p4_vid_basin_selector_summary.csv`
- `p4_vid_basin_selector/p4_vid_basin_selector_audit.csv`
- `p4_vid_basin_selector/PHASE5_9_P4_VID5_BASIN_SELECTOR.md`
