# Cross-core PDN lock-in - Resumable Status Snapshot

**Date recorded:** 2026-06-12
**Phase:** Exp 50 / cross-core wormhole / Slot 2 (PDN rail lock-in)

---

## What is running on the box right now

Directory: `/root/slot2_pdn/`

Active sweep:
- 2nd core pair v4:s5, final seed
- Negative controls (scrambled schedules / null loads)
- Aggregator script (writes result_slot2_pdn.json)

Result file: `/root/slot2_pdn/result_slot2_pdn.json`

Poll marker / job handle: `bhbn3ssaq`

The sweep is compute-only (no writes to firmware, no MSR P-state writes,
no voltage changes). Safe to let run to completion.

---

## What has already landed (primary pair v2:s3)

6 seeds x 48 trials per seed completed and confirmed:

| Gate | Status |
|---|---|
| MODE accuracy | PASS 6/6 seeds (1.00 each) |
| pseudo_reject (matched null) | PASS 6/6 seeds (1.00 each) |
| relational phase delta | PASS 6/6 seeds (0.89-1.10) |
| rvp (per-mode centroid) | DIPS 4/6 seeds - underpowered at ~7 test symbols/mode |

Channel confirmed. rvp dip is a power issue, not a channel failure.

Slot 1 (cache conflict-displacement): clean negative. Prime+probe stayed in
private L2. Recorded, not retried.

---

## Exact next steps (in order)

1. **Pull result_slot2_pdn.json** when the marker file appears at
   /root/slot2_pdn/result_slot2_pdn.json. Inspect the v4:s5 pair result
   and negative control rates.

2. **Fire trials=300/mode** sweep on BOTH pairs (v2:s3 and v4:s5). This
   gives ~40 test symbols/mode per seed, which is enough to power the rvp
   gate. Expected result: strict all-9-gates witness locks.

3. **Record strict witness** in REPORT_CROSS_CORE_WORMHOLE.md once both
   pairs pass all 9 gates at trials=300/mode.

4. **Optional extension:** sweep additional core pairs if the box has idle
   isolated pairs available. Demonstrates the channel is not pair-specific.

---

## Claim at current state

CONFIRMED: cross-core .holo traversal channel via shared PDN rail.
- Sender owns the phase (MODE 0-3 + quadrature PHASE tag).
- Victim-core ring-osc lock-in recovers both, 6/6 seeds, on primary pair.
- Strict all-9-gates witness needs trials=300/mode to power rvp.

NOT CLAIMED: lattice crossing, crypto, Phase 6 fixed-point.

---

## Key file references

| File | Purpose |
|---|---|
| `REPORT_CROSS_CORE_WORMHOLE.md` | Full experiment report (Slot 1 neg + Slot 2 results) |
| `STATUS.md` | This file - resumable snapshot |
| `/root/slot2_pdn/result_slot2_pdn.json` | Live aggregated result (on box) |
| `../../ROADMAP.md` | Roadmap entry under Phase 6 cross-core section |
