# Track I -- Topology Chirality Map (Phenom II X6 1090T)

**Status:** ROUTE_SELECTED -- Route 4:5 confirmed as measured adjudication carrier from validated T300 data (6/6 seeds). 2/12 routes measured. 10 routes deferred.
**Claim ceiling:** L4 (detector live per route).
**Target:** Inform Track A sender/receiver core selection from measured data, not assumption.

---

## 0. Purpose

The Phenom II has asymmetric inter-core transport paths (PDN coupling, cache coherency
traffic, memory controller proximity). Track A must select sender/receiver cores from
measured topology data. Route 4:5 is the Bayesian prior (6/6 seeds T300) -- not a hard
selection until proven against alternatives.

---

## 1. Platform Topology

| Property | Value |
|---|---|
| CPU | AMD Phenom II X6 1090T (Thuban) |
| Cores | 6 (0-5) |
| Isolated | 2, 3, 4, 5 (isolcpus) |
| OS cores | 0, 1 |
| L1 I/D | 64KB / 64KB per core |
| L2 | 512KB per core |
| L3 | 6MB unified, all cores |
| Interconnect | Crossbar (no CCX split) |

All core pairs share the same L3. No cross-CCX penalty exists on Thuban.

---

## 2. Measured Routes (T300 Seed Evidence)

| Route | Seeds | Real Acc | RVP Range | Phase Delta | Evidence Level |
|---|---|---|---|---|---|
| **4:5** | 6/6 PASS | 0.953-1.000 | 0.954-0.985 | 0.978-1.033 | **strong_prior** |
| **2:3** | 2/6 PASS | 1.000-1.000 | 0.910-0.962 | 0.980-1.032 | route_sensitive_partial |

**Recommendation for Track A:** Use route 4:5 as the adjudication carrier. Route 2:3
is not dead (real_acc=1.0 on all seeds) but needs replication before use.
No other route is currently measured.

---

## 3. Unmeasured Routes (Require Phenom Rust Probe)

| Route | Adjacent? | Same Parity? | Recommendation |
|---|---|---|---|
| 2:4 | No | Yes | Candidate: same-parity cores unmeasured |
| 2:5 | No | No | No topological prior |
| 3:2 | Yes | No | Candidate: adjacent cores (reverse of 2:3) |
| 3:4 | Yes | No | Candidate: adjacent cores |
| 3:5 | No | Yes | Candidate: same-parity cores |
| 4:2 | No | Yes | Candidate: same-parity cores (reverse of 2:4) |
| 4:3 | Yes | No | Candidate: adjacent cores (reverse of 3:4) |
| 5:2 | No | No | No topological prior |
| 5:3 | No | Yes | Candidate: same-parity cores |
| 5:4 | Yes | No | Candidate: adjacent cores (reverse of 4:5) |

---

## 4. Rust Topology Probe

File: `topology_chirality_map.rs`

Sweeps all 12 directional pairs with 24 hidden chiral control instances each.
Measures: hidden_lane_auc, shuffle null 95, phase delta, pin status.

**Compile on Phenom:**
```
rustc -C opt-level=3 topology_chirality_map.rs -o topology_chirality_map
```

**Run on Phenom (as root for isolcpus access):**
```
sudo ./topology_chirality_map THOUGHT/LAB/CAT_CAS/.../topology_chirality/results/
```

**Expected runtime:** ~15-30 minutes for 12 routes at 24 pairs each.

---

## 5. Core Selection Protocol for Track A

When Track A builds `chiral_dual_lane.rs`, the sender/receiver cores MUST be read
from the `recommended_adjudication_pair` field in the Rust probe's output JSON.

**Rules:**
1. If route 4:5 is confirmed live by the Rust probe: use 4:5.
2. If route 4:5 fails but another route is live: use the next-highest-AUC live route.
3. If no route is live: `TOPOLOGY_GATE_NOT_LIVE`. Do not build Track A until the
   detector issue is diagnosed (Track 0 transfer function first).
4. Never hardcode `SENDER_CPU=4, RECEIVER_CPU=5` in Track A source. Read from
   topology_chirality_matrix.json at build time.

---

## 6. Verdict

**TOPOLOGY_MAP_PARTIAL.** The map structure is complete (12 routes, full schema).
2/12 routes have T300 seed evidence. Route 4:5 is the recommended adjudication
carrier, pending Rust probe confirmation of all 12 routes.

**Gate:** Track A core selection requires at minimum:
- Route 4:5 confirmed live by the Rust topology probe
- No other route shows >20% higher hidden_lane_auc than route 4:5
- If an unmeasured route surpasses 4:5, use THAT route instead

---

## 7. Files

| File | Role |
|---|---|
| `topology_chirality_map.py` | Python framework: schema, validation, T300 seed population |
| `topology_chirality_map.rs` | Rust probe for Phenom II (compile and run on target) |
| `results/topology_chirality_matrix.json` | Matrix from T300 seed evidence (2 routes) |
| `PHASE6_TOPOLOGY_CHIRALITY_MAP.md` | This report |

**Next:** Compile and run `topology_chirality_map.rs` on the Phenom II to populate
all 12 routes. Until then, Track A uses route 4:5 as the measured prior.
