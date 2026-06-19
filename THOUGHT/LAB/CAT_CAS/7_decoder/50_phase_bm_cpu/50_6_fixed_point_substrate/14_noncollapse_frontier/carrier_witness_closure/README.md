# Phase 6B.5 — PDN Carrier-Witness Closure

**Branch:** `phase6b/carrier-witness-closure`  
**Predecessor:** Phase 6B coherence repair, squash commit `9de22d1e3c076537973abbab2a9e50b21ee8f791`  
**Status:** `CONTRACT_AND_TOOLING_PREPARED__PHYSICAL_EXECUTION_PENDING`  
**Claim ceiling:** reconstructable channel-level carrier evidence only

---

## Purpose

The existing T300 result established a selected-route PDN carrier at the scored-summary level. It retained compact JSON summaries and reported that the raw matrix CSVs remained on the Phenom host, but it did not preserve the per-window timestamp/ring-period arrays from which each lock-in I/Q value was computed.

This stage closes that evidence gap.

A closed carrier witness must make this chain independently reconstructable:

```text
source + binary + host/configuration
→ deterministic sender schedule
→ absolute TSC-aligned raw ring-period samples
→ per-window lock-in I/Q and off-bin floor
→ per-symbol complex vectors
→ per-run matched-null metrics
→ route/session aggregate
→ bounded channel-level claim
```

A compact score without the raw predecessor does not close the witness.

---

## Current evidence boundary

Supported now:

```text
selected Phenom route 4:5 carried sender-owned mode and relational phase
compact T300 summaries survived six seeds and matched controls
```

Still missing from the repository witness:

```text
raw per-window TSC samples
raw ring-period samples
absolute t0 and capture deadlines per run
complete schedule serialization
source/binary/configuration hashes
raw-to-summary reconstruction proof
thermal/frequency telemetry tied to each window
cryptographic root manifest
```

The existing result remains valid channel evidence. This stage upgrades provenance and reconstructability; it does not retroactively promote the claim.

---

## Files

| File | Role |
|---|---|
| `CARRIER_WITNESS_CONTRACT.md` | Binding scientific and evidentiary contract |
| `RAW_ARTIFACT_SCHEMA.md` | Portable raw bundle and binary record format |
| `carrier_witness_validate.py` | Hash, structure, raw lock-in reconstruction, and closure validator |
| `audit_existing_t300.py` | Non-destructive audit of old host/repository artifacts |
| `SSH_AGENT_HANDOFF.md` | Exact physical-machine execution handoff |

The original acquisition/scoring stack remains under:

```text
10_cross_core_wormhole/slot2_pdn/
```

The original imported T300 summaries remain under:

```text
12_chiral_lane_frontier/pdn_slot2_t300/
```

---

## Required order

```text
1. Audit old host artifacts without modifying them.
2. Determine whether raw per-window arrays exist anywhere.
3. If absent, instrument the receiver to emit the raw bundle defined here.
4. Restore thermal observability before any new full acquisition.
5. Run preflight and one short raw smoke capture.
6. Prove raw-to-I/Q reconstruction on the smoke capture.
7. Freeze campaign ID, source commit, binary hash, routes, seeds, controls, and thresholds.
8. Acquire the full campaign.
9. Validate every run and regenerate every summary.
10. Issue a route-scoped closure verdict.
```

No observability/operator acquisition begins in this stage.

---

## Closure states

```text
PENDING
PARTIAL
CLOSED_ROUTE_4_5
CLOSED_MULTI_ROUTE
INVALID
```

`CLOSED_ROUTE_4_5` is allowed when route `4:5` is fully reconstructable and the route scope is explicitly frozen. Route `2:3` may remain a partial topology comparator.

---

## Forbidden promotion

Carrier-witness closure does not establish:

```text
physical HoloGeometry
observable relational state sufficiency
identified state-transition operator
physical path memory
physical restoration
public target coupling
fold-odd invariant
orientation recovery
Small Wall crossing
```

The next gate after closure is the external L4B.5B0 human design review, followed by a separate authorization decision for observability acquisition.
