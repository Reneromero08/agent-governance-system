# EXP50 PHASE 5.10D - CACHE/ADDRESS TOPOLOGY LIVE RESULT

**Status:** `VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL`

**Scope:** software-only address-layout carrier probe on the owned Phenom II target. No flash, no voltage writes, no board modification, and no physical instrumentation.

## A. Verdict

5.10D found a reproducible software-topology carrier candidate.

This is a clean additional witness datapoint, not a crossing. The effect is controlled by address layout and survives a swapped core assignment. It is still below the encoding wall because it is a scalar/shared-resource timing channel: it shows software-readable substrate state, but not a phase-resolving/quadrature readout and not an answer-predictive fixed-point basin.

## B. Runs

| Run | Output | Analyzer report | Verdict |
| --- | --- | --- | --- |
| victim core 2 / aggressor core 3 | `results/live_5_10d_cache_addr/phase5_10d_cache_address_topology.csv` | `results/live_5_10d_cache_addr/phase5_10d_cache_address_topology_report.json` | `PHASE5_10D_TOPOLOGY_PREP_CANDIDATE` |
| victim core 5 / aggressor core 2 | `results/live_5_10d_cache_addr_swap/phase5_10d_cache_address_topology.csv` | `results/live_5_10d_cache_addr_swap/phase5_10d_cache_address_topology_report.json` | `PHASE5_10D_TOPOLOGY_PREP_CANDIDATE` |

## C. Evidence

Primary run:

- rows: 240
- same-address median cycles/touch: `319.294829`
- random-address median cycles/touch: `312.9943195`
- different-address median cycles/touch: `293.607014`
- compute-only median cycles/touch: `270.0120805`
- no-aggressor median cycles/touch: `270.1871645`
- same-vs-controls absolute median effect: `36.9428405`
- permutation p-value: `0.0004997501`
- family sign agreement: `1.0`

Swapped-core run:

- rows: 240
- same-address median cycles/touch: `321.2311055`
- random-address median cycles/touch: `308.2863225`
- different-address median cycles/touch: `291.2463445`
- compute-only median cycles/touch: `266.638125`
- no-aggressor median cycles/touch: `266.585976`
- same-vs-controls absolute median effect: `32.3227925`
- permutation p-value: `0.0004997501`
- family sign agreement: `1.0`

Per-family effects were positive in all 12 tested family/core-layout combinations.

## D. Interpretation

The carrier appears to be an address-topology/shared-resource effect, not compute-only load. That is useful because it is directly software-controllable and does not depend on the rejected VID, passive-strobe, or free-tone rail story.

The current data does not prove boundary-state preparation. It proves that same-address layout reliably moves the timing carrier relative to matched controls. Because the channel is scalar/real, it corroborates the 5.10 witness story but does not break the encoding wall diagnosed by the canonical 5.10 report.

## E. Classification

```text
VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL
```

This means:

- valid: the channel separates cleanly and reproduces across core layouts;
- scalar: the readout is timing/cache-address geometry, not quadrature/phase-resolving;
- below wall: it cannot by itself recover the odd/disambiguating component required for fixed-point crossing;
- witness only: it strengthens observability, not Phase 6 readiness.

## F. Optional Next Gate

Required next test:

- split address families into train and held-out sets;
- freeze thresholds on train only;
- predict held-out same/different/random layout class;
- run shuffled-layout label nulls;
- require held-out separation and null rejection before Phase 6 can consume the basin.

Even if this optional gate passes, it remains a scalar-substrate result unless the readout is re-encoded into a phase-resolving/quadrature channel.

## G. Do-Not-Overclaim

Do not call this CPU-sings evidence yet.

Do not call this physical Kuramoto, physical Ising, voltage control, rail-state preparation, or fixed-point crossing.

Do not open Phase 6 from this artifact alone.
