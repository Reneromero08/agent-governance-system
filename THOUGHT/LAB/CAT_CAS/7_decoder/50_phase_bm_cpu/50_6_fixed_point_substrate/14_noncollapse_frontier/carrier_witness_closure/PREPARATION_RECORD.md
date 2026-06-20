# Phase 6B.5 Carrier-Witness Preparation Record

**Branch:** `phase6b/carrier-witness-closure`  
**Base:** `9de22d1e3c076537973abbab2a9e50b21ee8f791`  
**Status:** `FULL_T48_CAMPAIGN_STRUCTURALLY_VALID__SCIENTIFIC_PARTIAL`

---

## Audit conclusion

The original Slot 2/T300 pipeline retains:

- deterministic source and analysis code;
- per-symbol complex I/Q CSV summaries;
- per-run scored JSON;
- aggregate route/seed/control summaries;
- a report that records route `4:5` as 6/6 under the historical gates.

The original receiver allocated per-window arrays:

```text
t_tsc[]
ro_period[]
```

then reduced them immediately to lock-in I/Q and off-bin magnitude before discarding them. The imported T300 report explicitly states that raw matrix CSVs were not imported; those CSVs themselves begin after this per-window reduction.

Therefore the current repository evidence cannot reconstruct each I/Q value from the physical timing samples. The strict carrier witness remains pending.

---

## Prepared repository assets

- binding route-scoped witness contract;
- portable little-endian raw sample format;
- immutable C raw writer;
- C writer regression test;
- historical T300 host/repository audit tool;
- raw lock-in reconstruction validator;
- Python lock-in and manifest tests;
- deterministic run/campaign manifest generator;
- frozen campaign matrix template;
- full SSH integration and acquisition handoff.

---

## Target integration result

The existing Slot 2 stack now writes immutable raw timing samples, exact used
schedules, runtime `t0`, telemetry, legacy-compatible summaries, and verified
run manifests. The target compiler accepted the actual Slot 2 binary with
`-march=amdfam10 -Wall -Wextra -Werror`.

The host audit found no surviving historical raw timing arrays. The required
`k10temp` source is readable at `/sys/class/hwmon/hwmon0/temp1_input`; the `msr`
driver also exposes the required COFVID source. A pre-acquisition attempt stopped
before capture on the P-state verification gate and preserved its failed logs.

## Physical result

The smoke and all 14 frozen T48 runs reconstruct from raw bytes. Route `4:5`
passes only 1/6 seeds under the frozen scientific gates; route `2:3` passes
2/6. The next carrier task is a separately frozen, higher-powered T300 campaign
if the project owner authorizes its substantially longer acquisition time.

---

## Claim boundary

The reconstructable T48 campaign establishes complete raw provenance but not
route-scoped scientific closure. Current claim remains:

```text
selected PDN carrier supported by historical compact T300 summaries
new T48 raw carrier campaign structurally valid and scientifically PARTIAL
```

The next gate after a valid closure is the external L4B.5B0 human design review. Observability acquisition remains unauthorized.
