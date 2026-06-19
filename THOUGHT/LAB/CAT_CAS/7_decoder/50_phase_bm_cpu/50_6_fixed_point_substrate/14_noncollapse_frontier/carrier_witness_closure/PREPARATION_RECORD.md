# Phase 6B.5 Carrier-Witness Preparation Record

**Branch:** `phase6b/carrier-witness-closure`  
**Base:** `9de22d1e3c076537973abbab2a9e50b21ee8f791`  
**Status:** `REPOSITORY_PREPARATION_COMPLETE__SSH_EXECUTION_PENDING`

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

## Physical work still required

The SSH agent must:

1. audit `/root/slot2_pdn` without altering historical files;
2. restore a valid thermal sensor or stop;
3. integrate the raw writer into the existing Slot 2 receiver;
4. serialize the exact sender/receiver schedule and telemetry;
5. prove one smoke bundle reconstructs itself;
6. freeze source, binary, configuration, routes, seeds, and controls;
7. acquire the route `4:5` campaign and route `2:3` comparator;
8. regenerate summaries and analysis from raw bytes;
9. issue an honest route-scoped closure report.

---

## Claim boundary

Repository preparation does not establish new physical evidence.

Current claim remains:

```text
selected PDN carrier supported at compact channel-summary level
strict reconstructable carrier witness pending
```

The next gate after a valid closure is the external L4B.5B0 human design review. Observability acquisition remains unauthorized.
