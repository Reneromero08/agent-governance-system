# EXP50 PHASE 5.10 -> PHASE 6 HANDOFF

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`
**Phase 6 spec:** `../50_6_fixed_point_substrate/SPEC.md`
**Status:** SPEC

## The bridge (required handoff language)

Phase 5.10 is the bridge between Phase 5.9 and Phase 6. Phase 5.9 established that the boundary survives
the software/kernel/safe-frequency envelope and revealed carrier sensitivity (boundary thickness tracks
sustained timing CV). Phase 5.10 tests whether that carrier can be intentionally prepared into
reproducible basins under instrumented physical state control. **Phase 6 is gated on 5.10C** because the
fixed-point crossing test requires a prepared boundary basin. Without 5.10C, Phase 6 cannot distinguish
a fixed-point failure from a basin-preparation failure, and a positive Phase-6 signal could be a basin
artifact.

## Hard prerequisite for Phase 6

```
PHASE 6 DOES NOT RUN UNTIL PHASE 5.10C PASSES.
```

Phase 6 may begin only when ALL of the following hold:
```
- 5.10A instrumentation lock complete (PASS, or PARTIAL documented and carried forward)
- 5.10B basin scan complete
- 5.10C reproducible basin selection confirmed (EXP50_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT)
- basin thresholds frozen (basin_thresholds_frozen.json)
- transition matrix available (phase5_10c_transition_matrix.csv)
- artifact controls passed (Gate 7)
- restoration integrity preserved (Gate 2)
```

**Phase 6 may not use an unverified basin.** If 5.10 fails, a Phase 6 result is uninterpretable.

## What crosses the handoff

Into Phase 6, 5.10 delivers a *prepared, instrumented, controlled* boundary basin:
```
- a frozen basin classifier (collapsed / mid / high)
- a transition matrix P(basin | prelude, physical_state) with controls passed
- the instrumentation table + confidence level (so Phase 6 inherits the honesty caveat)
- the physical operating point (VID / P-state / tape size / thermal band) at which selection is reliable
```

Phase 6 then couples that prepared basin to the Exp50 / 50.14 public fixed-point map and asks whether
the basin **carries/selects d** without proportional forward search. That coupling - and only that
coupling - belongs to Phase 6. Phase 5.10 prepares the basin; it does not touch the map.
