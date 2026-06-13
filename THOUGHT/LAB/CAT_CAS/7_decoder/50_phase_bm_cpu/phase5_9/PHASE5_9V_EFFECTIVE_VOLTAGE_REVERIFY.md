# PHASE5_9V_EFFECTIVE_STATE_REVERIFY

## Verdict

`REQUESTED_P4_SETTING_CARRIER_BASIN__EFFECTIVE_STATE_UNPROVEN`

The Phase 5.9V requested-setting series was rechecked against scripts and
tracked result artifacts. The series supports that requested P4 definition
changes and timing/CV carrier changes were observed. It does not prove that the
decoded lower VID became the effective silicon operating state.

## Source Findings

- Phase 5.9V scripts record P4 definition set/readback steps and decoded VID
  labels.
- The 5.9V series does not record COFVID status (`0xC0010071`) as a required
  per-row effective-state readback.
- Prior Phase 1/Phase 2 runtime evidence already showed VID clamping behavior.
- A fresh read-only P4 asymmetry oracle found current runtime state:
  - P4 definition: all cores `0x8000013540003440`, decoded VID `0x1A`
  - COFVID VID: `0x12` under the saved load matrix
  - no P4-definition asymmetry oracle candidate

## Hardened Interpretation

Allowed claim:

```text
Requested P4 VID-definition changes and/or P4-state setup correlate with
timing-CV carrier basin changes in the Phase 5.9V runs.
```

Disallowed inherited claim:

```text
Decoded VID labels reached the effective silicon operating state.
```

The current label should be treated as:

```text
REQUESTED_P4_SETTING_CARRIER_BASIN
```

not:

```text
EFFECTIVE_STATE_BASIN_SWITCHING
```

## Required Next Proof

`P4_EFFECTIVE_STATE_CONTROL_PROOF`

Before using requested VID as a physical cause, a future P4-scoped run must record:

- P4 definition before/after
- PSTATE_STATUS before/after
- COFVID_STATUS before/after
- timing/CV carrier metrics
- restore status
- same run row IDs joining MSR state to carrier output

No uncontrolled setting changes. No P0-P3 modification. No platform image action.
