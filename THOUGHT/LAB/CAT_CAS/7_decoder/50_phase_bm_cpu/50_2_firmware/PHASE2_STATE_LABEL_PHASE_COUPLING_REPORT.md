# PHASE2_STATE_LABEL_PHASE_COUPLING_REPORT

## Verdict

`STATE_LABEL_PHASE_COUPLING_CANDIDATE_NOT_REPRODUCED`

The read-only state-label coupling probe found a timing/coupling candidate in
one seed window, but the candidate did not reproduce in the second seed window.
Restoration held in both runs.

This keeps the software route alive, but it is not `CPU_SINGS`.

## Artifacts

- Runner: `50_1_subthreshold_msr/src/msr_state_label_phase_coupling_probe.py`
- Initial run: `50_2_firmware/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE.json`
- Seed 5000 run: `50_2_firmware/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_SEED5000.json`
- Long-row run: `50_2_firmware/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_LONG.json`
- Hardened seed 1000 run: `50_2_firmware/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_BALANCED_SEED1000.json`
- Hardened seed 5000 run: `50_2_firmware/PHASE2_STATE_LABEL_PHASE_COUPLING_PROBE_BALANCED_SEED5000.json`

## Hardened Results

| Run | Verdict | Rows | Restore failures | Elapsed balanced accuracy | State-label balanced accuracy | Mode control balanced accuracy | Core control balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| seed 1000 | `STATE_LABEL_PHASE_COUPLING_CANDIDATE` | 360 | 0 | 0.658338 | 0.504080 | 0.426568 | 0.492351 |
| seed 5000 | `STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED` | 360 | 0 | 0.594862 | 0.500000 | 0.464427 | 0.551877 |

The original unbalanced seed 1000 run showed elapsed holdout accuracy `0.677778`
against core/mode controls `0.522222` / `0.411111`. The long-row run collapsed
into class-balance behavior, so the runner was hardened to use balanced
accuracy for verdicts.

## Interpretation

The positive edge is in elapsed timing, not in the internally computed carrier
bit and not in state-label majority voting. That matters:

- restoration is stable
- VID remains a control, not a cause
- state labels are observable
- elapsed timing can separate one seed window above controls
- the effect is not seed-stable yet

Current status:

```text
timing edge observed -> not reproduced -> needs stability sweep
```

## Route Impact

Route 5 advances from:

```text
READ_ONLY_EFFECTIVE_STATE_SELECTOR_FOUND__VID_STILL_FIXED
```

to:

```text
STATE_LABEL_PHASE_COUPLING_CANDIDATE_NOT_REPRODUCED
```

Follow-up compact sweep:

```text
STATE_LABEL_TIMING_EDGE_NOT_STABLE_YET
```

`50_2_firmware/PHASE2_STATE_LABEL_TIMING_EDGE_STABILITY_SWEEP.md` ran six compact
seed/duration cases. Two of six were candidates and all had zero restore
failures, but the acceptance threshold was three candidate runs. The timing edge
therefore remains live but unstable.

This is not:

- `CPU_SINGS`
- `BYTE_READY_HUMAN_REVIEW`
- `SOFTWARE_FIRMWARE_TRUE_WALL`

## Next Exact Action

`STATE_LABEL_TIMING_EDGE_NARROWING_SWEEP`

Run a narrower bounded sweep over:

- seed windows
- row duration
- tape size
- core subset
- load selector

Acceptance criteria:

- zero restoration failures
- balanced accuracy used for all classifier verdicts
- elapsed/state-label model beats mode/core controls by at least `0.10`
- effect reproduces across at least three seed windows
- state labels and timing rows remain joined in the same artifact
- prefer the `rounds=4096` region first; `rounds=8192` did not preserve the
  compact-sweep edge

## Boundary

- No platform setting changes.
- No P0-P3 modification.
- No candidate image construction.
- No external instrumentation.
