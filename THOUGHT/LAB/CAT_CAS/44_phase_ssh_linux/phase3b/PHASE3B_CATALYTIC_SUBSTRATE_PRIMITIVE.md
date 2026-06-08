# PHASE3B_CATALYTIC_SUBSTRATE_PRIMITIVE

## Verdict

`RELATIONAL_INVARIANT_CONFIRMED`

The Phase 3B four-snapshot invariant probe ran on the Phenom II target and passed the current CAT_CAS primitive gate for the constructed catalytic carrier test. This does not prove physical Kuramoto, quantum coherence, Landauer violation, microscopic entropy reduction, zero heat, or physical holography.

The accepted claim is narrower:

```text
CAT_CAS computes through reversible relational invariants in this harness: a restored tape preserves an answer-predictive carrier structure at T1/T2 that beats destructive and random reversible nulls.
```

## Target Run

Command shape:

```bash
gcc -O2 catalytic_invariant_probe.c -lm -o catalytic_invariant_probe
./catalytic_invariant_probe
```

Target stdout:

```text
=== PHASE 3B: CATALYTIC INVARIANT PROBE ===
families=3 seeds_per_family=8

Rows accepted: 24/24
CSV: phase3b/results/invariant_probe_summary.csv
Same-final-hash wrong-answer control answer-corr: 0.000
=== VERDICT: RELATIONAL_INVARIANT_CONFIRMED ===
```

## Artifacts

| Artifact | Purpose |
|---|---|
| `session_scripts/phase3b/catalytic_invariant_probe.c` | Four-snapshot invariant probe with destructive, random reversible, random-answer, shuffled-schedule, and same-final-hash/wrong-answer controls. |
| `phase3b/results/invariant_probe_summary.csv` | Target-run metrics for 3 problem families x 8 seeds plus null summaries. |
| `PHASE3B_CATALYTIC_SUBSTRATE_PRIMITIVE.md` | This report. |

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| T0 before tape | Pass | `strength_t0=1.000` for all 24 catalytic cases. |
| T1 disturbed / working tape | Pass | Catalytic carrier strength `strength_t1=1.000` for all 24 cases. |
| T2 answer-extracted tape | Pass | Catalytic carrier + answer slot strength `strength_t2=1.000` for all 24 cases. |
| T3 restored tape | Pass | `restored=1.000`, `strength_t3=1.000` for all 24 catalytic cases. |
| Answer prediction | Pass | `answer_correct=1.000`, `answer_corr=1.000` in catalytic summary. |
| Destructive null | Pass | `destructive_write` summary has `restored=0.000`, `answer_correct=0.000`, and carrier strength `0.125`. |
| Random reversible null | Pass | `random_reversible_write` restores final tape (`restored=1.000`) but has no catalytic carrier at T1/T2 (`strength_t1=0.000`, `strength_t2=0.000`). |
| Random answer null | Pass | Restores final tape, but `answer_correct=0.000` and `answer_corr=0.000`. |
| Shuffled schedule null | Pass | Carrier rejected and restoration fails for the shuffled carrier order in this harness. |
| Same final hash but wrong answer | Pass | Final tape restores (`restored=1.000`) but answer correlation is zero (`answer_corr=0.000`). |

## Interpretation

This result separates tape integrity from answer-carrying structure. The same-final-hash/wrong-answer control proves that restoration alone is not accepted as a primitive: the final hash can match while the extracted answer is wrong and the invariant-answer correlation fails.

The survivor in this harness is a relational carrier derived from graph/parity/Walsh-style transforms of the problem slots and written into reversible work slots during T1/T2. Random reversible writes can restore the tape but do not reproduce the carrier. Destructive writes lose the source structure. Wrong-answer controls keep the final hash but fail the answer gate.

## Decision Gate

The same transform family repeatedly identifies the same answer-predictive survivor across 24/24 cases.

Decision:

```text
PROMOTE_TO_PHASE4_3_RESIDUAL_CHANNEL_DESIGN
```

## Boundaries

- Do not claim physical Kuramoto.
- Do not claim CPU-sings physical phase lock.
- Do not claim quantum coherence.
- Do not claim Landauer violation.
- Do not claim microscopic entropy reduction.
- Do not claim zero heat on CMOS.
- Do not claim physical holography.

## Next Action

Use the confirmed carrier invariant as the seed for Phase 4.3 residual-channel design:

```text
PHASE4_3_RESIDUAL_CHANNEL
```

The next test should determine whether the answer-predictive carrier can be compressed into `.holo` residual tags while preserving layer individuality and still restoring the tape.
