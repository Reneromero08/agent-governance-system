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
CSV: 50_3b_substrate_primitive/results/invariant_probe_summary.csv
Same-final-hash wrong-answer control answer-corr: 0.000
=== VERDICT: RELATIONAL_INVARIANT_CONFIRMED ===
```

## Artifacts

| Artifact | Purpose |
|---|---|
| `50_3b_substrate_primitive/src/catalytic_invariant_probe.c` | Four-snapshot invariant probe with destructive, random reversible, random-answer, shuffled-schedule, and same-final-hash/wrong-answer controls. |
| `50_3b_substrate_primitive/results/invariant_probe_summary.csv` | Target-run metrics for 3 problem families x 8 seeds plus null summaries. |
| `50_3b_substrate_primitive/src/phase3b_angle_rescue_probe.py` | Post-audit rescue verifier that excludes the extracted answer slot and tests non-formula features plus full carrier words on holdout rows. |
| `50_3b_substrate_primitive/results/angle_rescue/PHASE3B_ANGLE_RESCUE_PROBE.md` | Hardening report for the carrier-vs-residual angle. |
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

## Angle Rescue Hardening

The current attack surface is that the original `answer_corr` metric is a formula
oracle: it uses the same relation/Walsh/graph family as `expected_answer`. That
does not collapse the experiment, but it narrows what the result can mean. The
surviving structure must be treated as an encoded relational carrier unless a
non-formula residual feature independently predicts the answer.

Follow-up command:

```bash
python 50_3b_substrate_primitive/src/phase3b_angle_rescue_probe.py
```

Hardening result:

```text
VERDICT: ENCODED_RELATIONAL_CARRIER_RESCUE
rows: 768
restore_rate: 1.000000
gf2_carrier_train_accuracy: 1.000000
gf2_carrier_holdout_accuracy: 1.000000
gf2_carrier_same_model_wrong_accuracy: 0.000000
gf2_carrier_same_model_shuffled_accuracy: 0.562500
gf2_carrier_effect_vs_null: 0.437500
slot24_leak_holdout_accuracy: 1.000000
```

The non-formula scalar residual lane did not separate on holdout rows
(`best_nonformula_holdout_accuracy=0.505208`). The full T1/T2 carrier-word lane
did separate over GF(2) while excluding the extracted answer slot. This rescues
the hypothesis as a carrier claim: the tape can carry answer-predictive
relational structure through reversible work slots. The next proof must make
that carrier less hand-authored and more substrate-discovered.

## Interpretation

This result separates tape integrity from answer-carrying structure. The same-final-hash/wrong-answer control proves that restoration alone is not accepted as a primitive: the final hash can match while the extracted answer is wrong and the invariant-answer correlation fails.

The survivor in this harness is a relational carrier derived from graph/parity/Walsh-style transforms of the problem slots and written into reversible work slots during T1/T2. Random reversible writes can restore the tape but do not reproduce the carrier. Destructive writes lose the source structure. Wrong-answer controls keep the final hash but fail the answer gate.

After hardening, the precise reading is:

```text
ENCODED_RELATIONAL_CARRIER_RESCUE
```

That is stronger than tape integrity and weaker than a naturally discovered
public residual primitive. It says the constructed reversible carrier is
answer-predictive without relying on the answer slot, and it beats same-model
wrong-answer and shuffled-label controls.

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
