# PHASE3B_L0_L3_REVERIFY

## Verdict

`CATALYTIC_LAYER_RESCUED_AS_ENCODED_CARRIER_NOT_LOW_LEVEL_PRIMITIVE`

The recheck items were verified against local source. They are real for the older
Phase 3 / Phase 2B bridge code. They do not kill the later encoded-carrier
track, but they do restrict what the early layers are allowed to claim.

## Source Findings

| Item | Source | Finding | Hardened interpretation |
|---|---|---|---|
| L0 immediate restore | `session_scripts/phase3_catalytic/catcas_phase3.c` | `catcas_compute_parity`, `catcas_compute_hash_fragment`, and `catcas_compute_fsm_transition` place the computed value and immediately restore the original value before hash verification. | API/logical-restore demonstration only; not independent catalytic proof. |
| L1 snapshot restore | `session_scripts/phase2b/active_catalytic_ising.c` | Mode C snapshots spins with `memcpy(restore_buf, tape, N*8)` and restores with `memcpy(tape, restore_buf, N*8)`. | Active solver plus memory restore; not borrowed-tape evidence by itself. |
| L2/L3 perfect fidelity | Phase 5.8 / 5.9 reports and harness design | Large-trial "0 failures" claims validate reversible mask self-inversion and memory stability under load. They do not prove computation by themselves. | Memory-stability / restoration-survival evidence, not standalone catalytic-boundary primitive proof. |
| 3B formula coupling | `session_scripts/phase3b/catalytic_invariant_probe.c` | Original `answer_corr` uses the same relation/Walsh/graph family as `expected_answer`. | Rescued by `phase3b_angle_rescue_probe.py` only as encoded relational carrier evidence. |

## Hardening Boundary

Allowed claims:

- Phase 3 L0/L1 establishes usable tape APIs and restoration mechanics.
- Phase 5.8/5.9 large-trial runs establish restoration survival and timing-carrier observability under load.
- Phase 3B hardening establishes `ENCODED_RELATIONAL_CARRIER_RESCUE`: full T1/T2 carrier words predict the answer on holdout rows while the same learned model misses wrong-answer controls.

Disallowed inherited claims:

- Do not treat immediate restore as a catalytic primitive.
- Do not treat snapshot restore as borrowed-tape evidence.
- Do not treat reversible mask self-inversion as computation proof.
- Do not use perfect fidelity alone as proof that a catalytic boundary primitive was discovered.

## Live Hypothesis After Reverify

The strongest surviving interpretation is:

```text
CAT_CAS carrier evidence lives in encoded relational structure, not in raw
restore success. Restoration is the necessary substrate gate; carrier prediction
and null separation are the primitive gate.
```

## Next Push

The next catalytic proof should discover the carrier from modal structure rather
than writing it by construction:

- Walsh slices
- graph Laplacian buckets
- `.holo` eigen slots
- same-final-hash wrong-answer controls
- shuffled/operator-order controls
