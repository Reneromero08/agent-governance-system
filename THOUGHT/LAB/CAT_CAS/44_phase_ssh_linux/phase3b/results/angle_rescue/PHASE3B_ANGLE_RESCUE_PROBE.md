# Phase 3B Angle Rescue Probe

Verdict: `ENCODED_RELATIONAL_CARRIER_RESCUE`

Objective: test the catalytic invariant hypothesis while excluding the answer-generating relation/Walsh/graph formula from the public predictor.

## Source-Level Findings

- Original Phase 3B `answer_corr` is a formula oracle: it uses relation/Walsh/graph, the same transform family used by `expected_answer`.
- Original `strength_t1` and `strength_t2` prove the carrier slots were written and survived through extraction, but they are not independent answer prediction by themselves.
- The first rescue angle is non-formula survivorship: parity/correlation/MI/holo/checksum/phase features must predict the answer on holdout rows better than shuffled/wrong-answer controls.
- The second rescue angle is encoded relational carrier survivorship: full T1/T2 carrier words, excluding the extracted answer slot, must predict the answer over GF(2) better than wrong/shuffled controls.

## Results

- `rows`: `768`
- `restore_rate`: `1.000000`
- `formula_oracle_accuracy`: `1.000000`
- `best_nonformula_feature`: `checksum_t0`
- `best_nonformula_bit`: `0`
- `best_nonformula_train_accuracy`: `0.553819`
- `best_nonformula_holdout_accuracy`: `0.505208`
- `shuffled_holdout_accuracy`: `0.739583`
- `wrong_answer_holdout_accuracy`: `0.505208`
- `holdout_effect_vs_null`: `-0.234375`
- `slot24_leak_holdout_accuracy`: `1.000000`
- `gf2_nonformula_status`: `INCONSISTENT`
- `gf2_nonformula_train_accuracy`: `0.000000`
- `gf2_nonformula_holdout_accuracy`: `0.000000`
- `gf2_carrier_status`: `SOLVED`
- `gf2_carrier_train_accuracy`: `1.000000`
- `gf2_carrier_holdout_accuracy`: `1.000000`
- `gf2_carrier_same_model_wrong_accuracy`: `0.000000`
- `gf2_carrier_same_model_shuffled_accuracy`: `0.562500`
- `gf2_carrier_effect_vs_null`: `0.437500`
- `gf2_carrier_t2_status`: `SOLVED`
- `gf2_carrier_t2_holdout_accuracy`: `1.000000`

## Interpretation Boundary

This probe does not downgrade the hypothesis. It separates two meanings of invariant: public non-formula residual structure versus encoded relational carrier structure. A carrier rescue means the tape can carry answer-predictive relational structure without relying on the answer slot, but the next proof still has to make that carrier less hand-authored and more substrate-discovered.

## Next Angle

Replace scalar bit probes with modal carriers over the full carrier word basis: Walsh slices, graph Laplacian buckets, and tape-restoration eigen slots, then require holdout separation against same-hash wrong-answer controls.
