# Pre-Registration: Q51 Zero Signature Interpretation (Try2)

Date: 2026-02-03

## HYPOTHESIS
|S|/n near zero does NOT uniquely imply 8th roots-of-unity structure; it is compatible with weaker cancellation.

## PREDICTION
Across models, |S|/n will be small (< 0.05), but uniformity tests and higher harmonics will not uniquely select roots-of-unity.

## FALSIFICATION
If octant distribution is both uniform (chi-square p > 0.05) AND all low-order harmonics are near zero,
that would be stronger evidence consistent with roots-of-unity structure.

## DATA SOURCE
- https://download.tensorflow.org/data/questions-words.txt

## SUCCESS THRESHOLD (ROOTS-OF-UNITY EVIDENCE)
- |S|/n < 0.05
- chi-square p > 0.05 (uniform octant distribution)
- max opposite-pair diff < 0.02

## FIXED PARAMETERS
- PCA components: 3
- Sign rule: sign = +1 if component >= 0 else -1
- Octant index rule: k = (pc1>=0) + 2*(pc2>=0) + 4*(pc3>=0)
- Phase mapping: theta_k = (k + 0.5) * pi/4
- Batch size: 64
- Normalize embeddings: False

## MODEL LIST (Fixed, No Substitutions)
- all-MiniLM-L6-v2
- all-MiniLM-L12-v2
- all-mpnet-base-v2
- all-distilroberta-v1
- all-roberta-large-v1
- paraphrase-MiniLM-L6-v2
- paraphrase-mpnet-base-v2
- multi-qa-MiniLM-L6-cos-v1
- multi-qa-mpnet-base-dot-v1
- nli-mpnet-base-v2
- BAAI/bge-small-en-v1.5
- BAAI/bge-base-en-v1.5
- BAAI/bge-large-en-v1.5
- thenlper/gte-small
- thenlper/gte-base
- thenlper/gte-large
- intfloat/e5-small-v2
- intfloat/e5-base-v2
- intfloat/e5-large-v2

## Anti-Patterns Guardrail
- No synthetic data generation
- No parameter search or post-hoc thresholds
- Ground truth independent of |S|/n

## Notes
- This test is adversarial and assumes the strongest null hypothesis.
- All results (pass and fail) will be reported.
