# Pre-Registration: Q51 Phase Arithmetic Validity (Try2)

Date: 2026-02-03

## Hypothesis
Phase-difference consistency holds for external analogy data when phases are computed
from a global PCA fit on a training split of the analogy vocabulary.

## Prediction
For each evaluated embedding model, mean absolute phase error < pi/4 and pass rate > 0.60.

## Falsification
Mean absolute phase error >= pi/4 OR pass rate <= 0.60 for a model.

## Data Source
- https://download.tensorflow.org/data/questions-words.txt

## Success Threshold
- pass_rate > 0.60 AND mean_error < pi/4 (0.785398...) per model
- overall: at least 60% of successfully evaluated models meet the threshold

## Fixed Parameters
- PCA components: 2
- Phase definition: theta = atan2(PC2, PC1)
- Phase error threshold: 0.785398 radians
- Train/test split ratio: 0.8
- Split seed: 1337
- Batch size: 64
- Normalize embeddings: False

## Model List (Fixed, No Substitutions)
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
- No parameter search or post-hoc threshold changes
- Ground truth is independent of phase metrics

## Notes
- PCA is fit only on the training split vocabulary to preserve train/test separation.
- All results (pass and fail) will be reported.
