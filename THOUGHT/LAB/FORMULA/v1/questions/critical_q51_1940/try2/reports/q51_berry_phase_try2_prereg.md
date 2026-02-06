# Pre-Registration: Q51 Berry Phase / Holonomy Correctness (Try2)

Date: 2026-02-03

## HYPOTHESIS
Gamma computed as sum of angle(z_{i+1}/z_i) is not invariant across projection choices (local vs global PCA); quantization may be projection-induced.

## PREDICTION
Gamma values will vary across projection bases and reflections beyond a small tolerance, and quantization will not be stable.

## FALSIFICATION
If gamma is invariant (mean |Δγ| < 0.2 rad) across projection choices (local vs global PCA) and reflections, and quantization scores remain high,
that would support a valid topological interpretation.

## DATA SOURCE
- https://download.tensorflow.org/data/questions-words.txt

## SUCCESS THRESHOLD (HOLONOMY EVIDENCE)
- mean |Δγ| across bases < 0.2 rad
- quant_score_1_8 > 0.8 across bases

## FIXED PARAMETERS
- Loop size: 4 words
- Loops per category max: 3
- PCA components: 2
- Rotation angles: [0.0, 0.39269908169872414, 0.7853981633974483, 1.5707963267948966]
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
- Loop construction is deterministic from external dataset order

## Notes
- Loops are formed by grouping consecutive words in each category.
- All results (pass and fail) will be reported.
