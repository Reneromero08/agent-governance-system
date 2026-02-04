# Pre-Registration: Q51b Complex-Linearity Stress Test (Try3)

Date: 2026-02-03

## HYPOTHESIS
Complex-linear structure is not intrinsic; phase arithmetic and winding effects will weaken under random bases and nonlinear distortions.

## PREDICTION
Exp1: Median phase error will exceed π/4 in many random bases.
Exp2: |λ_ab−λ_cd| will not be consistently smaller than random baselines.
Exp3: Winding γ will change materially under sign/tanh distortions.
Exp4: Complex probe will not consistently outperform phase-only probe.

## FALSIFICATION
If Exp1 passes on ≥50 bases, Exp2 beats baseline on ≥60% of bases, Exp3 survives distortions on ≥60% of bases,
and Exp4 complex probe beats phase probe on ≥60% of bases, then intrinsic complex structure is supported.

## DATA SOURCE
- https://download.tensorflow.org/data/questions-words.txt

## SUCCESS THRESHOLD
- Exp1: median phase error < π/4 on ≥50 bases
- Exp2: median |λ_ab−λ_cd| ratio < 0.8 on ≥60% of bases
- Exp3: median |Δγ| < 0.2 on ≥60% of bases
- Exp4: complex probe error ratio < 0.9 on ≥60% of bases

## FIXED PARAMETERS
- Random seed: 1337
- Random bases: 50
- Batch size: 64
- Normalize embeddings: False
- Train split ratio: 0.8
- Loop size: 3
- Loops per category max: 3
- Distortion epsilon: 0.01
- Probe ridge: 1e-06

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
- Random bases fixed by seed

## Notes
- No PCA or data-dependent bases are used.
- All results (pass and fail) will be reported.
