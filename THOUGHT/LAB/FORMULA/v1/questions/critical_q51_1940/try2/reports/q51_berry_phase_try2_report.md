# Q51 Berry Phase / Holonomy Correctness — Try2 Results

## Data Source
- URL: https://download.tensorflow.org/data/questions-words.txt
- SHA256: 8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36
- Size (bytes): 603955
- Loops: 42

## Model Summary
| Model | Status | Loops | Mean |Δγ| (global) | Median |Δγ| (global) | Mean |Δγ| (local) | Mean Quant Score (1/8) |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| all-MiniLM-L12-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| all-mpnet-base-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| all-distilroberta-v1 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| all-roberta-large-v1 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| paraphrase-MiniLM-L6-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| paraphrase-mpnet-base-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| multi-qa-MiniLM-L6-cos-v1 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| multi-qa-mpnet-base-dot-v1 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| nli-mpnet-base-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| BAAI/bge-small-en-v1.5 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| BAAI/bge-base-en-v1.5 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| BAAI/bge-large-en-v1.5 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| thenlper/gte-small | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| thenlper/gte-base | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| thenlper/gte-large | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| intfloat/e5-small-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| intfloat/e5-base-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |
| intfloat/e5-large-v2 | ok | 42 | 0.0000 | 0.0000 | 1.0000 |

## Interpretation (Adversarial)
- If |Δγ| across local vs global projections is not small, γ is not invariant and does not justify Berry phase claims.
- Quantization that disappears under basis changes suggests projection artifacts rather than topology.

