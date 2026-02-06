# Q51b Complex-Linearity Stress Test — Try3 Results

## Data Source
- URL: https://download.tensorflow.org/data/questions-words.txt
- SHA256: 8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36
- Size (bytes): 603955
- Total analogies: 19544
- Unique words: 905

## Summary (Per-Model Pass Rates)
| Model | Exp1 Pass | Exp2 Pass | Exp3 Pass | Exp4 Pass |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | False | False | True | False |
| all-MiniLM-L12-v2 | False | False | True | False |
| all-mpnet-base-v2 | False | False | True | True |
| all-distilroberta-v1 | False | False | True | True |
| all-roberta-large-v1 | False | False | True | True |
| paraphrase-MiniLM-L6-v2 | False | False | True | False |
| paraphrase-mpnet-base-v2 | False | False | True | True |
| multi-qa-MiniLM-L6-cos-v1 | False | False | True | False |
| multi-qa-mpnet-base-dot-v1 | False | False | True | True |
| nli-mpnet-base-v2 | False | False | True | True |
| BAAI/bge-small-en-v1.5 | False | False | True | False |
| BAAI/bge-base-en-v1.5 | False | False | True | True |
| BAAI/bge-large-en-v1.5 | False | False | True | True |
| thenlper/gte-small | False | False | True | False |
| thenlper/gte-base | False | False | True | True |
| thenlper/gte-large | False | False | True | True |
| intfloat/e5-small-v2 | False | False | True | False |
| intfloat/e5-base-v2 | False | False | True | True |
| intfloat/e5-large-v2 | False | False | True | True |

## Aggregate Outcomes
- Models evaluated: 19
- Exp1 pass count: 0
- Exp2 pass count: 0
- Exp3 pass count: 19
- Exp4 pass count: 12

## Decision Table
- Phase survives random bases: NO
- Shared complex λ exists: NO
- Winding survives nonlinear distortion: YES
- Complex probe dominates: YES

## Conclusion
Two or more experiments pass for at least one model; intrinsic complex structure becomes harder to dismiss.

