# Q51 Phase Arithmetic Validity — Try2 Results

## Data Source
- URL: https://download.tensorflow.org/data/questions-words.txt
- SHA256: 8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36
- Size (bytes): 603955

## Split
- Total analogies: 19544
- Train analogies: 15635
- Test analogies: 3909

## Model Results
| Model | Status | Pass Rate | Mean Error (rad) | Median Error (rad) | Model Pass |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | ok | 0.822 | 0.4858 | 0.2552 | True |
| all-MiniLM-L12-v2 | ok | 0.848 | 0.4302 | 0.2398 | True |
| all-mpnet-base-v2 | ok | 0.900 | 0.3493 | 0.2052 | True |
| all-distilroberta-v1 | ok | 0.810 | 0.4686 | 0.2695 | True |
| all-roberta-large-v1 | ok | 0.892 | 0.3503 | 0.1979 | True |
| paraphrase-MiniLM-L6-v2 | ok | 0.747 | 0.5950 | 0.3374 | True |
| paraphrase-mpnet-base-v2 | ok | 0.851 | 0.4360 | 0.2532 | True |
| multi-qa-MiniLM-L6-cos-v1 | ok | 0.778 | 0.5324 | 0.3378 | True |
| multi-qa-mpnet-base-dot-v1 | ok | 0.890 | 0.3693 | 0.2222 | True |
| nli-mpnet-base-v2 | ok | 0.680 | 0.6707 | 0.4618 | True |
| BAAI/bge-small-en-v1.5 | ok | 0.824 | 0.4858 | 0.2886 | True |
| BAAI/bge-base-en-v1.5 | ok | 0.842 | 0.4392 | 0.2884 | True |
| BAAI/bge-large-en-v1.5 | ok | 0.912 | 0.3514 | 0.2425 | True |
| thenlper/gte-small | ok | 0.811 | 0.4875 | 0.3058 | True |
| thenlper/gte-base | ok | 0.805 | 0.5314 | 0.3406 | True |
| thenlper/gte-large | ok | 0.806 | 0.5084 | 0.3333 | True |
| intfloat/e5-small-v2 | ok | 0.829 | 0.4756 | 0.2468 | True |
| intfloat/e5-base-v2 | ok | 0.894 | 0.3769 | 0.2424 | True |
| intfloat/e5-large-v2 | ok | 0.888 | 0.3731 | 0.2258 | True |

## Summary
- Models attempted: 19
- Models evaluated: 19
- Models passing: 19
- Fraction passing: 1.000
- Overall hypothesis confirmed: True

## Anti-Pattern Check
- ground_truth_independent_of_R: True
- parameters_fixed_before_results: True
- no_grid_search: True
- would_report_failures: True
- no_goalpost_shift: True

## Notes
- PCA was fit on training split vocabulary only.
- No synthetic data or parameter search was used.

