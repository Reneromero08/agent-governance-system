# Q51 Zero Signature Interpretation — Try2 Results

## Data Source
- URL: https://download.tensorflow.org/data/questions-words.txt
- SHA256: 8c29b3332afc46f3fb8be04cb5297bf96f39aa7131272dff57869b4485b22a36
- Size (bytes): 603955
- Unique words: 905

## Model Results (|S|/n + cancellation metrics)
| Model | |S|/n | Null E[|S|/n] | Max Pair Diff | chi-square p | dft_m2 | dft_m3 | Roots-of-Unity Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.0547 | 0.0295 | 0.0674 | 0.0000 | 0.0860 | 0.0929 | False |
| all-MiniLM-L12-v2 | 0.0494 | 0.0295 | 0.0541 | 0.0000 | 0.1807 | 0.0752 | False |
| all-mpnet-base-v2 | 0.1881 | 0.0295 | 0.0906 | 0.0000 | 0.1236 | 0.0913 | False |
| all-distilroberta-v1 | 0.1500 | 0.0295 | 0.2000 | 0.0000 | 0.1698 | 0.2904 | False |
| all-roberta-large-v1 | 0.1521 | 0.0295 | 0.0950 | 0.0000 | 0.0577 | 0.1411 | False |
| paraphrase-MiniLM-L6-v2 | 0.1321 | 0.0295 | 0.0939 | 0.0000 | 0.0458 | 0.0776 | False |
| paraphrase-mpnet-base-v2 | 0.0584 | 0.0295 | 0.0751 | 0.0000 | 0.0620 | 0.1010 | False |
| multi-qa-MiniLM-L6-cos-v1 | 0.2030 | 0.0295 | 0.1878 | 0.0000 | 0.0389 | 0.2745 | False |
| multi-qa-mpnet-base-dot-v1 | 0.0078 | 0.0295 | 0.0674 | 0.0000 | 0.1380 | 0.1287 | False |
| nli-mpnet-base-v2 | 0.0305 | 0.0295 | 0.0586 | 0.0000 | 0.1618 | 0.1313 | False |
| BAAI/bge-small-en-v1.5 | 0.0385 | 0.0295 | 0.0608 | 0.0000 | 0.0665 | 0.1022 | False |
| BAAI/bge-base-en-v1.5 | 0.0936 | 0.0295 | 0.1028 | 0.0000 | 0.0577 | 0.1527 | False |
| BAAI/bge-large-en-v1.5 | 0.0760 | 0.0295 | 0.0530 | 0.0000 | 0.0724 | 0.0482 | False |
| thenlper/gte-small | 0.0355 | 0.0295 | 0.0442 | 0.0000 | 0.1122 | 0.0551 | False |
| thenlper/gte-base | 0.0943 | 0.0295 | 0.1149 | 0.0000 | 0.1459 | 0.1416 | False |
| thenlper/gte-large | 0.0218 | 0.0295 | 0.0354 | 0.0000 | 0.1780 | 0.0554 | False |
| intfloat/e5-small-v2 | 0.1325 | 0.0295 | 0.1282 | 0.0000 | 0.0442 | 0.3167 | False |
| intfloat/e5-base-v2 | 0.0392 | 0.0295 | 0.0530 | 0.0005 | 0.0438 | 0.0932 | False |
| intfloat/e5-large-v2 | 0.1601 | 0.0295 | 0.1238 | 0.0000 | 0.0115 | 0.1315 | False |

## Summary
- Models attempted: 19
- Models evaluated: 19
- Roots-of-unity evidence count: 0

## Interpretation (Adversarial)
- |S|/n near zero only constrains the first Fourier component of the octant distribution; it does NOT imply discrete roots-of-unity clustering.
- Small |S|/n is consistent with uniform random phase assignments and with simple opposite-pair cancellation.
- Additional evidence required: (1) octant uniformity, (2) low higher-order harmonics, and (3) direct phase clustering near 8 centers.

