#!/usr/bin/env python3
"""Apply Haiku-extracted headers to OK papers."""

from pathlib import Path

md_dir = Path(r'd:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\perception\research\papers\markdown')
god_dir = Path(r'd:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\perception\research\papers\god_tier')

papers = {
    '2406.17639.md': [
        {'line': 1, 'h': '# Experiments'},
        {'line': 3, 'h': '## Training Dataset and Setup'},
        {'line': 21, 'h': '## Cross-Modal Alignment'},
        {'line': 41, 'h': '## Classification'},
        {'line': 49, 'h': '## Robustness to Natural Distribution Shift'},
        {'line': 60, 'h': '## Multi-Modal Retrieval'},
        {'line': 84, 'h': '## Ablation Study'},
        {'line': 95, 'h': '## Results Analysis'}
    ],
    '2410.17891.md': [
        {'line': 7, 'h': '## Continuous-time Discrete Diffusion Processes'},
        {'line': 29, 'h': '## Unifying Language Modeling Objectives'},
        {'line': 38, 'h': '## Adaptation'},
        {'line': 62, 'h': '#### Attention Mask Annealing'},
        {'line': 66, 'h': '#### Shift Operation'},
        {'line': 70, 'h': '#### Time-Embedding-Free Architecture'},
        {'line': 74, 'h': '## Sampling'}
    ],
    '2509.19453.md': [
        {'line': 1, 'h': '# Astronomy and the Platonic Representation Hypothesis'},
        {'line': 11, 'h': '# Astronomical Data as Imperfect Phenomena of Forms'},
        {'line': 21, 'h': '#### Model architectures'},
        {'line': 29, 'h': '# Convergence Toward Shared Representations'},
        {'line': 61, 'h': '# Acknowledgements'},
        {'line': 68, 'h': '# Full Crossmodal Results Table'},
        {'line': 76, 'h': '# SDSS Spectra and Domain Shift Challenges'},
        {'line': 84, 'h': '# Further Information On Used Datasets and Models'}
    ],
    '2010.11386.md': [
        {'line': 19, 'h': '###### Abstract'},
        {'line': 27, 'h': '## 1 Introduction'},
        {'line': 54, 'h': '## 2 Background'},
        {'line': 130, 'h': '## 3 Methodology'},
        {'line': 140, 'h': '### 3.1 Knowledge Distillation'},
        {'line': 178, 'h': '### 3.2 Hybrid Dense-Sparse Ranking'},
        {'line': 231, 'h': '## 4 Experimental Setup'},
        {'line': 257, 'h': '## 5 Results'},
        {'line': 326, 'h': '## 6 Conclusions'},
        {'line': 333, 'h': '## Acknowledgements'},
        {'line': 338, 'h': '## References'}
    ],
    '2407.04803.md': [
        {'line': 19, 'h': '# Methods'},
        {'line': 23, 'h': '## Quantization'},
        {'line': 27, 'h': '## Pruning'},
        {'line': 35, 'h': '# Experiments'},
        {'line': 37, 'h': '## Experimental Settings'},
        {'line': 49, 'h': '## Quantization'},
        {'line': 53, 'h': '### Average Return'},
        {'line': 117, 'h': '### Resource Utilization'},
        {'line': 126, 'h': '## Pruning'},
        {'line': 218, 'h': '# Discussions and Findings'},
        {'line': 234, 'h': '# Conclusion'}
    ],
    '2004.05150.md': [
        {'line': 1, 'h': '# Pretraining and Finetuning'},
        {'line': 7, 'h': '#### Attention Pattern'},
        {'line': 11, 'h': '#### Position Embeddings'},
        {'line': 28, 'h': '#### Continued MLM Pretraining'},
        {'line': 55, 'h': '#### Frozen RoBERTa Weights'},
        {'line': 59, 'h': '# Tasks'},
        {'line': 65, 'h': '## Question answering'},
        {'line': 73, 'h': '## Coreference Resolution'},
        {'line': 77, 'h': '## Document Classification'},
        {'line': 81, 'h': '## Results'},
        {'line': 83, 'h': '#### Main Result'},
        {'line': 87, 'h': '#### -large for QA'},
        {'line': 115, 'h': '## Ablations on WikiHop'}
    ],
    '2402.12784.md': [
        {'line': 5, 'h': '# Introduction'},
        {'line': 27, 'h': '# The Vec2Text Method'},
        {'line': 41, 'h': '# Reproduction of vec2text'},
        {'line': 43, 'h': '## Experimental Methodology'},
        {'line': 98, 'h': '## Reproduction Results'},
        {'line': 124, 'h': '# Understanding what impacts Vec2Text effectiveness'},
        {'line': 146, 'h': '## Distance Metric and Pooling Method'},
        {'line': 160, 'h': '## Zero-shot Regime and Bottleneck Pre-training'},
        {'line': 188, 'h': '## Embedding Dimensionality and Quantization'},
        {'line': 227, 'h': '# Mitigation Strategies'},
        {'line': 231, 'h': '## Noise Injection'},
        {'line': 241, 'h': '## Embedding Transformation'},
        {'line': 251, 'h': '# Conclusion'}
    ],
    '1908.10084.md': [
        {'line': 1, 'h': '# Introduction'},
        {'line': 17, 'h': '# Related Work'},
        {'line': 31, 'h': '# Model'},
        {'line': 73, 'h': '## Training Details'},
        {'line': 77, 'h': '# Evaluation - Semantic Textual Similarity'},
        {'line': 81, 'h': '## Unsupervised STS'},
        {'line': 91, 'h': '## Supervised STS'},
        {'line': 149, 'h': '## Argument Facet Similarity'},
        {'line': 199, 'h': '## Wikipedia Sections Distinction'},
        {'line': 221, 'h': '# Evaluation - SentEval'},
        {'line': 266, 'h': '# Ablation Study'},
        {'line': 312, 'h': '# Computational Efficiency'},
        {'line': 334, 'h': '# Conclusion'},
        {'line': 342, 'h': '# Acknowledgments'}
    ],
    '2207.06881.md': [
        {'line': 1, 'h': '# Introduction'},
        {'line': 23, 'h': '# Related work'},
        {'line': 33, 'h': '# Recurrent Memory Transformer'},
        {'line': 78, 'h': '# Experiments'},
        {'line': 96, 'h': '# Results'},
        {'line': 252, 'h': '# Conclusions'},
        {'line': 268, 'h': '# Checklist'},
        {'line': 316, 'h': '# Training details and additional results'},
        {'line': 318, 'h': '## Algorithmic tasks'},
        {'line': 344, 'h': '## Associative retrieval'},
        {'line': 348, 'h': '## Quadratic equations'},
        {'line': 368, 'h': '## Enwik8'},
        {'line': 400, 'h': '## WikiText-103'},
        {'line': 467, 'h': '# Operations with Memory'}
    ],
    '2404.13950.md': [
        {'line': 1, 'h': '# Introduction'},
        {'line': 7, 'h': '# Related Works'},
        {'line': 9, 'h': '#### Efficient Late Interaction Retrieval'},
        {'line': 13, 'h': '#### Hybrid Models'},
        {'line': 17, 'h': '# Method'},
        {'line': 29, 'h': '#### Adapting Representations'},
        {'line': 33, 'h': '#### SPLATE'},
        {'line': 55, 'h': '#### Efficient Candidate Generation for Late Interaction'},
        {'line': 69, 'h': '# Experiments'},
        {'line': 71, 'h': '#### Setting'},
        {'line': 85, 'h': '#### Latency Results'},
        {'line': 91, 'h': '#### Approximation Quality'},
        {'line': 99, 'h': '#### Overall Results'},
        {'line': 158, 'h': '# Conclusion'}
    ]
}


def apply_headers(filename, headers):
    src = md_dir / filename
    dst = god_dir / filename

    content = src.read_text(encoding='utf-8', errors='replace')
    lines = content.split('\n')

    headers_sorted = sorted(headers, key=lambda x: x['line'], reverse=True)

    for h in headers_sorted:
        line_num = h['line']
        header_text = h.get('h', '')
        if line_num < len(lines):
            lines.insert(line_num, '')
            lines.insert(line_num, header_text)

    result = '\n'.join(lines)
    dst.write_text(result, encoding='utf-8')
    return len(headers)


if __name__ == '__main__':
    for fname, hdrs in papers.items():
        try:
            count = apply_headers(fname, hdrs)
            print(f'{fname}: {count} headers applied')
        except Exception as e:
            print(f'{fname}: ERROR - {e}')
