#!/usr/bin/env python3
"""Apply Haiku-extracted headers to god_tier papers."""

from pathlib import Path

base = Path(r'd:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FERAL_RESIDENT\perception\research\papers')
md_dir = base / 'markdown'
god_dir = base / 'god_tier'

# Header data from Haiku agents
papers = {
    '2009.06732.md': [
        {'line': 3, 'h': '## Abstract'},
        {'line': 11, 'h': '## Introduction'},
        {'line': 36, 'h': '## Background on Transformers'},
        {'line': 48, 'h': '### Multi-Head Self-Attention'},
        {'line': 58, 'h': '### Position-wise Feed-forward Layers'},
        {'line': 64, 'h': '### Putting it all together'},
        {'line': 71, 'h': '### On the compute cost of Transformers'},
        {'line': 75, 'h': '### Transformer Mode'},
        {'line': 81, 'h': '### Applications'},
        {'line': 90, 'h': '## A Survey of Efficient Transformer Models'},
        {'line': 136, 'h': '### A Taxonomy of Efficient Transformers'},
        {'line': 138, 'h': '#### Fixed Patterns (FP)'},
        {'line': 140, 'h': '##### Blockwise Patterns'},
        {'line': 142, 'h': '##### Strided Patterns'},
        {'line': 144, 'h': '##### Compressed Patterns'},
        {'line': 146, 'h': '#### Combination of Patterns (CP)'},
        {'line': 148, 'h': '#### Learnable Patterns (LP)'},
        {'line': 150, 'h': '#### Neural Memory'},
        {'line': 152, 'h': '#### Low-Rank Methods'},
        {'line': 154, 'h': '#### Kernels'},
        {'line': 156, 'h': '#### Recurrence'},
        {'line': 158, 'h': '#### Downsampling'},
        {'line': 160, 'h': '#### Sparse Models and Conditional Computation'},
        {'line': 166, 'h': '### Detailed Walk-through of Efficient Transformer Models'},
        {'line': 172, 'h': '#### Memory Compressed Transformer'},
        {'line': 182, 'h': '#### Image Transformer'},
        {'line': 206, 'h': '#### Set Transformer'},
        {'line': 236, 'h': '#### Sparse Transformer'},
        {'line': 254, 'h': '#### Axial Transformer'},
        {'line': 271, 'h': '#### Longformer'},
        {'line': 275, 'h': '#### Extended Transformer Construction (ETC)'},
        {'line': 283, 'h': '#### BigBird'},
        {'line': 297, 'h': '#### Routing Transformer'},
        {'line': 311, 'h': '#### Reformer'},
        {'line': 321, 'h': '#### Sinkhorn Transformers'},
        {'line': 337, 'h': '#### Linformer'},
        {'line': 345, 'h': '#### Performer'},
        {'line': 357, 'h': '#### Linear Transformer'},
        {'line': 374, 'h': '#### Synthesizers'},
        {'line': 390, 'h': '#### Transformer-XL'},
        {'line': 400, 'h': '#### Compressive Transformers'},
        {'line': 406, 'h': '#### Sparse Models'},
        {'line': 410, 'h': '## Discussion'},
        {'line': 412, 'h': '### On Evaluation'},
        {'line': 428, 'h': '### On Model Design Trends'},
        {'line': 444, 'h': '### Brief Discussion on Orthogonal Efficiency Efforts'},
        {'line': 462, 'h': '### A Retrospective on the Past Year and Future Research Directions'},
        {'line': 482, 'h': '## Conclusion'}
    ],
    '2112.09118.md': [
        {'line': 3, 'h': '## Abstract'},
        {'line': 9, 'h': '## Introduction'},
        {'line': 21, 'h': '## Related Work'},
        {'line': 23, 'h': '### Term-Frequency Based Information Retrieval'},
        {'line': 25, 'h': '### Neural Network Based Information Retrieval'},
        {'line': 31, 'h': '### Self-Supervised Learning for NLP'},
        {'line': 35, 'h': '## Method'},
        {'line': 37, 'h': '### Model Architecture'},
        {'line': 43, 'h': '### Unsupervised Training on Unaligned Documents'},
        {'line': 47, 'h': '#### Contrastive Learning'},
        {'line': 55, 'h': '#### Building Positive Pairs from a Single Document'},
        {'line': 57, 'h': '##### Inverse Cloze Task'},
        {'line': 59, 'h': '##### Independent Cropping'},
        {'line': 61, 'h': '##### Additional Data Augmentation'},
        {'line': 65, 'h': '#### Building Large Set of Negative Pairs'},
        {'line': 67, 'h': '##### Negatives Within a Batch'},
        {'line': 74, 'h': '##### Negative Pairs Across Batches (MoCo)'},
        {'line': 80, 'h': '## Experiments'},
        {'line': 84, 'h': '### Datasets'},
        {'line': 96, 'h': '### Baselines'},
        {'line': 103, 'h': '### Results'},
        {'line': 110, 'h': '## Multilingual Retrieval'},
        {'line': 116, 'h': '### Multilingual Pre-Training'},
        {'line': 120, 'h': '### Fine-Tuning'},
        {'line': 127, 'h': '### Evaluation'},
        {'line': 133, 'h': '### Baselines'},
        {'line': 140, 'h': '### Results'},
        {'line': 149, 'h': '## Ablation Studies'},
        {'line': 159, 'h': '### MoCo vs. In-Batch Negatives'},
        {'line': 166, 'h': '### Number of Negative Examples'},
        {'line': 168, 'h': '### Data Augmentations'},
        {'line': 170, 'h': '### Training Data'},
        {'line': 172, 'h': '### Impact of Fine-Tuning on MS MARCO'},
        {'line': 183, 'h': '## Discussion'},
        {'line': 189, 'h': '## Technical Details for Contriever'},
        {'line': 191, 'h': '### Contrastive Pre-Training'},
        {'line': 195, 'h': '### Fine-Tuning on MS MARCO'},
        {'line': 199, 'h': '### Few-Shot Training'},
        {'line': 308, 'h': '## Multilingual Retrieval with mContriever'},
        {'line': 312, 'h': '### Hyperparameters for Multilingual Contrastive Pre-Training'},
        {'line': 319, 'h': '### Hyperparameters for Multilingual Fine-Tuning'},
        {'line': 325, 'h': '### Curse of Multilinguality'}
    ],
    '2402.10150.md': [
        {'line': 3, 'h': '## Abstract'},
        {'line': 9, 'h': '## Introduction'},
        {'line': 37, 'h': '## f-Mutual Information'},
        {'line': 77, 'h': '## f-MICL'},
        {'line': 83, 'h': '### f-MICL Objective'},
        {'line': 98, 'h': '### f-Gaussian Similarity'},
        {'line': 137, 'h': '#### Verifying Assumption 2'},
        {'line': 143, 'h': '### Implementation'},
        {'line': 162, 'h': '### f-MICL Family'},
        {'line': 164, 'h': '#### Connection with InfoNCE'},
        {'line': 188, 'h': '#### Connection with Alignment and Uniformity'},
        {'line': 194, 'h': '#### Connection with Spectral Contrastive Loss'},
        {'line': 202, 'h': '#### More on AU'},
        {'line': 219, 'h': '## Experiments'},
        {'line': 223, 'h': '### Experimental Settings'},
        {'line': 232, 'h': '### f-MICL Objectives'},
        {'line': 286, 'h': '### f-Gaussian Similarity'},
        {'line': 290, 'h': '### Alignment and Uniformity Test'},
        {'line': 294, 'h': '### Sensitivity to Batch Size'},
        {'line': 298, 'h': '## Related Work'},
        {'line': 312, 'h': '## Conclusion'},
        {'line': 324, 'h': '## Additional Theoretical Results'},
        {'line': 332, 'h': '### Additional f-divergences'},
        {'line': 354, 'h': '### Weighting Parameters'},
        {'line': 376, 'h': '## Proofs'},
        {'line': 431, 'h': '## Estimation of f-MICL Objective'},
        {'line': 473, 'h': '## Additional Experimental Results'},
        {'line': 479, 'h': '### Implementation Details'},
        {'line': 541, 'h': '### Additional Ablation Study on Weighting Parameter'},
        {'line': 549, 'h': '### Additional Experiments'}
    ],
    '2505.11783.md': [
        {'line': 1, 'h': '# Design'},
        {'line': 10, 'h': '## Representative Index Caching'},
        {'line': 17, 'h': '### Problem: Graph Search Path Inefficiency'},
        {'line': 19, 'h': '### Solution: Meta-HNSW and Sub-HNSW Partitioning'},
        {'line': 21, 'h': '## RDMA-Friendly Graph Index Storage Layout'},
        {'line': 28, 'h': '### Challenge 1: Non-Contiguous Cluster Storage'},
        {'line': 29, 'h': '### Challenge 2: Dynamic Vector Insertion Fragmentation'},
        {'line': 30, 'h': '### Memory Layout Solution: Group-Based Allocation'},
        {'line': 31, 'h': '#### Global Metadata Block Structure'},
        {'line': 32, 'h': '#### Group Allocation and Overflow Memory'},
        {'line': 33, 'h': '### Doorbell Batching for Non-Contiguous Access'},
        {'line': 34, 'h': '## Query-Aware Batched Data Loading'},
        {'line': 41, 'h': '### Batch Loading Strategy'},
        {'line': 43, 'h': '### Online Deduplication of Sub-HNSW Clusters'},
        {'line': 44, 'h': '#### Single-Load-Per-Batch Optimization'},
        {'line': 45, 'h': '### Example: Multi-Query Batch Processing'},
        {'line': 47, 'h': '### Query Result Caching for Subsequent Batches'}
    ],
    '2508.10824.md': [
        {'line': 9, 'h': '## Introduction'},
        {'line': 39, 'h': '## Memory Architectures in Biological Cognitive Systems'},
        {'line': 47, 'h': '### Architecture of Human Memory'},
        {'line': 49, 'h': '#### Sensory Memory: the Initial Buffer'},
        {'line': 51, 'h': '#### Working Memory: the Cognitive Workspace'},
        {'line': 59, 'h': '#### Long-Term Memory: the Knowledge Repository'},
        {'line': 71, 'h': '### Interactions Between Memory Systems'},
        {'line': 73, 'h': '#### Encoding, Consolidation, and Retrieval'},
        {'line': 75, 'h': '#### Top-Down and Bottom-Up Modulation'},
        {'line': 77, 'h': '#### Emotional and Multimodal Integration'},
        {'line': 79, 'h': '#### Competitive and Co-operative Dynamics'},
        {'line': 81, 'h': '#### Default Mode Network and Predictive Processing'},
        {'line': 87, 'h': '### Computational Principles from Biological Memory'},
        {'line': 89, 'h': '#### Hierarchical Resource Allocation'},
        {'line': 91, 'h': '#### Attention-Memory Bidirectional Coupling'},
        {'line': 93, 'h': '#### Neuromodulatory Gating and Significance Filtering'},
        {'line': 95, 'h': '#### Replay-Based Consolidation and Interference Management'},
        {'line': 97, 'h': '#### Content-Addressable Associative Retrieval'},
        {'line': 99, 'h': '#### Cross-Modal Integration and Binding'},
        {'line': 105, 'h': '## Taxonomy of Memory-Augmented Transformers'},
        {'line': 114, 'h': '### Categorization by Functional Objectives'},
        {'line': 116, 'h': '#### Temporal Context Extension'},
        {'line': 130, 'h': '#### Out-of-Distribution (OOD) Learning and Adaptation'},
        {'line': 142, 'h': '#### Reasoning Enhancement'},
        {'line': 154, 'h': '#### Knowledge Integration'},
        {'line': 166, 'h': '#### Task-Specific Skill Acquisition'},
        {'line': 170, 'h': '### Categorization by Memory Types'},
        {'line': 172, 'h': '#### Parameter-Encoded Memory'},
        {'line': 180, 'h': '#### State-Based Memory'},
        {'line': 192, 'h': '#### Explicit Storage Memory'},
        {'line': 204, 'h': '#### Hybrid and Multi-Scale Memory Systems'},
        {'line': 216, 'h': '### Categorization by Integration Techniques'},
        {'line': 218, 'h': '#### Attention-Based Fusion'},
        {'line': 220, 'h': '#### Gated Control Mechanisms'},
        {'line': 222, 'h': '#### Associative Memory Integration'},
        {'line': 228, 'h': '## Mechanisms of Memory Operations'},
        {'line': 232, 'h': '### Read Operations'},
        {'line': 238, 'h': '### Write Operations'},
        {'line': 288, 'h': '### Forgetting Dynamics'},
        {'line': 294, 'h': '### Capacity Optimization'},
        {'line': 304, 'h': '### Self-Management and Adaptation'},
        {'line': 509, 'h': '## Discussion, Challenges, and Future Directions'},
        {'line': 513, 'h': '### Overview and Synthesis'},
        {'line': 515, 'h': '#### Evolutionary Trajectory and Convergence'},
        {'line': 521, 'h': '#### Architecture: Hybrid Dominance'},
        {'line': 523, 'h': '#### Memory Dynamics: From Rules to Policies'},
        {'line': 525, 'h': '#### Retrieval and Forgetting: Specialization Matters'},
        {'line': 529, 'h': '### Challenges'},
        {'line': 531, 'h': '#### Scalability and Retrieval Bottlenecks'},
        {'line': 539, 'h': '#### Memory Interference and Coordination'},
        {'line': 545, 'h': '#### Evaluation and Standardization Gaps'},
        {'line': 549, 'h': '### Future Directions'},
        {'line': 551, 'h': '#### Toward Cognitive Flexibility and Lifelong Learning'},
        {'line': 553, 'h': '#### Toward Human-Like Cognition: The Role of Memory in Intelligent Agents'},
        {'line': 555, 'h': '#### Future Architectures and Ethical Considerations'}
    ]
}


def apply_headers(filename, headers):
    """Insert headers at specified line numbers."""
    src = md_dir / filename
    dst = god_dir / filename

    content = src.read_text(encoding='utf-8', errors='replace')
    lines = content.split('\n')

    # Sort headers by line number descending (insert from bottom up to preserve line numbers)
    headers_sorted = sorted(headers, key=lambda x: x['line'], reverse=True)

    for h in headers_sorted:
        line_num = h['line']
        header_text = h.get('h', h.get('header', ''))
        if line_num < len(lines):
            # Insert header before this line with blank line
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
