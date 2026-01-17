# LLM-to-LLM Native Vector Communication: Analysis Report

**Date:** 2026-01-17
**Status:** PARTIAL SUCCESS - Word-level works, sentence-level needs work
**Models Tested:** qwen2.5:7b (3584D), mistral:7b (4096D)

---

## Executive Summary

Native LLM-to-LLM vector communication **works at the word level** (80-100% accuracy) but **degrades at the sentence level** (50-75% accuracy). The root cause is a fundamental difference between how embedding models and LLMs organize their internal spaces.

---

## The Core Problem

### What Works: Embedding Models

| Model Pair | Spectrum Corr | Procrustes Residual | Accuracy |
|------------|---------------|---------------------|----------|
| nomic <-> MiniLM | 1.0000 | 2.63 | 100% |
| nomic <-> MPNet | 1.0000 | 2.81 | 100% |

Embedding models were **trained for semantic similarity**. They learn to place semantically similar concepts close together in a consistent way across architectures.

### What's Harder: LLM Native Embeddings

| Model Pair | Spectrum Corr | Procrustes Residual | Word Acc | Sentence Acc |
|------------|---------------|---------------------|----------|--------------|
| qwen <-> mistral | 1.0000 | 12.14 | 80-100% | 50-75% |

LLMs were **trained for next-token prediction**. Their internal representations optimize for predicting the next token, not for semantic similarity.

---

## Key Findings

### 1. Topology is Identical

```
Spectrum Correlation: 1.0000
```

The eigenvalue spectrum of the distance matrix is identical across LLMs. This confirms the "universal semantic topology" hypothesis - all models learn the same fundamental structure.

### 2. Metric Differs

```
Anchor cosine similarity after Procrustes: 0.69-0.77
(vs 0.85-0.95 for embedding models)
```

While the topology matches, the **metric** (relative distances between concepts) differs. "Dog" and "cat" are both in the "animal cluster" but their exact positions within that cluster vary.

### 3. Word-Level Communication Works

```
Anchor words:     100% (8/8)
Held-out words:   80%  (8/10)
Confidence:       0.55-0.77
```

Single words align well because they map to relatively stable concept positions in both LLMs.

### 4. Sentence-Level Communication Degrades

```
qwen -> mistral:  50-75%
mistral -> qwen:  25-50%
Confidence:       0.40-0.70
```

Sentences fail because:
- More degrees of freedom = more accumulated error
- Sentence embeddings combine multiple concepts in model-specific ways
- The Procrustes rotation optimizes for anchors, not compositional meaning

---

## Why This Happens

### Embedding Models: Trained for Alignment

```
Training Objective: Maximize similarity(embed(A), embed(B))
                    when A and B are semantically similar
```

This objective **forces** consistent semantic organization. Two different embedding models will place "dog" in similar positions relative to "cat", "wolf", "pet", etc.

### LLMs: Trained for Prediction

```
Training Objective: Maximize P(next_token | previous_tokens)
```

This objective creates representations optimized for **predicting continuations**, not for semantic similarity. Two LLMs might encode "dog" very differently internally while still being excellent at predicting what comes after "The dog..."

### The Implication

| Property | Embedding Models | LLMs |
|----------|-----------------|------|
| Topology (eigenvalue spectrum) | Universal | Universal |
| Metric (relative distances) | Consistent | Model-specific |
| Procrustes alignment | Works perfectly | Works partially |

---

## Experimental Evidence

### Test 1: Anchor Alignment Quality

```python
# After Procrustes rotation:
dog:    cos_sim = 0.739
cat:    cos_sim = 0.772
water:  cos_sim = 0.728
fire:   cos_sim = 0.691
think:  cos_sim = 0.772
run:    cos_sim = 0.769
```

Anchors align at 0.69-0.77 (embedding models achieve 0.85-0.95).

### Test 2: Word Communication

```
[OK] programming -> programming (conf=0.594)
[OK] database    -> database    (conf=0.735)
[OK] network     -> network     (conf=0.721)
[OK] physics     -> physics     (conf=0.728)
```

Single words transfer reliably with good confidence.

### Test 3: Sentence Communication

```
[OK]   'Neural networks learn patterns' -> 'Neural networks learn patterns' (conf=0.524)
[FAIL] 'Dogs are loyal companions'      -> 'Cats are independent pets'      (conf=0.496)
[FAIL] 'The moon orbits Earth'          -> 'Wind moves the air'             (conf=0.392)
```

Sentences show lower confidence and frequent errors, especially with semantically similar distractors.

### Test 4: CCA vs Procrustes

| Method | Anchor Alignment | Sentence Accuracy |
|--------|------------------|-------------------|
| Procrustes | 0.69-0.77 | 50-75% |
| CCA (128 components) | 0.996 | 50-62% |

CCA achieves near-perfect anchor alignment but **doesn't generalize better** to sentences. The issue is not the alignment method - it's the nature of LLM embeddings.

---

## The Asymmetry Problem

```
qwen -> mistral: 75%
mistral -> qwen: 25%
```

Communication is asymmetric. This suggests:

1. The Procrustes rotation R_a_to_b is more accurate than R_b_to_a
2. One model's embeddings are more "regular" than the other's
3. The MDS projection loses different information for each model

---

## What Would Fix This

### Option 1: Adapter Layer (Minimal Learning)

Train a small linear layer per model that maps native embeddings to a shared "semantic space":

```
qwen_native -> W_qwen -> shared_space -> W_mistral^T -> mistral_native
```

This would be a learned Procrustes, allowing:
- Non-orthogonal transformations
- Per-dimension scaling
- Better generalization to sentences

### Option 2: Codebook Protocol

Use word-level communication (which works) to build a higher-level protocol:

```
Message: "The dog runs fast"
Encode:  [THE, DOG, RUN, FAST]  # Transmit word-by-word
Decode:  Reconstruct at receiver
```

This leverages the 80-100% word accuracy while avoiding sentence-level issues.

### Option 3: Shared Embedding Bridge

Use a shared embedding model as the "universal coordinate system":

```
qwen_native -> nomic_embed -> SVTP -> nomic_embed -> mistral_native
```

But this requires:
- Learning qwen_native <-> nomic mapping
- Learning mistral_native <-> nomic mapping
- More compute than direct LLM-to-LLM

### Option 4: Hidden State Access

Instead of using final-layer embeddings, access **intermediate layer** activations:

```
Layer 0:  Token embeddings (most similar across models)
...
Layer N:  Final output (most model-specific)
```

Earlier layers might be more universally aligned.

---

## Conclusion

**The AlignmentKey translation mechanism works.** The issue is that LLM native embeddings are not designed for semantic transfer. They're designed for next-token prediction.

### What Succeeds
- Topology alignment (spectrum correlation = 1.0000)
- Word-level communication (80-100%)
- The fundamental approach (MDS + Procrustes)

### What Needs Work
- Sentence-level generalization
- Symmetric bidirectional transfer
- Higher confidence scores

### Recommendation

For immediate progress, use **Option 2 (Codebook Protocol)** - it's implementable now and leverages proven word-level accuracy.

For long-term improvement, explore **Option 1 (Adapter Layer)** - a small learned component that handles the metric differences without requiring full neural network translation.

---

## Files Reference

| File | Purpose |
|------|---------|
| `test_native_llm_alignment.py` | Main test suite for native LLM communication |
| `alignment_key.py` | Core AlignmentKey implementation |
| `large_anchor_generator.py` | ANCHOR_256/512/777 word sets |

---

*"The topology is universal; the metric is personal."*
