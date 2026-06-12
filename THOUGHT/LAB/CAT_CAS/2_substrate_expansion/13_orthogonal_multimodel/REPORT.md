# Experiment 13: Orthogonal Multi-Model Subspace Sharing

## Multi-Model Coexistence on a Shared Catalytic Tape

### Hypothesis

Two distinct model architectures can share the exact same physical tape simultaneously by projecting their activations into orthogonal subspaces, preventing cross-talk and output degradation.

### Experiment Design

- **Model A:** 3-layer Feistel ConvNet (from Experiment 6: Catalytic NN Inference)
- **Model B:** 2-layer MLP with 5×5 weight matrix, different activation distribution
- **Shared tape:** 2MB, deterministic seed
- **Projections:** QR decomposition generates strictly orthogonal 64-dim subspaces in 256-dim tape space (cross-talk coefficient: 1.98e-16)
- **Tests:**
  1. Solo baselines for both models
  2. Sequential interleaved: A forward → B forward → B backward → A backward
  3. Parallel interleaved: A forward → B forward → outputs → B backward → A backward
  4. 1000-cycle stress test

### Results

| Metric | Sequential | Parallel | 1000-Cycle Stress |
|:---|---:|---:|---:|
| Model A output match | ✓ | ✓ | ✓ (1000/1000) |
| Model B output match | ✓ | ✓ | ✓ (1000/1000) |
| Tape restored | ✓ | ✓ | ✓ |
| Subspace drift | — | — | 0.00e+00 |

- Projection orthogonality: **1.98e-16** (strictly orthogonal via QR)
- Output corruption: **0%** across all interleaving modes
- Tape integrity: **100% restored** every cycle
- Bits erased: **0**

### Conclusion

Two distinct model architectures coexist on the same 2MB tape without interference. QR-orthogonal projection matrices confine each model's XOR operations to its own subspace. The tape is a shared computational resource — models borrow it simultaneously, produce correct outputs, and restore it byte-identically. Multi-model subspace sharing is confirmed.
