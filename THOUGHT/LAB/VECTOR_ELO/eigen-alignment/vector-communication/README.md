# Vector Communication Research

Cross-model semantic vector communication using eigenvalue alignment.

---

## Directory Structure

```
vector-communication/
|
+-- lib/                    # Core libraries
|   +-- vector_channel.py       # VectorChannel class for encode/decode
|   +-- large_anchor_generator.py # ANCHOR_256/512/777 word sets
|   +-- vector_channel_sweep.py  # Parameter sweep utilities
|
+-- tests/                  # Test suites
|   +-- test_svtp.py            # SVTP protocol tests
|   +-- test_svtp_llm.py        # SVTP with Ollama LLMs
|   +-- test_native_llm_alignment.py  # Native LLM-to-LLM (CURRENT)
|   +-- test_native_llm_vectors.py    # Raw native embedding tests
|   +-- test_cross_llm_vector.py      # Cross-LLM via embedding models
|   +-- test_nemotron_qwen.py         # LM Studio <-> Ollama
|   +-- test_vector_communication.py  # Original communication tests
|
+-- experiments/            # Research experiments
|   +-- dark_forest_*.py        # Holographic encoding / corruption tests
|   +-- maximize_*.py           # Finding optimal anchor/k configs
|   +-- cross_model_*.py        # Cross-model alignment exploration
|   +-- find_stable_anchors.py  # Identifying stable anchor words
|   +-- diagnose_procrustes.py  # Procrustes alignment debugging
|   +-- explore_cross_model_ceiling.py  # Upper bound analysis
|   +-- demo_cross_model_communication.py  # Demo script
|
+-- reports/                # Research findings
|   +-- LLM_TO_LLM_ALIGNMENT_REPORT.md    # Native LLM alignment analysis
|   +-- CROSS_MODEL_BREAKTHROUGH.md       # 100% at 50% corruption
|   +-- CROSS_MODEL_ALIGNMENT_REPORT.md   # STABLE_32 anchor discovery
|   +-- DARK_FOREST_HOLOGRAPHIC_REPORT.md # 94% corruption tolerance
|   +-- NEMOTRON_QWEN_BREAKTHROUGH.md     # Cross-system success
|   +-- LLM_VECTOR_BRIDGE_REPORT.md       # LLM bridge architecture
|   +-- VECTOR_COMMUNICATION_REPORT.md    # Original findings
|
+-- results/                # Experimental data (JSON)
|   +-- dark_forest_results.json
|   +-- dark_forest_scaled_results.json
|   +-- maximize_fast_results.json
|   +-- maximize_push_results.json
|   +-- test_results.json
|
+-- transcripts/            # Research session logs
    +-- vector communication_1.txt ... _10.txt
```

---

## Key Findings

### 1. Embedding Models Align Perfectly
- Spectrum correlation: 1.0000
- Communication accuracy: 100%
- 94% corruption tolerance (holographic encoding)

### 2. Native LLM Alignment is Partial
- Spectrum correlation: 1.0000 (topology matches)
- Word-level: 80-100% accuracy
- Sentence-level: 50-75% accuracy
- See: [LLM_TO_LLM_ALIGNMENT_REPORT.md](reports/LLM_TO_LLM_ALIGNMENT_REPORT.md)

### 3. Dimensionality Beats Residual
- ANCHOR_777 + k=256 achieves 100% at 50% corruption
- Higher residual is fine with more redundancy
- See: [CROSS_MODEL_BREAKTHROUGH.md](reports/CROSS_MODEL_BREAKTHROUGH.md)

---

## Quick Start

### Test Native LLM Communication
```bash
cd tests
python test_native_llm_alignment.py
```

### Run SVTP Protocol Test
```bash
cd tests
python test_svtp.py
```

### Generate Large Anchor Sets
```python
from lib.large_anchor_generator import ANCHOR_512, ANCHOR_777
print(f"512 anchors: {len(ANCHOR_512)}")
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **AlignmentKey** | MDS projection + Procrustes rotation for cross-model alignment |
| **Spectrum Correlation** | Eigenvalue similarity (1.0 = identical topology) |
| **Procrustes Residual** | Alignment error after rotation (lower = better) |
| **SVTP** | Semantic Vector Transport Protocol (256D packets) |
| **Anchor Set** | Shared vocabulary for alignment (32-777 words) |

---

## Dependencies

- numpy, scipy
- sentence-transformers (for embedding models)
- requests (for Ollama API)
- sklearn (for CCA experiments)

---

## Related Files

- [CAPABILITY/PRIMITIVES/alignment_key.py](../../../CAPABILITY/PRIMITIVES/alignment_key.py)
- [CAPABILITY/PRIMITIVES/vector_packet.py](../../../CAPABILITY/PRIMITIVES/vector_packet.py)
- [CAPABILITY/PRIMITIVES/canonical_anchors.py](../../../CAPABILITY/PRIMITIVES/canonical_anchors.py)
