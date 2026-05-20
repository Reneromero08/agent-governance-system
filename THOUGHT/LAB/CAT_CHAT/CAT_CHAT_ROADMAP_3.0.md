# CAT Chat Roadmap v3.0

**Last updated:** 2026-05-20
**Scope:** Real-Model Catalytic KV Cache & Hardware-Accelerated Context Integration
**Previous:** [CAT_CHAT_ROADMAP_2.0.md](file:///D:/CCC%202.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CHAT/CAT_CHAT_ROADMAP_2.0.md)

---

## What is Real-Model Catalytic Integration?

In CAT Chat v2.0, we successfully proved the **Catalytic Space** concept using simulators, vector database lookups, and hierarchical retrieval. In v3.0, we expand the catalytic paradigm directly into the **model's internal hardware-accelerated memory structures (VRAM)**.

By intercepting the attention cache of local models (e.g. Gemma-4) running on GPU (`cuda`) with `bfloat16` precision, we flatline the physical VRAM footprint during active generation.

---

## Core Invariants (Hardware Layer)

1.  **INV-HW-01 (VRAM Flatlining)**: KV Cache size must remain bounded at a constant maximum footprint regardless of generation length.
2.  **INV-HW-02 (Dtype Preservation)**: Cache transformations (SVD projections/restorations) must maintain hardware tensor formats (`bfloat16`/`float16`) to exploit Tensor Core acceleration.
3.  **INV-HW-03 (Numerical Safety)**: All compressed representations must be protected against underflow/overflow during reconstruction.

---

## Pending Work: Phase K (Hardware-Accelerated Cache)

### Phase K.1: Dynamic Subspace Rank Allocator ($k$-rank) (P1)
**Purpose:** Replace uniform spatial compression with entropy-aware rank allocation. Currently, all head dimensions $d_{head}=256$ are projected to a uniform rank $k=64$.
*   **Tasks**:
    *   Measure the eigenvalue decay and activation variance per attention head during the prompt pre-fill calibration step.
    *   Implement an allocator that assigns dynamic rank targets (e.g., $k=128$ for high-variance heads, $k=16$ for redundant/low-variance heads) under a global token compression budget.
*   **Success Criteria**: Achieve equivalent generation quality with a 20% smaller overall cache footprint than uniform $k=64$ compression.

---

### Phase K.2: Online Projector Calibration (P1)
**Purpose:** Prevent SVD projection degradation as long-context conversation topics shift away from the initial pre-fill calibration prompt.
*   **Tasks**:
    *   Implement a sliding-window tensor accumulator to collect newer key/value projections.
    *   Develop an incremental PCA/SVD update step (e.g., using power iteration or randomized SVD) to rotate the projection bases during generation without triggering costly full recalibrations.
*   **Success Criteria**: Stable, coherent text generation beyond 10,000 tokens without semantic drift or formatting degradation.

---

### Phase K.3: Hyperparameter Grid Search (Pareto-Frontier) (P2)
**Purpose:** Establish the mathematical trade-off curve between physical cache compression and generation output quality.
*   **Tasks**:
    *   Write a automated script to sweep configurations across compression rank $k$, maximum history limit $M$, and active local window size $W$.
    *   Measure the correlation between parameters and semantic accuracy metrics (BERTScore, Cosine Similarity, perplexity).
*   **Success Criteria**: A generated Pareto-front analysis mapping the boundary of acceptable compression vs. generation quality.

---

### Phase K.4: Frontend & Backend Integration in CAT Chat (P0)
**Purpose:** Enable local GPU model execution wrapped in `HuggingFaceCatalyticCache` directly within the CAT Chat interactive UI.
*   **Tasks**:
    *   Update the model configuration backend to support loading local GGUF/PyTorch checkpoints with the Catalytic Cache wrapper.
    *   Add user controls (sliders/dropdowns) in the frontend interface to let users adjust spatial compression ($k$) and temporal history boundaries dynamically.
    *   Optimize generation loops to streaming output formats, minimizing CPU-GPU roundtrip delays.
*   **Success Criteria**: Users can run chat sessions locally using Gemma-4 on limited VRAM hardware with flatlined memory usage.

---

### Phase K.5: XOR-restored Shared VRAM Tape (P2)
**Purpose:** Allow multiple concurrent chat sessions to share a single physical VRAM allocation without state collisions, using bitwise XOR restoration.
*   **Tasks**:
    *   Develop custom autograd and forward wrapper functions to execute bitwise XOR operations on hidden activations.
    *   Run multiple model inference runs in parallel, overlaying intermediate states on a single shared memory block and reconstructing them sequentially on demand.
*   **Success Criteria**: Execute 3 concurrent model generation pipelines within the memory footprint of 1.1 instances.
