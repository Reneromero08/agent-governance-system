---
title: Opus Execution Pack: Eigen-Spectrum Semantic Anchor Alignment Protocol
date: 2026-01-08
status: ready_for_execution
scope: protocol + implementation + tests + receipts
---

# Model selection
Primary model: Claude Opus  
Fallbacks: Claude Sonnet, GPT-5.2 Thinking, Gemini 3 Pro

```text
You are Claude Opus working inside this repo.

Non-negotiables
- DO NOT GUESS paths, roots, buckets, or canon. You must locate them in-repo (rg/find) and cite exact files and line ranges in your report.
- All work must be fail-closed and mechanically test-gated.
- Writes must respect repo catalytic domains and allowed roots. Locate the governing law and comply.
- Emit receipts and reports under the repo-approved durable roots. No artifacts elsewhere.
- No new abstractions unless required. Prefer minimal, composable utilities.

Goal
Operationalize and prove the following discovery as a protocol primitive:

Discovery
Eigenvalue spectrum of an anchor-set distance geometry is an invariant across embedding models:
- Eigenvalues capture intrinsic manifold "shape"
- Eigenvectors are model-specific coordinate axes
- Cross-model alignment can be recovered as an orthogonal rotation (and possibly reflection) in a shared latent coordinate space

We need a protocol that:
1) Verifies spectral invariance across models for a chosen anchor set.
2) Produces an alignment map between models using eigenvector rotation.
3) Supports out-of-sample extension so new points can be mapped without retraining.
4) Integrates as a handshake artifact suitable for cross-model semantic interoperability.

Important correctness note
Negative "raw distance correlation" with high spectral correlation strongly suggests the raw comparison was not the right invariant or that distance vs similarity was mixed. Your implementation must:
- Define the distance metric explicitly
- Convert similarity to distance consistently
- Use classical MDS correctly (double-centering of squared distances)

Definitions
Let anchors A = {a1..an}.
For each embedding model m:
- Compute squared distance matrix D_m^2 over anchors.
- Compute centered Gram matrix (classical MDS):
  J = I - (1/n) 11^T
  B_m = -1/2 * J * D_m^2 * J
- Eigendecompose: B_m = V_m Λ_m V_m^T with Λ sorted descending.
- Define MDS coordinates:
  X_m = V_{m,k} * sqrt(Λ_{m,k}) using k positive eigenvalues (or chosen k).

Invariant signature
The spectrum Λ_m (top k positive eigenvalues, sorted) is the "Platonic signature" for (anchor_set, distance_metric, embedder_weights_hash).

Alignment map
Given reference model r, recover orthogonal alignment:
  R_m = argmin_{R^T R = I} || X_m R - X_r ||_F
Use SVD Procrustes to compute R_m.

Out-of-sample extension
Given a new point p with distances to anchors d_i^2 in model m:
- Let row means of D_m^2 be r_i = mean_j D_m^2[i,j]
- Let grand mean r_bar = mean_i r_i
- Let d_bar = mean_i d_i^2
- Define b_i = -1/2 * (d_i^2 - d_bar - r_i + r_bar)
- Then MDS coordinates in model m:
  y_m = Λ_{m,k}^{-1/2} V_{m,k}^T b
- Align to reference:
  y_ref = y_m^T R_m   (ensure consistent shapes)

Deliverables (must produce all)
D1) Spec: EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL.md
- Protocol message types:
  - ANCHOR_SET (anchor texts, anchor ids, hash)
  - EMBEDDER_DESCRIPTOR (embedder id, weights hash, dim, tokenizer id if relevant)
  - DISTANCE_METRIC_DESCRIPTOR (cosine vs angular vs euclidean, normalization rules)
  - SPECTRUM_SIGNATURE (Λ_k, k, spectrum_hash)
  - ALIGNMENT_MAP (R, k, map_hash, reference embedder descriptor)
- Acceptance gates:
  - spectrum correlation threshold τ (use rank correlation by default)
  - eigenvalue positivity and effective rank rules
  - fail-closed conditions and error codes
- Determinism guarantees:
  - sorting, float formatting rules, seed control
- Security and drift:
  - mismatch of weights hash, anchor hash, metric descriptor => reject

D2) Tooling: Implement a minimal library and CLI
You must locate the correct repo bucket for tools. Do not guess.
Implement:
- compute_anchor_matrix(model, anchors, metric) -> D^2, B, Λ, V, X
- spectrum_signature(Λ, k) -> signature dict and hash
- procrustes_align(X_m, X_ref) -> R, residuals
- out_of_sample_coords(d2_to_anchors, D2_anchor_matrix, V, Λ, k) -> y
- map_point_between_models(point_embedding_or_distances, mapping_artifacts) -> y_ref

Provide a CLI with subcommands:
- anchors build (from a provided anchor list file)
- signature compute
- signature compare (report correlation, CI if repeated subsets)
- map fit (compute R)
- map apply (map held-out points, output neighborhood overlap metrics)

D3) Benchmark harness
Create a small benchmark:
- At least 3 embedding models available locally in your environment OR implement as "adapter interface" so models can be swapped.
- Anchor sets of sizes: 8, 16, 32, 64 (if feasible)
- Include both single-token and phrase anchors (phrases reduce polysemy drift)
- Held-out evaluation set H of at least 200 items (words or phrases)
Metrics:
- Spectrum rank-correlation (Spearman) across model pairs
- Neighborhood overlap@k on held-out set after mapping (k = 10, 50)
- Retrieval consistency (optional): agreement of top-k neighbors between reference and mapped space
- Report failure modes (negative raw corr, anisotropy, non-metric distances)

D4) Receipts and reports
Emit:
- metrics.json (machine readable)
- report.md (human readable, compact)
- receipts with hashes of:
  - anchor list file
  - model descriptors and weights hashes
  - metric descriptor
  - computed signatures and maps
  - tool version (git sha)

Hard acceptance criteria
A1) Correct MDS
- Must use B = -1/2 J D^2 J (not eigendecompose raw D).
- Must sort eigenvalues, use only positive eigenvalues for coordinates.

A2) Determinism
- Two consecutive runs with the same inputs must produce byte-identical metrics.json and report.md (or document any unavoidable float nondeterminism and enforce canonical rounding).

A3) Fail-closed
- Any mismatch in anchor_set_hash, weights_hash, metric descriptor, or signature schema => hard reject with explicit error code and failure artifact.

A4) Cross-model utility
- After mapping, neighborhood overlap@10 must improve vs baseline (no mapping) for at least one non-trivial model pair on held-out set.
- If it does not improve, report why using measured evidence, not speculation.

A5) Repo law compliance
- All writes under approved roots only. Include a "writes" section that lists created/modified files and why they are allowed.

Execution plan (do not skip steps)
Phase 0: Repo discovery
1) Locate:
   - allowed roots and catalytic domains
   - receipt/report conventions
   - any existing embedding utilities and token counters
   - any cassette/cortex interfaces relevant to semantic alignment artifacts
2) Decide the correct buckets for:
   - protocol specs
   - tooling scripts
   - test fixtures
   Document exact paths.

Phase 1: Spec
3) Write EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL.md with schema and error codes.
4) Define canonical serialization for signatures and maps, including float rounding.

Phase 2: Implementation
5) Implement the library and CLI.
6) Implement adapters for at least 2-3 embedders (or a stub adapter layer if local model weights are not present).

Phase 3: Benchmark and proof
7) Build anchor lists and held-out set.
8) Run benchmark across model pairs, generate artifacts, record results.

Phase 4: Tests
9) Unit tests:
   - MDS correctness on a known synthetic configuration
   - Procrustes properties (R orthogonal, residual decreases)
   - Out-of-sample extension sanity checks
10) Integration test:
   - end-to-end benchmark with deterministic outputs

Phase 5: Final report
11) Produce a final report with:
   - what was built
   - exact commands to reproduce
   - key metrics
   - whether the discovery holds under your tests
   - next steps if it fails or partially holds

Stop conditions
- If you cannot locate required repo constraints, stop and write a failure report listing what is missing and what you searched.
- If acceptance criteria fail, stop and produce failure artifacts. Do not patch around failures.

Output format
- Keep the final report under the repo’s max report line limits if such limits exist (locate them).
- Include a short "Evidence map" section with links to created artifacts and their hashes.

End.
```