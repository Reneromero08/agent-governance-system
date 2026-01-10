# Eigen-Spectrum Alignment Protocol Specification

**Version:** 1.0.0
**Status:** Draft
**Date:** 2026-01-10

---

## 1. Overview

This specification defines the Eigen-Spectrum Alignment Protocol (ESAP) for cross-model semantic alignment using eigenvalue spectrum invariance.

### 1.1 Scope

This protocol covers:
- Protocol message types and schemas
- Alignment computation procedures
- Acceptance gates and error codes
- Determinism guarantees
- Security and drift detection

### 1.2 Definitions

| Term | Definition |
|------|------------|
| **Anchor Set** | A fixed set of words/phrases used as reference points |
| **Spectrum Signature** | The eigenvalue spectrum of an anchor distance matrix |
| **Alignment Map** | An orthogonal rotation matrix aligning coordinate systems |
| **MDS** | Multidimensional Scaling - method for deriving coordinates from distances |
| **Procrustes** | Method for finding optimal rotation between point sets |

---

## 2. Protocol Messages

### 2.1 ANCHOR_SET

Defines the reference anchor words for alignment.

```json
{
  "type": "ANCHOR_SET",
  "version": "1.0.0",
  "anchors": [
    {"id": "a001", "text": "dog"},
    {"id": "a002", "text": "love"},
    {"id": "a003", "text": "up"}
  ],
  "anchor_hash": "sha256:abc123...",
  "created_at": "2026-01-10T00:00:00Z"
}
```

**Fields:**
- `anchors`: Array of anchor objects with `id` and `text`
- `anchor_hash`: SHA-256 of canonical anchor list (sorted by id, joined with newlines)

### 2.2 EMBEDDER_DESCRIPTOR

Describes the embedding model used.

```json
{
  "type": "EMBEDDER_DESCRIPTOR",
  "version": "1.0.0",
  "embedder_id": "sentence-transformers/all-MiniLM-L6-v2",
  "weights_hash": "sha256:def456...",
  "dimension": 384,
  "normalize": true,
  "prefix": null
}
```

**Fields:**
- `embedder_id`: Model identifier (HuggingFace path or unique ID)
- `weights_hash`: SHA-256 of model weights (for drift detection)
- `dimension`: Embedding dimension
- `normalize`: Whether embeddings are L2-normalized
- `prefix`: Query prefix if required (e.g., "query: " for E5)

### 2.3 DISTANCE_METRIC_DESCRIPTOR

Specifies the distance metric used.

```json
{
  "type": "DISTANCE_METRIC_DESCRIPTOR",
  "version": "1.0.0",
  "metric": "cosine",
  "normalization": "l2",
  "squared": true
}
```

**Fields:**
- `metric`: Distance metric (`cosine`, `euclidean`, `angular`)
- `normalization`: Embedding normalization (`l2`, `none`)
- `squared`: Whether distances are squared (required for MDS)

### 2.4 SPECTRUM_SIGNATURE

The eigenvalue spectrum - the invariant across models.

```json
{
  "type": "SPECTRUM_SIGNATURE",
  "version": "1.0.0",
  "anchor_set_hash": "sha256:abc123...",
  "embedder_id": "all-MiniLM-L6-v2",
  "eigenvalues": [0.8234, 0.5123, 0.3456, 0.2345, 0.1234],
  "k": 5,
  "effective_rank": 4.2,
  "spectrum_hash": "sha256:789xyz...",
  "computed_at": "2026-01-10T00:00:00Z"
}
```

**Fields:**
- `eigenvalues`: Top k positive eigenvalues (sorted descending)
- `k`: Number of eigenvalues retained
- `effective_rank`: Sum of eigenvalues / max eigenvalue
- `spectrum_hash`: SHA-256 of canonical eigenvalue representation

### 2.5 ALIGNMENT_MAP

The rotation matrix for cross-model alignment.

```json
{
  "type": "ALIGNMENT_MAP",
  "version": "1.0.0",
  "source_embedder": "all-MiniLM-L6-v2",
  "target_embedder": "intfloat/e5-large-v2",
  "anchor_set_hash": "sha256:abc123...",
  "rotation_matrix": [[0.9, 0.1, ...], ...],
  "k": 5,
  "procrustes_residual": 0.0234,
  "map_hash": "sha256:map123...",
  "computed_at": "2026-01-10T00:00:00Z"
}
```

**Fields:**
- `source_embedder`: Source model ID
- `target_embedder`: Target (reference) model ID
- `rotation_matrix`: k×k orthogonal rotation matrix R
- `procrustes_residual`: Frobenius norm of alignment error
- `map_hash`: SHA-256 of canonical rotation matrix

---

## 3. Computation Procedures

### 3.1 Squared Distance Matrix

For normalized embeddings E (n × d):

```
D²[i,j] = 2(1 - E[i] · E[j])
```

For cosine similarity to squared Euclidean distance conversion.

### 3.2 Classical MDS

Given squared distance matrix D² (n × n):

1. **Centering matrix:** J = I - (1/n)11^T
2. **Double-centered Gram:** B = -½ J D² J
3. **Eigendecomposition:** B = VΛV^T (eigenvalues sorted descending)
4. **Retain positive eigenvalues:** Keep k eigenvalues where λ_i > ε (ε = 1e-10)
5. **MDS coordinates:** X = V_k √Λ_k

### 3.3 Spectrum Signature

The spectrum signature is:
- Top k positive eigenvalues from classical MDS
- Normalized by largest eigenvalue (optional, for comparison)
- Hashed for integrity verification

### 3.4 Procrustes Alignment

Given MDS coordinates X_source (n × k) and X_target (n × k):

1. **SVD of cross-covariance:** X_source^T X_target = UΣV^T
2. **Optimal rotation:** R = VU^T
3. **Aligned coordinates:** X_aligned = X_source R
4. **Residual:** ||X_aligned - X_target||_F

### 3.5 Out-of-Sample Extension (Gower's Formula)

Given new point with squared distances d² to anchors:

1. **Row means of anchor D²:** r_i = mean_j D²[i,j]
2. **Grand mean:** r̄ = mean_i r_i
3. **Mean of new distances:** d̄ = mean_i d²_i
4. **Gower projection:** b_i = -½(d²_i - d̄ - r_i + r̄)
5. **MDS coordinates:** y = Λ^(-½) V^T b
6. **Aligned coordinates:** y_aligned = y R

---

## 4. Acceptance Gates

### 4.1 Spectrum Correlation Threshold

For two spectrum signatures to be considered compatible:

- **Spearman correlation ≥ 0.95** (default threshold τ)
- Pearson correlation reported but not gating

### 4.2 Eigenvalue Positivity

- At least k eigenvalues must be positive (λ > 1e-10)
- Effective rank must be ≥ k/2

### 4.3 Procrustes Residual

- Residual / ||X_target||_F < 0.5 (50% relative error maximum)
- Lower is better; typical good alignment < 0.1

---

## 5. Error Codes

| Code | Name | Description |
|------|------|-------------|
| E001 | ANCHOR_MISMATCH | anchor_hash does not match |
| E002 | EMBEDDER_MISMATCH | weights_hash does not match |
| E003 | METRIC_MISMATCH | Distance metric descriptor mismatch |
| E004 | SPECTRUM_MISMATCH | Spectrum correlation below threshold |
| E005 | INSUFFICIENT_RANK | Not enough positive eigenvalues |
| E006 | ALIGNMENT_FAILED | Procrustes residual too high |
| E007 | SCHEMA_INVALID | Message does not match schema |
| E008 | VERSION_UNSUPPORTED | Protocol version not supported |

### 5.1 Fail-Closed Behavior

On ANY error:
1. Reject the operation
2. Return explicit error code and message
3. Log the failure with full context
4. Do NOT fall back to degraded operation

---

## 6. Determinism Guarantees

### 6.1 Canonical Serialization

- JSON keys sorted alphabetically
- Floats formatted to 10 decimal places
- Arrays maintain insertion order
- UTF-8 encoding, no BOM

### 6.2 Hash Computation

All hashes computed as:
```
SHA-256(canonical_json_bytes)
```

### 6.3 Reproducibility

Two runs with identical inputs MUST produce:
- Byte-identical spectrum signatures
- Byte-identical alignment maps
- Identical hashes

Floating-point operations use double precision with canonical rounding.

---

## 7. Security Considerations

### 7.1 Drift Detection

- Any change in `weights_hash` invalidates all signatures and maps
- Any change in `anchor_hash` invalidates all signatures and maps
- Signatures MUST be recomputed after model updates

### 7.2 Integrity Verification

- All messages include content hashes
- Verify hashes before processing
- Reject on hash mismatch (E001, E002)

### 7.3 Version Compatibility

- Major version changes are breaking (reject E008)
- Minor version changes are backward-compatible
- Patch version changes are internal only

---

## 8. References

- Classical MDS: Torgerson (1952)
- Procrustes analysis: Gower (1975)
- Out-of-sample MDS: Gower (1968)
- Platonic Representation Hypothesis: arXiv:2405.07987
