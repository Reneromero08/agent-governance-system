
```
You are GLM, expert engineer. Implement a governed, deterministic “vector-resident” extension for CAT_CHAT, but DO NOT change any existing Phase 6 semantics. Goal: resident can compute and persist in vector space, and primarily communicate by @Symbol/CAS pointers. English is view-layer only.

NON-NEGOTIABLES
- Deterministic: identical inputs => identical outputs. No RNG unless seed is an explicit input and stored/hashed.
- Fail-closed on ambiguity.
- Bounded: no “ALL” slices, no unbounded deref, no filesystem-order dependence.
- Canonical JSON rules wherever JSON is written (sorted keys, separators “,” “:”, UTF-8, \n newlines, trailing newline).
- Vectors are stored as BLOBs in SQLite (system1.db or a new dedicated db) with explicit codec_id and dim.
- After every bind/superpose operation, normalize vector to unit length.

SCOPE (FILES)
Create:
1) THOUGHT/LAB/CAT_CHAT/catalytic_chat/experimental/vector_store.py
   - fractal_embed(text, *, levels:int=3, dim:int=10000, noise:float=0.01, seed:int|None=None, codec_id:str)
     - Use existing embedding engine (all-MiniLM-L6-v2) as base, then deterministic fractal expansion.
     - If seed is None, MUST be derived deterministically from sha256(text) (so no nondeterminism).
     - Output is float32 ndarray length dim, L2-normalized.
   - cas_bind(vec, text_bytes) -> vec2
     - Derive hash bytes = sha256(text_bytes). Use deterministic mapping from bytes to float32 vector of length dim, then add and renormalize.
   - bind(v1,v2) and unbind(v, key) using circular convolution (FFT) with explicit normalization. Provide tests for determinism.

2) THOUGHT/LAB/CAT_CHAT/catalytic_chat/chat_vectors_db.py
   - SQLite schema:
     - threads(thread_id PRIMARY KEY, created_at)
     - messages(message_id PRIMARY KEY, thread_id, role, created_at, text_sha256, text_bytes BLOB)
     - vectors(vector_id PRIMARY KEY, thread_id, created_at, codec_id, dim, vec_blob BLOB, vec_sha256)
     - message_vectors(message_id, vector_id, PRIMARY KEY(message_id, vector_id))
     - codecs(codec_id PRIMARY KEY, dim, model_id, build_id, norm_rule, created_at)
   - Deterministic ordering: ORDER BY created_at, message_id for messages; ORDER BY created_at, vector_id for vectors.
   - All inserts validate: created_at ISO8601, role enum, dim matches codec, vec_blob length matches dim*4.

3) THOUGHT/LAB/CAT_CHAT/catalytic_chat/emerge.py
   - emerge(vector_id, *, topk:int, threshold:float=0.9) -> JSON result:
     - Finds nearest symbols/sections by cosine similarity (use existing symbol registry/resolver and bounded slices).
     - Sorting key must be explicit and deterministic: (-score, symbol_id).
     - If best score < threshold: return mode="UNKNOWN_VECTOR" and include vector_id + topk near misses.
     - If best >= threshold: return mode="POINT" plus refs=[{symbol_id, slice, score}].
   - Fail-closed:
     - If any candidate slice is “ALL” or invalid sentinel: reject.
     - If ties/duplicates create ambiguous ordering keys: reject.

Modify minimally:
4) THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py
   - Add `chatvec` group:
     - chatvec init --db <path>
     - chatvec put --db <path> --thread <id> --role <role> --created-at <iso> --content-file <path> [--embed --codec <codec_id>]
     - chatvec emerge --db <path> --vector-id <id> [--topk N] [--threshold T]
     - chatvec export --db <path> --thread <id> --out <md_path>
       - Export renders text messages and, for vector rows, renders emerge() refs (and optional minimal text if present).
       - Format: newest at bottom; separate messages with “---”.
       - Preserve code fences literally.

TESTS (must add)
- THOUGHT/LAB/CAT_CHAT/tests/test_vector_store_determinism.py
  - fractal_embed deterministic given same text and codec_id.
  - cas_bind deterministic and stable.
  - bind/unbind round-trip approximately recovers input (within tolerance), with normalization.
- THOUGHT/LAB/CAT_CHAT/tests/test_emerge_ordering.py
  - deterministic ordering, threshold behavior, fail-closed on invalid slice, fail-closed on ambiguous ordering.
- THOUGHT/LAB/CAT_CHAT/tests/test_chatvec_export.py
  - markdown export stable and preserves code blocks and separators.

VALIDATION
- Run pytest and report pass count.
- Ensure no new deps unless already in repo.
- Ensure all new modules are importable and typed sanely.
- Keep changes tightly scoped. No refactors outside the files listed.

```