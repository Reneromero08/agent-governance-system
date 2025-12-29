# TODO: Phase 4 — Discovery: FTS + vectors (candidate selection only)

**Roadmap Phase:** Phase 4 (Not Started)

## Tasks

- [ ] Add FTS index over sections (title + body).
- [ ] Add embeddings table for sections (vectors stored in DB only).
- [ ] Implement `search(query, k)` returning **section_ids/symbol_ids only**.
- [ ] Implement hybrid search: combine FTS + vector scores (bounded).
- [ ] Store retrieval receipts:
  - [ ] query_hash
  - [ ] topK ids
  - [ ] thresholds
  - [ ] timestamp/run_id

## Exit Criteria

- [ ] Search returns stable candidates for repeated queries on unchanged corpus.
- [ ] No vectors are ever emitted into model prompts (only ids + optionally tiny snippets).

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `CAT_CHAT_ROADMAP_V1.md` (Phase 4)
