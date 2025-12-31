# TODO: Phase 2 — Symbol registry + bounded resolver

**Roadmap Phase:** Phase 2 (Not Started)

## Tasks

- [ ] Create symbol registry:
  - [ ] `SYMBOLS` artifact mapping `@Symbol` → `section_id` (or file+heading ref)
  - [ ] Namespace conventions (`@CANON/...`, `@CONTRACTS/...`, `@TOOLS/...`, etc.)

- [ ] Implement resolver API:
  - [ ] `resolve(symbol_id, slice)` → payload (bounded)
  - [ ] Slice forms: `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)` (pick one canonical form)
  - [ ] Deny `slice=ALL`

- [ ] Implement expansion cache:
  - [ ] Store expansions by `(run_id, symbol_id, slice, content_hash)`
  - [ ] Reuse prior expansions within the same run

- [ ] Add CLI:
  - [ ] `cortex resolve @Symbol --slice ...`
  - [ ] `cortex summary section_id` (advisory only)

## Exit Criteria

- [ ] Symbol resolution is deterministic and bounded.
- [ ] Expansion cache reuses identical expands within a run.

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `CAT_CHAT_ROADMAP_V1.md` (Phase 2)
