# TODO: Phase 3 — Message cassette (LLM-in-substrate communication)

**Roadmap Phase:** Phase 3 (Not Started)

## Tasks

- [ ] Add tables / files for messaging:
  - [ ] `messages` (planner + worker requests)
  - [ ] `jobs` / `steps` (claimable units)
  - [ ] `receipts` (append-only)

- [ ] Implement job lifecycle:
  - [ ] `post(message)` → job created
  - [ ] `claim(job_id, worker_id)` → exclusive lock
  - [ ] `complete(job_id, receipt)` → stored + immutable

- [ ] Enforce: message payload must be structured (refs/ops/budgets), not prose-only.

- [ ] Provide minimal "ant" runtime contract:
  - [ ] reads a job
  - [ ] resolves only allowed symbols/slices
  - [ ] executes ops
  - [ ] writes receipt + outputs

## Exit Criteria

- [ ] A job can be posted, claimed, executed, and completed with receipts.
- [ ] A worker cannot expand beyond budgets.

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `CAT_CHAT_ROADMAP_V1.md` (Phase 3)
