# ROADMAP Patch: Semiotic Compression Layer (SCL)
Drop this into your current roadmap as a new phase or thread.

## Phase 1.x: Semiotic Compression Layer (SCL) MVP
**Goal:** reduce frontier-model token burn by introducing a symbolic IR that deterministically expands into JobSpec JSON and tool-call plans.

### Deliverables
- [ ] `SCL/CODEBOOK.json` symbol dictionary (macro -> meaning -> expansion templates)
- [ ] `SCL/GRAMMAR.md` syntax rules + examples
- [ ] `SCL/decode.py` symbolic IR -> expanded JSON JobSpec(s) + audit rendering
- [ ] `SCL/validate.py` symbolic syntax + expanded schema validation (error vectors)
- [ ] `SCL/tests/` paired fixtures (symbolic -> expanded) with expected pass/fail
- [ ] `scl` CLI: `decode`, `validate` (and later `run`)

### Tests
- [ ] determinism: same symbolic program => same expanded JSON hash
- [ ] schema: expanded JobSpecs validate
- [ ] negative fixtures: invalid symbol programs fail with clear error vectors
- [ ] token benchmark: show reduction vs baseline “natural governance” prompt

### Success demo
- [ ] a real AGS/Cat-DPT task is completed using <20 tokens of symbolic instructions + deterministic expansion, without re-pasting governance text.

### Notes
- Hash-based externalization (CAS/SPECTRUM) stays as-is; SCL is an additional control-plane compression for repeated semantics.
- Tiny models may be used to emit symbolic programs under strict validation, but are not required for MVP.
