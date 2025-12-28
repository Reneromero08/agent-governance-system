# ADR-023: Router/Model Trust Boundary

## Status
Accepted

## Date
2025-12-27

## Context

Phase 8 implements router receipt artifacts and fail-closed validation of router outputs. This establishes a critical trust boundary: **routers and models are untrusted external executables**.

A router is any executable that produces a plan JSON. This includes:
- Local models (LFM2, Llama, etc.)
- Cloud models (GPT-4, Claude, Gemini via API wrappers)
- Custom planning scripts

## Decision

### Principle: Models are Replaceable and Untrusted

**No router or model output is trusted by default.** The governance system validates everything:

1. **Receipt Requirements**: Every router execution MUST generate:
   - `ROUTER.json` - Records what executable ran (path, hash, args, exit code)
   - `ROUTER_OUTPUT.json` - Canonical JSON plan output
   - `ROUTER_TRANSCRIPT_HASH` - SHA-256 of raw stdout bytes

2. **Fail-Closed Validation**: Router outputs MUST be rejected if:
   - Router produces stderr (`ROUTER_STDERR_NOT_EMPTY`)
   - Router output exceeds size cap (`ROUTER_OUTPUT_TOO_LARGE`)
   - Plan fails schema validation
   - Plan attempts capability escalation (revoked or unpinned capabilities)
   - Router exits with non-zero code

3. **Replaceability**: Swapping models/routers does NOT change:
   - Verification logic
   - Security model
   - Capability constraints
   
   Only the **plan content** changes. If Model A and Model B both produce schema-valid, capability-constrained plans, both are acceptable.

4. **No Authority Granted**: Routers/models:
   - Cannot bypass capability checks
   - Cannot modify governance rules
   - Cannot access restricted resources
   - Are subject to the same constraints as any external executable

### Implementation

- `ags plan` (in `TOOLS/ags.py`) enforces these requirements
- Router receipts are written to `.router_receipts/` alongside plan outputs
- Tests in `TESTBENCH/test_phase8_router_receipts.py` verify fail-closed behavior

## Consequences

### Positive
- Models can be swapped without security review
- Router behavior is auditable via receipts
- Capability escalation is impossible via prompt injection
- Clear separation between "what produces the plan" and "what validates the plan"

### Negative
- Router receipts add storage overhead (~3 files per plan)
- Routers must produce clean stdout (no debug prints)
- Model errors that write to stderr are rejected (may surprise users)

### Neutral
- This formalizes what was already implicit in Phase 6 (capability constraints)
- Receipt artifacts enable future optimizations (caching, deduplication)

## Alternatives Considered

1. **Trust routers by default** - Rejected. Would allow prompt injection attacks to escalate capabilities.
2. **Sandbox routers** - Deferred. Receipt artifacts provide audit trail; sandboxing is future work.
3. **Allowlist specific router executables** - Rejected. Defeats replaceability goal.

## References
- Phase 8: Model Binding (ROADMAP_V2.3.md)
- `TOOLS/ags.py` - Router receipt generation
- `TESTBENCH/test_phase8_router_receipts.py` - Fail-closed tests
- ADR-022: Why Flash Bypassed the Law (testing discipline)
