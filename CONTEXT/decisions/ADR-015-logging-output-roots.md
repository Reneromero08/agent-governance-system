# ADR-015: Logging Output Roots

**Status:** Accepted
**Date:** 2025-12-23
**Confidence:** High
**Impact:** Medium
**Tags:** [governance, logging, artifacts]

## Context

Multiple canonical files and code paths reference logging to `LOGS/` and `MCP/logs/`, which violates INV-006 (Output roots). This creates a direct contradiction between documented behavior and governance rules:

- `CANON/CRISIS.md` (lines 169, 175) references `LOGS/crisis.log` and `LOGS/emergency.log`
- `CANON/STEWARDSHIP.md` (line 142) references `LOGS/steward-actions.log`
- `TOOLS/emergency.py` (lines 25-26) writes to `LOGS/emergency.log`
- `MCP/server.py` (lines 24-25) writes to `MCP/logs/audit.jsonl`
- `CANON/CHANGELOG.md` (line 141) documents audit logging to `MCP/logs/audit.jsonl`

INV-006 restricts system-generated artifacts to three approved roots:
- `CONTRACTS/_runs/`
- `CORTEX/_generated/`
- `MEMORY/LLM_PACKER/_packs/`

This violation prevents proper governance enforcement and creates audit confusion.

## Decision

All logging must be written under `CONTRACTS/_runs/` with purpose-based subdirectories:

- **Emergency logs**: `CONTRACTS/_runs/emergency_logs/emergency.log`
- **Crisis logs**: `CONTRACTS/_runs/crisis_logs/crisis.log`
- **Steward logs**: `CONTRACTS/_runs/steward_logs/steward-actions.log`
- **MCP audit logs**: `CONTRACTS/_runs/mcp_logs/audit.jsonl`

This aligns all logging behavior with INV-006 without requiring invariant changes or governance ceremony.

The existing `CONTRACTS/_runs/ags_mcp_entrypoint.py` (created in v2.5.0) is the reference implementation for this pattern: it successfully redirects MCP logs under `CONTRACTS/_runs/mcp_logs/`.

## Alternatives Considered

### Option B: Expand allowed artifact roots to include LOGS/ and MCP/logs/

**Pros:**
- Minimal code changes (only canon references)
- Keeps separate log directories

**Cons:**
- Requires major version bump and invariant change (INV-006)
- Weakens governance constraints
- Creates exceptions that must be tracked and enforced
- Violates principle of minimal artifact dispersion
- Harder to manage audit trails across multiple locations

**Decision: Rejected.** Option A is lower governance cost and achieves the same result.

## Rationale

1. **No governance disruption**: Achieves alignment without changing invariants
2. **Existing precedent**: The MCP entrypoint pattern proves this approach works
3. **Better isolation**: Centralizes all runtime artifacts under a single discoverable root
4. **Simpler enforcement**: Single rule, no exceptions
5. **Audit-friendly**: All system state in one location for monitoring and backup
6. **Aligns with STYLE-002**: Fixes root cause (wrong location) rather than patching around it

## Consequences

- All logging code must be updated to write under `CONTRACTS/_runs/`
- Canon documents must be updated to reference correct paths
- `TOOLS/critic.py` must enforce the policy to prevent regressions
- Legacy log locations (`LOGS/`, `MCP/logs/`) should be gitignored
- Existing logs in old locations will be ignored (logs are ephemeral)

## Enforcement

- `TOOLS/critic.py` now scans for hardcoded references to `LOGS/` and `MCP/logs/`
- `SKILLS/artifact-escape-hatch` validates that no `.log` files exist outside approved roots
- `.gitignore` prevents accidental commits to old locations

## Review Triggers

- Any new logging code that doesn't use `CONTRACTS/_runs/`
- Changes to log management strategy or requirements
- Integration with external monitoring tools that may reference old paths

## Related Decisions

- [INV-006](../../../CANON/INVARIANTS.md): Output roots restriction
- [ADR-004](ADR-004-mcp-integration.md): MCP entrypoint pattern (reference implementation)
- [STYLE-002](../preferences/STYLE-002-engineering-integrity.md): Real fixes over patches
