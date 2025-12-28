# Gemini note triage report (untracked)

## Inputs (current locations)
- `CATALYTIC-DPT/LAB/RESEARCH/AUTOSTART_SETUP.md`
- `CATALYTIC-DPT/LAB/RESEARCH/CONNECTION_STATUS.md`
- `CATALYTIC-DPT/LAB/RESEARCH/F3_STRATEGY.md`
- `CATALYTIC-DPT/LAB/RESEARCH/MULTI_AGENT_MCP_COORDINATION.md`

## Findings

### AUTOSTART_SETUP.md
- Content: step-by-step Windows PowerShell instructions for `MCP/autostart.ps1`, mentions `MCP/autostart_config.json`, logs under `CONTRACTS/_runs/mcp_logs/`, and includes a Claude Desktop config snippet.
- Placement: this is **AGS MCP operational documentation**, not CAT-DPT research.
- Proposed destination: `MCP/AUTOSTART_SETUP.md` (copied to `proposed/MCP/AUTOSTART_SETUP.md`).
- Notes: overlaps with existing MCP docs; treat as optional append/merge, not authoritative.

### CONNECTION_STATUS.md
- Content: a snapshot-style status report (timestamps, counts, claims of tools/resources/tests passing) plus “how to start” instructions.
- Placement: if kept, it belongs under MCP operational docs; however it reads like a **point-in-time report** and may go stale quickly.
- Proposed destination: `MCP/CONNECTION_STATUS.md` (copied to `proposed/MCP/CONNECTION_STATUS.md`).
- Notes: consider deleting instead of tracking, unless you explicitly want historical status reports.

### MULTI_AGENT_MCP_COORDINATION.md
- Content: long architecture write-up of MCP multi-agent coordination, audit logs, governance gates (preflight/admission/critic), and startup tiers.
- Placement: this is **AGS/MCP architecture documentation** (not CAT-DPT).
- Proposed destination: `MCP/MULTI_AGENT_MCP_COORDINATION.md` (copied to `proposed/MCP/MULTI_AGENT_MCP_COORDINATION.md`).
- Notes: contains strong claims (counts, tool lists, file paths) — should be reviewed before making it canonical.

### F3_STRATEGY.md
- Content: CAS + manifest approach for token-cost reduction, and notes about future integration with `MEMORY/LLM_PACKER`.
- Placement: this is **CAT-DPT research** (Lane F3) and is correctly scoped under CAT-DPT.
- Proposed destination: keep under `CATALYTIC-DPT/LAB/RESEARCH/F3_STRATEGY.md` (copied to `proposed/CATALYTIC-DPT/LAB/RESEARCH/F3_STRATEGY.md`).
- Notes: references `CATALYTIC-DPT/LAB/PROTOTYPES/f3_cas_prototype.py` — verify it exists before treating as implemented.

## Recommended next action (no changes applied to originals)
- If you want these moved into `MCP/`, I need you to either move them manually or authorize a repo move (current agent mutation rules restrict edits outside allowed roots).
- If you want them deleted, confirm which ones and I’ll remove them.
