# ADR-029: Headless Swarm Execution (Terminal Prohibition)

**Status:** Accepted
**Date:** 2025-12-28
**Confidence:** High
**Impact:** High
**Tags:** [security, execution, terminal, invariants]

## Status
**Accepted**
The CATALYTIC-DPT Swarm Orchestrator originally supported two execution modes:
1.  **Programmatic (Headless)**: Background processes with no visible UI.
2.  **Interactive (Terminal Bridge)**: VSCode terminal tabs via `launch-terminal` skill.

The Interactive mode caused significant friction:
-   Terminals spawned unexpectedly, cluttering the user's workspace.
-   Process cleanup was unreliable (zombie terminals).
-   The `local-agent-server` MCP tool spawned visible windows when calling `spawn_worker`.
-   Cognitive overhead for the user to manage multiple terminal panes.

## Decision
We **permanently prohibit** visible terminal spawning within the AGS ecosystem.

### Rules
1.  **INV-012 (Visible Execution)**: All execution MUST be headless or via the Antigravity Bridge (which is considered "always on" and invisible to the user unless explicitly opened).
2.  The `launch-terminal` skill is **DELETED**.
3.  The `mcp-startup` skill is **DELETED**.
4.  Any `subprocess.Popen` call on Windows MUST use `creationflags=0x08000000` (`CREATE_NO_WINDOW`).
5.  Agents are **FORBIDDEN** from using `start`, `xterm`, `gnome-terminal`, or any command that creates a new visible console window.

### Enforcement
-   `TOOLS/terminal_hunter.py` scans the repo for violations.
-   Pre-commit hooks may integrate this in CI.
-   Violations are a **hard reject** for PRs.

## Consequences
1.  **UX**: The user's IDE remains clean. All agent work happens in the background.
2.  **Debugging**: Developers must use logs (`CONTRACTS/_runs/`) to inspect agent behavior.
3.  **External MCP Servers**: ✅ **FIXED**: The `local-agent-server` (`d:/CCC 2.0/AI/AGI/MCP/server.py`) has been patched to use headless `subprocess.Popen` with `CREATE_NO_WINDOW` instead of the Antigravity Bridge terminal API.
4.  **Documentation**: `AGENTS.md` updated to reflect this prohibition.
