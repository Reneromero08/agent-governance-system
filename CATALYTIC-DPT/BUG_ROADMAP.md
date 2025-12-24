# CATALYTIC-DPT Bug Roadmap

> Generated: 2025-12-24

## Critical Bugs

- [x] **agent_loop.py:44** — Subprocess list+shell=True conflict
- [x] **PRIMITIVES/__init__.py** — Exports non-existent classes (added stubs)

## Medium Severity

- [x] **MCP/server.py:352-359** — Schema validation now validates required fields + enums
- [x] **swarm_config.json:47** — Wrong ledger path (`CATALYTIC-DPT/CONTRACTS` → `CONTRACTS`)
- [x] **launch_swarm.ps1** — Prompts now passed to CLIs
- [x] **SKILLS/governor/run.py:44** — Invalid Gemini CLI invocation
- [x] **SKILLS/file-analyzer/run.py:52** — Same Gemini CLI issue

## Minor Issues

- [x] **SKILLS/ant-worker/run.py** — Docstring says "grok-executor"
- [x] **SKILLS/governor/run.py** — Docstring says "gemini-executor"
- [x] **mcp_client.py** — Missing exit code on invalid command

## Verification ✓

- [x] Run `ant-worker` skill with test fixture (5/5 passed)
- [x] Test MCP server import
- [x] Test PRIMITIVES import
- [x] Verify `mcp_client.py` exit code (returns 1)
