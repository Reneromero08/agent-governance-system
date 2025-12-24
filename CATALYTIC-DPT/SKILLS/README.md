# CATALYTIC-DPT Skills

**Architecture**: Governor → Ant Workers (model-agnostic)
**Governance**: MCP-mediated, template-bound, ledger-logged

---

## Skill Directory

| Skill | Role | CLI | Status |
|-------|------|-----|--------|
| **governor/** | Dispatch JobSpecs to workers | Gemini CLI | ✅ Ready |
| **ant-worker/** | Execute templates, no drift | Kilo Code (Grok Fast) | ✅ Ready |
| **file-analyzer/** | Analyze file structure | Gemini CLI | ✅ Ready |
| **templates/** | Strict task templates | N/A | ✅ Created |

---

## Execution Flow

```
Higher Authority (Claude/User)
    ↓
Governor (analyzes + dispatches)
    ├─ Breaks goal into subtasks
    ├─ Creates ant-task.json for each worker
    └─ Dispatches to Ant Workers
        ↓
Ant Workers (execute templates)
    ├─ Ant-1: File operations
    └─ Ant-2: Code adaptation
        ↓
MCP (governance + logging)
    └─ CONTRACTS/_runs/<task_id>/
```

---

## Quick Start

### Run Ant Worker Test
```bash
cd CATALYTIC-DPT/SKILLS/ant-worker
python test_ant_worker.py
```
Expected: 5/5 tests pass.

### Execute Ant Task
```bash
python ant-worker/run.py fixtures/file_copy_task.json output.json
```

---

## Templates

Ants receive strict templates from `templates/ant-task.template.json`:

```json
{
  "task_id": "{{TASK_ID}}",
  "task_type": "file_operation",
  "on_error": "STOP_AND_REPORT"
}
```

> [!IMPORTANT]
> Ants ONLY fill `{{placeholders}}`. They cannot modify template structure.
> Schema-valid output or HARD FAIL.

---

## Model Configuration

### Ant Workers (Kilo Code)
`~/.kilocode/cli/config.json`:
```json
{
  "kilocodeModel": "x-ai/grok-code-fast-1"
}
```

### Governor (Gemini CLI)
```bash
gemini --experimental-acp
```

Both are swappable to local models when ready.

---

## Governance Rules

1. **Template-Bound**: Workers execute templates only
2. **MCP-Mediated**: All operations via MCP server
3. **Hash-Verified**: SHA-256 integrity checks
4. **Immutable Ledger**: Every action logged
5. **Escalation**: Uncertain → STOP, don't guess

---

**Status**: Ready for 2-worker deployment
**Workers**: 2 Ant Workers (configurable)
