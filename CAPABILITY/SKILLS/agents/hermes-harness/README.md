# Hermes Harness Skill Folder

Drop-in repo folder that lets Hermes Agent act as a task harness for other agents.

It gives you three operating modes:

1. **Inside Hermes**: install `skills/hermes-harness/` as a Hermes skill and invoke `/hermes-harness`. Use `persistent_worker` mode with named conversations for multi-phase continuity, or use `delegate_task` for disposable isolated subagents.
2. **From another agent or script**: call `skills/hermes-harness/scripts/hermes_harness.py`. It sends a structured task request via `/v1/responses` to a local Hermes API server with named `conversation` support for persistent multi-turn context.
3. **Persistent workers**: Set `mode: persistent_worker` and a `conversation` name. The worker continues from prior context without spawning `delegate_task`. Use this for phase 1 → 2 → 3 → pause → 5 continuity.

This folder is intentionally conservative: named conversations for persistent workers, flat delegation for disposable subagents, explicit context packets, and final synthesis by the parent agent.

## Design principles

- Use `persistent_worker` mode + named conversation for multi-phase continuity.
- Use `delegate_task` only for disposable, isolated subtasks.
- Subagents receive explicit context only. Never assume they know the parent conversation.
- Parent synthesis is mandatory. Never paste raw subagent outputs as the final answer.
- Durable long-running work should use Hermes cron or background terminal jobs, not synchronous delegation.
- `--model` is a cosmetic label — actual model selection is server-side in Hermes config.

## Usage examples

```bash
# Persistent worker — multi-phase audit with named conversation
python scripts/hermes_harness.py run \
  --task "Continue phase 5 of the CAT_CAS audit." \
  --mode persistent_worker \
  --conversation "ccc:ags:catcas-auditor"

# New conversation from scratch (appends timestamp)
python scripts/hermes_harness.py run \
  --task "Start a fresh audit of the roadmap." \
  --mode persistent_worker \
  --conversation "ccc:ags:new-audit" \
  --conversation-new

# Stateless disposable task — no conversation, no persistence
python scripts/hermes_harness.py run \
  --task "What files changed in src/ today?" \
  --mode audit

# Dry-run — print the prompt without calling the API
python scripts/hermes_harness.py run \
  --task "Audit the repo" \
  --mode audit \
  --dry-run

# Prompt-only — generate the harness prompt for manual review
python scripts/hermes_harness.py prompt \
  --task-file examples/task.audit.json
```

## Folder map

```text
skills/hermes-harness/
├── SKILL.md
├── README.md
├── config/
│   └── hermes-harness.yaml
├── examples/
│   ├── task.audit.json
│   ├── task.research.json
│   └── commands.md
├── scripts/
│   ├── hermes_harness.py
│   └── hermes_task.sh
├── templates/
│   ├── delegation_brief.md
│   ├── external_agent_prompt.md
│   ├── synthesis_report.md
│   └── task_matrix.yaml
└── tests/
    └── test_contracts.py
```

## References

- Hermes skills use `SKILL.md`, references, templates, scripts, and assets, and can be loaded as slash commands.
- Hermes `delegate_task` creates isolated child agents whose only inherited context is what the parent sends in `goal` and `context`.
- Hermes API server exposes `/v1/responses` (stateful, named conversations) and `/v1/chat/completions` (stateless). This skill uses `/v1/responses` with `conversation` for persistent worker context.
