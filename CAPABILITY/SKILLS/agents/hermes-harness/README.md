# Hermes Harness Skill Folder

Drop-in repo folder that lets Hermes Agent act as a task harness for other agents.

It gives you two operating modes:

1. **Inside Hermes**: install `skills/hermes-harness/` as a Hermes skill and invoke `/hermes-harness`. The parent Hermes agent uses `delegate_task` to split work into isolated subagents, supervise them, and synthesize the result.
2. **From another agent or script**: call `skills/hermes-harness/scripts/hermes_harness.py`. It sends a structured task request to a local Hermes API server, asking Hermes to perform the fan-out and return a consolidated report.

This folder is intentionally conservative: it defaults to flat delegation, explicit context packets, limited parallelism, and final synthesis by the parent agent.

## Install in a repo

```bash
# From your repo root after unzipping this package
mkdir -p skills
cp -R hermes-harness-skill/skills/hermes-harness skills/
```

Then either copy/symlink it into Hermes:

```bash
mkdir -p ~/.hermes/skills/orchestration
ln -s "$PWD/skills/hermes-harness" ~/.hermes/skills/orchestration/hermes-harness
```

Or add your repo skill directory as an external skill directory in `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ${REPO_ROOT}/skills
```

Set `REPO_ROOT` before starting Hermes:

```bash
export REPO_ROOT="$PWD"
```

## Start Hermes API server mode

The external script expects the Hermes Agent API server to be available at an OpenAI-compatible endpoint.

```bash
# In another terminal, start Hermes normally with API server enabled per your Hermes config
hermes gateway

# Defaults expected by the script
export HERMES_API_BASE="http://127.0.0.1:8642/v1"
export HERMES_API_KEY="change-me-local-dev"
```

## Use from Hermes chat

```text
/hermes-harness audit this repo for the smallest refactor that improves reliability
```

## Use from another agent, CI step, or shell

```bash
python skills/hermes-harness/scripts/hermes_harness.py run \
  --task "Audit this repo for dead code, brittle tests, and missing docs" \
  --workspace "$PWD" \
  --mode audit \
  --max-workers 3
```

Print a handoff prompt without calling Hermes:

```bash
python skills/hermes-harness/scripts/hermes_harness.py prompt \
  --task "Split the roadmap into implementation tickets" \
  --workspace "$PWD"
```

Run from a task JSON file:

```bash
python skills/hermes-harness/scripts/hermes_harness.py run \
  --task-file skills/hermes-harness/examples/task.audit.json
```

## Design principles

- The parent agent is the harness, not a worker.
- Subagents receive explicit context only. Never assume they know the parent conversation.
- Subagents should be leaf workers by default.
- Parallelism is useful only for independent tasks.
- Parent synthesis is mandatory. Never paste raw subagent outputs as the final answer.
- Durable long-running work should use Hermes cron or background terminal jobs, not synchronous delegation.

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
- Hermes API server exposes OpenAI-compatible `/v1/chat/completions` for external clients.
