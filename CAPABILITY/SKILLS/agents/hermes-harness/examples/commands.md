# Example commands

## Print a prompt for manual handoff

```bash
python skills/hermes-harness/scripts/hermes_harness.py prompt \
  --task "Plan the migration from my old agent pipeline to Hermes subagent delegation" \
  --workspace "$PWD" \
  --mode plan
```

## Send a repo audit to local Hermes API server

```bash
python skills/hermes-harness/scripts/hermes_harness.py run \
  --task "Audit the repo for unsafe automation, flaky tests, and missing README setup" \
  --workspace "$PWD" \
  --mode audit \
  --max-workers 3
```

## Save the result

```bash
python skills/hermes-harness/scripts/hermes_harness.py run \
  --task-file skills/hermes-harness/examples/task.audit.json \
  --output .hermes-harness/audit-result.md
```

## Dry run payload

```bash
python skills/hermes-harness/scripts/hermes_harness.py run \
  --task "Map the repo architecture" \
  --workspace "$PWD" \
  --dry-run
```
