Use Hermes Agent as the harness for the task below. Load or follow the `hermes-harness` skill. Act as the parent coordinator: decompose the task, delegate independent subtasks with `delegate_task` when useful, then synthesize a final result.

Task:
{{task}}

Workspace:
{{workspace}}

Mode:
{{mode}}

Constraints:
{{constraints}}

Maximum concurrent subagents:
{{max_workers}}

Allowed toolsets:
{{toolsets}}

Output style:
- Start with the result.
- Include what was delegated.
- Include findings, changes, verification, conflicts, uncertainty, and next move.
- Do not paste raw subagent logs unless needed for evidence.
