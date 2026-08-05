# AGS LAB ISOLATION CONTRACT V1

This is the complete cross-agent contract for work rooted in `THOUGHT/LAB`.
It replaces repository-root governance for normal Lab sessions.

## Scope

The immutable Git boundary is `THOUGHT/LAB/`. The `.agent-root` file controls
instruction discovery only and never grants Git authority.

Paths have three classes:

- `LAB_PAYLOAD`: ordinary paths below `THOUGHT/LAB/`.
- `LAB_BOUNDARY_CONTROL`: `.agent-root`, this contract, this directory's
  `AGENTS.md`, `opencode.json`, and everything below `.codex/`.
- `MAIN`: every other repository path.

During normal Lab work, edit only `LAB_PAYLOAD`. Boundary-control files are
read-only. Main source may be read when needed, except the repository-root
`AGENTS.md`, which is outside the Lab instruction boundary and must not be
opened or searched.

Never edit the root `CHANGELOG.md`, LAW, CANON, CAPABILITY, root hooks, or CI to
finish a Lab task. If a main-system fix is genuinely required, stop and hand it
off as a separate main-maintenance task. Do not repair or revert unrelated main
changes automatically.

## Git behavior

- A Lab payload commit may contain only `LAB_PAYLOAD`.
- Boundary-control changes require an explicit boundary-maintenance task, the
  full isolation acceptance suite, and no experiment payload in the commit.
- A payload/main or payload/boundary mixture fails with `LAB SCOPE VIOLATION`
  and exact offending paths.
- Lab push history must be linear; rebase instead of merging.
- Use the normal hooks. Never use `--no-verify` to bypass a scope failure.
- Lab payload does not require the root changelog, main commit ceremony, or a
  main verification receipt.

Run the shared checks from anywhere below the Lab root:

```text
python <repo>/CAPABILITY/TOOLS/utilities/lab_scope.py doctor --agent <agent> --cwd "$PWD"
python <repo>/CAPABILITY/TOOLS/utilities/lab_scope.py audit --lab-session
```

## Runtime writes

Direct agent edit/write/patch operations outside the Lab must be denied by the
agent adapter. Codex additionally enforces the filesystem split with the
`ags-lab` permission profile. OpenCode's shell is not an OS sandbox: its
managed-Scratch Git shim runs the scope audit before every Git command, so a
shell mutation outside the Lab blocks Git; do not describe the original shell
write as physically impossible. Do not blanket-deny ordinary OpenCode shell
mutators: Lab agents retain normal Bash capability for experiments.

Route `TMPDIR`, `TEMP`, `TMP`, `XDG_CACHE_HOME`, `PYTHONPYCACHEPREFIX`, and
`COVERAGE_FILE` to the managed, disk-backed Scratch payload supplied by the Lab
launcher. Never use `/dev/shm`, `/run/shm`, or another RAM filesystem.

## Safety inherited independently of repository governance

Do not manipulate the host desktop, input devices, USB, PCI, GPU, or VM viewer.
Do not create worktrees, clones, checkout copies, or RAM-backed scratch. Never
permanently delete files; removal requires the user's exact approval and a
recoverable trash operation.

If the Lab marker, permission profile, project trust, exact prompt boundary, or
Git root cannot be verified, fail closed and do not begin work.
