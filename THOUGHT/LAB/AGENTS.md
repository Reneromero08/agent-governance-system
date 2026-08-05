# AGS LAB ISOLATION CONTRACT V1

This Codex instruction root is the compact form of `LAB_CONTRACT.md`.

- The immutable Git boundary is `THOUGHT/LAB/`; `.agent-root` controls only
  instruction discovery and grants no Git authority.
- Normal Lab work may edit only experiment payload below `THOUGHT/LAB/`.
- `.agent-root`, this file, `LAB_CONTRACT.md`, `opencode.json`, and `.codex/**`
  are boundary controls and are read-only outside an explicit
  boundary-maintenance task.
- Every path outside `THOUGHT/LAB/` is main scope. Main source may be read when
  useful, but do not open or search the repository-root `AGENTS.md` and do not
  edit root `CHANGELOG.md`, LAW, CANON, CAPABILITY, hooks, or CI for a Lab task.
- A Lab payload commit contains only payload. Payload mixed with main or
  boundary paths fails with `LAB SCOPE VIOLATION` and exact paths. Boundary
  controls never share a commit with experiment payload.
- Use the normal Git hooks. Never bypass a scope failure with `--no-verify`.
  Lab payload needs no root changelog, main ceremony, or main push receipt.
- Lab push history is linear; rebase instead of merging.
- Codex physically restricts writes with `ags-lab`. OpenCode denies direct
  out-of-Lab edit/write/patch tools; its managed-Scratch Git shim audits before
  every Git command, so shell mutations block Git without being misdescribed as
  physically impossible. Ordinary OpenCode Bash remains unrestricted for Lab
  experimentation.
- Route caches and temporary output to launcher-provided managed disk-backed
  Scratch. Never use `/dev/shm`, `/run/shm`, or other RAM-backed scratch.
- Do not manipulate VM viewers, the desktop, input, USB, PCI, or GPU. Do not
  create worktrees, clones, or checkout copies. Never permanently delete;
  removal requires exact user approval and recoverable trash.
- If project root, prompt input, permission profile, project trust, marker, or
  Git root cannot be verified, fail closed and do not begin work.

Deeper `AGENTS.md` files may add experiment-specific directions but cannot
widen this boundary. Run `lab_scope.py doctor` and `audit --lab-session` as
documented in `LAB_CONTRACT.md` before consequential Git operations.
