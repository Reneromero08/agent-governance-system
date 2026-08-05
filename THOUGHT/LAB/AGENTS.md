# AGS LAB ISOLATION CONTRACT V1

This Codex instruction root is the compact form of `LAB_CONTRACT.md`.

- The immutable Git boundary is `THOUGHT/LAB/`; `.agent-root` controls only
  instruction discovery and grants no Git authority.
- Normal Lab commits contain only experiment payload below `THOUGHT/LAB/`.
- `.agent-root`, this file, `LAB_CONTRACT.md`, `opencode.json`, and `.codex/**`
  are boundary controls and are read-only outside an explicit
  boundary-maintenance task.
- Every path outside `THOUGHT/LAB/` is main scope. Agents retain their normal
  filesystem capability, but a Lab task must not turn an incidental main edit
  into part of its Lab commit. Do not open or search the repository-root
  `AGENTS.md`, and do not update the root `CHANGELOG.md` merely for Lab payload.
- A Lab payload commit contains only payload. Payload mixed with main or
  boundary paths fails with `LAB SCOPE VIOLATION` and exact paths. Boundary
  controls never share a commit with experiment payload.
- Use the normal Git hooks. Never bypass a scope failure with `--no-verify`.
  Lab payload needs no root changelog, main ceremony, or main push receipt.
- Lab push history is linear; rebase instead of merging.
- Codex and OpenCode retain their normal file and Git access. Repository hooks
  audit commit and push scope; they do not pretend that out-of-Lab files are
  inaccessible.
- Route caches and temporary output to launcher-provided managed disk-backed
  Scratch. Never use `/dev/shm`, `/run/shm`, or other RAM-backed scratch.
- Do not manipulate VM viewers, the desktop, input, USB, PCI, or GPU. Do not
  create worktrees, clones, or checkout copies. Never permanently delete;
  removal requires exact user approval and recoverable trash.
- If project root, prompt input, marker, or Git root cannot be verified, fail
  closed and do not begin work.

Deeper `AGENTS.md` files may add experiment-specific directions but cannot
widen this boundary. Run `lab_scope.py doctor` and `audit --lab-session` as
documented in `LAB_CONTRACT.md` before consequential Git operations.
