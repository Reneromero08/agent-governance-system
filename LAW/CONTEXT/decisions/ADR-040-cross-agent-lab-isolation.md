---
id: "ADR-040"
title: "Cross-Agent Lab Isolation Boundary"
status: "Accepted"
date: "2026-08-05"
confidence: "High"
impact: "High"
tags: ["architecture", "agents", "git", "isolation", "lab"]
---

<!-- CONTENT_HASH: 5b2034feab6336671df8d3987399ba73b18982f2c41b608bbd9027633630ac52 -->
# ADR-040: Cross-Agent Lab Isolation Boundary

## Context

Repository-root instructions cannot opt out after an agent has loaded them.
Codex previously discovered the Git root, concatenated root-to-current
`AGENTS.md` files, and therefore exposed Lab agents both to an early instruction
to ignore main governance and a later universal changelog rule. The model had
already received both statements before it could obey either.

Git enforcement amplified the problem. Pre-commit always ran main governance,
and the canon checker combined staged and unstaged paths. A preserved or
unrelated main diff could therefore make an otherwise Lab-only commit request a
root changelog change. Pre-push similarly required a main receipt for every
branch and used a repository temp file with irreversible cleanup.

The boundary must work for more than one agent product. Discovery behavior and
edit controls differ between Codex and OpenCode, while Git path authorization
must remain identical and independent of agent prose.

## Decision

Define `THOUGHT/LAB/` as the immutable V1 Git prefix and classify every path as
one of:

- `LAB_PAYLOAD`: ordinary experiment paths beneath the prefix.
- `LAB_BOUNDARY_CONTROL`: the Lab marker, shared contracts, agent configs, and
  `.codex/**` enforcement files.
- `MAIN`: every other repository path.

The `.agent-root` marker is only a Codex discovery marker. It never authorizes
Git scope. Normal Lab work may write payload only. Boundary controls are
read-only during normal sessions, require an explicit boundary-maintenance
task and full isolation acceptance when changed, and cannot share a commit
with experiment payload. Payload mixed with main is also rejected with exact
paths and no automatic repair.

Codex uses a user-defined `ags-lab` permission profile selected only by the
trusted Lab project config. The profile reads main source, writes the complete
Lab subtree, carves boundary controls back to read-only, and denies root
`AGENTS.md`. No legacy `sandbox_mode` configuration is combined with it.

OpenCode uses the same checked-in contract plus a dispatcher-generated inline
permission layer. The dispatcher overwrites inherited
`OPENCODE_CONFIG_CONTENT`, denies direct edit/write/patch outside Lab and on
boundary controls, and denies root `AGENTS.md` reads. Because OpenCode Bash is
not an OS filesystem sandbox, arbitrary shell writes are described accurately:
they are detected by a managed-Scratch Git shim before every Git command, which
then fails closed until the out-of-scope mutation is handed off. Ordinary Bash
remains available; the adapter does not globally deny mutation commands merely
because those commands could also target a main path.

The canonical Python scope engine routes pre-commit, pre-push, and CI. It uses
repository-relative Git paths, retains both rename endpoints, rejects
non-canonical paths and Lab merge commits, and calculates new-branch commits as
the head minus commits already reachable from the configured remote. Pre-push
reads ref updates once into memory. Lab payload receives focused syntax and
structure checks without main ceremony, root changelog, or main receipt.

## Alternatives considered

- **Instruction-only opt-out in root `AGENTS.md`:** rejected because discovery
  has already injected the entire file before the model reads the opt-out.
- **A separate Lab repository:** rejected because it fragments history and is
  unnecessary for the required read-main/write-Lab relationship.
- **Arbitrary nearest-marker Git scope:** rejected because an accidental marker
  elsewhere could grant authority outside Lab.
- **Codex-only project-root handling:** rejected because it leaves other agents
  and Git hooks with different boundaries.
- **Claiming OpenCode Bash is physically isolated:** rejected without an
  OS-level sandbox. The contract states detection and Git blocking precisely.
- **Legacy Codex workspace-write plus permission profiles:** rejected because
  the systems do not compose and the legacy model cannot express the carveouts
  as directly.

## Rationale

Instruction discovery, runtime filesystem access, and Git authorization are
separate control planes. Giving each one an explicit responsibility removes
the contradiction that caused the original contamination. An immutable Git
prefix prevents agent-created markers from widening authority, while one scope
engine keeps Codex, OpenCode, hooks, and CI in agreement.

Boundary controls require stronger treatment than ordinary Lab files because a
Lab agent able to replace its own marker or policy could silently reactivate
root discovery on the next session. The separate class closes that bootstrap
hole without making normal experiment work main-governed.

## Consequences

Lab agents no longer receive repository-root governance through normal Codex
discovery and no longer enter main changelog or receipt flows for payload-only
changes. A deep experiment still receives write access to the full Lab root.

Boundary maintenance is intentionally heavier and remains a main-governed
change. OpenCode shell commands remain capable at the operating-system level;
out-of-scope shell mutations fail the audit and all consequential Git routes.
Additional agent products require small adapters, but do not require another
scope framework.

Existing contaminated worktrees are not mutated automatically. Their main
patches must be handed off or resolved explicitly before a Lab commit or push
passes the audit.

## Enforcement

- `CAPABILITY/TOOLS/utilities/lab_scope.py` is the canonical classifier,
  range/push router, audit, and doctor.
- `CAPABILITY/TOOLS/utilities/ags_lab.py` and the user launchers adapt Codex and
  OpenCode without changing the Git authority definition.
- `.githooks/pre-commit`, `.githooks/pre-push`, and
  `.github/workflows/contracts.yml` route on the same scope result.
- `CAPABILITY/TESTBENCH/01_core/test_lab_scope.py` covers path normalization,
  boundary classification, rename endpoints, new branches, merge rejection,
  realpath escapes, and OpenCode direct-edit controls.
- `lab_isolation_acceptance.py` checks the exact Codex prompt and profile, the
  effective OpenCode config, tracked/untracked main state, deletion/rename
  state, and boundary hashes before and after verification.

## Review triggers

Review this ADR when Codex changes project-root or permission-profile behavior,
OpenCode adds an enforceable OS filesystem sandbox, another agent product is
installed for Lab work, the canonical Lab prefix moves, or the active nested
instruction chain needs a separate size-reduction migration.
