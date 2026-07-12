# AGENTS.md - Small Wall Build-First Protocol

**Scope:** this directory and every descendant.

This is a LAB workspace. Repository-wide AGS governance does not apply here. When a
parent instruction conflicts with this file, this file controls work in this subtree.

## Prime directive

**Break the Small Wall. Build first. Verify after there is something coherent to
verify.**

Spend effort in this order:

1. the mechanism;
2. an end-to-end implementation;
3. the shortest useful experimental feedback loop;
4. integrity testing of the completed slice;
5. compact provenance;
6. publication ceremony, only when explicitly requested.

Provenance matters, but it is not an admission gate. Tests are a finishing and
falsification tool, not an entrance exam. CI is an integration tool, not an editor
lock.

## Default operating mode

Design, implementation, refactoring, compilation, simulation, fake-transport work,
non-driving harness work, and local iteration are all **BUILD MODE**.

Building code does not automatically trigger verification mode. Enter verification
mode only when one of these is true:

- the user explicitly asks to test, verify, audit, qualify, publish, or claim a result;
- a coherent build slice exists and the user asks to take it through validation;
- a scientific result is about to be promoted from an observation to a claim.

Until then, advance the implementation.

## The build loop

1. Read only enough current source and status material to locate the active engineering
   boundary. Do not reopen historical evidence unless the implementation depends on it.
2. Name the next falsifiable engineering bottleneck in one sentence.
3. Implement the thinnest end-to-end slice that attacks it.
4. Use only the feedback needed to keep building: compile, a narrow local check, a fake,
   or a focused smoke test.
5. Keep iterating until the mechanism is wired end to end or a real scientific,
   physical, or technical blocker is exposed.
6. Then run focused integrity tests for the changed slice. Expand testing only when a
   failure or integration boundary justifies it.
7. Record compact provenance and update status once, after the checkpoint is real.

Do not interrupt this loop to create process artifacts.

## What is not required before building

For changes contained in this subtree, do not require or create any of the following
unless the user explicitly asks for it:

- Cortex or CANON startup;
- an ADR, governance review, or new authority framework;
- a skill wrapper for ordinary engineering work;
- a separate worktree or a new branch;
- a dedicated pull request;
- a roadmap addendum for every intermediate step;
- a full-repository test run;
- hosted CI;
- an exact-head evidence packet;
- independent review;
- pre-implementation test scaffolding;
- commit-by-commit ceremony or micro-commits.

Do not add a validator, workflow, wrapper, schema, receipt, or policy unless it directly
enables the experiment or the user explicitly requests it. Prefer removing or bypassing
LAB-local process machinery over expanding it.

If Git hosting unexpectedly rejects a normal fast-forward push, report the exact remote
constraint. Do not automatically create a pull request, run full CI, or build more
process around the rejection.

## Testing policy

- **Before a coherent build:** no broad tests by default.
- **While building:** run only the compiler, focused test, simulation, or smoke check
  that gives useful engineering feedback.
- **After a coherent build:** run changed-path tests first, then the smallest relevant
  integration test.
- **Full repository CI:** run only for an explicit user request, a release, or a change
  that crosses out of this LAB subtree into shared production code.
- **Unrelated failure:** record it once and keep it out of scope. Do not spend the LAB
  session repairing unrelated governance or CI failures.
- **Documentation-only change:** inspect the diff and check formatting if useful. Do not
  run the engineering suite.

Never claim that an untested build is proven. Say `built, validation pending`. Once the
build exists, test it hard enough to support the claim actually being made, not every
claim the repository could conceivably make.

## Minimum viable provenance

Use Git history and the experiment's own outputs as the primary record. At each
meaningful build or run checkpoint, retain only what is needed to reconstruct it:

- starting and final commit or a clear dirty-diff marker;
- exact command;
- relevant input, seed, and configuration;
- raw output or result path;
- observed outcome or failure;
- files changed.

Put this in one compact existing result record or run log. Do not create a packet of
duplicated manifests, reviews, ledgers, and status documents unless the experiment
itself requires them. Capture provenance after the engineering checkpoint; do not make
the checkpoint wait for provenance paperwork.

## Small Wall focus

Choose work by one test:

> Does this action create a mechanism, produce a measurement, or reduce uncertainty at
> the next Small Wall boundary?

If not, do not do it.

Favor direct progress through the actual chain: observable carrier state, controlled
physical evolution, exact restoration, target-to-carrier coupling, a fold-odd
observable beyond the fold-even public interface, a killing null/no-smuggle check, and
repeatable adjudication.

Do not substitute candidate ranking, another scalar recovery loop, governance
architecture, status churn, or more test infrastructure for that chain.

## Live-system boundary

This file removes workflow bottlenecks; it does not itself authorize contact with a
target or a live hardware action. SSH, SCP, sender start, physical capture, MSR access,
frequency or voltage writes, and hardware execution require an explicit user directive
for that live action. Live-capable code may be built and tested through dependency
injection, fakes, and local non-driving surfaces without inventing an authority system.

## Git discipline

- Work in the current checkout unless the user requests isolation.
- Do not create a branch, worktree, or PR solely because process documentation says to.
- Keep a substantive change in one coherent commit when the user asks to commit.
- An explicit `commit` request authorizes one coherent commit of the inspected LAB
  change. An explicit `push` request authorizes its ordinary fast-forward push. Do not
  add another confirmation ceremony.
- Before committing or pushing, inspect the worktree and preserve unexpected work. For
  a push, fetch and confirm that the update is fast-forward. These are safety checks,
  not reasons to start a test suite.
- For a roadmap or documentation-only change, `git diff --check` is sufficient unless
  the user explicitly requests more validation. Do not run unit tests, the repository
  critic, the full local gate, or hosted CI.
- For LAB engineering code, run focused changed-component tests after the coherent
  build exists. Do not make full CI a condition of an ordinary push.
- Do not create a PR merely because a local hook expects a full-suite receipt. For an
  authorized LAB push, bypass that local hook after the scoped integrity check and
  report the bypass plainly.
- Do not fix unrelated files while preparing the change.

## Safety boundary and `MASTER_OVERRIDE`

Normal LAB work must use fast-forward history. The standing remote safety floor is:

- `main` cannot be deleted;
- `main` cannot be force-pushed or rewritten;
- ordinary fast-forward pushes are allowed;
- pull requests and full CI are not globally required.

Do not weaken that floor during ordinary work. The following actions require the exact
token `MASTER_OVERRIDE` in the current user prompt:

- force-pushing or otherwise rewriting published history;
- changing repository or branch protection;
- deleting a protected branch or tag;
- destructive cleanup outside the exact paths named by the user;
- bypassing an explicit live-target or hardware boundary.

An override is single-use and action-specific. Before an overridden Git action, record
the expected old remote SHA, use an exact lease, change only the minimum necessary
protection, perform the one authorized action, and restore protection immediately even
if the action fails. `MASTER_OVERRIDE` never expands the scientific or hardware scope
of the prompt.

Do not ask for `MASTER_OVERRIDE` for a normal commit, a normal fast-forward push, a
roadmap update, a documentation edit, focused testing, or ordinary LAB engineering.

## Stop conditions

Stop and ask only when:

- the next action would contact a target or drive hardware without explicit permission;
- the action is destructive or irreversible;
- a missing scientific choice would materially change the mechanism;
- no useful implementation path remains after inspecting the relevant code and trying
  the smallest viable alternative.

Tests, CI, governance, provenance formatting, branch ceremony, and roadmap polish are
not reasons to stop engineering.

## Final instruction

Build the shortest real path to the next observation. Preserve enough provenance to
reconstruct it. Test integrity when the slice exists. Return to engineering immediately
after the test tells you what is actually broken.
