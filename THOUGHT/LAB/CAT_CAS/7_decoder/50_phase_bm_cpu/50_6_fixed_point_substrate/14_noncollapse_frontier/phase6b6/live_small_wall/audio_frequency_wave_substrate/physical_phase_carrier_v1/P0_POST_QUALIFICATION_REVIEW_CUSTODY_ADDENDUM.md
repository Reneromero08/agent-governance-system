# P0 post-qualification review-custody addendum

**Parent audit:** `P0_POST_QUALIFICATION_AUDIT.md`  
**Audited candidate root:** `844ef33332dcbfdfca6a4332f958894ee4dc02cb39087d82d3c6b15a0289ef2e`  
**Decision effect:** `P0_BUILD_READINESS_BLOCKED` remains unchanged  
**Authority effect:** none

## Audit finding A-05: independent-review custody is asserted, not externally checkable

**Severity:** material governance finding  
**Status:** open

The committed review object records four entries with:

```text
agent_id
independent = true
review_id
reviewed_root
role
verdict
findings
```

The agent identities are local path-like strings:

```text
/root/p0sp_circuit_carrier
/root/p0sp_path_sourceoff
/root/p0sp_analyzer_controls
/root/p0sp_claims_authority
```

The review report contains short closure summaries, but the packet does not bind immutable review-task prompts, full review responses, model identities, start/end times, parent commit, transcript or receipt hashes, or another mechanism that lets an external auditor establish that four independent executions actually occurred and inspected the claimed scope.

Therefore the artifacts establish:

```text
four root-bound PASS declarations with distinct declared roles
```

They do not by themselves establish:

```text
externally reproducible independence custody
```

This finding does not imply that the reviews did not occur. It limits what the committed evidence proves.

## Required repair

The next exact-root review packet must bind for every reviewer:

```text
review_id
role
model and effort route
operation
exact task/prompt bytes and SHA-256
starting commit and candidate root
read-only authority and prohibited actions
start and completion UTC
full response or immutable response artifact and SHA-256
normalized findings
verdict
reviewer receipt SHA-256
```

The four prompts must be materially independent in focus and must explicitly inspect the resonance/load law, dynamic preparation and ringdown model, continuous uncertainty enclosure, common-mode observability, no-smuggle boundary, and claim ceiling.

A boolean `independent` field is descriptive metadata, not proof of independence.

Closure requires the package validator to reject:

```text
missing task or response bytes
reused task hash across roles
reused response hash
wrong candidate root
review completed before candidate freeze
review with mutation authority
review transcript omitted while claiming external checkability
```

Until this finding and the parent audit findings close, `P0_BUILD_READINESS_PACKET_FROZEN` remains withdrawn.
