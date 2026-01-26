# Phase 9: Swarm Architecture Roadmap

**Version:** 1.0.0
**Status:** IN PROGRESS (experimental until proven)
**Created:** 2026-01-25
**Goal:** Build a governed swarm of local models for parallel task execution
**Priority:** P1 (core infrastructure)
**Identifier:** Z.6 / D.1-D.2

---

## Overview

Phase 9 establishes the swarm architecture for distributed task execution using local models. The goal is to create a governed, deterministic system where tiny models (0.5B-3B) can execute delegated subtasks under the supervision of a Governor pattern, with full receipt-based verification.

### Core Principles

1. **Determinism First**: All swarm operations must produce identical outputs given identical inputs
2. **Patch-First Outputs**: Tiny models produce patches, not direct writes (unless explicitly allowlisted)
3. **Fail-Closed Verification**: Any mismatch in verification fails the entire operation
4. **Receipt-Based Audit**: Every operation emits a receipt with inputs, outputs, and hashes
5. **Governed Escalation**: Clear chain of command with no ad-hoc escalation logic

### Architecture Overview

```
Claude (President)
    |
    v
Governor (Conductor) - analyzes, decomposes, dispatches
    |
    +---> Ant Worker 1 (Tier 1: 0.5B model)
    |         |
    |         v
    |     Patch + Receipt
    |
    +---> Ant Worker 2 (Tier 2: 3B model)
    |         |
    |         v
    |     Patch + Receipt
    |
    v
Verifier (validates patches, applies if clean)
    |
    v
CONTRACTS/_runs/ (receipt storage)
```

---

## Current Infrastructure Status

### Operational Components

| Component | Location | Status |
|-----------|----------|--------|
| Caddy Deluxe Orchestrator | `MODELS_1/swarm_orchestrator_caddy_deluxe.py` | OPERATIONAL |
| The Professional Orchestrator | `MODELS_1/swarm_orchestrator_professional.py` | OPERATIONAL |
| Failure Dispatcher | `failure_dispatcher.py` | OPERATIONAL |
| Governor Skill | `AGENTS_SKILLS_ALPHA/governor/` | ACTIVE |
| Ant Worker Skill | `AGENTS_SKILLS_ALPHA/ant-worker/` | ACTIVE |
| Swarm Directive Skill | `AGENTS_SKILLS_ALPHA/swarm-directive/` | DRAFT |
| Swarm Orchestrator Skill | `AGENTS_SKILLS_ALPHA/swarm-orchestrator/` | ACTIVE |
| Agent Activity Tracker | `AGENTS_SKILLS_ALPHA/agent-activity/` | ACTIVE |
| Task Coordination | `MODELS_1/Coordinator/` | OPERATIONAL |

### Model Tiers (Caddy Deluxe Architecture)

| Tier | Model | Role | Capabilities |
|------|-------|------|-------------|
| 1 (Ant) | `qwen2.5-coder:0.5b` | Lightspeed syntax fixes | Simple logic, formatting |
| 2 (Foreman) | `qwen2.5-coder:3b` | Chain-of-Thought reasoning | Complex analysis |
| 3 (Expert) | `qwen2.5-coder:7b` | Deep troubleshooting | Root cause analysis |
| 4 (Professional) | Claude/Large | Strategic direction | Architecture decisions |

---

## Phase 9.1: MCP Tool Calling Test (Z.6.1)

**Goal:** Verify that 0.5B models can reliably use MCP tools.

### Tasks

- [ ] 9.1.1 Test basic tool invocation with qwen2.5-coder:0.5b
  - Tools to test: file read, file write, grep, glob
  - Success criteria: >90% correct tool calls on 100 test prompts
  - Failure handling: log and retry with clarified prompt

- [ ] 9.1.2 Test tool argument parsing accuracy
  - Verify JSON argument extraction
  - Test edge cases (special characters, unicode, long paths)
  - Measure argument error rate

- [ ] 9.1.3 Test multi-step tool chains
  - Read file -> Analyze -> Write patch
  - Verify state consistency between steps
  - Test rollback on mid-chain failure

- [ ] 9.1.4 Benchmark latency and throughput
  - Measure tool call latency (target: <500ms per call)
  - Measure throughput (target: 10+ calls/sec with Ollama)
  - Document model loading overhead

### Exit Criteria

- [ ] 100 test prompts pass at >90% accuracy
- [ ] Tool argument error rate <5%
- [ ] Multi-step chains complete successfully
- [ ] Performance benchmarks documented

### Fixtures

```
AGENTS_SKILLS_ALPHA/mcp-tool-test/
  fixtures/
    basic/
      input.json    # 100 test prompts
      expected.json # Expected tool calls
    edge_cases/
      input.json    # Unicode, special chars, long paths
      expected.json
```

---

## Phase 9.2: Task Queue Primitives (Z.6.2)

**Goal:** Implement dispatch/ack/complete primitives for task management.

### Tasks

- [ ] 9.2.1 Implement task dispatch primitive
  - JSON schema for task spec:
    ```json
    {
      "task_id": "TASK-YYYY-MM-DD-NNN",
      "created_at": "ISO-8601",
      "priority": "HIGH|MEDIUM|LOW",
      "task_type": "file_operation|code_adapt|validate|research",
      "target_files": ["path/to/file.py"],
      "allowed_writes": ["path/to/output.patch"],
      "timeout_seconds": 60
    }
    ```
  - Write to `PENDING_TASKS/` directory
  - Update `DISPATCH_LEDGER.json`

- [ ] 9.2.2 Implement task acknowledge primitive
  - Atomic move from `PENDING_TASKS/` to `ACTIVE_TASKS/`
  - Set `assigned_to` field with agent identifier
  - Lock mechanism to prevent double-claim

- [ ] 9.2.3 Implement task complete primitive
  - Move from `ACTIVE_TASKS/` to `COMPLETED_TASKS/`
  - Attach result receipt:
    ```json
    {
      "task_id": "...",
      "completed_at": "ISO-8601",
      "status": "success|failure",
      "artifacts": [{"type": "patch", "hash": "sha256:..."}],
      "verification": {"tests_passed": true, "lint_passed": true}
    }
    ```
  - Update `DISPATCH_LEDGER.json`

- [ ] 9.2.4 Implement task failure handling
  - Increment `attempts` counter
  - If `attempts < max_attempts`: return to `PENDING_TASKS/`
  - If `attempts >= max_attempts`: move to `FAILED_TASKS/`
  - Log failure reason with stack trace

- [ ] 9.2.5 Implement queue monitoring
  - Real-time dashboard showing queue depth
  - Alert on stuck tasks (active >10min)
  - Metrics: throughput, latency, failure rate

### Exit Criteria

- [ ] All primitives have fixture-backed tests
- [ ] No task can be claimed by two agents
- [ ] Retry logic works correctly
- [ ] Monitoring dashboard operational

### Fixtures

```
LAW/CONTRACTS/fixtures/task_queue/
  dispatch/
    input.json    # Task to dispatch
    expected.json # Ledger state after dispatch
  acknowledge/
    input.json    # Task to claim
    expected.json # Ledger state after claim
  complete/
    input.json    # Task result
    expected.json # Ledger state after complete
```

---

## Phase 9.3: Chain of Command (Z.6.3)

**Goal:** Implement escalate/directive/resolve primitives for hierarchical control.

### Tasks

- [ ] 9.3.1 Implement escalate primitive
  - Agent escalates task to next tier
  - Attach escalation reason:
    ```json
    {
      "task_id": "...",
      "escalated_from": "Ant-1",
      "escalated_to": "Foreman",
      "reason": "complexity_exceeded|error_unrecoverable|timeout",
      "context": {...}
    }
    ```
  - Preserve full task history

- [ ] 9.3.2 Implement directive primitive
  - Governor sends directive to worker:
    ```json
    {
      "directive_id": "DIR-YYYY-MM-DD-NNN",
      "target_agent": "Ant-1",
      "command": "execute|pause|abort|reconfigure",
      "payload": {...}
    }
    ```
  - Workers poll for directives
  - Acknowledge receipt

- [ ] 9.3.3 Implement resolve primitive
  - Mark escalation as resolved
  - Propagate resolution back to original requester
  - Close escalation chain

- [ ] 9.3.4 Implement escalation timeout handling
  - Auto-escalate after configurable timeout
  - Log timeout events
  - Alert on repeated timeouts

- [ ] 9.3.5 Test multi-tier escalation
  - Ant -> Foreman -> Expert -> Professional
  - Verify context preservation at each tier
  - Test resolution propagation

### Exit Criteria

- [ ] Escalation works across all 4 tiers
- [ ] Context is fully preserved during escalation
- [ ] Resolution propagates correctly
- [ ] Timeouts trigger auto-escalation

### Architecture

```
Escalation Flow:
  Ant Worker (0.5B)
      | (cannot complete)
      v
  Foreman (3B)
      | (still stuck)
      v
  Expert (7B)
      | (complex issue)
      v
  Professional (Claude)
      |
      v
  Resolution flows back down
```

---

## Phase 9.4: Governor Pattern (Z.6.4)

**Goal:** Implement central Governor for orchestrating ant workers.

### Tasks

- [ ] 9.4.1 Implement Governor core
  - Poll MCP for new directives from Claude
  - Decompose directives into subtasks
  - Dispatch subtasks to appropriate tier
  - Aggregate results

- [ ] 9.4.2 Implement task decomposition
  - Analyze task complexity
  - Split into parallelizable subtasks
  - Determine appropriate tier per subtask
  - Emit decomposition receipt

- [ ] 9.4.3 Implement result aggregation
  - Collect results from all subtasks
  - Merge patches (detect conflicts)
  - Generate summary report
  - Return to Claude

- [ ] 9.4.4 Implement health monitoring
  - Track worker heartbeats
  - Detect stuck/dead workers
  - Redistribute tasks from dead workers
  - Alert on unhealthy swarm

- [ ] 9.4.5 Implement Governor SOP (Standard Operating Procedure)
  - Location: `AGENTS_SKILLS_ALPHA/governor/assets/GOVERNOR_SOP.json`
  - Define decision trees for common scenarios
  - Document escalation thresholds
  - Specify timeout policies

### Exit Criteria

- [ ] Governor can decompose and dispatch complex tasks
- [ ] Results aggregate correctly
- [ ] Health monitoring detects failures
- [ ] SOP covers all common scenarios

### Governor SOP Schema

```json
{
  "version": "1.0.0",
  "decision_trees": {
    "task_classification": {
      "simple_syntax": "route_to_tier_1",
      "logic_change": "route_to_tier_2",
      "architecture": "escalate_to_professional"
    },
    "failure_handling": {
      "timeout": "retry_once_then_escalate",
      "error": "log_and_escalate",
      "conflict": "pause_and_request_human"
    }
  },
  "timeouts": {
    "tier_1": 30,
    "tier_2": 60,
    "tier_3": 120,
    "tier_4": 300
  }
}
```

---

## Phase 9.5: Delegation Protocol (D.1)

**Goal:** Define schemas for delegated subtasks and worker receipts.

### Tasks

- [ ] 9.5.1 Define JSON Directive Schema
  ```json
  {
    "directive_id": "string",
    "task_id": "string",
    "model_class": "tiny|medium|large",
    "allowed_paths": ["path/to/allowed/**"],
    "read_paths": ["path/to/readable/**"],
    "deliverable_types": ["patch", "analysis", "test"],
    "required_verifications": ["lint", "test", "hash"],
    "timeout_seconds": 60,
    "max_retries": 3
  }
  ```

- [ ] 9.5.2 Define Worker Receipt Schema
  ```json
  {
    "receipt_id": "string",
    "directive_id": "string",
    "worker_id": "string",
    "touched_files": ["sorted array of paths"],
    "produced_artifacts": [
      {"type": "patch", "cas_ref": "sha256:..."}
    ],
    "patch_ref": "sha256:...",
    "assumptions": ["array of assumptions made"],
    "errors": ["sorted array of errors"],
    "verdict": "success|failure|escalated"
  }
  ```

- [ ] 9.5.3 Implement patch-first output enforcement
  - Tiny models MUST output patches (not direct writes)
  - Medium models MAY output patches or direct writes if allowlisted
  - Large models MAY use any output mode
  - Verifier checks compliance before applying

- [ ] 9.5.4 Implement Verifier requirements
  - Validate worker stayed within allowlists
  - Apply patch deterministically
  - Run specified tests and greps
  - Emit verification receipt
  - Fail-closed on any mismatch

- [ ] 9.5.5 Document delegation protocol in CANON
  - Location: `LAW/CANON/SWARM/DELEGATION_PROTOCOL.md`
  - Normative specification
  - Examples for each model class

### Exit Criteria

- [ ] Schemas are fully specified with JSON Schema validation
- [ ] Patch-first enforcement is implemented
- [ ] Verifier catches all violations
- [ ] Protocol documented in CANON

### Fixtures

```
LAW/CONTRACTS/fixtures/delegation/
  directive/
    valid_tiny.json    # Valid directive for tiny model
    valid_medium.json  # Valid directive for medium model
    invalid_paths.json # Directive with disallowed paths (should fail)
  receipt/
    valid_success.json # Valid success receipt
    valid_failure.json # Valid failure receipt
    invalid_fields.json # Missing required fields (should fail)
```

---

## Phase 9.6: Delegation Harness (D.2)

**Goal:** End-to-end fixture-backed delegation testing.

### Tasks

- [ ] 9.6.1 Implement "golden delegation" test
  - Input: Fixed directive + deterministic prompt
  - Tiny worker produces patch + receipt
  - Governor verifies + applies
  - Tests pass
  - Receipts deterministic across re-runs

- [ ] 9.6.2 Implement negative tests - scope violation
  - Worker touches out-of-scope file
  - Verifier MUST reject
  - Correct error message emitted
  - Task marked as failed

- [ ] 9.6.3 Implement negative tests - missing receipt fields
  - Worker omits required receipt field
  - Verifier MUST reject
  - Correct error message emitted
  - Task marked as failed

- [ ] 9.6.4 Implement negative tests - non-deterministic ordering
  - Worker produces unsorted touched_files
  - Verifier MUST reject
  - Correct error message emitted
  - Enforce sorted arrays

- [ ] 9.6.5 Implement end-to-end integration test
  - Full pipeline: Claude -> Governor -> Worker -> Verifier -> Apply
  - Multiple workers in parallel
  - Verify all receipts and audit trail
  - Deterministic replay from cold start

### Exit Criteria

- [ ] Golden delegation passes consistently
- [ ] All negative tests catch violations
- [ ] End-to-end integration verified
- [ ] Deterministic replay proven

### Golden Delegation Fixture

```
LAW/CONTRACTS/fixtures/delegation_harness/
  golden/
    directive.json     # The directive to execute
    prompt.txt         # Fixed prompt for worker
    expected_patch.diff # Expected patch output
    expected_receipt.json # Expected receipt
    expected_tests.json   # Expected test results
  negative_scope/
    directive.json     # Directive with narrow scope
    worker_output.json # Worker that violates scope
    expected_error.json # Expected verifier error
  negative_fields/
    worker_output.json # Receipt missing fields
    expected_error.json # Expected verifier error
  negative_ordering/
    worker_output.json # Unsorted arrays
    expected_error.json # Expected verifier error
```

---

## Dependencies

### Blocking Dependencies

| Dependency | Status | Impact |
|------------|--------|--------|
| Phase 5 (Vectors) | COMPLETE | Symbol references for compression |
| Phase 6 (Cassettes) | COMPLETE | Receipt storage infrastructure |
| Phase 7 (ELO) | CORE COMPLETE | Worker performance tracking |
| Phase 8 (Resident) | COMPLETE | Geometric reasoning foundation |

### Soft Dependencies

| Dependency | Status | Impact |
|------------|--------|--------|
| Ollama Service | Required | Local model execution |
| MCP Server | Required | Tool calling infrastructure |
| CONTRACTS filesystem | Required | Receipt and audit storage |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| MCP tool accuracy (0.5B) | >90% | 100 test prompts |
| Task queue throughput | >10 tasks/min | Parallel workers |
| Escalation success rate | >95% | Context preserved |
| Delegation verification | 100% | No false passes |
| Patch determinism | 100% | Same input = same output |
| Receipt completeness | 100% | All fields present |
| End-to-end latency | <30s | Simple tasks |

---

## Risk Mitigations

### Risk: 0.5B models cannot use MCP reliably

**Mitigation:**
- Start with simplified tool schemas
- Use few-shot examples in prompts
- Fall back to 3B tier if accuracy <80%

### Risk: Race conditions in task queue

**Mitigation:**
- Atomic file moves for claims
- Lock files with timeouts
- Idempotent operations

### Risk: Patch conflicts in parallel execution

**Mitigation:**
- Governor detects file overlap before dispatch
- Serialize conflicting tasks
- Abort and retry on detected conflicts

### Risk: Escalation storms (everything escalates)

**Mitigation:**
- Tier-appropriate task routing
- Escalation quotas per timeframe
- Auto-rejection of trivially complex tasks

---

## References

- **Main Roadmap:** [AGS_ROADMAP_MASTER.md](../../../AGS_ROADMAP_MASTER.md) - Phase 9 section
- **Governor Skill:** [AGENTS_SKILLS_ALPHA/governor/SKILL.md](AGENTS_SKILLS_ALPHA/governor/SKILL.md)
- **Ant Worker Skill:** [AGENTS_SKILLS_ALPHA/ant-worker/SKILL.md](AGENTS_SKILLS_ALPHA/ant-worker/SKILL.md)
- **Coordinator Docs:** [MODELS_1/Coordinator/COORDINATOR.md](MODELS_1/Coordinator/COORDINATOR.md)
- **Workflow Status:** [AGENT_WORKFLOW_STATUS.md](AGENT_WORKFLOW_STATUS.md)
- **Caddy Deluxe:** [MODELS_1/swarm_orchestrator_caddy_deluxe.py](MODELS_1/swarm_orchestrator_caddy_deluxe.py)
- **The Professional:** [MODELS_1/swarm_orchestrator_professional.py](MODELS_1/swarm_orchestrator_professional.py)

---

## Appendix A: Model Compatibility Matrix

| Model | MCP Tools | Patch Output | Direct Write | Chain-of-Thought |
|-------|-----------|--------------|--------------|------------------|
| qwen2.5-coder:0.5b | Limited | Required | Forbidden | No |
| qwen2.5-coder:1.5b | Good | Required | Forbidden | Limited |
| qwen2.5-coder:3b | Good | Preferred | Allowlisted | Yes |
| qwen2.5-coder:7b | Excellent | Optional | Allowlisted | Yes |
| llama3.2:1b | Limited | Required | Forbidden | No |
| Claude | Excellent | Optional | Full | Yes |

---

## Appendix B: Quick Start Guide

### Starting the Swarm

```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Pull required models
ollama pull qwen2.5-coder:0.5b
ollama pull qwen2.5-coder:3b

# 3. Start the Governor
python THOUGHT/LAB/TURBO_SWARM/AGENTS_SKILLS_ALPHA/swarm-orchestrator/scripts/poll_and_execute.py --role Governor

# 4. Start Ant Workers (in separate terminals)
python THOUGHT/LAB/TURBO_SWARM/AGENTS_SKILLS_ALPHA/swarm-orchestrator/scripts/poll_and_execute.py --role Ant-1
python THOUGHT/LAB/TURBO_SWARM/AGENTS_SKILLS_ALPHA/swarm-orchestrator/scripts/poll_and_execute.py --role Ant-2

# 5. Send a directive
python THOUGHT/LAB/TURBO_SWARM/AGENTS_SKILLS_ALPHA/swarm-directive/run.py input.json output.json
```

### Monitoring

```bash
# Watch task queue
tail -f INBOX/agents/Local\ Models/DISPATCH_LEDGER.json

# Watch directive flow
tail -f LAW/CONTRACTS/_runs/mcp_ledger/directives.jsonl

# Check swarm health
python THOUGHT/LAB/TURBO_SWARM/MODELS_1/swarm_orchestrator_caddy_deluxe.py --status
```
