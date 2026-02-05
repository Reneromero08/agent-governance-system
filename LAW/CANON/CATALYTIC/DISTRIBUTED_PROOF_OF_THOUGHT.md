# DISTRIBUTED_PROOF_OF_THOUGHT.md

**Version:** 0.1.0
**Status:** Diffusing
**Created:** 2026-02-05

---

## 1. Purpose

This document specifies the Distributed Proof of Thought (DPT) architecture for the Agent Governance System. DPT enables multiple agents to collaborate on complex tasks while maintaining cryptographic accountability, restoration guarantees, and governance compliance.

DPT makes the system **catalytic**: agents can use the entire codebase as scratch space while proving they restored it to its original state.

---

## 2. Scope

This specification defines:
- The DPT agent hierarchy (Governor, Manager, Ant)
- Distributed execution model
- Proof chain semantics
- Invariants governing DPT operations
- Integration with SPECTRUM specifications

This specification does NOT define:
- Specific model implementations
- Network protocols
- User interface details
- Key management (see SPECTRUM-04)

---

## 3. Definitions

| Term | Definition |
|------|------------|
| **DPT** | Distributed Proof of Thought - architecture for multi-agent collaboration with cryptographic accountability |
| **Governor** | SOTA-tier agent that handles complex strategy, governance, and analysis |
| **Manager** | Mid-tier agent that coordinates execution and breaks tasks into subtasks |
| **Ant** | Mechanical executor that follows templates with zero creativity |
| **Swarm** | Collection of agents executing a distributed task |
| **Proof Chain** | SPECTRUM-03 compliant chain of bundle hashes proving execution integrity |
| **Restoration Proof** | Cryptographic evidence that catalytic domains were restored |

---

## 4. Core Concept: Making the System Catalytic

The foundation of DPT is catalytic computing (see `CATALYTIC_COMPUTING.md`). The key insight:

> Large disk state can be used as powerful scratch space if you guarantee restoration.

DPT extends this to distributed execution:
- Multiple agents share the same catalytic workspace
- Each agent produces restoration proofs
- The swarm's combined output forms a valid SPECTRUM-03 chain
- Final state is cryptographically bound to initial state

This enables token-efficient distributed computation where:
- **Governor** (big brain) analyzes and strategizes
- **Manager** (mid brain) coordinates
- **Ants** (small brains) execute mechanically
- **Proofs** (cryptographic) bind everything together

---

## 5. DPT Architecture

### 5.1 Agent Hierarchy

```text
                    PRESIDENT (Human / User)
                    - The Source of Intent
                    - Final Authority
                           |
                           v
                    GOVERNOR (SOTA AI)
                    - Complex decisions, governance, strategy
                    - Delegates to Manager for execution
                    - Monitors via MCP ledger
                           |
                           v
                    MANAGER (Mid-tier AI)
                    - Receives tasks from Governor
                    - Breaks into mechanical subtasks
                    - Distributes to Ants
                           |
              +------------+------------+
              |            |            |
              v            v            v
           ANT 1        ANT 2        ANT N
           (Local)      (Local)      (Local)
              |            |            |
              +------------+------------+
                           |
                           v
                    MCP LEDGER FILES
                    - directives.jsonl
                    - task_queue.jsonl
                    - task_results.jsonl
                    - escalations.jsonl
```

### 5.2 Role Definitions

| Role | Model Tier | Capabilities | Responsibilities |
|------|-----------|--------------|------------------|
| **President** | Human | Absolute authority | Intent, judgment, override |
| **Governor** | SOTA (Claude Sonnet 4.5) | Full - complex analysis | Strategy, governance, delegation |
| **Manager** | Mid (Qwen 7B) | Limited - coordination | Task breakdown, dispatch, monitoring |
| **Ant** | Tiny (Local) | Mechanical - templates | Execution, pass/fail reporting |

### 5.3 Capability Boundaries

**Governor CAN:**
- Analyze complex problems
- Make strategic decisions
- Design governance and architecture
- Monitor Manager via MCP ledger

**Governor CANNOT:**
- Micromanage execution
- Execute tasks directly

**Manager CAN:**
- Break tasks into mechanical steps
- Dispatch tasks to Ants
- Monitor results and report

**Manager CANNOT:**
- Make strategic decisions
- Analyze complex problems
- Perform governance functions

**Ant CAN:**
- Poll for tasks
- Execute strict templates
- Report pass/fail

**Ant CANNOT:**
- Make decisions
- Deviate from templates
- Access resources outside task scope

---

## 6. Relationship to SPECTRUM Specifications

DPT integrates with the SPECTRUM family of specifications:

| Specification | Role in DPT |
|--------------|-------------|
| SPECTRUM-02 (Resume Bundle) | Enables swarm restart without execution history |
| SPECTRUM-03 (Chain Verification) | Provides temporal integrity for proof chains |
| SPECTRUM-04 (Identity Signing) | Binds signatures to agent identity |
| SPECTRUM-05 (Verification Law) | Defines 10-phase verification procedure |
| SPECTRUM-06 (Restore Runner) | Ensures atomicity of restoration |

### 6.1 SPECTRUM-04 Integration

All DPT signatures use the domain separation prefix:

```text
CAT-DPT-SPECTRUM-04-v1:BUNDLE:<payload>
```

This ensures:
- No signature collision with other systems
- Clear identification of DPT-originated signatures
- Compatibility with SPECTRUM-04 verification

---

## 7. Distributed Execution Model

### 7.1 Communication Flow

All agents communicate via MCP ledger files, not direct subprocess calls:

```text
President -> send_directive()   -> directives.jsonl   -> Governor reads
Governor  -> dispatch_task()    -> task_queue.jsonl   -> Ants read
Ants      -> report_result()    -> task_results.jsonl -> Governor reads
Governor  -> escalate()         -> escalations.jsonl  -> President reads
```

### 7.2 MCP as Single Source of Truth

**Problem:** Multiple agents editing files leads to drift and conflicts.
**Solution:** MCP server mediates ALL state changes.

**Rules:**
1. No agent directly modifies files (ideally via MCP tools)
2. All changes via MCP tools
3. MCP logs every change
4. Conflicts resolved by MCP (last-write-wins or merge)
5. Terminal access shared (President sees Governor's output)

### 7.3 Headless Execution Constraint

Per ADR-029, all DPT execution MUST be headless:

**PROHIBITED:**
- `Start-Process wt` (Windows Terminal)
- `subprocess.Popen` with external windows
- Any process the user cannot see

**REQUIRED:**
- All `subprocess.Popen` on Windows uses `creationflags=0x08000000` (`CREATE_NO_WINDOW`)
- Execution via Antigravity Bridge (port 4000) or internal MCP calls

---

## 8. Proof Chain Semantics

### 8.1 Swarm Output Structure

A DPT swarm run produces a proof chain with the following structure:

```text
CONTRACTS/_runs/<swarm_id>/
    SWARM_MANIFEST.json      # Swarm-level metadata
    bundles/
        <ant_1_run_id>/      # Standard CMP-01 bundle
            TASK_SPEC.json
            STATUS.json
            OUTPUT_HASHES.json
            VALIDATOR_IDENTITY.json
            SIGNED_PAYLOAD.json
            SIGNATURE.json
            PROOF.json
        <ant_2_run_id>/
            ...
    CHAIN_MANIFEST.json      # SPECTRUM-03 chain linking bundles
    CHAIN_SIGNATURE.json     # Governor signature over chain
```

### 8.2 Chain Integrity

The chain manifest links all ant bundles:

```json
{
  "bundle_roots": ["<hash_1>", "<hash_2>", ...],
  "run_ids": ["ant_1_run_id", "ant_2_run_id", ...]
}
```

The chain root is computed per SPECTRUM-03:

```text
chain_root = sha256(canonical_json(chain_manifest))
```

### 8.3 Restoration Aggregation

Each ant produces a restoration proof. The swarm aggregates these:

1. Collect all `PROOF.json` from ant bundles
2. Verify each ant restored its catalytic domain
3. Compute aggregate restoration hash
4. Governor signs the chain manifest

---

## 9. Invariants

The following invariants govern DPT operations. All are machine-verified.

### INV-DPT-01: Distributed Restoration

```text
Every ant run must produce verified restoration proof

forall run R in swarm S:
  exists proof P in R.outputs:
    P.verified = true AND
    P.pre_manifest_hash = P.post_manifest_hash
```

**Enforcement:** Ant bundles without valid PROOF.json are rejected.

### INV-DPT-02: Chain Integrity

```text
Swarm output must be valid SPECTRUM-03 chain

verify_chain(S.chain_manifest) = ACCEPT
```

**Enforcement:** Chain verification runs before swarm acceptance.

### INV-DPT-03: Identity Binding

```text
All signatures use CAT-DPT-SPECTRUM-04-v1: prefix

forall signature sig in swarm:
  sig.message.startswith("CAT-DPT-SPECTRUM-04-v1:")
```

**Enforcement:** Signatures without correct prefix are rejected.

### INV-DPT-04: Headless Execution

```text
No visible terminal spawning

forall process P spawned by swarm:
  P.visible = false
```

**Enforcement:** Terminal hunter scans for violations (per ADR-029).

### INV-DPT-05: MCP Mediation

```text
All state changes go through MCP server

forall file_write W by agent A:
  W.via_mcp = true OR W.path in allowed_direct_paths
```

**Enforcement:** File system guard enforces write boundaries.

### INV-DPT-06: Capability Boundaries

```text
Governor/Manager/Ant have strict capability limits

Governor.can_execute_directly = false
Manager.can_strategize = false
Ant.can_deviate = false
```

**Enforcement:** Role-specific prompts and tool restrictions.

---

## 10. Domain Separation Prefix Registry

DPT uses the following domain separation prefixes:

| Prefix | Usage |
|--------|-------|
| `CAT-DPT-SPECTRUM-04-v1:` | All DPT bundle/chain signatures |
| `CAT-DPT-SWARM-v1:` | Swarm-level metadata hashes |
| `CAT-DPT-ANT-v1:` | Ant-specific task receipts |

These prefixes ensure cryptographic domain separation per SPECTRUM-04 section 7.4.

---

## 11. Evolution Path

### Phase 1: Foundation (Current)
- Single-machine swarm execution
- MCP-mediated state management
- Manual swarm initiation

### Phase 2: Distributed
- Multi-machine swarm coordination
- Network protocol for agent communication
- Distributed MCP state

### Phase 3: Autonomous
- Self-healing swarms
- Dynamic agent allocation
- Automated escalation chains

---

## 12. References

### Canon
- [CATALYTIC_COMPUTING.md](CATALYTIC_COMPUTING.md) - Foundation theory
- [CMP-01_CATALYTIC_MUTATION_PROTOCOL.md](CMP-01_CATALYTIC_MUTATION_PROTOCOL.md) - Operational protocol
- [SPECTRUM-03_CHAIN_VERIFICATION.md](SPECTRUM-03_CHAIN_VERIFICATION.md) - Chain integrity
- [SPECTRUM-04_IDENTITY_SIGNING.md](SPECTRUM-04_IDENTITY_SIGNING.md) - Signing law
- [SPECTRUM-05_VERIFICATION_LAW.md](SPECTRUM-05_VERIFICATION_LAW.md) - Verification procedure

### ADRs
- `LAW/CONTEXT/decisions/ADR-029-headless-swarm-execution.md` - Headless constraint

### Skills
- `CAPABILITY/SKILLS/agents/ant-worker/SKILL.md` - Ant worker specification

### Historical
- `INBOX/2025-12/Week-52/12-28-2025-23-52_SWARM_ARCHITECTURE.md` - Original swarm design

---

*This document consolidates DPT concepts from scattered sources into a single canonical specification.*
