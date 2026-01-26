# TURBO_SWARM Changelog

All notable changes to the TURBO_SWARM LAB project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `PHASE_9_SWARM_ROADMAP.md` - Comprehensive Phase 9 roadmap with 6 sub-phases
- This `CHANGELOG.md` file

### Changed
- Nothing yet

### Removed
- Nothing yet

---

## [0.5.0] - 2026-01-25

### Added
- **Phase 9 Roadmap Creation**
  - Full breakdown of 9.1 (MCP Tool Calling), 9.2 (Task Queue), 9.3 (Chain of Command)
  - Detailed 9.4 (Governor Pattern), 9.5 (Delegation Protocol), 9.6 (Delegation Harness)
  - JSON schema specifications for directives and receipts
  - Exit criteria and success metrics per phase
  - Risk mitigations documented
  - Quick start guide appended

---

## [0.4.0] - 2025-12-30

### Added
- **Agent Workflow Integration Status** (`AGENT_WORKFLOW_STATUS.md`)
  - Documented operational status of all swarm components
  - Created integration checklist for pipeline adoption
  - Defined model tier requirements (Tier 1-4)
  - Added verification commands for health checks

### Changed
- **Caddy Deluxe Orchestrator** - Verified operational status
- **The Professional Orchestrator** - Verified operational status

### Fixed
- Manifest system consolidated to v4 format

---

## [0.3.0] - 2025-12-30

### Added
- **MCP Swarm Coordination** (ADR-024 implementation)
  - Integrated MCP Message Board into Failure Dispatcher
  - Agent Inbox governance via `agent_inbox_list`, `agent_inbox_claim`, `agent_inbox_finalize`
  - The Professional v2.0 with inbox awareness

- **Swarmctl Commands**
  - `solo` command for manual high-tier task execution
  - `troubleshoot` command using qwen2.5-coder:7b for root cause analysis
  - `broadcast` command for real-time tactical guidance

### Changed
- Failure Dispatcher now writes to MCP ledger instead of file-based queue
- Governor linked to MCP message board for coordination

---

## [0.2.0] - 2025-12-29

### Added
- **Caddy Deluxe Architecture** - Multi-tiered local swarm
  - Tier 1 (Ant): qwen2.5-coder:0.5b - lightspeed syntax fixes
  - Tier 2 (Foreman): qwen2.5-coder:3b - Chain-of-Thought reasoning
  - Tier 3 (Expert): qwen2.5-coder:7b - deep troubleshooting
  - Tier 4 (Professional): Claude - strategic direction

- **AGENTS_SKILLS_ALPHA Directory**
  - `governor/` - Governor skill with SOP assets
  - `ant-worker/` - Distributed task executor
  - `swarm-directive/` - Claude-to-swarm interface
  - `swarm-orchestrator/` - Swarm launcher and coordinator
  - `agent-activity/` - Activity tracking
  - `agi-hardener/` - Security hardening
  - `qwen-cli/` - Qwen model CLI wrapper

- **MODELS_1 Directory**
  - `swarm_orchestrator_caddy_deluxe.py`
  - `swarm_orchestrator_professional.py`
  - `swarm_orchestrator_entropy_hackers.py`
  - `swarm_orchestrator_hand_of_midas.py`
  - `Coordinator/` subdirectory with coordination docs

- **MODELS_BETA Directory**
  - Experimental orchestrator variants
  - `swarm_orchestrator_turbo.py`
  - `swarm_orchestrator_super_turbo.py`
  - `swarm_orchestrator_super_turbo_deluxe.py`
  - `swarm_orchestrator_entropy_crusher.py` variants

- **MODELS_ALPHA Directory**
  - `skill_runtime.py` - Skill execution runtime
  - `tiny_agent_real_agi.py` - Minimal agent implementation

### Changed
- Task format standardized to JSON with explicit schemas
- Manifest system moved to v3/v4 format

---

## [0.1.0] - 2025-12-28

### Added
- **Initial TURBO_SWARM LAB Structure**
  - Created `THOUGHT/LAB/TURBO_SWARM/` directory
  - `event_logger.py` - Event logging infrastructure
  - `agent_reporter.py` - Agent activity reporting
  - `tui_dashboard.py` - Terminal UI dashboard
  - `failure_dispatcher.py` - Test failure scanning and dispatch
  - `generate_swarm_manifest_2.py` - Manifest generation

- **Manifest System**
  - `SWARM_MANIFEST.json` (v1)
  - `SWARM_MANIFEST_V2.json`
  - `SWARM_MANIFEST_V3.json`
  - Manifest generation and conversion scripts

### Notes
- Initial experimental infrastructure
- Targeting test fixing automation as primary use case

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.5.0 | 2026-01-25 | Phase 9 Roadmap created |
| 0.4.0 | 2025-12-30 | Agent Workflow Status documented |
| 0.3.0 | 2025-12-30 | MCP Swarm Coordination (ADR-024) |
| 0.2.0 | 2025-12-29 | Caddy Deluxe + AGENTS_SKILLS_ALPHA |
| 0.1.0 | 2025-12-28 | Initial LAB structure |

---

## Relationship to Main Roadmap

This CHANGELOG tracks development in the TURBO_SWARM LAB, which implements **Phase 9: Swarm Architecture** from `AGS_ROADMAP_MASTER.md`.

### Phase 9 Tasks (Z.6 + D.1-D.2)

| Task | Status | Notes |
|------|--------|-------|
| 9.1 MCP Tool Calling Test | NOT STARTED | Needs 0.5B model validation |
| 9.2 Task Queue Primitives | PARTIAL | Basic dispatch works, needs formalization |
| 9.3 Chain of Command | PARTIAL | Escalation exists, needs protocol |
| 9.4 Governor Pattern | OPERATIONAL | Caddy Deluxe + Professional active |
| 9.5 Delegation Protocol | NOT STARTED | Schema design needed |
| 9.6 Delegation Harness | NOT STARTED | Fixture tests needed |

---

## References

- **Main Roadmap:** [AGS_ROADMAP_MASTER.md](../../../AGS_ROADMAP_MASTER.md)
- **Phase 9 Roadmap:** [PHASE_9_SWARM_ROADMAP.md](PHASE_9_SWARM_ROADMAP.md)
- **Workflow Status:** [AGENT_WORKFLOW_STATUS.md](AGENT_WORKFLOW_STATUS.md)
- **Coordinator Docs:** [MODELS_1/Coordinator/COORDINATOR.md](MODELS_1/Coordinator/COORDINATOR.md)

---

## Contributing

When making changes to TURBO_SWARM:

1. Update this CHANGELOG under `[Unreleased]`
2. Follow the Keep a Changelog format
3. Reference the related Phase 9 task (9.1-9.6)
4. Include fixture references where applicable
5. Update `AGENT_WORKFLOW_STATUS.md` if operational status changes

---

## Graduation Criteria

TURBO_SWARM will graduate from LAB when:

- [ ] All Phase 9 tasks (9.1-9.6) are complete
- [ ] 100% fixture test coverage for delegation protocol
- [ ] Deterministic replay verified from cold start
- [ ] Production-scale stress testing passed (100+ concurrent tasks)
- [ ] Documentation complete in CANON
