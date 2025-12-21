# Roadmap

This file outlines the high-level milestones for developing the Agent Governance System (AGS). Because AGS is intended as a reusable kernel for many projects, the roadmap focuses on capabilities rather than application-specific features.

## v0.1 - First usable kernel ✅

- [x] Finalise the canon: write the constitutional contract, define invariants, versioning policy and glossary.
- [x] Implement the shadow cortex: generate `CORTEX/_generated/cortex.json` and provide a query API.
- [x] Define and implement basic skills: file reading, querying the cortex, and simple write operations under strict controls.
- [x] Establish the runner and initial fixtures: enforce no raw path access, determinism and canonical output rules.
- [x] Provide example ADRs and context records to demonstrate decision compounding.

## v0.2 - Governance automation ✅

- [x] Add critic scripts to automatically lint canon, skills and fixtures.
- [x] Integrate continuous integration workflows for running the runner and critics on pull requests.
- [x] Implement versioning and deprecation machinery for tokens.

## v0.3 - Extended memory and indexing ✅

- [x] Introduce a persistent memory store and summarisation tools for long running agents.
- [x] Support incremental packer updates and manifest integrity checking.
- [x] Provide migration skills for older versions of the canon.

## v1.0 - Stable release ✅

- [x] Freeze the core canon and invariants.
- [x] Publish comprehensive documentation and examples.
- [x] Harden security: least privilege execution, sandboxing and cryptographic provenance.

## v1.1 - Maintenance & Refinement

- [x] Complete "meta in the split" logic for LLM_PACKER.
- [x] Refine "O(1) Cortex" by implementing a proper indexing database (SQLite).
- [x] Implement Genesis Prompt (`CANON/GENESIS.md`): Bootstrap prompt that ensures agents load CANON first.
- [x] Add Context Query Tool (`CONTEXT/query-context.py`): CLI to search decisions by tag, status, review date.
- [ ] Add Context Review Tool (`CONTEXT/review-context.py`): Flags overdue ADR reviews.

## v1.2 - Protocol Integration & Validation

- [ ] Implement initial MCP (Model Context Protocol) seams for external tool access.
- [ ] Add "Crisis Mode" procedures for automated isolation in case of governance failure.
- [ ] Add provenance headers to all generated files (generator version, input hashes, timestamp).
- [ ] Implement JSON Schema validation for canon, skills, context records, and cortex index.

## v2.0 - Advanced Autonomy

- [ ] Advanced conflict resolution: arbitration for contradictory canon rules.
- [ ] Constitutional license: formalized legal-protective layer for agentic systems.
- [ ] Reversible token economics: lossless compression for audit-ready handoffs.
- [ ] Automated "Stewardship" alerts for human escalation when canon-bound loops fail.
- [ ] Canon codebook addressing: stable IDs (`@C1`, `@M7`) so prompts reference IDs not prose.

