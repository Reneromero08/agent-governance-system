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

## v1.0 - Stable release

- [ ] Freeze the core canon and invariants.
- [ ] Publish comprehensive documentation and examples.
- [ ] Harden security: least privilege execution, sandboxing and cryptographic provenance.
