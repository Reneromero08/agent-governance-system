# Contracts

The `CONTRACTS/` directory contains the machinery that enforces rules within the Agent Governance System.

- **Fixtures** (`LAW/CONTRACTS/fixtures/`, `SKILLS/*/fixtures/`) - Concrete test cases that capture laws and precedents. Each fixture directory typically contains an `input.json` and an `expected.json`. The runner executes the relevant skill or validation script to ensure the actual output matches expectations.
- **Schemas** (`CONTRACTS/schemas/`) - JSON schemas that describe the structure of canon, skill manifests, context files and cortex indices. These can be used to validate files before they are processed.
- **Runner** (`runner.py`) - A script that discovers and runs fixtures. The runner must exit with non-zero status if any fixture fails.

Contracts are mechanical: they do not rely on human judgment or heuristics. If a change causes any fixture to fail, the change must be rejected until the fixture is updated or the change is reconsidered.
