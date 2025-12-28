# CAT-DPT Schema Versioning Policy

Status: ACTIVE
Version: 1.0.0

## 1. Goal
Ensure that historical catalytic runs remains verifiable indefinitely, even as schemas for JobSpecs, Ledgers, and Proofs evolve.

## 2. Immutability of Major Versions
- Any change that is not purely additive or non-breaking (e.g., adding an optional field) MUST increment the schema version.
- Historical schemas MUST remain in the repository (e.g., `jobspec.v1.schema.json`) or be reachable via a stable content-addressed reference.

## 3. Backward Compatibility
- Verifiers MUST select the correct schema based on the version declared in the artifact (e.g., `dag_version` in `PIPELINE_DAG.json`).
- If a verifier encounter an unknown schema version, it MUST fail closed.

## 4. Migration Paths
- When a schema version is bumped, a migration script (or skill) MUST be provided if existing artifacts need to be "upgraded" for new tooling.
- Upgrading historical artifacts MUST NOT invalidate their original cryptographic proofs. Typically, this means producing a NEW artifact alongside the old one.

## 5. Release Ceremony for Schema Changes
Changing a schema is a "Law Change" and requires:
1. An ADR documenting the rationale.
2. Updates to `SCHEMAS/` with appropriate versioning.
3. Verification that historical runs still pass with old schemas.
4. Incrementing the `validator_semver` in `PipelineRuntime`.
