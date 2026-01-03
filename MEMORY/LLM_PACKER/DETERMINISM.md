<!-- CONTENT_HASH: 2835948d02cfe366c4b5b97a83e6632ed1167264cd01362d3f30d7c2b2653348 -->

# Pack Determinism Contract

This document records the **B2 determinism contract** for AGS memory packs. The goal is to ensure that packs built from the same repo state are reproducible, traceable, and reject mismatched governance versions.

## Identity & Cache Keys

- The primary cache key is the `manifest_digest` derived from `meta/REPO_STATE.json`. Only file content hashes, sizes, and paths feed the digest; timestamps are ignored.
- The digest determines the default pack directory name (`llm-pack-<scope>-<digest[:12]>`).

## Deterministic Outputs

The following metadata files are deterministic by construction:

1. `meta/FILE_INDEX.json` – sorted list of `path`, `hash`, `size`.
2. `meta/FILE_TREE.txt` – deterministic tree text generated from the same file listing.
3. `meta/REPO_STATE.json` – manifest with `canon_version`, `grammar_version`, and sorted `files`.
4. `meta/PACK_INFO.json` – reports `scope`, `title`, `stamp`, and `version`.
5. `NAVIGATION/CORTEX/_generated/SECTION_INDEX.json` and `NAVIGATION/CORTEX/_generated/SUMMARY_INDEX.json` (via repository upstream) are regenerated deterministically before packaging.
6. `LITE/PACK_MANIFEST.json` – deterministic CAS-addressed manifest (P.2), with entries sorted by `path`.
7. `LITE/RUN_REFS.json` – deterministic refs for run records (P.2).

Some artifacts (e.g., `meta/BUILD_TREE.txt` and optional `FULL/*` outputs) may include derived stats. They are explicitly **not** part of the digest and should not be used for cache keys.

## Version Fields & Mismatch Behavior

- `canon_version` is read from `LAW/CANON/VERSIONING.md`.
- `grammar_version` is currently hard-coded to `1.0`. It describes the packing grammar (what files are included, hash formats, etc.).
- When consuming a pack, verify both versions:
  - If `meta/REPO_STATE.json` `canon_version` differs from the repo’s `canon_version`, rebuild or reject the pack to avoid applying stale governance rules.
  - If `meta/REPO_STATE.json` `grammar_version` differs from `1.0`, treat the pack as incompatible and trigger a regeneration that understands the newer grammar.

Pack creation writes `canon_version` to `meta/REPO_STATE.json` and includes a `version` field in `meta/PACK_INFO.json` for quick inspection.

## Observable Variance

- The only permitted variance between packs from identical repo states is in files that are excluded from the digest (e.g., `meta/PROVENANCE.json`, optional `FULL/*` outputs, and optional ZIP archives). These are safe because lookup & verification use the deterministic metadata listed above.
- If additional deterministic files are added, update this document and the manifest digest to include them explicitly.
