<!-- CONTENT_HASH: 12624a977f71ab48ded52f330899bac291b0fa1e5a6c87691c564714daf93934 -->

# Pack Determinism Contract

This document records the **B2 determinism contract** for AGS memory packs. The goal is to ensure that packs built from the same repo state are reproducible, traceable, and reject mismatched governance versions.

## Identity & Cache Keys

- The primary cache key is the `manifest_digest` derived from `meta/REPO_STATE.json`. Only file content hashes, sizes, and paths feed the digest; timestamps are ignored. The digest determines the pack directory name (`llm-pack-<digest[:12]>`) and the `PACK_INFO.repo_digest` field.
- Pack consumers (skills, CI, pack readers) must treat `repo_digest` as the source of truth for identity. If the digest changes, the pack must be regenerated.

## Deterministic Outputs

The following metadata files are deterministic by construction:

1. `meta/FILE_INDEX.json` – sorted list of `path`, `hash`, `size`.
2. `meta/FILE_TREE.txt` – deterministic tree text generated from the same file listing.
3. `meta/REPO_STATE.json` – manifest with `canon_version`, `grammar_version`, and sorted `files`.
4. `meta/PACK_INFO.json` – reports `mode`, `profile`, `canon_version`, `grammar_version`, `repo_digest`, `included_paths`, and `deleted_paths`.
5. `meta/PROVENANCE.json` – provenance manifest built from deterministic metadata (content hash of canonical JSON).
6. `CORTEX/_generated/SECTION_INDEX.json` and `SUMMARY_INDEX.json` (via repository upstream) are regenerated deterministically before packaging.

Some artifacts (e.g., `meta/BUILD_TREE.txt`, combined `COMBINED/*` outputs, `ENTRYPOINTS.md` footers) may include timestamps or derived stats. They are explicitly **not** part of the digest and should not be used for cache keys.

## Version Fields & Mismatch Behavior

- `canon_version` is read from `CANON/VERSIONING.md`.
- `grammar_version` is currently hard-coded to `1.0`. It describes the packing grammar (what files are included, hash formats, etc.).
- When consuming a pack, verify both versions:
  - If `PACK_INFO.canon_version` differs from the repo’s `canon_version`, rebuild or reject the pack to avoid applying stale governance rules.
  - If `PACK_INFO.grammar_version` differs from `1.0`, treat the pack as incompatible and trigger a regeneration that understands the newer grammar.

Pack creation writes both fields to `meta/PACK_INFO.json` and `meta/REPO_STATE.json`, allowing downstream systems (including future pack validators) to enforce compatibility.

## Observable Variance

- The only permitted variance between packs from identical repo states is in files that are excluded from the digest (e.g., `COMBINED/FULL-*` timestamped outputs and optional ZIP archives). These are safe because lookup & verification use the deterministic metadata listed above.
- If additional deterministic files are added, update this document and the manifest digest to include them explicitly.
