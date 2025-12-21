# Skill: canon-migration

**Version:** 0.1.0

**Status:** Template

**required_canon_version:** ">=0.1.0 <1.0.0"

## Purpose

This skill handles migrations when breaking changes occur to the canon or system structure.

## Trigger

Run this skill when:
- Canon version has a major bump (e.g., 0.x.x → 1.x.x)
- Invariants are changed (requires exceptional process per INV-* rules)
- File structure changes (per INV-001)

## Inputs

- `source_version`: The canon version the pack was created under
- `target_version`: The canon version to migrate to
- `pack_path`: Path to the pack directory to migrate

## Outputs

- `migrated_files`: List of files that were transformed
- `migration_log`: Detailed log of changes made
- `warnings`: Any compatibility warnings

## Migration Process

1. **Version Detection**: Read `meta/PACK_INFO.json` for source version
2. **Migration Chain**: Apply migrations in sequence (0.1 → 0.2 → 0.3 → ...)
3. **Validation**: Run fixtures to verify migrated pack
4. **Manifest Update**: Regenerate manifests with new hashes

## Example Migrations

### 0.1.x → 0.2.x
- No structural changes required
- Update manifest version field

### Future: 0.x → 1.0
- Evaluate invariant changes
- Apply file structure transformations
- Update token grammar if deprecated

## Constraints

- Must not lose data during migration
- Must maintain referential integrity
- Must log all transformations

## Fixtures

- `fixtures/basic/` - Test migration from previous version
- `fixtures/roundtrip/` - Verify data survives migration cycle
