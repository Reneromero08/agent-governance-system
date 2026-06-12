# redteam_slug_dir_mismatch

Attack: VERDICT.md lives in q01_example/ but its frontmatter says slug: q01_elpmaxe. The slug is well-formed (q<NN>_<name>) and the dir IS in the catalog, so only the slug-vs-directory comparison catches it.

Expected: exit 1 with E_SCHEMA (slug-dir mismatch is explicitly listed under E_SCHEMA, not E_CATALOG).

Spec basis: VERDICT_SCHEMA.md section 3 (slug MUST match the directory name) and section 9 (E_SCHEMA: slug-dir mismatch).
