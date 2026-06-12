# redteam_unknown_frontmatter_key

Attack: An extra frontmatter key 'analyst: redteam' is smuggled in between method_summary and registry_ids. All required keys are present and valid, so a lazy validator that only checks required-key presence wrongly passes.

Expected: exit 1 with E_SCHEMA. Frontmatter is STRICT: unknown keys are validation errors.

Spec basis: VERDICT_SCHEMA.md section 3 (STRICT: unknown keys are validation errors) and section 9 (E_SCHEMA: unknown field).
