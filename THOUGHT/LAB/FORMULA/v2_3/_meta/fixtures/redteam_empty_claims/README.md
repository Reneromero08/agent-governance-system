# redteam_empty_claims

Attack: claims: [] with frontmatter status UNSUPPORTED. A lazy validator that skips the MIN recomputation when the claim list is empty (or treats min() over nothing as vacuously fine) wrongly passes.

Expected: exit 1 with E_SCHEMA. Reasoning for the code choice: claims is a required field and the verdict status is DEFINED as MIN over claims[].status (sections 1 and 3); with zero claims the MIN does not exist, so the status field cannot be validated and the verdict asserts nothing. That is a degenerate value for a required field, i.e. a schema error - not E_STATUS_NOT_MIN, which presupposes a computable MIN.

Spec basis: VERDICT_SCHEMA.md sections 1, 3, 4 and section 9 (E_SCHEMA: missing field / bad value).
