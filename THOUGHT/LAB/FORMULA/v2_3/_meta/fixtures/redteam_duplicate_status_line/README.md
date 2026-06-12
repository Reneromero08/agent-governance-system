# redteam_duplicate_status_line

Attack: Two body lines begin with **Status:** (one in ## Status, one smuggled into ## Provenance) and BOTH match the frontmatter status. A lazy validator that checks 'at least one matching Status line exists' wrongly passes.

Expected: exit 1 with E_BODY_MISMATCH. Exactly one **Status:** line is allowed in the body; duplication is an error even when the tokens agree.

Spec basis: VERDICT_SCHEMA.md section 8 (no other line anywhere in the body may begin with **Status:**) and section 9 (E_BODY_MISMATCH: duplicated).
