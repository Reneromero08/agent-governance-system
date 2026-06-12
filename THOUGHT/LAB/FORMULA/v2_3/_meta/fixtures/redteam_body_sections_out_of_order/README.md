# redteam_body_sections_out_of_order

Attack: All six required H2 sections are present but ## Method and ## Claims are swapped. A lazy validator that only checks set membership of headings wrongly passes.

Expected: exit 1 with E_SCHEMA. The spec mandates the exact order Hypothesis, Claims, Method, Results, Status, Provenance; order violations are structural schema errors (task spec assigns this case to E_SCHEMA).

Spec basis: VERDICT_SCHEMA.md section 8 (exact H2 headings, in this exact order).
