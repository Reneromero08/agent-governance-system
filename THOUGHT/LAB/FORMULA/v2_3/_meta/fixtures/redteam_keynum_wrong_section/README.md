# redteam_keynum_wrong_section

Attack: key_results string 'pass rate = 1.000' appears verbatim in the ## Method section but NOT in the ## Results section. A lazy validator that greps the whole body (or the whole file) finds it and wrongly passes.

Expected: exit 1 with E_KEYNUM. The key_results search must be scoped to the ## Results section only.

Spec basis: VERDICT_SCHEMA.md section 4 (each string must appear VERBATIM in the body ## Results section) and section 9 (E_KEYNUM).
