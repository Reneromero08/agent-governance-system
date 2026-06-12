# redteam_claim_evidence_not_subset

Attack: Claim C1 (VERIFIED) cites evidence/ghost_log.txt, which is NOT in evidence_manifest (the manifest lists only evidence/run_log.txt, which exists with a correct hash). The floor rule is satisfied (the claim lists 1 evidence path), so a lazy validator that only checks non-emptiness wrongly passes.

Expected: exit 1 with E_SCHEMA. Reasoning for the code choice: the subset constraint is part of the claims field schema (section 4: evidence must be a subset of evidence_manifest paths). E_FLOOR is reserved for the two numbered floor rules in section 7, and E_MANIFEST_MISSING is reserved for evidence_manifest path checks (section 5); a claim citing a path outside the manifest is therefore a schema violation on the claims field.

Spec basis: VERDICT_SCHEMA.md section 4 (evidence: must be a subset of evidence_manifest paths) and section 9 (E_SCHEMA).
