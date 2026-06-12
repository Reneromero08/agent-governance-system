# redteam_manifest_traversal

Attack: Manifest path ../outside_evidence.txt escapes the question dir. The target file EXISTS at the escaped location and the sha256 in the manifest is its real hash, so a lazy validator that just joins the path, checks existence, and checks the hash will wrongly pass.

Expected: exit 1 with E_MANIFEST_MISSING (containment check, not existence or hash).

Spec basis: VERDICT_SCHEMA.md section 5 (paths must stay inside the question dir, no .. escapes) and section 9 (E_MANIFEST_MISSING: escapes the question dir).
