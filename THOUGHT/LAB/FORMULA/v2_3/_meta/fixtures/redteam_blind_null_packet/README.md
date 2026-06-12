# redteam_blind_null_packet

Attack: verification: blind with packet_sha256: null and no _packets/ dir. Everything else is valid (prediction_ids non-empty, P-001 exists).

Expected: exit 1 with E_BLIND. A lazy validator that only checks packet hashes when non-null misses this.

Spec basis: VERDICT_SCHEMA.md section 3 (packet_sha256 MUST be non-null when blind) and section 9 (E_BLIND: null hash).
