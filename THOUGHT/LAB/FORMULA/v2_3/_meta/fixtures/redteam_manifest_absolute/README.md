# redteam_manifest_absolute

Attack: Manifest path /etc/fixture_evidence.txt is absolute (POSIX-rooted). On any platform it is also absent relative to the question dir, so both the absolute-path check and the existence check map to the same code.

Expected: exit 1 with E_MANIFEST_MISSING. Reporting E_HASH here would mean the validator tried to hash an out-of-tree absolute path, which is wrong.

Spec basis: VERDICT_SCHEMA.md section 5 (must be relative, no absolute paths) and section 9 (E_MANIFEST_MISSING: absolute).
