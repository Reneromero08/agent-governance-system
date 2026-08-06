# Raw evidence retention policy for imported audio verification

The V1, V2, and V3 audio verification branches preserved large raw execution
trees containing stdout, stderr, generated binaries, object files, sanitizer
artifacts, temporary build products, and reproduced package outputs.

Those raw trees were copied into this tomography worktree for local inspection,
but they are intentionally ignored and not committed here:

- `audio_phase_catvm_candidate_unverified_v1/raw_logs/`
- `audio_phase_catvm_candidate_unverified_v1/raw_outputs/`
- `audio_phase_catvm_candidate_unverified_v2/raw_logs/`
- `audio_phase_catvm_candidate_unverified_v2/raw_outputs/`
- `audio_phase_noncollapse_candidate_unverified_v3/raw_logs/`
- `audio_phase_noncollapse_candidate_unverified_v3/raw_outputs/`

The committed import keeps the durable scientific package:

- final decisions and closure files;
- source receipts, transfer manifests, and hash ledgers;
- claims, verification plans, and reports;
- independent verifier source and tests;
- source snapshots required to understand or rerun the verification;
- corrective V3 audit artifacts.

Complete raw evidence remains recoverable from the task branch heads recorded in
`AUDIO_VERIFICATION_IMPORT_READINESS.md`:

- `task/audio-catvm-independent-verification`
  @ `25e58a26395549c703d430f0e37410db2b7f54a0`
- `task/audio-noncollapse-independent-verification-v3`
  @ `d628dbf918d94b9015f7ae026bbb842145788f6b`

This retention policy keeps the tomography branch reviewable while preserving
the raw evidence lineage.
