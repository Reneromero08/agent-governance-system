# Phase 6B.6 Non-Hardware Target Qualification Evidence

Status: PHASE6B6_NONHARDWARE_TARGET_QUALIFICATION_EVIDENCE_COMPLETE

The first procedure stopped because Git was unavailable.

Git was installed under a superseded target-Git procedure.

That procedure was rejected because the Phenom is a copied-file target.

Git was removed.

The first copy-only target qualification passed positively.

Its mandatory 0644-to-0600 negative check unexpectedly passed.

Evidence collection stopped because this exposed a source validator defect.

PR #32 repaired canonical package-mode enforcement.

A fresh archive was exported from merge commit 0c5f1cbfb30ce9d6f8242674a1fc86f44372fa56.

The complete Phenom qualification was rerun.

The repaired verifier rejected the real-target unchanged-byte 0644-to-0600 mutation.

All required positive and negative non-hardware qualification checks passed.

## Final Binding

- Target: root@192.168.137.100, hostname catcas, Debian GNU/Linux 13 trixie
- Portable export commit: 0c5f1cbfb30ce9d6f8242674a1fc86f44372fa56
- Portable export tree: 940187df10582ec5560c5e13420685107d21e8c3
- Portable archive SHA-256: affbc0b3e9725de62aa946774e3e8830399f9af12414713b1bfbc68547765ca4
- Portable manifest SHA-256: 59e5c5927cfa7f19bdaafdd740cb350f5819e81741b62821a22f2eb80ecd4676
- Target final-result digest: c2d1bf3c78e2a9318f51e06d27ac39a49fe7a49e3cd49c0c8850cd6c85a07f7f
- Mandatory 0644-to-0600 rejection: PASS, exit 2
- Target mutation negatives: 15 rejected
- Forbidden option negatives: 9 rejected

## Authority Boundary

Physical acquisition remains NOT AUTHORIZED and NOT RUN.

No hardware experiment ran. No sender workload ran. No MSR access occurred. No frequency or voltage control occurred. No campaign session was generated. No calibration occurred. No restoration occurred. No target coupling occurred. No acquisition authority artifact was created.

The next boundary is independent evidence review.
