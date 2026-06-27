# Phase 6 V2 Exact-Head Sealed Snapshot Qualification

Generated head `500f7dfcd198e6e70dc3f999248aa61224d530cd` was qualified on the Phenom target through a sealed Git archive snapshot, not a target Git checkout.

Bindings:
- source commit: `ba48125d15009a044bb869b5716c412b1a8baa1b`
- source review: `4584742973`
- generated commit: `500f7dfcd198e6e70dc3f999248aa61224d530cd`
- generated review: `4584795315`
- plan SHA-256: `3c1b8d3da4d24e97a4395747dc8f587f60d21ef6d789bd27da8cd95908b7ebb3`
- source-bundle SHA-256: `bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f`
- archive SHA-256: `974ee0d030b74d2515ed897e82e278d9385f96c91a0a7b5974cb95bf24d1efcf`
- recursive manifest SHA-256: `a59b53235854b6173ebb5f7c2b7eac32d0a5f1bdfe8265d67d94eb0d078730c8`

Hosted workflows at the generated head passed: Governance `[DEPRECATED]` run `28284598855`, Contracts run `28284598856`, Phase 6 V2 Strict Qualification run `28284598853`, and Phase 6 Combined Campaign Plan run `28284598854`.

Target software qualification passed strict C compile, functional runtime tests, separate ASan, separate UBSan, and V2 Python contracts/analyzer tests using the target's existing Python/NumPy environment. Pre-test and post-test recursive manifest verification both passed with no changed, missing, or unexpected persistent committed snapshot files.

No hardware path ran. Calibration, acquisition, restoration, target coupling, Small Wall work, Gate R entry, and Phase 6B.6 remain unauthorized.

## Corrective Evidence Custody Addendum

Independent evidence review `4585030778` identified sealed-snapshot provenance gaps in commit `3c1fa7dc5e2d6a380d7ff310c1160dfc642f6a32`, which is preserved as incomplete evidence provenance. This corrective package recovers and commits the exact target verifier, reproduces the archive and recursive manifest bridge, and corrects the Windows status claim.

The recovered verifier is committed at `target/verify_manifest.py` with SHA-256 `43a5fe14b581a889b33f4ddefd49de1c78efd2973088f1696020aa861783a199`. The target and local verifier digests match. The target manifest verifier was rerun once for identity correction only and reported `MANIFEST_VERIFICATION_PASS` with 8350 expected files, 8350 actual files, and zero missing, extra, or changed files. No strict compile, functional, ASan, UBSan, or V2 Python test suite was repeated.

The original Windows status logs are not altered and truthfully show the untracked evidence package. The archive identity remains unaffected because `git archive` was generated from explicit immutable commit `500f7dfcd198e6e70dc3f999248aa61224d530cd`.

The snapshot bridge was reproduced in a fresh temporary directory. The reproduced archive SHA-256 is `974ee0d030b74d2515ed897e82e278d9385f96c91a0a7b5974cb95bf24d1efcf`; the reproduced recursive manifest SHA-256 is `a59b53235854b6173ebb5f7c2b7eac32d0a5f1bdfe8265d67d94eb0d078730c8`; the reproduced file count is 8350. The original transfer command and timestamp remain `not_recorded`; current target-side SHA queries prove that the archive, manifest, and verifier bytes presently on the target match the committed identities.

`COMMANDS.jsonl` is the canonical complete command ledger with 55 rows. Component ledgers are explicitly labeled in `LEDGER_ROLES.md`.

Sealed-snapshot evidence custody is corrected. Fresh independent evidence review remains pending, Gate R remains blocked, Phase 6B.6 is not entered, and all hardware/scientific authorization fields remain false.
