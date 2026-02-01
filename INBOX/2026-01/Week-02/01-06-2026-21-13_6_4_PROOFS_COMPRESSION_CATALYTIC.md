---
uuid: 00000000-0000-0000-0000-000000000000
title: Proof Suite Additions (Compression and Catalytic)
section: report
bucket: "reports/v4/section_6_4_9\u20136_4_11"
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: Proof Suite Additions (Compression and Catalytic)'
tags:
- catalytic
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: 363327b90df60ea3acf512176637cb9eb6bdd1580febf68245759e1ea94ec3e8 -->

# Report: Proof Suite Additions (Compression and Catalytic)

Date: 2026-01-05
Roadmap Section: 6.4.9–6.4.11

## Source inputs used
- COMPRESSION_PROOF_AGENT_PROMPT.md
- CRYPTO_SAFE docs (protected artifacts and sealing rules)

## Goal
Turn two claims into mechanical, reproducible proofs:
1) Compression rate (token, symbol, or representation reduction) backed by auditable artifacts.
2) Catalytic safety (restore and purity) backed by receipts and deterministic verification.

These proofs must run fresh during pack runs and be sealable for public distribution.

## Proof regimes (Elo-aware)
Maintain two baseline regimes:
- Pre-Elo baseline: pure similarity retrieval, deterministic selection.
- Post-Elo baseline: Elo-tier filtered retrieval once Elo loop is green.

This prevents mixing “retrieval policy changes” with “compression math claims.”

## proof_compression_run
### Outputs (minimum)
Directory: `NAVIGATION/PROOFS/COMPRESSION/`
- `COMPRESSION_DATA.json` (machine)
- `COMPRESSION_REPORT.md` (human)
- receipts:
  - tokenizer/version
  - corpus anchors (hashes of source inputs)
  - retrieved hashes and selection spec
  - formula and computed metrics
  - environment info relevant to determinism

### Requirements
- Never mutate source notes.
- Deterministic token counting and stable ordering.
- Fail closed if inputs cannot be verified.

## proof_catalytic_run
Directory: `NAVIGATION/PROOFS/CATALYTIC/`
- `RESTORE_PROOF.json` (from Phase 4.2)
- `PURITY_SCAN.json` (from Phase 4.3)
- `CATALYTIC_REPORT.md` summarizing:
  - roots, exclusions, digests
  - verdicts
  - reproduction commands

## Binding into pack generation
- Proofs are regenerated each pack run (freshness boundary).
- Proof outputs are added to `PROOF_MANIFEST.json`.
- If proof outputs contain protected details, they are treated as protected artifacts and sealed for public packs under CRYPTO_SAFE.

## Negative tests
- Corrupt one proof file and verify verifier fails.
- Insert plaintext protected artifact and verify crypto-safe verifier fails.
- Force restore mismatch and verify catalytic proof fails.

## Acceptance criteria
- Proofs are reproducible and mechanically verifiable.
- Proofs are included in pack outputs and manifests.
- Public packs contain no plaintext protected artifacts.
