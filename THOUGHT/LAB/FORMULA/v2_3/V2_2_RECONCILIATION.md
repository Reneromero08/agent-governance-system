# V2_2 RECONCILIATION

**Date:** 2026-06-12
**Scope:** THOUGHT/LAB/FORMULA/v2_2 (FROZEN)
**Author:** v2_3 reconciliation pass (automated audit, spot-verified against v2_2 files on disk)

## Purpose

v2_2 is frozen history. It is not edited, not deleted, not rewritten -- this document
is the bridge record between the two versions. v2_3 starts fresh: every question
enters v2_3 with status OPEN regardless of what v2_2 claimed. The audit below
documents every discrepancy found between the v2_2 INDEX.md, the per-question
verdict files, and the actual evidence on disk, so that v2_3 work can cite v2_2
material with full knowledge of its reliability. Nothing in this file changes any
v2_2 file; it only records what is there.

Sources audited:

- `THOUGHT/LAB/FORMULA/v2_2/INDEX.md` (header dated 2026-05-17)
- `THOUGHT/LAB/FORMULA/v2_2/V2_2_DISCOVERY_REPORT_2026-05-19.md` (total 56 Qs)
- `THOUGHT/LAB/FORMULA/v2_2/AGENT_HANDOFF.md` (dated 2026-05-18, total 54 Qs)
- All 30 question directories `q01_grad_s/` ... `q57_mera_holography/`

Column key for the table below:

- **INDEX** = status in v2_2 INDEX.md.
- **Headline** = `**Status:**` line at the top of the directory's VERDICT file.
- **Bottom** = final `## Verdict` (or equivalent closing) status in the same file.
- **Evidence** = count of .py scripts / count of files under results-type dirs,
  as found on disk. "results/ EMPTY" means a results directory exists with zero files.
- PV = PARTIALLY VERIFIED. "(same)" = bottom matches headline or no separate
  bottom verdict block exists.

## Per-question reconciliation (30 directories)

| Q | v2_2 INDEX | Verdict headline | Verdict bottom | Evidence on disk | Anomaly summary |
|---|---|---|---|---|---|
| Q1 (q01_grad_s) | VERIFIED | PV (CV=0.55) | (same) | 1 script / results/ EMPTY | INDEX upgrades PV to VERIFIED. INDEX adds a claim ("alpha gap is sigma error, not nabla_S error") found in no verdict file. Empty results/ despite quoted R2=0.94. Listed OPEN in 05-19 discovery report; no session report documents the promotion. |
| Q6 (q06_iit) | VERIFIED | VERIFIED | (same) | 1 script / 0 results | Statuses agree, but listed OPEN in 05-19 discovery report; promoted with no session report. No results files behind the claim. |
| Q7 (q07_multiscale) | VERIFIED | PV | PV (closing text says "Q7 confirmed at d=2") | 2 scripts / 0 results | VERDICT.md and FABRIC_UPDATE.md both say PARTIALLY VERIFIED; INDEX says VERIFIED. |
| Q8 (q08_topology) | VERIFIED | PV | PV | 4 scripts / 0 results | VERDICT.md says PARTIALLY VERIFIED top and bottom. The upgrade to VERIFIED exists only in Q57_CONNECTION.md; INDEX adopts it. |
| Q10 (q10_alignment) | PV | PV | PV | 3 scripts / 0 results | Consistent. |
| Q12 (q12_phase_transitions) | PV | FALSIFIED | PV (quantitative, not qualitative) | 4 scripts / 0 results | Headline FALSIFIED vs bottom PARTIALLY VERIFIED in the same file. INDEX sides with the bottom. |
| Q15 (q15_bayesian) | VERIFIED | VERIFIED | (same) | 0 scripts / 0 results | VERIFIED with zero scripts and zero results files; directory contains only VERDICT.md. Listed OPEN in 05-19 discovery report. |
| Q17 (q17_governance) | PV | PV | PV (verdict paragraph asserts gate success) | 3 scripts / 0 results | Verdict paragraph cites 94.8% and p=0.04 -- numbers appearing nowhere in the body. Body table shows the R-GATED variant FAILING (90.5%, p=0.13). The verdict describes a phase_coh gate result not documented in the body. |
| Q21 (q21_rate_of_change) | PV | PV | PV | 2 scripts / 0 results | Consistent. |
| Q25 (q25_sigma) | VERIFIED | VERIFIED | VERIFIED (v2; v1 noted FALSIFIED) | 1 script / results/ EMPTY | Empty results/ despite quoted R2=1.0000. Listed OPEN in 05-19 discovery report; no session report documents the promotion. |
| Q28 (q28_attractors) | CONFIRMED | PV | CONFIRMED | 4 scripts / 0 results | Headline PV vs bottom CONFIRMED. Bottom cites 73-126% dropout recovery and 10 seeds; body documents 0-3% recovery ("basin is shallow") and 5 seeds. Bottom verdict contradicts its own body. |
| Q31 (q31_compass) | PV | CONFIRMED | PV | 3 scripts / 0 results | Headline CONFIRMED vs bottom PARTIALLY VERIFIED. Verdict paragraph describes a different experiment (Native Eigen C^2 compass) than the body. |
| Q32 (q32_meaning_field) | PV | PV | PV | 16 scripts / results/ EMPTY | Statuses consistent, but results/ is empty despite quoted numerics (R2=0.63, AUROC 0.64-0.69). |
| Q33 (q33_emergence) | PV | PV | (same; closing line says "emergence confirmed") | 1 script / 0 results | Consistent status; closing "confirmed" wording vs PV headline is vocabulary drift, not a status conflict. |
| Q34 (q34_platonic) | PV | PV | (same; no bottom verdict block) | 7 scripts / 0 results | Consistent. |
| Q36 (q36_bohm) | VERIFIED | VERDICT.md: PV (2026-05-19) | VERDICT_BOHM.md: VERIFIED (2026-05-21) | 4 scripts / results/ EMPTY | Two verdict files with different statuses testing different operationalizations (hardened 5/7 checks vs surface-code argument); neither references the other. INDEX silently adopts the later VERIFIED. Empty results/ despite quoted numerics. |
| Q38 (q38_noether) | VERIFIED | VERIFIED (v1 SLERP tautology FALSIFIED, geodesic truth VERIFIED) | Updated Verdict: PV (Claim A FALSIFIED, Claim B VERIFIED) | 3 scripts + 5 graveyard tests / 1 results file | Three documents, three statuses: VERDICT.md headline VERIFIED, REPORT.md PARTIALLY VERIFIED, AGENT_HANDOFF.md FALSIFIED. Only directory in v2_2 preserving a failed_exploratory/ graveyard. |
| Q40 (q40_error_correction) | VERIFIED | CONFIRMED | "PARTIALLY VERIFIED -> CONFIRMED" | 1 script / 1 results file | INDEX says VERIFIED, verdict says CONFIRMED (vocabulary never defined). Listed OPEN in 05-19 discovery report; verdict self-describes the upgrade ("Upgraded from OPEN") with no session report. |
| Q42 (q42_bell) | VERIFIED | NO VERDICT FILE | NO VERDICT FILE | 1 script (verify_q42.py) / 0 results | INDEX says VERIFIED; directory contains ONLY verify_q42.py -- no verdict file of any kind, no results. The 05-19 discovery report still listed Q42 as OPEN. Largest status/evidence gap in v2_2. |
| Q43 (q43_qgt) | CONFIRMED (boundary) | CONFIRMED (with boundary condition) | CONFIRMED with boundary condition | 2 scripts / 0 results | Consistent. |
| Q44 (q44_born_rule) | CONFIRMED | CONFIRMED (with boundary condition) | CONFIRMED with boundary condition | 3 scripts / 0 results | Consistent; INDEX drops the boundary qualifier. |
| Q45 (q45_geometry_nav) | PV | PV | PV | 4 scripts / 0 results | Consistent. |
| Q48 (q48_riemann) | CONFIRMED | CONFIRMED | CONFIRMED | 17 scripts / 0 results | Statuses consistent; no results files on disk despite quoted KS statistics across 10 seeds. |
| Q49 (q49_why_8e) | FALSIFIED | FALSIFIED | FALSIFIED | 8 scripts / 1 results file | Consistent. Clean falsification; v1 adversarial verdict sustained. |
| Q50 (q50_completing_8e) | PV | PV (8e value FALSIFIED, architecture invariance CONFIRMED) | PV | 2 scripts / 0 results | Consistent. |
| Q51 (q51_complex_plane) | CONFIRMED (boundary) | CONFIRMED | CONFIRMED (extrinsic) | 4 scripts / 0 results | Hypothesis claims INTRINSIC complex structure; verdict explicitly finds the structure is EXTRINSIC (holonomy = 0, emerges only under Hilbert complexification) -- yet the question is labeled CONFIRMED. The confirmed finding is not the hypothesis as posed. |
| Q54 (q54_energy_conservation) | VERIFIED | VERIFIED | VERIFIED | 0 scripts / 0 results | Zero local scripts/results; directory contains only VERDICT.md, which cites the external CAT_CAS/18 Hawking Decompressor experiment (that experiment does exist). Listed OPEN in 05-19 discovery report. |
| Q55 (q55_kuramoto_heads) | PV | PV | (same; no bottom verdict block) | 2 scripts / 0 results | Consistent. |
| Q56 (q56_entangled_heads) | PV | PV | (same; no bottom verdict block) | 8 scripts / 0 results | Consistent. |
| Q57 (q57_mera_holography) | VERIFIED | VERIFIED | VERIFIED (v7) | 3 scripts / 5 results files | Consistent. Best-evidenced directory in v2_2. |

## Systemic findings

1. **Empty results/ directories despite quoted numerics.** q01, q25, q32, q36 each
   have a results/ directory containing zero files, while their verdicts quote
   precise numbers (R2=0.94, R2=1.0000, AUROC 0.64-0.69, hardened-check tallies).
   Across all 30 directories only four hold any results files at all: q38 (1),
   q40 (1), q49 (1), q57 (5). Quoted numerics are therefore mostly unreproducible
   from the frozen tree -- the scripts exist but their outputs were not preserved.
2. **INDEX header is stale.** INDEX.md is dated 2026-05-17 but contains statuses
   from verdicts dated 05-21/22 (e.g. Q36 VERDICT_BOHM.md 2026-05-21). The header
   was never updated as statuses were edited.
3. **Tier 5 row-count mismatch.** The Tier 5 header says 19 Qs; the table has 17 rows.
4. **Question-total drift.** AGENT_HANDOFF.md (05-18) counts 54 questions, the
   discovery report (05-19) counts 56, INDEX.md counts 57. Q55-Q57 were added over
   time with no reconciliation note.
5. **Silent promotions.** The 05-19 discovery report lists Q1, Q6, Q15, Q25, Q40,
   Q42, Q54 as OPEN; INDEX marks all seven VERIFIED. No session report documents
   any of these promotions. Q42 was promoted with no verdict file at all.
6. **Status vocabulary undefined.** VERIFIED vs CONFIRMED vs CONFIRMED (boundary)
   vs PARTIALLY VERIFIED is never specified anywhere in v2_2. INDEX and verdicts
   use the terms interchangeably (Q40: CONFIRMED in verdict, VERIFIED in INDEX;
   Q44: boundary qualifier dropped in INDEX).
7. **Headline/bottom contradiction pattern.** Q12, Q28, Q31, Q38 each contain a
   verdict file whose headline status disagrees with its own bottom verdict.
   In Q17 and Q28 the bottom verdict additionally cites numbers that contradict
   or do not appear in the body of the same document. The pattern indicates
   verdict paragraphs were appended or edited without synchronizing headlines
   or re-checking body tables.
8. **Upgrades recorded outside verdict files.** Q8's upgrade lives only in
   Q57_CONNECTION.md; Q36's upgrade lives in a second verdict file that does not
   reference the first; Q40's upgrade is an inline self-note. There is no single
   authoritative status location per question.
9. **Negative results not preserved.** Only 1 of 30 directories (q38_noether)
   keeps a failed_exploratory/ graveyard. Failed approaches elsewhere were
   deleted or never committed, removing falsification context.
10. **Hypothesis/finding mismatch labeled as confirmation.** Q51's verdict finds
    the opposite of the hypothesis as posed (extrinsic, not intrinsic, structure)
    yet carries a CONFIRMED label; INDEX propagates it with a "(boundary)" tag
    that is never defined.

## Disposition

All questions enter v2_3 as OPEN regardless of v2_2 status; v2_2 evidence may be
cited as predecessor context but confers no status.
