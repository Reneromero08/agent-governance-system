# Family 10h Carrier Tomography Final Subagent Review

Package: `family10h_carrier_tomography_v1`

Review mode: read-only, offline, no target contact, no PMU execution, no sensor inventory, no live authority.

Audited commit:

```text
a27e4cf1f41d1b12f146ecb347f24e666e5f160b
```

Audited hash set:

```text
manifest_file_sha256=b016ec5cfc3cee811069ebbba0b35b8569e4b07bc2bd1e4639c3abd4f792825b
manifest_canonical_sha256=ac4f6ce09e7562a58d5e4a82ce538851f9f8cd96580edeb1f02bff71903171a9
source_hashes_sha256=6c7fd544762b8324c6396db9833ed52751b59fd7110c54268d77887024baed31
source_bundle_sha256=5b5d296802c443047e0d058367cf1ec59d89e8749438a601a7028f2661a86562
runtime_binary_sha256=e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89
```

## Reviewer Verdicts

| Role | Agent ID | Final verdict |
|---|---|---|
| Physical carrier-state auditor | `019f7447-b882-7c31-a59a-40e03d65d3e6` | `NO_MATERIAL_BLOCKER` |
| Experimental-design/operator auditor | `019f7447-6c10-7d93-b22e-f081211d10a1` | `NO_MATERIAL_BLOCKER` |
| Custody/evidence auditor | `019f7448-0de8-7bd0-9c6b-6cc00582065e` | `NO_MATERIAL_BLOCKER` |
| Claim-boundary adjudicator | `019f7448-58a5-7e21-a2db-3c4d5ca9c9ea` | `NO_MATERIAL_BLOCKER` |

## Physical Carrier-State Auditor

Agent ID: `019f7447-b882-7c31-a59a-40e03d65d3e6`

```text
reviewer_role: physical_sensor_authority_auditor
agent/thread id: agent unavailable; thread 019f5815-7922-7a53-b022-32f2c8d8c03f
model: GPT-5 Codex
checked_commit: a27e4cf1f41d1b12f146ecb347f24e666e5f160b; clean worktree, local HEAD, upstream-tracking ref, and live pushed branch all matched.
verdict = NO_MATERIAL_BLOCKER

findings: None. PHYS-REVIEW-01 remains repaired. The package remains intentionally blocked with no active sensor authority or live-contact authorization.

evidence paths/functions/lines:
- family10h_carrier_tomography_target.py:2414 operational_pin_capability_failures lines 2414-2454 requires exact CPU 4/5 entries, singleton requests/readbacks, restoration, pass status, and zero forbidden work.
- family10h_carrier_tomography_target.py:2457 lines 2457-2610 enforce k10temp, temp1_input, label absent-or-exactly-Tctl, AMD PCI identity, canonical driver/subsystem paths, descriptor pinning, and repeated identity checks.
- run_family10h_carrier_tomography_v1.py:1677 consumer lines 1677-1957 independently recompute physical, canonical-path, pin, identity, candidate-selection, sample, descriptor, challenge, and counter predicates.
- CARRIER_TOMOGRAPHY_TARGET_SELF_TEST.json:111 records passing wrong-sensor, label, drift, descriptor, noncanonical-path, and missing/null/non-list/non-exact affinity regressions.
- CARRIER_TOMOGRAPHY_CONTROLLER_SELF_TEST.json:3724 lines 3724-3749 record consumer rejection of missing/null pin readback, /tmp/k10temp, /tmp/pci, wrong hwmon, and forged authority.
- CARRIER_TOMOGRAPHY_SOURCE_HASHES.json:32 independently recomputed to the three supplied hashes; all 9 source entries and all 9 archive members matched.
- CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json:273 remains PACKAGE_BLOCKED; active contact/inventory/live/PMU counters are zero.
- Final parse commit left the target blob unchanged and added 43/0 lines only to package_json_parse_audit and its gates. Current-tree AST, 48 JSON files, and 2 JSONL files parsed cleanly; receipt and manifest digests matched.

attempted attacks: recomputed-digest identities using /tmp/k10temp and /tmp/pci; missing, null, string, and wrong singleton affinity readbacks; wrong hwmon/label/driver/subsystem/modalias; non-CPU-first selection; class-path substitution, descriptor swap, and identity drift; self-authored/rehashed authority without correct challenge or transport. These were audited through committed source and digest-verified receipts, not rerun.

concise recommendations: Accept this exact head for the physical sensor-authority review. Preserve PACKAGE_BLOCKED and make no target contact until four distinct exact-head C6 reviews are bound and the pre-contact gate passes. Any source change requires new hashes and fresh reviews.
```

## Experimental-Design Operator Auditor

Agent ID: `019f7447-6c10-7d93-b22e-f081211d10a1`

```text
reviewer_role: experimental-design/operator-law final reviewer
agent/thread_id: 019f7447-6c10-7d93-b22e-f081211d10a1
model: GPT-5 (Codex)
checked_commit: a27e4cf1f41d1b12f146ecb347f24e666e5f160b; clean HEAD, local branch, tracking ref, and live GitHub branch all match
verdict = NO_MATERIAL_BLOCKER

findings:
- OBS-01 (non-material): The stored controller parse audit counts 46 JSON files because prepare_only runs it before writing the controller receipt and manifest. The final tree contains 48; independent strict parsing of all 48 JSON files, both JSONL files/6 rows, and all 3 Python ASTs succeeded. validate_only also re-audits the completed tree.

evidence_paths/functions/lines:
- family10h_carrier_tomography_public.py:43 query identities 43-54; query_family 425-469; separate schedule construction 606-700; raw-response model 1286-1422 and 1822-1852; classifier 1496-1553; order-identifiability checks 1556-1682; strict holdouts 1753-1775; lifetime gates/vocabulary 1943-2082; fail-closed gate/adjudication 2133-2235.
- CARRIER_TOMOGRAPHY_OPERATOR_ANALYSIS_SELF_TEST.json:4 diagonal 7x7 confusion matrix, 1.0 balanced accuracy versus fixed 0.95, negative-classifier regressions, and order-collapse downgrades; strict holdout level sets/counts at 5159-5223.
- CARRIER_TOMOGRAPHY_CONTRACT.md:122 query/record law 122-175; held-out operator law and exclusive adjudication 228-278; complete five-class lifetime vocabulary 305-317.
- run_family10h_carrier_tomography_v1.py:1453 strict package parse audit 1453-1486; final-tree validation gate 6876-6905; generation order underlying OBS-01 at 6024-6037.
- CARRIER_TOMOGRAPHY_SOURCE_HASHES.json:29 all nine source hashes/sizes verified; canonical source digest at line 74. Tarball members are byte-identical; bundle and runtime hashes match the supplied identities.
- CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json:255 package remains blocked at 255/273; bound identities at 472-475; authority absent and active counters zero at 573-603.
- Parent-to-HEAD diff leaves the public model, contract, schedule artifacts, thresholds, target source, runtime sources, and runtime binary unchanged. The failed zero-byte affinity output is an R100 rename from misleading .json to .stdout.

attempted_attacks:
- Recomputed the schedule: 8,320 unique tuple IDs; all four active queries remain distinct, with 640 persistence rows each and separate ordered factorial rows.
- Traced level-leakage and in-sample-rescue attacks across replicate 1, mapping map1, and delay 10ms; each tested level is absent from training and participates in the hard gate.
- Traced confused, imbalanced-trivial, and training-memorized classifier attacks; each downgrades rather than supporting observed status.
- Traced query-order collapse, equal-contrast/different-raw-response, session/mapping/query confounding, confidence, polarity, and one-stratum aggregate-rescue attacks.
- Recomputed all supplied identities, compared all tar members, strict-parsed the final package, and audited the parent diff for repair widening.
- No target, SSH, SCP, ping, PMU, sensor-inventory, live-authority, or lab-device command was run.

recommendations:
- Accept this exact commit and hash set for the experimental/operator-law surface.
- Preserve PACKAGE_BLOCKED and no-live status until separate source-authority quorum, physical temperature authority, and final exact-object verification pass.
- Future regeneration should persist a post-manifest parse receipt so the stored count covers all 48 final JSON artifacts.
```

## Custody Evidence Auditor

Agent ID: `019f7448-0de8-7bd0-9c6b-6cc00582065e`

```text
reviewer_role: custody/evidence final reviewer, read-only
agent/thread id: 019f5815-7922-7a53-b022-32f2c8d8c03f
model: GPT-5 Codex
checked_commit: a27e4cf1f41d1b12f146ecb347f24e666e5f160b; local HEAD, tracking ref, and hosted branch all match
verdict = NO_MATERIAL_BLOCKER

findings: none. CUSTODY-EVIDENCE-01 and CUSTODY-EVIDENCE-02 are repaired for the bound commit and hash set.

evidence paths/functions/lines:
- Manifest canonical/file hashes match recomputation: CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.sha256:2 and CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json:263.
- Source receipt canonical digest 6c7fd544... and runtime authority e4005546... validate: CARRIER_TOMOGRAPHY_SOURCE_HASHES.json:32 and CARRIER_TOMOGRAPHY_SOURCE_HASHES.json:74.
- Deterministic nine-file bundle reconstruction equals 5b5d2968...; actual bundle hash matches.
- All 12 committed self-test receipts pass their canonical digest and cross-link checks. Runtime receipt binds committed and isolated-compile hashes byte-exactly: CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST.json:43.
- Offline validation and deployment layout pass: CARRIER_TOMOGRAPHY_OFFLINE_VALIDATE.json:12 and CARRIER_TOMOGRAPHY_DEPLOYMENT_LAYOUT_SELF_TEST.json:2.
- Documented validate_only() returned exit 0, passed=true, no failures/missing artifacts, 8,320 tuples, and exact zero active counters: run_family10h_carrier_tomography_v1.py:6835.
- Recursive strict parse passed for all 48 committed JSON files and 2 JSONL files containing 6 rows: run_family10h_carrier_tomography_v1.py:1453.
- The sole zero-byte committed artifact is .stdout; its receipt names that extension and binds empty-content SHA-256: AFFINITY_CAPABILITY_OBSERVATION/family10h_affinity_capability_observation_v1.sha256.json:3.
- Worktree porcelain is empty; staged and unstaged package diffs are empty.

attempted_attacks:
- HEAD/tracking/hosted-ref drift.
- Manifest canonical and file-hash substitution.
- Source receipt, source member, bundle reconstruction, runtime authority, and receipt-link mismatch.
- Invalid/duplicate/non-finite JSON, malformed/blank-row JSONL, and zero-byte JSON masquerading.
- Active counter smuggling and live-authority environment presence.
- Dependency on staged, unstaged, untracked, or post-commit package mutation.

concise recommendations: Accept both custody findings as closed at a27e4cf1. Preserve the current hash set and the intentional FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED decision; this review does not authorize sensor acquisition, target contact, or carrier activation.
```

## Claim-Boundary Adjudicator

Agent ID: `019f7448-58a5-7e21-a2db-3c4d5ca9c9ea`

```text
reviewer_role: claim-boundary / live-authority final reviewer
agent/thread id: unavailable
model: GPT-5 Codex
checked_commit: a27e4cf1f41d1b12f146ecb347f24e666e5f160b (HEAD and local origin/codex/family10h-tomography-repair tracking ref; clean worktree)
verdict: NO_MATERIAL_BLOCKER

findings: none

evidence paths/functions/lines:
- CARRIER_TOMOGRAPHY_CONTRACT.md:3 PACKAGE_BLOCKED; claim classes and prohibitions at lines 15-36; complete lifetime vocabulary at 305-317; C6 claim ceiling and custody gates at 340-435.
- TOMOGRAPHY_REPAIR_BOOTSTRAP.md:208 all four reviews are required before freeze; freeze does not authorize contact; prohibited repair-phase contact at lines 234-244.
- CARRIER_TOMOGRAPHY_IMPLEMENTATION_MANIFEST.json:273 blocked decision; missing final evidence at 65-130; failed source-authority quorum at 439-475; zero active live counters and non-authoritative historical contacts at 573-720.
- run_family10h_carrier_tomography_v1.py:370 source_audit_quorum; exact four-role/hash/custody checks through line 572. acquire_temperature_sensor_authority blocks before contact at 2686-2717. Package/final gates remain fail-closed at 5818-5917 and 6407-6832.
- run_family10h_carrier_tomography_v1.py:1453 parse audit is read-only validation, connected only to self-test/validate paths at 5647-5753 and 6893-6911.
- family10h_carrier_tomography_target.py:3161 discovery rejects live-authority environment; repaired fixture-aware sensor construction at line 3222 does not relax canonical production behavior. Live execution independently requires exact authority at 4002-4114.
- family10h_carrier_tomography_public.py:764 structured/text forbidden-claim scanners; complete lifetime classification at 1855-2082; exclusive four-class adjudication and injected boundary attacks at 2155-2335.
- Independent read-only checks reproduced all three bound identities, all nine source member hashes, the deterministic bundle membership, and successful parsing of all 48 JSON files plus 2 JSONL files.

attempted attacks:
- Searched for positive SMALL_WALL_CROSSED, OrbitState, and frozen-status promotion.
- Tried missing, malformed, duplicate, stale, and hash-mismatched source-review/final-receipt paths.
- Traced direct/default discovery and live-execution entrypoints for authorization bypasses.
- Tested lifetime result completeness, confounder downgrades, and forbidden extra result classes.
- Parsed every committed JSON/JSONL artifact and checked the renamed zero-byte stdout receipt.
- No target-contact, live-authority, PMU, sensor-inventory, SSH, SCP, ping, or device command was run.

concise recommendations:
- Keep the package PACKAGE_BLOCKED.
- Any future transition requires fresh four-role C6 source-authority evidence bound to this exact commit and identities, followed by separately authorized discovery and final exact-object evidence.
- Treat any later frozen status as readiness only; live acquisition must retain its independent explicit authorization gate.
```

## Parent Disposition

All four final read-only reviewers returned `NO_MATERIAL_BLOCKER` for the repaired package at commit `a27e4cf1f41d1b12f146ecb347f24e666e5f160b`.

The repaired material findings are:

```text
PHYS-REVIEW-01
OPER-REVIEW-01
OPER-REVIEW-02
OPER-REVIEW-03
OPER-REVIEW-04
CUSTODY-EVIDENCE-01
CUSTODY-EVIDENCE-02
```

The retained non-material observation is:

```text
OBS-01
```

The correct package decision remains:

```text
FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED
```

Reason: the real provenance-bound temperature sensor authority receipt, C6 source-authority review package, live sensor acquisition, and final exact-object evidence remain unperformed and unauthorized in this repair branch. No live execution, PMU execution, sensor inventory, target contact, OrbitState claim, or Small Wall promotion is authorized by this review archive.
