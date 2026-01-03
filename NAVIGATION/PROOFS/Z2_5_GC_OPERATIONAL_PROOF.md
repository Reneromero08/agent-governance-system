---
type: operational_proof
title: Z.2.5 Garbage Collection Operational Proof
date: 2026-01-02
executor: Antigravity Agent
executor_uuid: 09f19580-20c7-4e45-aa1a-06faa957b585
system: Windows / PowerShell
tags:
  - Z.2.5
  - GC
  - CAS
  - proof
---

<!-- CONTENT_HASH: 115f7a97d56fcb0d9dbf2428e591d558780d0a143db40553233642b4156abe16 -->

## 1. Objective
Run an operational proof that Z.2.5 GC does not propose deleting rooted blobs, operates deterministically, and fails closed on empty roots.

## 2. Dirty State Preparation
To simulate a live environment, the CAS storage was populated using the existing test suite.

**Command:**
```powershell
pytest CAPABILITY/TESTBENCH/cas/test_cas.py
```

**State:**
- `CAPABILITY/CAS/storage` populated with ~87 test blobs.
- Target Root: `dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f` ("Hello, World!")

## 3. Execution & Determinism Check
The GC was executed twice in `dry_run=True, allow_empty_roots=False` mode against a mock `RUN_ROOTS.json`.

**Setup:**
```powershell
New-Item -ItemType Directory -Force -Path "LAW/CONTRACTS/_runs/_tmp/gc_proof"
Set-Content "LAW/CONTRACTS/_runs/_tmp/gc_proof/RUN_ROOTS.json" '["dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"]'
```

**Execution:**
Two consecutive runs were performed to generate independent receipt files.

**Metrics:**
| Metric | Count |
| :--- | :--- |
| **Roots Identified** | 1 |
| **Reachable Hashes** | 1 |
| **Candidate Hashes (Garbage)** | 86 |

**Determinism Result:**
The two receipts `receipt_1.json` and `receipt_2.json` were compared.
> **Result: Byte-Identical MATCH**

## 4. Fail-Closed Verification
The GC was run with an empty roots source to test Policy B (POLICY LOCK).

**Condition:** `roots=[]`, `allow_empty_roots=False`

**Result:**
- **Deleted Hashes:** 0
- **Error Output:**
  > `POLICY_LOCK: Empty roots detected and allow_empty_roots=False. Fail-closed: no deletions.`

## 5. Artifacts
The receipts and proof context are located at:
`LAW/CONTRACTS/_runs/_tmp/gc_proof/`
- `receipt_1.json`
- `receipt_2.json`

## 6. Conclusion
The Operational Proof confirms:
1.  **Safety:** Rooted blobs are preserved (1 reachable).
2.  **Correctness:** Unreferenced blobs are correctly identified as candidates (86 candidates).
3.  **Determinism:** Execution is strictly deterministic (identical receipts).
4.  **Policy Compliance:** The system fails closed when roots are missing.
