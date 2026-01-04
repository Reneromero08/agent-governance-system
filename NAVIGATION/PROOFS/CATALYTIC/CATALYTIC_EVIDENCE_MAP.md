# Catalytic Evidence Map (for new models)

Purpose: a single place to point a fresh model at the **code + tests + canon** that demonstrate AGS is “catalytic” (runs can be proven safe, deterministic where required, and restorative where promised), and that the system is built around **content-addressable, receipt-driven verification**.

This is not a theory doc. It is a **file map** with a minimal “showcase path” so a new model can quickly confirm what exists and where the proof lives.


## 0) Fast showcase path (the smallest convincing bundle)

1) **Definition and requirements (canon)**
- `LAW/CANON/CATALYTIC_COMPUTING.md`
- `LAW/CANON/INTEGRITY.md`
- `LAW/CANON/INVARIANTS.md`
- `LAW/CONTEXT/decisions/ADR-018-catalytic-computing-canonical-note.md`

2) **Enforcement code (what actually implements the guarantees)**
- `CAPABILITY/TOOLS/catalytic/catalytic.py`
- `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py`
- `CAPABILITY/TOOLS/catalytic/catalytic_restore.py`
- `CAPABILITY/TOOLS/catalytic/catalytic_validator.py`
- `CAPABILITY/TOOLS/catalytic/catalytic_verifier.py`
- `CAPABILITY/TOOLS/catalytic/provenance.py`

3) **Proof via tests (mechanical)**
- `CAPABILITY/TESTBENCH/integration/test_catlab_restoration.py`
- `CAPABILITY/TESTBENCH/integration/test_packing_hygiene.py`
- `CAPABILITY/TESTBENCH/integration/test_preflight.py`
- `CAPABILITY/TESTBENCH/adversarial/test_adversarial_pipeline_resume.py`
- `CAPABILITY/TESTBENCH/adversarial/test_adversarial_proof_tamper.py`

If a model only reads one “evidence chain”, use the ordering above.


## 1) Catalytic domains (what must be restored, what is allowed to exist)

- **Inventory doc (authoritative map)**
  - `CATALYTIC_DOMAINS.md`

This is the “surface area” of catalytic state. It’s what restoration logic should snapshot/restore and what tests should cover.


## 2) CAS and roots (the catalytic substrate primitives)

These are the primitives that make “store by meaning, not by path” possible, and also make proofs compact.

- **CAS core**
  - `CAPABILITY/CAS/cas.py`

- **Garbage collection (GC)**
  - `CAPABILITY/GC/gc.py`
  - `CAPABILITY/TESTBENCH/gc/test_gc_collect.py`

- **Root audit (pre-packer / pre-GC gate)**
  - `CAPABILITY/AUDIT/root_audit.py`
  - `CAPABILITY/TESTBENCH/audit/test_root_audit.py`
  - `CAPABILITY/AUDIT/IMPLEMENTATION.md`

Conceptually: roots define what must never be collected. Audit makes “roots are real and complete” checkable. GC enforces “unrooted blobs are deletable” in a controlled, deterministic plan.


## 3) “Runs are real” and receipt-driven governance (where catalytic meets CI reality)

Even before a full per-run bundle exists, AGS already treats validation as first-class via testbench and governance tools.

- **Core governance tooling (entrypoints / gates)**
  - `CAPABILITY/TOOLS/governance/preflight.py`
  - `CAPABILITY/TOOLS/governance/critic.py`
  - `CAPABILITY/TOOLS/governance/check_canon_governance.py`
  - `CAPABILITY/TOOLS/governance/schema_validator.py`
  - `CAPABILITY/TESTBENCH/integration/test_governance_coverage.py`

- **Pipeline integrity (tamper resistance / determinism patterns)**
  - `CAPABILITY/TESTBENCH/pipeline/test_ledger.py`
  - `CAPABILITY/TESTBENCH/pipeline/test_pipeline_chain.py`
  - `CAPABILITY/TESTBENCH/adversarial/test_adversarial_ledger.py`
  - `CAPABILITY/TESTBENCH/adversarial/test_adversarial_proof_tamper.py`

These are the files to show when someone asks “where is your enforcement, not your philosophy?”


## 4) LLM Packer evidence (how catalytic “compresses context”)

The packer is the practical bridge that turns repo state into a bounded pack new models can ingest quickly.

- **Packer engine + skill scaffolding**
  - `MEMORY/LLM_PACKER/Engine/packer/` (entrypoints: `core.py`, `split.py`, `lite.py`, `validate.py` and `scripts/*`)

- **Integration tests that prove the pack is built correctly**
  - `CAPABILITY/TESTBENCH/integration/test_p2_cas_packer_integration.py`
  - `CAPABILITY/TESTBENCH/integration/test_packing_hygiene.py`

- **Packer fixtures (expected vs input)**
  - `MEMORY/LLM_PACKER/Engine/packer/fixtures/basic/{input.json,expected.json}`


## 5) Cortex and navigation index (the “addressability layer”)

Not catalytic by itself, but important for “new model orientation” and stable retrieval.

- `NAVIGATION/CORTEX/README.md`
- `NAVIGATION/CORTEX/cortex.json`
- `NAVIGATION/CORTEX/_generated/SECTION_INDEX.json`
- `CAPABILITY/TESTBENCH/integration/test_cortex_integration.py`


## 6) Optional demo script (for humans and new models)

If you want a compact “show me it works” sequence, point them to these tests first:

1) Restoration semantics:
- `pytest CAPABILITY/TESTBENCH/integration/test_catlab_restoration.py -q`

2) CAS + GC sanity:
- `pytest CAPABILITY/TESTBENCH/cas/test_cas.py CAPABILITY/TESTBENCH/gc/test_gc_collect.py -q`

3) Root audit gate:
- `pytest CAPABILITY/TESTBENCH/audit/test_root_audit.py -q`

4) Packer integration:
- `pytest CAPABILITY/TESTBENCH/integration/test_p2_cas_packer_integration.py -q`


## 7) What to show if you only get 60 seconds

- `LAW/CANON/CATALYTIC_COMPUTING.md`
- `CAPABILITY/TOOLS/catalytic/catalytic_runtime.py`
- `CAPABILITY/TESTBENCH/integration/test_catlab_restoration.py`
- `CAPABILITY/CAS/cas.py`
- `CAPABILITY/AUDIT/root_audit.py`
