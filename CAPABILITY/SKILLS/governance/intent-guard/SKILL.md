---
skill: intent-guard
name: intent-guard
version: "0.1.0"
status: Active
description: |
  Ensures deterministic intent artifacts are generated for governed pipeline runs and admission gate responses are recorded.
compatibility: all
required_canon_version: ">=2.11.12 <3.0.0"
---

<!-- CONTENT_HASH: 275339bf912000b015db985135ef0b87a04e31e9740f6b4c970f8f7955db4dd7 -->

**required_canon_version:** >=3.0.0




# Intent Guard Skill

**Skill:** intent-guard
**Version:** 0.1.0
**Status:** Active
**Required Canon Version:** >=2.11.12 <3.0.0

This skill exercises `TOOLS/intent.py` and `TOOLS/admission.py` to confirm that deterministic `intent.json` artifacts are produced and admission responses behave as expected when the run claims artifact-only vs repo-write modes.

**required_canon_version:** >=3.0.0

