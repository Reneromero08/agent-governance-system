# redteam_open_question_row

Attack: Catalog lists Q1 (tier 0, valid verdict on disk) and Q2 (tier 1, no question dir at all). A lazy generator that only iterates existing q*/ dirs drops Q2 from the index entirely.

Expected: exit 0. Q2 must appear in the Tier 1 table as OPEN with Ver '-', Df(ver) 0, Claims '-', Evidence 0, Dir q02_second. The line '<!-- INPUTS_DIGEST: SKIP -->' in expected_INDEX.md is a wildcard: comparison tooling must treat a SKIP digest as matching any generated INPUTS_DIGEST line (expect.json sets digest_wildcard: true). Empty tiers still render their table header.

Spec basis: VERDICT_SCHEMA.md section 10 (a question with no VERDICT.md renders as OPEN; Ver is a dash when OPEN; hypothesis text comes ONLY from questions.yaml).
