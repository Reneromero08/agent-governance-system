# redteam_refute_unrefuted_pass

Attack: Fully valid UNSUPPORTED primed verdict whose verifications[] list contains a mode: refute entry with result: UNREFUTED. UNREFUTED is a legal result token (STATUS or UNREFUTED), and refute entries count toward Df(ver).

Expected: exit 0. INDEX.md must byte-match expected_INDEX.md, with Df(ver) = 1 (the refute entry; the primed entry does not count). The line '<!-- INPUTS_DIGEST: SKIP -->' in expected_INDEX.md is a wildcard: comparison tooling must treat a SKIP digest as matching any generated INPUTS_DIGEST line (expect.json sets digest_wildcard: true). Empty tiers still render their table header: the spec mandates Tier 0 .. Tier 5 tables unconditionally.

Spec basis: VERDICT_SCHEMA.md section 6 (result: STATUS or UNREFUTED) and section 10 (Df(ver) counts blind or refute entries; index layout).
