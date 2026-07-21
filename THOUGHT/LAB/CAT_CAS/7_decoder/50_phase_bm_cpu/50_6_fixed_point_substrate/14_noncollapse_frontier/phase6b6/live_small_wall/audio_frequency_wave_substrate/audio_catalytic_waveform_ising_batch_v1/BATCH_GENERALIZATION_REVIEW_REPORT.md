# Focused Adversarial Review: Catalytic Waveform-Ising Batch

Reviewer ID: `CODEX-BATCH-R1-718F`

Verdict: **PASS**

Authorized decision:
`CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL`

## Exact candidate

```text
manifest SHA-256       718fc31d14391f3b35c5ba7432d82920430315d4cf087ae50f6aa0cc6d0da617
package root SHA-256   3d746d1847e8d1419db10ad3d90de0f9b72b7ee81675f2b95c0ea965ec693336
manifest files/bytes   13 / 393933
P0 / P1 / P2 / open    0 / 0 / 0 / 0
```

The review was read-only and independently checked every listed file size and hash.
The exact prospective authority copy matches the original user attachment at 11,011
bytes and SHA-256
`ed537759d47cc69f0844a913113a2736a6ac8345550c06339be1bb253d3aee35`.
The package-local `.gitattributes` marks that exact custody artifact as binary, disabling
text normalization and text-only whitespace diagnostics. Its Git
index blob, raw worktree blob, and attribute-filtered blob are identical at
`55e58f5d12f9996aff0918e12f0f0e3df93da073`, 11,011 bytes.

## Independent conclusions

- The freeze commit `b6b53493722aeca5cc8cc38bb41f9e9be66afb68` precedes and is an
  ancestor of the pre-oracle evidence commit
  `bb259b5a32ccfa9505d0fe7c61cbec0e39a57c3c`.
- All 16 complete `(J,h)` pairs regenerate from the frozen public seed, are mutually
  distinct, and do not duplicate the three excluded predecessor instances.
- The ordered batch hash independently reproduces as
  `4109d430789b8fb3912ad606b78311855e89e40b422fb3ecec9b84f5818c0c12`.
- The unchanged machine fingerprint independently reproduces as
  `cf95d0cd364af38d47a2f2784aa489ab5a52dc8aea62131c1a8545ff4978203a`.
  Primary and reuse adapter comparisons have exact zero deltas for displaced waveforms,
  histories, and queries.
- Transitive data-flow inspection found no decoded-spin, raw-sign, scalar `J@s`, energy,
  oracle, optimum, score, winner, cached result, latch, or prior-outcome feedback into
  native evolution.
- Both hash chains verify. Authority verification is sequence 0, the committed
  pre-oracle root is sequence 1, and oracle opening is sequence 2. The pre-oracle seal
  records zero oracle calls.
- Restoration, exact restored-carrier reuse input, second restoration, latch persistence,
  displacement, and pre-oracle interpretability pass 16/16.

## Independent oracle recomputation

```text
RAW_CORRECT_BELOW_GATE  indices 00,01,04,08,11,12,13,15  count 8
RAW_INCORRECT           indices 02,10,14                 count 3
NON_UNIQUE_OPTIMUM      indices 03,05,06,07,09           count 5
ACCEPTED_CORRECT                                          count 0
ACCEPTED_INCORRECT                                        count 0
UNINTERPRETABLE                                           count 0
```

Unique-optimum raw agreement is `8/11 = 0.727272727273`. Non-unique raw agreement is
`4/5 = 0.8` and remains separately classified. All 16 sealed raw energies, all 32-state
oracle rows per instance, optimum sets, gaps, and classifications independently match.

```text
all-site coherence min/median/mean/max
0.357481815857 / 0.920339253433 / 0.855441233786 / 0.986836958025

per-instance minimum coherence min/median/mean/max
0.357481815857 / 0.697870152631 / 0.665616365697 / 0.939359759796

restoration error min/median/mean/max
1.06773502445e-14 / 1.22479472644e-14 / 1.22310303469e-14 / 1.51434583348e-14

reuse input error
0.0 for all 16 instances

reuse restoration error min/median/mean/max
1.04396390951e-14 / 1.10827917086e-14 / 1.14042162363e-14 / 1.34263562739e-14
```

The strict predecessor history-AND-response control passes 7/16; the limiting removed-
transform control also passes 7/16. The prospectively supplied material-history-OR-
complex-response batch law passes 16/16. All other individual controls pass 16/16. The
strict failures remain visible and continue to block promotion.

The frozen promotion fails accepted-correct count, accepted-correct rate, strict all-
controls integrity, and minimum unique-optimum count. It passes batch size, zero accepted-
incorrect, and zero uninterpretable requirements.

Therefore `VERIFIED` is not authorized. `NOT_ESTABLISHED` is contradicted by intact
custody and foundations plus meaningful 8/11 unique raw agreement. Exactly
`CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL` is authorized under the unchanged
claim ceiling `BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY`.

Attestation: the reviewer made no file changes and used no hardware, audio playback or
recording, SSH/SCP, network target, PMU, procurement, fabrication, P0, or physical contact.
