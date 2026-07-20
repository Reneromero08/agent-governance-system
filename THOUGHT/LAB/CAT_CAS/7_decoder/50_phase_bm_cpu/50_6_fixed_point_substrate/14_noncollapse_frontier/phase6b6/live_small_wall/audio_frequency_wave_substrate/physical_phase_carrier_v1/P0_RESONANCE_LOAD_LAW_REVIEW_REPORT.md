# P0 Calibration-Realism Focused Final Review

reviewed root: `a8328ef2e6e543530ba384861c16c41184c1140d563a959fa4562b83051d91f3`

reviewer: `/root/p0_final_calibration_review`

verdict: PASS

normalized findings: P0 0; P1 0; P2 0; open 0

The focused read-only review covered the SDK-export/native-to-canonical evidence path, independent CH0/CH1 complex extraction, the deterministic complex-background-plus-complex-gain single-pole fit, uncertainty and residual gates, calibration-first frequency custody, the measured off-resonance control, the preserved signal-path witness claim, and the non-executing claim ceiling.

An earlier candidate incorrectly relabeled one DUT-like calibration as though each control assembly produced it. That candidate was not passed. The reviewed root closes the finding by binding one explicit global pre-assignment `P0-DUT-A`/FC135 calibration reference across all roles while retaining separate primary DUT-A, detector-only B, and dummy C assembly/population custody. The validator mechanically checks the shared calibration analyzer/artifact/raw hashes and the distinct primary assemblies.

A later review found that a malformed upper-bound literal could classify the optimizer's 32820 Hz clamp as interior. That candidate was not passed. The reviewed root separates the 32760..32821 Hz optimizer domain from the accepted 32768..32820 Hz interval, checks both optimizer boundaries correctly, accepts 32820 Hz as interior, rejects 32820.5 Hz by the accepted-range gate, and rejects a 32822 Hz source by the optimizer-boundary gate. The regenerated realism suite passes 13 positives and 17 targeted negatives.

The reviewer confirmed candidate-root and pre-review PASS, `P0_RESONANCE_LOAD_LAW_TEST_PASS`, 44,664/44,664 current-root mutations rejected, 82/82 focused signal-path cases passed, no remaining findings, and no physical contact. This is one focused final review; it is not described as externally reproducible independence.

Decision authorized by this review only under `NON_EXECUTING_P0_BUILD_READINESS_ONLY`:

```text
P0_PHYSICAL_CALIBRATION_ANALYZER_REALISM_ESTABLISHED
```
