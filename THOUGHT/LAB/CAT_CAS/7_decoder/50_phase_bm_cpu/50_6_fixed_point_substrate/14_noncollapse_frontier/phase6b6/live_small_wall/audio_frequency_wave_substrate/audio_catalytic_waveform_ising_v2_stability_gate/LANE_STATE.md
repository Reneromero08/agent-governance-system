# V2 Stability-Gate Lane State

Decision: `CATALYTIC_WAVEFORM_ISING_V2_STABILITY_BATCH_NOT_ESTABLISHED`
Review: `STABILITY-GATE-FINAL-REVIEW-01 PASS`
Claim ceiling: `BOUNDED_SOFTWARE_REJECT_ONLY_WAVEFORM_STABILITY_REFERENCE_ONLY`

```text
batch size                       64
unique optima                    56
non-unique optima                 8
nominal accepted correct         48
nominal accepted incorrect        5
nominal rejected incorrect        3
stability accepted correct       48
stability accepted incorrect      5
false-accept reduction             0
correct-result retention          1.0
strict and stability controls     64 / 64
restoration and reuse             64 / 64
```

The discriminator remained waveform-native, reject-only, restorable, and
reusable, but it did not change any unseen acceptance decision. Five nominal
accepted-incorrect results remained accepted, so the discriminator did not
provide credible unseen false-accept reduction and is not promoted.

The frozen V2 machine and all predecessor results remain unchanged. No hardware,
playback, recording, procurement, fabrication, physical contact, reliability
promotion, physical-computation claim, bit-replacement claim, advantage claim,
or Wall claim is authorized.
