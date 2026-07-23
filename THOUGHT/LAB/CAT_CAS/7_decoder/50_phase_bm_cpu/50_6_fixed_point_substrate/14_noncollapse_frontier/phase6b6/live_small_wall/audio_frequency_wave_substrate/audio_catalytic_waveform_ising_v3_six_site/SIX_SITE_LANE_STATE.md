# Catalytic Waveform-Ising V3 Six-Site Lane State

Decision: `CATALYTIC_WAVEFORM_ISING_V3_SIX_SITE_VERIFIED`
Claim ceiling: `BOUNDED_SOFTWARE_RECURSIVE_SPECTRAL_PHASE_REFERENCE_ONLY`

## Frozen custody

```text
machine fingerprint        5c2bd2883533c1ade8a6e6aad45dea2dd1cecc7fae9c03ee27d5e440112c85ec
ordered batch SHA-256      a5cf55486728a0fa9dc308f61ea87f477464af60bcd0c6896168c9c9386000c9
pre-oracle evidence SHA-256 0605ab493f6aa16e41443e6e2e98aed6d4a280c83e1e89230c24fb27867e72d2
oracle calls before seal   0
energy calls before seal   0
```

## Prospective result

```text
batch size                 256
unique optima              206
unique raw correct         206
accepted incorrect         0
rejected unique correct    0
non-unique rejected        50
promotion pass             True
```

## Independent verification

Reviewer: `V3-SIX-SITE-INDEPENDENT-REEXECUTION-01`
Verdict: `PASS`
Findings: `0`
States independently enumerated: `16384`

The bounded software mechanism uses a complete 64-mode recursive spectral phase
tree. It establishes correctness, reject-only handling of tied optima, exact
inverse restoration within the frozen tolerance, carrier reuse, and the frozen
anti-smuggling controls for the six-site reference domain.

It does not establish computational advantage, scalable complexity advantage,
physical waveform computation, hardware persistence, or bit replacement.
