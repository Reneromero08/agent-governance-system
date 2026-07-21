# Catalytic Waveform-Ising V2 Independent Adversarial Review

Reviewer ID: `CODEX-V2-BATCH-ADV-R1-C20F`

Verdict: `PASS`

Findings: `P0=0; P1=0; P2=0; open=0`

The review was read-only and independently checked the exact current candidate after
the remote freeze and pre-oracle commits. It regenerated all 32 instances from the
public seed, verified zero overlap with the 19-case development corpus, enumerated all
32 Ising states for every instance, reconstructed every classification and aggregate,
and reproduced the result and oracle-trace bytes.

Independent outcome:

```text
unique optima                 27
accepted correct              25
accepted incorrect             1
raw correct below gate         0
raw incorrect                  1
non-unique                     5
uninterpretable                0
unique raw optimum agreement  25 / 27
non-unique raw matches          5 / 5
```

All eleven controls passed on all 32 instances. Strict removed-transform causality
passed the history-and-response AND law on all 32; the independently recomputed minima
were `12.4028906568` history L2 and `0.0119151737615` response L2. Restoration, exact
restored-carrier reuse, second restoration, result persistence, and reuse output
reproduction passed 32/32.

The reviewer inspected the actual transitive native data flow through the recursive
tree renderers and found no decoded-spin, raw-sign, scalar `J@s`, energy, oracle,
winner, score, answer-cache, result-latch, or prior-outcome feedback into native
evolution. The five consolidation steps remain native complex-wave operations and are
part of the reversible history; they are not boundary cleanup.

`VERIFIED` is forbidden because the prospectively frozen accepted-incorrect maximum is
zero and one accepted-incorrect result occurred. The complete evidence exactly
authorizes:

`CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_PARTIAL`

Claim ceiling:

`BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY`
