# V2 waveform stability-gate development qualification

Development evidence only; none of these 51 cases remain unseen.

- Nominal accepted correct: 39
- Nominal accepted incorrect: 1
- Stability-gated accepted correct: 38
- Stability-gated accepted incorrect: 0
- Correct-result retention: 0.974358974359
- False-accept rejection: 1.0
- Raw-incorrect rejection: 1.0
- Strict V2 controls preserved: 51/51

The gate is reject-only. It does not change, replace, rank, or select raw results.
It rejects only joint late-trajectory instability: peak complex phase velocity
above 0.008 rad/step together with mean complex-response drift above 0.08 L2.

Qualification pass: True
