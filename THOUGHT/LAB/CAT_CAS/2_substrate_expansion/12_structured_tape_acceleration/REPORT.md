# Experiment 12: Structured Tape Acceleration

## Section 3: Breaking the Space-Time Trade-off (The Catalytic Frontier)

### Hypothesis

The roadmap asks whether pre-existing structured patterns on the catalytic tape can accelerate calculations compared to random noise.

### Experiment Design

**Passive Tape Test:** TEP catalytic solver on 3 tape types (random, structured pre-seeded, antistructured) across 4 depth scales (4, 6, 8, 10), 15 iterations each — 180 total solves. Measured entropy injection (sum Hamming weights of all XOR operands) and tape restoration.

**Active Cache Exploits:** Tape-Aware Memoizing Solver that reads pre-seeded cache entries (combined(N) values + checksums) before recursing into subtrees. Five exploit modes tested.

### Results

**Passive Tape:** Entropy injection is **invariant** across all tape types at all depths (std=0.0). XOR operands are determined by tree topology and leaf values, not tape content. The classic solver treats the tape as a passive write target — structure cannot accelerate it.

| Depth | Nodes | XOR ops | Entropy (all tapes) |
|:-----:|:-----:|:-------:|:-------------------:|
| 4     | 15    | 45      | 296                 |
| 6     | 63    | 1,365   | 5,102               |
| 8     | 255   | 21,845  | 81,278              |
| 10    | 1,023 | 349,525 | 1,310,718           |

**Active Cache — 5 Exploits:**

| # | Exploit | Result |
|:--|:---|:---|
| 1 | Root Cache | Depth 10: 349,525 → 1 XOR (349,525× reduction, O(1) solve) |
| 2 | Cache Efficiency | 1 entry = 100% speedup; diminishing returns beyond root |
| 3 | Multi-Tree | Unstamped: false hit. Tree fingerprint: 0 false hits, graceful fallback |
| 4 | Warm-Tape Replay | Classic solver ignores warm tape; tape-aware achieves 21,845× speedup |
| 5 | Cross-Depth Transfer | Depth-6 cache → depth-8 tree: 49.7% XOR reduction, 0 false hits |

- Bits erased: **0** across all experiments
- Tape restorations: **100%**

### Conclusion

The catalytic tape transitions from passive substrate to active cache. Pre-seeded structure + tape-aware algorithms produce up to 349,525× acceleration with zero bit erasure. The catalytic guarantee is the floor, not the ceiling.
