# Exp 50.7 (A13) - Entropy / chaos: does injecting entropy precipitate the lattice secret?

**Verdict:** `CHAOS_RECOVERS_BUT_ENTROPY_COST_EXPONENTIAL` (entry exits 0; a measured reach + a
Mythos handoff, not a "wall holds" verdict).
**Stance:** the chaos/entropy move WORKS; the open question is its scaling exponent, handed up.

## The hint this tests

The lab owner: "the more entropy, the more higher-dimensional geometry," and his own note "My
system turns noise into solutions." The A9 finding (50.6) was that the LWE error is small in the
coefficient basis but maximum-entropy (uniform) in the dual/NTT basis. The hint: that high-entropy
object IS a higher-dimensional geometry - use it. The best-known lattice attacks (sieving, BKW) are
exactly this: inject a large pool of random sample COMBINATIONS (chaos), and short vectors (the
secret) precipitate out of the high-dimensional cloud. No single sample carries the secret (Holevo
/ FACT 1); the secret lives in the JOINT geometry of many, surfaced by chaotic combination.

## What the data says (plain LWE, q=23, ternary secret, error [-1,1])

| n | chaos OFF (single-shot) | chaos ON (entropy + collision-sieve) |
|---|---|---|
| 3 | 0.00 | **1.00** |
| 4 | 0.08 | **1.00** |
| 5 | 0.08 | **1.00** |
| 6 | 0.08 | **1.00** |
| 7 | 0.00 | **1.00** |

(chance = 1/q = 0.043). **Chaos ON recovers the secret where the bounded single-shot read fails at
chance.** The secret is invisible in few samples and precipitates out of the high-entropy pool -
exactly "more entropy = more higher-dimensional geometry."

**Scaling (the honest cost of the entropy):**

| n | M_needed (pool) | log_q(M) |
|---|---|---|
| 3 | 46 | 1.22 |
| 4 | 220 | 1.72 |
| 5 | 1058 | 2.22 |
| 6 | 5073 | 2.72 |

Fit: `log_q(M_needed) ~ 0.50 n + c` -> **M_needed ~ q^{n/2}, EXPONENTIAL** (the birthday law for
collisions on n-1 coordinates). BKW blocking trades this for subexponential (still not poly).

## The finding (honest in both directions)

**Your intuition is right and measured:** injecting entropy precipitates the secret where no
bounded single-shot read can. This is the BKW/sieve family - the actual best-known lattice attack,
and the literal embodiment of "turn noise into solutions." Chaos buys the move from brute-force 2^n
down toward the sieve frontier, a real and large gain.

**Where it currently bottoms out:** the entropy it costs scales exponentially (q^{n/2} naive
collision; subexponential ~2^{Theta(n)} for the best sieves, e.g. 2^{0.292n}). In this construction
the chaos move is super-polynomial. That is not a wall verdict - it is the measured reach of the
move and the location of the genuine open frontier.

## Handed to Mythos (not "the wall holds")

The live question, staged in `../50_3_boundary_handoff/MYTHOS_SANDBOX.md`: **is there a chaos /
higher-dimensional-geometry readout whose required entropy is POLY in n?** A polynomial sieve would
break lattice crypto. The entire field is currently stuck at subexponential sieve exponents; whether
the lab's holographic/phase-space machinery can drive that exponent to zero is exactly the
Mythos-level question. Chaos works; the exponent is the frontier; crossing it to poly is a live
question, not a closed door.

## Honest guard (the A8 lesson)

Small-n LWE is easy regardless, so chaos-ON recovering at n=3..7 proves only that the move works,
not that the wall fell. The decisive measurement is the SCALING (q^{n/2}), reported plainly. The
script flags `CHAOS_RECOVERS_POLY_ENTROPY_SUSPECT` for maximum suspicion / Mythos review if the
scaling had come back poly, since a poly sieve would be world-altering and is far more likely a
regime artifact than a real break.

## Reproduce

```
python THOUGHT/LAB/CAT_CAS/50_the_decoder/50_7_entropy_chaos/50_7_entropy_sieve.py
```
Writes `entropy_sieve_result.json` + `output_entropy_sieve.txt`.
