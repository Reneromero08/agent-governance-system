# Memory / Symbol Survival

## Purpose

This track tests whether compressed, multi-depth symbols preserve meaning under
noise better than uncompressed or shallow messages.

## Hypothesis

High-`sigma`, high-`Df` messages survive noisy transmission better than messages
with lower compression or lower redundancy.

## Domain Mapping

| Symbol | Observable |
|---|---|
| `E` | meaning core |
| `grad_S` | forgetting, distortion, transmission noise |
| `sigma` | symbolic compression fidelity |
| `Df` | literal/moral/cultural layer count or independent faithful fragments |
| `R` | recall, faithful paraphrase survival, persistence |

## Candidate Experiments

1. proverb vs literal explanation recall
2. paraphrase-chain survival
3. translation-cycle survival
4. noisy summary survival
5. human or LLM cultural-transmission chain

## Baselines

- token length
- readability
- repetition count
- simple semantic similarity
- mutual information alone
- majority-vote redundancy

## Failure Criteria

The mapping fails if message length, familiarity, or simple repetition explains
survival as well as or better than the formula.
