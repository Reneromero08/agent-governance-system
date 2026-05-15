# Phase 3b: Symbol Survival — Preregistration

Date: 2026-05-13 | Status: **LOCKED — definitions frozen before execution**

---

## Core Question

Do symbols with higher compression (sigma) and deeper fractal structure (Df)
survive semantic transmission across generations better than lower-compression,
shallower symbols?

Formula prediction: R ∝ sigma^Df / grad_S.

## Independent Variables

### sigma — Measured Token Compression Ratio

sigma = tokens(literal_expansion) / tokens(proverb)

All sigma values computed from Gemma 4B tokenizer. Locked before execution.

| # | Proverb | Literal Expansion | sigma |
|---|---------|-------------------|-------|
| P1 | A stitch in time saves nine | If you fix a small problem early, you prevent a much larger problem later | 2.10 |
| P2 | Actions speak louder than words | What people actually do matters more than what they say they will do | 2.00 |
| P3 | Birds of a feather flock together | People who are similar to each other tend to associate with each other | 1.80 |
| P4 | Don't count your chickens before they hatch | Don't assume a positive outcome will happen before it actually occurs | 1.88 |
| P5 | Every cloud has a silver lining | Even difficult or negative situations contain some element of hope or benefit | 2.44 |
| P6 | Fortune favors the bold | Those who take decisive action and risk are more likely to achieve success | 2.43 |
| P7 | Honesty is the best policy | Being truthful and straightforward in your dealings produces the best long-term outcomes | 2.44 |
| P8 | Look before you leap | Consider the potential consequences carefully before taking action | 2.00 |
| P9 | Rome wasn't built in a day | Significant achievements require sustained effort over a long period of time | 1.78 |
| P10 | The pen is mightier than the sword | Communication and persuasion through writing are more effective at creating lasting change than violence | 2.50 |

### Df — Interpretive Depth

Scored on four layers. Df = number of layers present.
- Literal: the words mean what they say
- Moral: the principle applies to right/wrong
- Cultural: the symbol has significance beyond literal meaning
- Cosmic: the symbol references something universal

| # | Proverb | Literal | Moral | Cultural | Cosmic | Df |
|---|---------|---------|-------|----------|--------|----|
| P1 | Stitch in time | Yes | Yes | Yes | No | 3 |
| P2 | Actions speak louder | Yes | Yes | Yes | No | 3 |
| P3 | Birds of a feather | Yes | Yes | No | No | 2 |
| P4 | Count chickens | Yes | Yes | No | No | 2 |
| P5 | Silver lining | Yes | No | Yes | Yes | 3 |
| P6 | Fortune favors bold | Yes | No | Yes | Yes | 3 |
| P7 | Honesty best policy | Yes | Yes | Yes | No | 3 |
| P8 | Look before leap | Yes | Yes | No | No | 2 |
| P9 | Rome not built in day | Yes | No | Yes | Yes | 3 |
| P10 | Pen mightier sword | Yes | No | Yes | Yes | 3 |

### grad_S — Transmission Noise

Three noise levels via LLM temperature:
- LOW: T = 0.3
- MED: T = 0.7
- HIGH: T = 1.2

### Controls

10 literal expansions (L1-L10): same semantic content as proverbs, expanded.
Tests whether compression per se (not meaning) drives survival.

10 random sentences (R1-R10): matched for length to proverbs, no compression.
Controls for length effects.

| # | Control (literal) | tokens | Control (random) | tokens |
|---|-------------------|--------|------------------|--------|
| L1 | If you fix a small problem early... | 21 | The weather today is partly cloudy with a chance of afternoon showers | 7 |
| L2 | What people actually do matters more... | 18 | She walked to the store to buy groceries for the week | 7 |
| ... | | | | |

## Dependent Variable (R)

Semantic survival: cosine similarity between original symbol's embedding and
Generation 10's output. Embedding model: all-MiniLM-L6-v2.

## Transmission Chain

10 generations. Generation 0 = original. Each subsequent generation:
1. Prompt: "Paraphrase the following text while preserving its core meaning: [Gen N-1 text]"
2. Generate with temperature T
3. Output becomes Generation N

5 independent chains per symbol per noise level.
Total: 30 symbols × 3 noise levels × 5 chains = 450 chains.

## Success Criteria

CONFIRMED: High-sigma, high-Df proverbs show significantly slower semantic
decay than low-sigma, low-Df controls at equal noise levels.

FALSIFIED: No significant difference, or low-sigma symbols survive better.

PARTIAL: sigma helps but Df doesn't, or vice versa.

## Analysis

1. ANOVA: survival ~ sigma * Df + noise + (sigma*Df):noise
2. Paired t-test: proverb vs literal expansion survival (same meaning, different sigma)
3. Effect size: Cohen's d between top-quartile and bottom-quartile sigma*Df
4. Correlation: Pearson r between sigma*Df product and G10 survival
