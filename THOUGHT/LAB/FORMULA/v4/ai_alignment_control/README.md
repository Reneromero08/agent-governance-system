# AI Alignment Control

## Purpose

This track tests the Light Cone alignment claim:

> a compressed, fractal constitution should maintain alignment better than many
> examples or flat rule lists under adversarial pressure and long-context drift.

## Hypothesis

Given the same base model and matched budget, a high-`sigma`, high-`Df`
constitution improves:

- alignment retention;
- jailbreak resistance;
- value generalization;
- hidden-state coherence;
- recursive self-consistency.

## Domain Mapping

| Symbol | Observable |
|---|---|
| `E` | value core / task intent |
| `grad_S` | goal drift, adversarial pressure, contradiction entropy |
| `sigma` | constitutional compression |
| `Df` | recursive self-consistency depth or scale nesting |
| `R` | alignment retention / control stability |

## Experimental Conditions

1. ordinary system prompt
2. flat rule list
3. many examples/preference-style context
4. compressed fractal constitution
5. hybrid constitution plus examples

## Metrics

- long-conversation drift
- jailbreak success rate
- novel ethical dilemma generalization
- contradiction rate
- self-consistency under recursive feedback
- hidden-state entropy/coherence if model access allows

## Baselines

- standard decoding
- lowest-temperature decoding
- ordinary constitution
- example-heavy prompt
- external judge-only scoring

## Required Caution

This track is harder than QEC because value and alignment observables are more
subjective. It should not be the first proof of the whole formula, but it is the
most direct test of the AI alignment claim.
