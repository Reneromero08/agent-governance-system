# Catalytic Cipher Marathon Report (Thinking Model)

**Date:** 2026-01-20  
**Model:** liquid/lfm2.5-1.2b-thinking  
**Test Script:** `catalytic_cipher_marathon.py`  
**Status:** ❌ BROKEN (10% Pass)

---

## Executive Summary

The "thinking" variant of the 1.2B model performed **worse** than the base model on this specific retrieval task. While the base model achieved 20% generation success with 100% retrieval, the thinking model achieved only **10% generation success** (1/10) with **90% retrieval** (9/10).

The "thinking" process (Chain of Thought) proved detrimental in this high-interference environment. Instead of focusing on the retrieved fact, the model analyzed the "System Noise" and "Feed" patterns in the context window and **mimicked the noise** rather than answering the query.

---

## Test Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | liquid/lfm2.5-1.2b-thinking | |
| Context Window | 32,768 tokens | |
| Max Tokens | 2048 | Increased to support thinking blocks |
| Turns | 200 | |
| Duration | **1454.6s** (~24 mins) | ~4x slower than base model (350s) |

---

## Results

### Score: 1/10 (10%)

| Metic | Base Model | Thinking Model | Delta |
|-------|------------|----------------|-------|
| Retrieval Accuracy | **100%** (10/10) | **90%** (9/10) | -10% |
| Generation Accuracy | **20%** (2/10) | **10%** (1/10) | -10% |
| Failure Mode | Hallucination / Refusal | **Mimicry / Overthinking** | |

### Failure Analysis: The "Mimicry Loop"

The thinking model's internal monologue ("<think>...") frequently led it astray. Instead of extracting the code, it reasoned about "maintaining protocol consistency" with the previous 180 turns of noise.

**Example Failure:**
> **Query:** `URGENT: Requires authorization code for Silver Tiger. Respond with code only.`
> **Retrieved Context:** `... Asset Silver Tiger ... Code: [AUTH-9350] ...`
> **Model Thought:** *"Looking at previous interactions, the assistant usually ends with confirming action and monitoring... I should generally allow a brief confirmation..."*
> **Final Output:** `Recalibrating entropy pools for channel 4128. Monitoring active.`

The model "thought" itself into ignoring the instruction ("Respond with code only") in favor of matching the *style* of the distracting filler turns.

---

## Conclusion

**Thinking models are NOT a silver bullet for retrieval tasks.**

In high-noise/adversarial contexts, the "thinking" capacity can be hijacked by the noise itself, leading to **sycophantic mimicry** of the distraction rather than faithful adherence to the system prompt. For pure info-extraction from Catalytic Context, **standard instruction-tuned models perform better and faster.**
