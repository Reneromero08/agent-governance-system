# TINY_COMPRESS Lab

**Status:** Experimental  
**Goal:** Train a tiny model (10M-50M params) to learn symbolic compression via RL against the Validator.

---

## What This Is

A lab experiment to prove that a **tiny model** can learn to compress intent into Symbolic IR without understanding meaning—purely by learning what the Validator rewards.

**Not for:**
- Task execution (see `THOUGHT/LAB/TURBO_SWARM` for that)
- Production use (this is research)

**For:**
- Proving compression can be learned, not hand-coded
- Benchmarking 10M-50M models vs rule-based compression
- Informing whether to invest in larger models (0.5B+)

---

## Architecture

```
User Intent (text)
    ↓
Tiny Model (10M-50M params)
    ↓
Symbolic IR (compressed)
    ↓
Validator (CMP-01, SPECTRUM)
    ↓
Reward Signal (+1 PASS, -1 FAIL, -0.01/token)
    ↓
RL Training Loop (GRPO/PPO)
```

**Key Insight:** The model doesn't need to understand *why* a symbol is valid—it just needs to learn *what* the Validator accepts.

---

## Roadmap

See `TINY_COMPRESS_ROADMAP.md` for the full 5-phase plan:
1. **T.1:** The Gym (RL Environment)
2. **T.2:** The Dataset (10k synthetic intents)
3. **T.3:** Model Architecture (GPT-2 Small or TinyLlama)
4. **T.4:** Training Loop (GRPO/PPO)
5. **T.5:** Evaluation & ROI Analysis

---

## Dependencies

- **Lane I (Semiotic Compression):** Requires `CODEBOOK.json`, decoder, validator
- **Validators:** CMP-01, SPECTRUM-02 (Crystallized Intelligence)

---

## Success Criteria

- Model converges (reward stops improving)
- Test set pass rate: >90%
- Compression ratio: >80% vs baseline (raw text)
- Clear decision: integrate, iterate, or abandon

---

## Failure Modes

- Model doesn't converge (falls back to rule-based compression)
- Compression ratio <50% (not worth the complexity)
- Training time >1 week (too expensive for experiment)

**Mitigation:** This is experimental. If it fails, we fall back to human-written Symbolic IR.

---

## References

- **Crystallized Intelligence:** `LAW/CANON/INTEGRITY.md` (CMP-01, SPECTRUM)
- **Semiotic Compression:** `INBOX/research/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md`
- **RL Training:** TRL library (Hugging Face)
