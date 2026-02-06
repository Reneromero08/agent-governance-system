# N7: Can R Detect Real-World Semantic Phenomena?

## Why This Question Matters

Q16 showed R works for domain boundary detection on SNLI/ANLI. Q18 showed deception detection failed. Q32 showed misinformation detection works (but NLI model does the lifting). Where exactly is the boundary of R's practical utility? Nobody systematically mapped it.

## Hypothesis

**H0:** R discriminates some real-world semantic phenomena better than random and better than bare E, but the boundary is characterizable.

**Specific sub-hypotheses:**
- H0a: R detects misinformation (above baseline) on fact-checking benchmarks
- H0b: R detects sarcasm/irony (above baseline) on sarcasm benchmarks
- H0c: R detects stance (above baseline) on stance detection benchmarks
- H0d: R detects semantic shift over time (above baseline) on temporal corpora

**H1:** R does not outperform bare E on any practical semantic task.

## Pre-Registered Test Design

### Benchmarks (minimum 6 tasks)

| Task | Dataset | Source | Metric |
|------|---------|--------|--------|
| Fact verification | FEVER | HuggingFace | AUC (supported vs refuted) |
| Misinformation | LIAR | HuggingFace | AUC (true vs false) |
| Sarcasm | iSarcasm | HuggingFace | AUC (sarcastic vs not) |
| Stance detection | SemEval-2016 Task 6 | SemEval | F1 (favor/against/neither) |
| Paraphrase | MRPC | HuggingFace | AUC (paraphrase vs not) |
| Textual entailment | RTE | HuggingFace | AUC (entailment vs not) |

### Procedure

For each benchmark:
1. Encode texts with `all-MiniLM-L6-v2`
2. Compute E (bare cosine similarity) for each pair/group
3. Compute R_simple = E / grad_S
4. Compute R_full = (E / grad_S) * sigma^Df (if computable)
5. Evaluate each metric's ability to discriminate the task labels
6. Compare: E vs R_simple vs R_full vs random

### Success Criteria

- **R is useful:** R_simple or R_full outperforms bare E on >= 3/6 tasks (p < 0.05, Bonferroni corrected)
- **R has a niche:** R outperforms E on 1-2 specific task types (characterize which)
- **R adds nothing:** Bare E equals or beats R on >= 5/6 tasks

### Boundary Characterization

After testing, characterize:
- Which task types does R help with? (topical? logical? pragmatic?)
- Which task types does R hurt? (why? what does grad_S encode for those tasks?)
- Is there a task-property predictor for "R will outperform E here"?

### Implications

- If R is broadly useful: the formula has practical value for NLP
- If R has a niche: document the niche, explain why, and build tools for it
- If R adds nothing: the formula is theoretically interesting but practically redundant with cosine similarity

## Dependencies

- N1 results (E vs R head-to-head) provide context
- N2 results (what is grad_S) explain WHY R helps or hurts per task
- Independent of all v2 Qs

## Related

- v2/Q16 (Domain boundaries -- confirmed, the existence proof for R's utility)
- v2/Q18 (Deception -- failed, a known failure point)
- v2/Q32 (Meaning field -- partially confirmed on NLI tasks)
- N1 (E vs R -- the fundamental comparison)
