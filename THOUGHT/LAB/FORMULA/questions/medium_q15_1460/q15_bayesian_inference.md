# Question 15: Bayesian inference (R: 1460)

**STATUS: ✅ ANSWERED**

## Question
R seems to measure "when to trust local evidence." Is there a formal connection to posterior concentration or evidence accumulation?

## Answer

**YES AND NO (RESOLVED):** R is connected to **Likelihood Precision** (intensive), but NOT **Posterior Concentration** (extensive).

### The Distinction
- **Posterior Precision** ($\tau_{post} \approx n/\sigma^2$) grows with sample size $n$. You can achieve arbitrary certainty by averaging infinite noisy data.
- **R** ($R \propto 1/\sigma$) is **independent of sample size**. It measures the **quality** of the signal source itself.

### Key Finding
R correlates perfectly ($r=1.0$) with $\sqrt{\text{Likelihood Precision}}$. It is the **Evidence Density**.

**Why this matters:**
R works as a **Gate**, not a **Confidence interval**.
- A standard Bayesian agent becomes confident in a noisy channel given enough time ($n \to \infty$).
- An R-gated agent **ignores** the noisy channel forever ($R < \text{threshold}$), regardless of how much data accumulates.
- This prevents "false confidence via volume" – trusting a bad source just because you've heard it a lot.

### Test Results (2026-01-08)
Test: `questions/medium_q15_1460/tests\q15_proper_bayesian_test.py`

| Metric | Correlation with R | Meaning |
|--------|-------------------|---------|
| **Sqrt(Likelihood Precision)** | **1.0000** | R measures signal quality |
| **Posterior Precision** | -0.0937 | R ignores data volume |

### Conclusion
R is an **intensive** thermodynamic property (like temperature), while Bayesian posterior confidence is an **extensive** property (like heat). R measures the **temperature of the truth**.

Report: `questions/medium_q15_1460/tests\Q15_PROPER_TEST_RESULTS.md`
