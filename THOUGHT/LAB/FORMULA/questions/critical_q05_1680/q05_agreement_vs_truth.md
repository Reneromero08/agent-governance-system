# Question 5: Agreement vs. truth (R: 1680)

**STATUS: ANSWERED**

## Question
The formula measures agreement, not "objective truth." Is this a feature (truth IS agreement) or a limitation (consensus can be wrong)?

---

## TESTS
`questions/1/q1_deep_grad_s_test.py` + `questions/2/q2_echo_chamber_deep_test.py`

---

## FINDINGS

(Combined from Q1 and Q2 tests)

### 1. Agreement IS truth when observations are INDEPENDENT:
   - Independent + low dispersion -> R predicts accuracy: YES
   - Error: 0.05 (very low)

### 2. Consensus CAN be wrong when observations are CORRELATED:
   - Echo chamber + low dispersion -> R predicts accuracy: NO
   - Error: 0.24 (5x higher despite tighter agreement)

### 3. The formula correctly distinguishes:
   - Echo chambers have SUSPICIOUSLY high R (20x normal)
   - Adding fresh data crashes echo chamber R (93% drop)

---

## ANSWER

**BOTH are true.**

- **Feature:** For independent observers, agreement = truth (by definition)
- **Limitation:** For correlated observers, consensus can be wrong
- **Defense:** The formula's extreme R values (>95th percentile) signal potential echo chambers
