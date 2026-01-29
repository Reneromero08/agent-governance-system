# Question 2: Falsification criteria (R: 1750)

**STATUS: ANSWERED**

## Question
Under what conditions would we say the formula is wrong, not just "needs more context"?

---

## TESTS
`questions/2/`
- `q2_falsification_test.py` - attack attempts
- `q2_echo_chamber_deep_test.py` - echo chamber analysis

---

## FINDINGS

### 1. Echo chambers DO fool local R:

| Condition | Mean R | Mean Error | R predicts? |
|-----------|--------|------------|-------------|
| Independent | 0.15 | 0.26 | YES |
| Echo chamber | 3.10 | 2.44 | NO |

### 2. Detection: Suspiciously high R is a signal:
   - R > 95th percentile: 0% independent, 10% echo chambers
   - Echo chambers have 20x higher R than independent!

### 3. Defense: Fresh data breaks echo chambers:

| External obs added | Echo R drops to |
|--------------------|-----------------|
| 0 | 2.47 |
| 1 | 0.18 |
| 5 | 0.11 |
| 20 | 0.05 |

### 4. Bootstrap test works: 
Echo R drops 93% vs real drops 75% when fresh data added.

---

## ANSWER

Formula CAN be fooled by correlated observations (echo chambers).

**Falsification criteria:**
- Formula is CORRECT: It measures local agreement, which is what it claims
- Formula FAILS when: Observations are correlated (independence violated)
- Defense: Add fresh independent data; if R crashes, it was echo chamber

**Known limitation:** R assumes independence. This is not a bug - it's the epistemological boundary.
