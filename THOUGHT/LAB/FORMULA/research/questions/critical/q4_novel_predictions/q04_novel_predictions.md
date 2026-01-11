# Question 4: Novel predictions (R: 1700)

**STATUS: PARTIALLY ANSWERED**

## Question
What does the formula predict that we don't already know? Can we design an experiment where the formula makes a surprising, testable claim?

---

## TESTS
`open_questions/q4/`
- `q4_novel_predictions_test.py` - prediction validation

---

## FINDINGS

Predictions are **partially confirmed** (mixed strength):

| Prediction | Result | Numbers |
|------------|--------|---------|
| Low R predicts need for more context | WEAK SUPPORT | r = -0.11 |
| High R = faster convergence | CONFIRMED | 5.0 vs 12.4 samples |
| Threshold transfers across domains | CONFIRMED | Works on unseen distribution |
| R-gating improves decisions | CONFIRMED | 83.8% -> 97.2% accuracy |

---

## ANSWER

Yes, novel testable predictions exist, and several are validated — but not all are strong yet:

1. **Context prediction:** Initial R predicts samples needed to stabilize
2. **Convergence rate:** High R observations converge 2.5x faster
3. **Transfer:** R thresholds generalize to new domains
4. **Gating utility:** Abstaining when R is low improves accuracy by 16%
