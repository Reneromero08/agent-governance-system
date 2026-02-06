# Q33: Conditional Entropy / Semantic Density Shows Emergent Properties

## Hypothesis

R exhibits emergent properties when measured across scales -- specifically, macro-scale R values are not trivially predictable from micro-scale R values, and the relationship between conditional entropy H(X), symbolic compression sigma, and fractal depth Df reveals genuine emergent semantic density.

## Verification Status: OPEN (Improperly Tested)

**Category:** C1 -- Test used circular/tautological setup. Hypothesis never properly tested.

## v1 Evidence Summary

- The v1 document explicitly states: "This definition makes sigma^Df = N a tautology by construction"
- H(X) was incorrectly equated with token count
- Df can be negative (physically meaningless for fractal dimension)
- The "derivation" is a circular definition admitted by the document itself

## What Went Wrong With the Test

The test was **tautological by the document's own admission**. sigma^Df = N was true by construction, not by discovery. The test proved properties of the definition, not properties of semantic data. H(X) = token count is not conditional entropy in any information-theoretic sense.

**Crucially:** The test being tautological does NOT mean emergence doesn't exist in semantic systems. It means nobody tested for it properly.

## What a Proper Test Looks Like

### Design Requirements (per v2 METHODOLOGY.md)

1. **Consistent E:** Use E = mean pairwise cosine similarity throughout
2. **Real data:** Use real multi-scale text data (word, sentence, paragraph, document levels)
3. **Pre-registered criteria:**
   - Compute R at each scale independently
   - Predict macro-R from micro-R values using linear/polynomial regression
   - If residual > threshold (pre-register threshold), emergence is present
   - If macro-R is fully predictable from micro-R, no emergence
4. **Baseline:** Compare to bare cosine similarity at each scale
5. **Non-circular definition:** Do NOT define sigma^Df = N. Compute sigma and Df independently, then check whether sigma^Df correlates with any meaningful quantity.

### Specific Steps

1. Select 3+ real corpora (e.g., Wikipedia articles, legal documents, scientific papers)
2. Compute R at word-pair, sentence, paragraph, and document levels
3. Build predictive model: can document-level R be predicted from sentence-level R statistics?
4. If yes: R does not show emergence at this transition
5. If no: characterize the residual -- what information is gained at macro scale?
6. Report Df values -- if negative, explain or flag as problem

### Success Criteria

- **Confirmed:** Macro-R contains statistically significant information not present in micro-R (p < 0.01, effect size > 0.3)
- **Falsified:** Macro-R is fully predictable from micro-R (R^2 > 0.95)
- **Inconclusive:** Intermediate results

### Required Data Sources

- English Wikipedia (freely available)
- Legal corpus (e.g., SCOTUS opinions from CourtListener)
- Scientific abstracts (e.g., S2ORC or PubMed)

## Salvageable from v1

- The conceptual framework of multi-scale R measurement
- The observation that sigma and Df should relate to complexity (if measured properly)
- Nothing else -- the tautological test and circular definition must be discarded entirely
