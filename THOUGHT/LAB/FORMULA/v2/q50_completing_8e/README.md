# Q50: Df * alpha Is Conserved Across Architectures

## Hypothesis
The product Df * alpha is a universal conservation law holding across all trained embedding architectures, modalities (text, vision, audio, code), and languages. This conservation law is derivable from first principles, with 8 = 2^3 arising from Peirce's three irreducible semiotic categories and e arising as the natural information unit per octant. Human alignment (instruction tuning) systematically compresses this conserved quantity below its natural value. The Riemann zeta function's critical line at Re(s) = 1/2 is mathematically identical to the semantic spectral exponent alpha = 1/2.

## v1 Evidence Summary
- 24 models tested: mean Df * alpha = 21.57, CV = 6.93%.
- Mean error vs. 8e: 0.82%, range: 0.03% to 23.15%.
- Cross-modal: text models ~22, CLIP ~23.5 (7.9% error).
- Alignment distortion: 6/6 instruction-tuned models showed compression (6.8% to 34.2%), mean 27.7%.
- Riemann connection tests: functional equation FAILED (CV = 353%), zero spacing FAILED (33.5, not Riemann-like), special points FAILED (no relationship). Conclusion: "analogous, not identical."
- e-per-octant: Df * alpha / 8 = 2.72 (0.15% from e). Entropy test H/e = 0.70 (FAILED to confirm H = 1 nat per octant).
- Peirce's Reduction Thesis invoked for "why 3 dimensions."

## v1 Methodology Problems
1. **7% CV is not a conservation law.** Physical conservation laws hold to parts per billion. Zipf's law (~10-20% CV) is not called a conservation law. Calling a 7% statistical regularity a "conservation law" is a category error.
2. **Non-independent model samples.** Of 24 models, ~19 are transformer encoders trained on overlapping English web text with shared architectures. Effective independent observations: ~3-5.
3. **"Code models" were text models run on code snippets.** MiniLM-L6 on code input was labeled "code modality." These are not code-specific models.
4. **"Vision models" tested on text descriptions.** CLIP was tested on text strings like "a photo of a cat," not on actual images. The vision claim is unfounded.
5. **e-per-octant is tautological.** If Df * alpha = 8e, then Df * alpha / 8 = e. This is arithmetic, not an explanation. The report acknowledges this but presents it as a finding anyway.
6. **Entropy per octant test FAILED.** H/e = 0.70, not 1.0. The information-theoretic explanation for why e appears was empirically falsified but not counted as a failure.
7. **Riemann connection claims self-contradictory.** The question document says "analogous, not identical." The report says "This is not analogy. This is identity." Both are in the same project.
8. **Chern number derivation is circular.** It assumes embeddings live on CP^(d-1), then derives alpha = 1/2 from the Chern class of CP^n. This is the assumption, not a derivation. Real embeddings live on S^(d-1) (unit sphere), not CP^(d-1).
9. **Unfalsifiable framing.** Models matching 8e confirm the law. Models deviating confirm "alignment distortion." The law cannot be wrong.
10. **No pre-registered model list.** Without pre-registration, the 24-model sample may be the result of selection bias.

## v2 Test Plan

### Test 1: True Cross-Modal Universality
- Test Df * alpha on genuine non-text modalities:
  - **Vision:** Extract features from ResNet/ViT on ImageNet validation images (actual images, not text descriptions).
  - **Audio:** Extract features from wav2vec2 or Whisper on LibriSpeech audio segments.
  - **Code:** Extract features from CodeBERT or StarCoder on code from The Stack.
  - **Protein:** Extract features from ESM-2 on UniProt protein sequences.
  - **Chemistry:** Extract features from MolBERT on ZINC molecular representations.
- Each modality: 3+ models, 500+ samples, 3+ random subsets.
- Report per-modality Df * alpha with 95% confidence intervals.

### Test 2: Non-English Language Independence
- Test on independently trained (not multilingual or translated) monolingual models:
  - Chinese: Chinese word2vec trained on Chinese Wikipedia/Baidu corpus
  - Japanese: Japanese word2vec or BERT trained on Japanese corpus
  - Arabic: Arabic FastText or AraBERT
  - Hindi: Hindi embedding model
  - Each language on native vocabulary (not translated word lists).
- Compare Df * alpha to the English baseline.

### Test 3: Alignment Distortion -- Causal Test
- For the same base model (e.g., LLaMA-7B), compare:
  - Pre-trained base
  - RLHF-tuned version (LLaMA-chat)
  - DPO-tuned version
  - SFT-only version
- Track how Df * alpha changes at each alignment stage.
- Control for input formatting: test all versions with identical input strings (no instruction prefixes).
- Determine whether the compression is from alignment training or from input formatting conventions.

### Test 4: Independence from Vocabulary Size and Selection
- For 5 models, measure Df * alpha at N = 50, 100, 200, 500, 1000, 5000.
- Use frequency-ranked words, random words, domain-specific words, and mixed selections.
- If Df * alpha varies > 10% with vocabulary choice, it is a measurement artifact, not a constant.

### Test 5: Falsifiable Riemann Connection
- Pre-register: "The Riemann connection is falsified if zeta_sem has no functional equation, no zeros in the critical strip, and no Euler product."
- Test each property rigorously with proper numerical methods.
- If all three fail (as they did in v1), declare the Riemann connection falsified and remove it from the narrative.

### Test 6: Pre-Registered Constant Discrimination
- Pre-register three hypotheses: Df * alpha = 8e, Df * alpha = 7*pi, Df * alpha = 22.
- Collect 200+ (model, modality, vocabulary, language) measurements.
- Apply Bayesian model comparison with proper priors.
- Report honest result: which constant (if any) is preferred, or "approximately 22" if none is distinguished.

## Required Data
- **ImageNet** validation set (50,000 images) with ResNet-50 and ViT-B/16 features
- **LibriSpeech** (1000 hours) with wav2vec2 and Whisper features
- **The Stack** code dataset with CodeBERT and StarCoder features
- **UniProt** protein sequences with ESM-2 features
- **ZINC** molecular dataset with MolBERT features
- **Monolingual models:** Chinese word2vec, Japanese BERT, AraBERT, Hindi embeddings (independently trained)
- **LLaMA family:** base, chat, DPO variants for alignment comparison
- Standard English embedding models (GloVe, Word2Vec, BERT, Sentence-BERT) as baseline

## Pre-Registered Criteria
- **Success (universality):** Df * alpha CV < 10% across all modalities (text, vision, audio, code, protein) AND all languages. No modality or language systematically deviates.
- **Failure (universality):** Any modality or language has Df * alpha > 2 standard deviations from the cross-modality mean, OR overall CV > 20%.
- **Success (alignment distortion is real):** Df * alpha decreases monotonically through alignment stages (base -> SFT -> RLHF) with effect size > 10%, AND the effect persists when controlling for input formatting.
- **Failure (alignment distortion):** Effect disappears when controlling for input formatting (< 3% difference with identical inputs).
- **Success (Riemann connection):** At least one Riemann-like property confirmed (functional equation, zeros on a line, or Euler product) with p < 0.01.
- **Failure (Riemann connection):** All three properties fail. Declare falsified.
- **Inconclusive:** Mixed results across modalities, or ambiguous alignment effects.

## Baseline Comparisons
- **Per-modality random matrices** (structure-preserving randomization matched to each data type).
- **Untrained model features** (random initialization) as the "no learning" baseline.
- **Within-family vs. between-family CV** to distinguish "shared architecture" from "universal law."
- **Input-formatting control** for alignment distortion (same model, same text, with and without instruction prefix).

## Salvageable from v1
- test_q50_cross_modal.py has model loading infrastructure for multiple architectures, though the "code" and "vision" tests need to be replaced with actual cross-modal data.
- test_q50_alignment_distortion.py has a well-designed comparison framework (same model, different input formatting). The framework is good; the inputs need to be controlled better.
- The 8E_VS_7PI_COMPARISON.md analysis methodology (multi-metric, multi-dataset comparison) is the gold standard and should be expanded.
- The observation that instruction-tuning changes Df * alpha is interesting regardless of whether 8e is the "natural" value.
- The HONEST_FINAL_STATUS and SPECIFICATION correctly marking claims as OPEN/empirical should set the tone for v2.
