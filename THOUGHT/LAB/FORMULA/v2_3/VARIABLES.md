# VARIABLES.md - Operationalization Registry (v2_3)

## Purpose

This registry freezes the operationalizations of the five formula variables

    R = (E / nabla_S) * sigma(f)^Df

as they were actually used across the v2_2 question suite. Rules:

1. ONE canonical operationalization per variable per substrate. Where v2_2
   used several competing definitions on the same substrate, each gets its
   own ID; future verdicts must cite exactly one ID.
2. APPEND-ONLY. Entries are never edited or deleted. A new operationalization
   means a NEW entry with a new ID - never a silent redefinition of an
   existing ID.
3. VERDICTS CITE IDs. Any v2_3 verdict that reports R, E, nabla_S, sigma, or
   Df must name the registry ID used for each quantity.
4. This registry documents the lab as it IS. Nothing from v2_2 is
   retroactively unified, renamed, or dropped. Freezing means future work
   must pick an ID, not that history gets cleaned.

ID scheme: <VAR>-<SUBSTRATE>-<NN>

Substrates:
- QEC      : quantum error correction / surface-code experiments
- EMB      : embedding / transformer / Gram-matrix experiments
            (sentence-transformers, Native Eigen, hidden states)
- FEISTEL  : Feistel / XOR catalytic fabric experiments
- KURAMOTO : Kuramoto oscillator mappings
- ATTN     : attention-head architecture quantities
- GEN      : substrate-independent derived quantities

Date added is `undefined` for all v2_2-era entries (dates were not recorded
at time of first use; the column exists for entries appended from v2_3 on).

## WARNING: q57 symbol collision

In q57_mera_holography (and inside q06_iit's Feistel helpers), the symbol
`R` denotes NUMBER OF FEISTEL ROUNDS (e.g. `R = int(math.log2(N))` in
THOUGHT/LAB/FORMULA/v2_2/q06_iit/verify_q6.py line 19, and the `R`,
`R_SMALL`, `R_FULL` round-count parameters in
THOUGHT/LAB/FORMULA/v2_2/q57_mera_holography/test_mera_rt.py). It is NOT
resonance. When citing q57 or q06 internals, the round count maps to the
registry's DF-FEISTEL-01, never to any R-* entry.

## Spot verification

The following entries were verified against the cited v2_2 sources by
opening the files (exact paths, repo-relative):

- SIGMA-FEISTEL-01: THOUGHT/LAB/FORMULA/v2_2/q25_sigma/verify_q25.py
  line 67: `sigma_theory = 2.0 ** (-h)` with h = popcount(mask).
- SIGMA-EMB-01, R-EMB-01, NABLA_S-EMB-01, SIGMA-GEN-01:
  THOUGHT/LAB/FORMULA/v2_2/q28_attractors/verify_q28_push.py lines 22-34:
  Hermitian Gram matrix of phase-normalized complex weights, eigenvalues
  normalized; `sigma = 1.0/max(ev.sum()**2/(ev**2).sum(), 1e-10)` (IPR),
  `nabla = -sum(ev*log(ev))`, `phase_coh = 1 - nabla/log(n)`,
  `c_sem = sqrt(sigma/nabla_S)`.
- R-EMB-01: THOUGHT/LAB/FORMULA/v2_2/q31_compass/verify_q31.py
  lines 31-38: same pc = 1 - H/ln(n) on Gram eigenvalues.
- R-EMB-02, R-EMB-03: THOUGHT/LAB/FORMULA/v2_2/q07_multiscale/verify_q7_eigen.py
  lines 83-89 (circular order parameter) and lines 165-169
  (`pc_seq = 1.0 - H_seq` with H_seq = output entropy / log V).
- E-QEC-01: THOUGHT/LAB/FORMULA/v2_2/q01_grad_s/verify_q1.py
  lines 6, 27, 36: calibrated E = 0.0169 constant across (p, d).
- NABLA_S-QEC-01, SIGMA-QEC-01, R-QEC-01:
  THOUGHT/LAB/FORMULA/v2_2/q15_bayesian/VERDICT.md (no script survives in
  the q15 dir; the VERDICT table maps nabla_S -> sqrt(syndrome density),
  sigma -> fidelity factor from training slopes, R -> ~1/P_L).
- NABLA_S-QEC-02, R-QEC-02: THOUGHT/LAB/FORMULA/v2_2/q42_bell/verify_q42.py
  lines 11-12, 59-83: nabla_S = p; CHSH protection vs Df.
- E-EMB-01: THOUGHT/LAB/FORMULA/v2_2/q44_born_rule/verify_q44.py
  lines 38-41: E = mean(max(overlap,0)); P_born = mean(overlap^2).
- E-FEISTEL-01: THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_fabric.py
  lines 8-14, 57-58: E = initial peak-to-peak amplitude = 2.0.
- NABLA_S-EMB-01, NABLA_S-EMB-03:
  THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_entropy_mass.py
  lines 60-80: nabla_S = von Neumann entropy of evidence density matrix;
  density = mean pairwise cosine; sem_mass = nabla_S * density.
- R-GEN-01: THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_gap2.py
  line 77: `M = math.log(max(R, 1e-6))`.
- DF-QEC-01: THOUGHT/LAB/FORMULA/v2_2/q40_error_correction/harden_q40_stealth.py
  lines 10, 196, 216: D_f = t (t = 1 for 3-qubit code).
- DF-EMB-01: THOUGHT/LAB/FORMULA/v2_2/q50_completing_8e/verify_q50.py
  line 16: `Df = evnz.sum()**2 / (evnz**2).sum()` (participation ratio);
  same in q49_why_8e/verify_q49.py (compute_participation_ratio).
- DF-EMB-02: THOUGHT/LAB/FORMULA/v2_2/q36_bohm/verify_q36_bohm.py
  (participation_ratio, lines 102-116) and q36_bohm/VERDICT.md ("Df ratio
  (cpx/real)" 1.6x-1.9x).
- DF-ATTN-01, SIGMA-KURAMOTO-01, NABLA_S-KURAMOTO-01:
  THOUGHT/LAB/FORMULA/v2_2/q55_kuramoto_heads/verify_q55_heads.py lines 3-4
  ("K_c = nabla_S/sigma predicts minimum h"; independent heads = genuine
  D_f) and q55_kuramoto_heads/VERDICT.md (noise sweep section).
- DF-FEISTEL-01: THOUGHT/LAB/FORMULA/v2_2/q06_iit/verify_q6.py line 19
  (`R = int(math.log2(N))` rounds) and
  q57_mera_holography/test_mera_rt.py (R round-count parameter).
- R-FEISTEL-01: THOUGHT/LAB/FORMULA/v2_2/q06_iit/VERDICT.md ("Computed R as
  channel capacity I(Input;Output) in bits").
- R-EMB-04, NABLA_S-EMB-04: THOUGHT/LAB/FORMULA/v2_2/q17_governance/verify_q17.py
  lines 40-63: `R = E / max(nabla_S, 1e-8)` with nabla_S = std of
  per-sample prediction entropies.
- NABLA_S-EMB-02, SIGMA-EMB-02:
  THOUGHT/LAB/FORMULA/v2_2/q38_noether/tests/test_geodesic_proof.py
  lines 137-147: nabla_S = 2D Gaussian spread entropy (clamped),
  `sigma = 1.0 / max(1 - cos_sim, 0.01)`.
- R-GEN-02: THOUGHT/LAB/FORMULA/v2_2/q54_energy_conservation/VERDICT.md
  (conserved quantity = tape SHA-256; R as conserved U(1) charge).

Refinements found during verification (recorded, not unified):
- The "IPR sigma" attributed to q28 is also used verbatim in
  q12_phase_transitions/train_q12.py line 118, q17_governance/verify_q17.py
  line 58, and several q32 scripts (verify_q32_integrity.py line 21,
  verify_q32_gap2.py line 36, verify_q32_cv.py line 32). One ID covers all
  (SIGMA-EMB-01).
- q07's eigen script does NOT use the Gram-eigenvalue pc at token/attention
  scale; it uses a circular (Kuramoto-style) order parameter. That variant
  gets its own ID (R-EMB-03) rather than being silently folded into R-EMB-01.
- q17's nabla_S (std of prediction entropies) was not in the audit list but
  exists in the script; it is registered as NABLA_S-EMB-04 rather than
  dropped.

---

## E (signal / energy term)

| ID | substrate | definition (formula, plain ASCII) | first used by | source script (path) | date added |
|----|-----------|-----------------------------------|---------------|----------------------|------------|
| E-QEC-01 | QEC | Calibrated signal power, constant E = 0.0169 fitted from QEC training; held fixed across all (p, d) conditions | q01, q15, q40 | THOUGHT/LAB/FORMULA/v2_2/q01_grad_s/verify_q1.py | undefined |
| E-EMB-01 | EMB | Mean cosine overlap: E = mean(max(cos(psi, phi_i), 0)) over context set; Born companion P_born = mean(overlap^2) | q44 | THOUGHT/LAB/FORMULA/v2_2/q44_born_rule/verify_q44.py | undefined |
| E-FEISTEL-01 | FEISTEL | Fabric signal amplitude: E = initial peak-to-peak amplitude of injected sine wave (= 2.0); decay tested as amp ~ E * sigma^Df / nabla_S | q32 (fabric) | THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_fabric.py | undefined |
| E-FEISTEL-02 | FEISTEL | Channel input term: E = 1.0 (unit input) in R_formula = (E/nabla_S) * sigma^d on the N=4 Feistel TPM | q06 | THOUGHT/LAB/FORMULA/v2_2/q06_iit/verify_q6.py | undefined |

## NABLA_S (entropy gradient / disorder term)

| ID | substrate | definition (formula, plain ASCII) | first used by | source script (path) | date added |
|----|-----------|-----------------------------------|---------------|----------------------|------------|
| NABLA_S-EMB-01 | EMB | Von Neumann entropy of normalized Gram/density-matrix eigenvalues: ev = eigvalsh(H), ev >= 0, ev /= sum(ev); nabla_S = -sum(ev * ln(ev)) | q32, q28, q34 | THOUGHT/LAB/FORMULA/v2_2/q28_attractors/verify_q28_push.py; THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_entropy_mass.py | undefined |
| NABLA_S-EMB-02 | EMB | Semantic tension truth-vs-lie: 2D Gaussian spread entropy of the relation neighborhood (clamped at -10); low for true pairs, high for false pairs | q38 | THOUGHT/LAB/FORMULA/v2_2/q38_noether/tests/test_geodesic_proof.py | undefined |
| NABLA_S-EMB-03 | EMB | Composite semiotic mass: sem_mass = nabla_S * density, where density = mean pairwise cosine similarity of evidence set (uses NABLA_S-EMB-01 as input) | q32 | THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_entropy_mass.py | undefined |
| NABLA_S-EMB-04 | EMB | Std of per-sample prediction entropies: nabla_S = std_i(H(p_i)) + 1e-8 over model output distributions (added during spot-verification; was implicit in audit's q17 R entry) | q17 | THOUGHT/LAB/FORMULA/v2_2/q17_governance/verify_q17.py | undefined |
| NABLA_S-QEC-01 | QEC | Square root of syndrome density: nabla_S = sqrt(syndrome density) (evidence term P(D) in the Bayesian reading) | q15, q01 | THOUGHT/LAB/FORMULA/v2_2/q15_bayesian/VERDICT.md | undefined |
| NABLA_S-QEC-02 | QEC | Physical error rate: nabla_S = p (depolarizing noise strength) | q42 | THOUGHT/LAB/FORMULA/v2_2/q42_bell/verify_q42.py | undefined |
| NABLA_S-KURAMOTO-01 | KURAMOTO | Kuramoto noise width: task/label noise level entering the synchronization threshold K_c = nabla_S / sigma; more noise -> higher critical head count | q55, q56, q12 | THOUGHT/LAB/FORMULA/v2_2/q55_kuramoto_heads/verify_q55_heads.py; THOUGHT/LAB/FORMULA/v2_2/q55_kuramoto_heads/VERDICT.md | undefined |
| NABLA_S-FEISTEL-01 | FEISTEL | Fabric spectral entropy: spectral entropy of the signal on the XOR fabric (pure sine ~ 0, noise ~ max); governs survival rate | q32 (fabric) | THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/FABRIC_UPDATE.md | undefined |

## SIGMA (per-step survival / coupling factor)

| ID | substrate | definition (formula, plain ASCII) | first used by | source script (path) | date added |
|----|-----------|-----------------------------------|---------------|----------------------|------------|
| SIGMA-FEISTEL-01 | FEISTEL | Hash entropy per Feistel round: sigma_theory = 2^(-h), h = popcount(mask) = number of hash bits that must be zero per byte | q25 | THOUGHT/LAB/FORMULA/v2_2/q25_sigma/verify_q25.py | undefined |
| SIGMA-QEC-01 | QEC | Fidelity factor from training slopes: likelihood term P(D given H) fitted from QEC training-curve slopes | q15, q01 | THOUGHT/LAB/FORMULA/v2_2/q15_bayesian/VERDICT.md | undefined |
| SIGMA-EMB-01 | EMB | Inverse participation ratio of normalized Gram/density eigenvalues: sigma = 1 / max((sum ev)^2 / sum(ev^2), 1e-10) = sum(ev^2) for sum(ev)=1 | q28 (also q12, q17, q32 scripts) | THOUGHT/LAB/FORMULA/v2_2/q28_attractors/verify_q28_push.py; THOUGHT/LAB/FORMULA/v2_2/q12_phase_transitions/train_q12.py | undefined |
| SIGMA-EMB-02 | EMB | Statement compression truth-vs-lie: sigma = 1 / max(1 - cos_sim, 0.01) (inverse of semantic distance; truth compresses, lies stretch) | q38 | THOUGHT/LAB/FORMULA/v2_2/q38_noether/tests/test_geodesic_proof.py | undefined |
| SIGMA-KURAMOTO-01 | KURAMOTO | Kuramoto coupling K: per-head coupling strength in the synchronization threshold K_c = nabla_S / sigma | q12, q55, q56 | THOUGHT/LAB/FORMULA/v2_2/q55_kuramoto_heads/verify_q55_heads.py | undefined |
| SIGMA-GEN-01 | GEN | Derived semantic wave speed: c_sem = sqrt(sigma / nabla_S) (derived quantity, not a sigma operationalization itself; registered here because v2_2 reported it alongside sigma) | q32, q28 | THOUGHT/LAB/FORMULA/v2_2/q28_attractors/verify_q28_push.py; THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/VERDICT.md | undefined |

## DF (fractal dimension / redundancy exponent)

| ID | substrate | definition (formula, plain ASCII) | first used by | source script (path) | date added |
|----|-----------|-----------------------------------|---------------|----------------------|------------|
| DF-FEISTEL-01 | FEISTEL | Number of Feistel rounds: Df = rounds, with full-mix round count R = log2(N) for an N-cell tape. NOTE: the local symbol R in q57/q06 code means THIS round count, not resonance (see preamble warning) | q25, q57, q06 | THOUGHT/LAB/FORMULA/v2_2/q06_iit/verify_q6.py; THOUGHT/LAB/FORMULA/v2_2/q57_mera_holography/test_mera_rt.py | undefined |
| DF-QEC-01 | QEC | Correctable error count: Df = t = floor((d-1)/2) for code distance d (t = 1 for the 3-qubit code) | q15, q36, q40, q42 | THOUGHT/LAB/FORMULA/v2_2/q40_error_correction/harden_q40_stealth.py | undefined |
| DF-ATTN-01 | ATTN | Independent attention head count: Df = h for independent-weight heads; shared-weight heads collapse to Df = 1 (pseudo-redundancy) | q55, q56 | THOUGHT/LAB/FORMULA/v2_2/q55_kuramoto_heads/verify_q55_heads.py | undefined |
| DF-EMB-01 | EMB | Participation ratio / Quantum Darwinism redundancy count: Df = (sum ev)^2 / sum(ev^2) over nonzero covariance/density eigenvalues | q49, q50 | THOUGHT/LAB/FORMULA/v2_2/q50_completing_8e/verify_q50.py; THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verify_q49.py | undefined |
| DF-EMB-02 | EMB | Hilbert enfolding dimensionality ratio: Df_ratio = PR(complexified embedding) / PR(real embedding), PR as in DF-EMB-01; ~1.6x-1.9x observed | q36, q51 | THOUGHT/LAB/FORMULA/v2_2/q36_bohm/verify_q36_bohm.py; THOUGHT/LAB/FORMULA/v2_2/q36_bohm/VERDICT.md | undefined |

## R (resonance / output term)

| ID | substrate | definition (formula, plain ASCII) | first used by | source script (path) | date added |
|----|-----------|-----------------------------------|---------------|----------------------|------------|
| R-EMB-01 | EMB | Phase coherence on Gram eigenvalues: build complex Hermitian Gram matrix H of phase-normalized vectors, ev = eigvalsh(H) normalized; pc = 1 - (-sum(ev*ln(ev))) / ln(n) | q07, q10, q17, q21, q28, q31, q32, q33, q55, q56 | THOUGHT/LAB/FORMULA/v2_2/q28_attractors/verify_q28_push.py; THOUGHT/LAB/FORMULA/v2_2/q31_compass/verify_q31.py | undefined |
| R-EMB-02 | EMB | Sequence-level output coherence: pc_seq = 1 - H_out / ln(V), where H_out = entropy of mean softmax output distribution over vocab V | q07 | THOUGHT/LAB/FORMULA/v2_2/q07_multiscale/verify_q7_eigen.py | undefined |
| R-EMB-03 | EMB | Circular (Kuramoto-style) order parameter on phases: pc = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2) (q07 token/attention scales; added during spot-verification, previously folded into the pc family) | q07 | THOUGHT/LAB/FORMULA/v2_2/q07_multiscale/verify_q7_eigen.py | undefined |
| R-EMB-04 | EMB | Simple ratio: R = E / max(nabla_S, 1e-8) (no sigma^Df factor), nabla_S per NABLA_S-EMB-04 | q17 | THOUGHT/LAB/FORMULA/v2_2/q17_governance/verify_q17.py | undefined |
| R-QEC-01 | QEC | Bayesian posterior odds of logical survival: R ~ exp(log_suppression)/p ~ 1/P_L; log_R_empirical = log(1/P_L) | q15 | THOUGHT/LAB/FORMULA/v2_2/q15_bayesian/VERDICT.md | undefined |
| R-QEC-02 | QEC | CHSH protection level: persistence of Bell violation (CHSH > 2) under depolarizing noise; protected pair has effective error p^(Df+1), R quantifies nines of protection | q42 | THOUGHT/LAB/FORMULA/v2_2/q42_bell/verify_q42.py | undefined |
| R-FEISTEL-01 | FEISTEL | Channel capacity in bits: R = I(Input;Output) over the Feistel-fabric transition probability matrix | q06 | THOUGHT/LAB/FORMULA/v2_2/q06_iit/verify_q6.py; THOUGHT/LAB/FORMULA/v2_2/q06_iit/VERDICT.md | undefined |
| R-GEN-01 | GEN | Meaning field (log transform): M = log(max(R, 1e-6)); derived from any R operationalization, stabilizes multiplicative terms | q32, q12 | THOUGHT/LAB/FORMULA/v2_2/q32_meaning_field/verify_q32_gap2.py | undefined |
| R-GEN-02 | GEN | Conserved U(1) Noether charge: R as the invariant of the catalytic cycle, witnessed by tape SHA-256 hash invariance (forward + reverse = identity, zero net entropy change) | q54 | THOUGHT/LAB/FORMULA/v2_2/q54_energy_conservation/VERDICT.md | undefined |
