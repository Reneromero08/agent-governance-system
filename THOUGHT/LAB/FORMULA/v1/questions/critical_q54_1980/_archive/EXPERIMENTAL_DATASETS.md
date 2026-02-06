# Experimental Datasets for Testing Q54 R_mi Predictions

## Overview

This document catalogs **real experimental datasets** on Quantum Darwinism and decoherence that can be used to validate the Q54 hypothesis prediction that **R_mi (mutual information-based R metric) increases ~2x during decoherence**.

The goal is to move from "internally consistent simulations" to **externally validated science**.

---

## Category 1: Direct Quantum Darwinism Experiments

These experiments directly measure mutual information, redundancy, and the quantum-to-classical transition.

### 1.1 Superconducting Circuit Experiment (2025) - HIGHEST PRIORITY

**Citation:**
Zhu, Z., Salice, K., Touil, A., Bao, Z.-H., Song, Z., Zhang, P., Li, H.-P., Wang, Z., Song, C., Guo, Q., Wang, H., & Mondaini, R. (2025). Observation of quantum Darwinism and the origin of classicality with superconducting circuits. *Science Advances*, 11, eadx6857.

**DOI:** 10.1126/sciadv.adx6857

**What was measured:**
- Quantum mutual information I(S:F) between system and environment fragments
- Saturation/plateau of mutual information (classical plateau)
- Branching quantum states supporting classicality
- Discord measurements

**Raw data availability:**
- **YES - Data available at Zenodo:** https://doi.org/10.5281/zenodo.15702784
- Supplementary materials included with paper

**Platform:** Superconducting quantum circuits (transmon qubits)

**Relevance to Q54:**
- Direct measurements of mutual information during decoherence
- Can extract R_mi values from the plateau behavior
- System-environment entanglement dynamics measured

**How to use for R_mi validation:**
1. Extract I(S:F) curves from Zenodo dataset
2. Calculate R_mi at different stages of decoherence
3. Compare ratio of R_mi values to Q54 prediction of ~2x increase

---

### 1.2 NV Center in Diamond Experiment (2019)

**Citation:**
Unden, T. K., Louzon, D., Zwolak, M., Zurek, W. H., & Jelezko, F. (2019). Revealing the Emergence of Classicality Using Nitrogen-Vacancy Centers. *Physical Review Letters*, 123, 140402.

**DOI:** 10.1103/PhysRevLett.123.140402
**arXiv:** 1809.10456

**What was measured:**
- NV spin system interacting with ~4 carbon-13 nuclear spins (environment)
- Information transfer to environment fragments
- Dynamical decoupling sequences to control decoherence
- Fluorescence measurements revealing environment state

**Raw data availability:**
- Data may need to be extracted from figures
- Contact authors (Jelezko group at Ulm University) for raw data
- Supplementary materials may contain additional data

**Platform:** Nitrogen-vacancy centers in diamond at room temperature

**Relevance to Q54:**
- First solid-state demonstration of quantum Darwinism
- Natural decoherence from carbon-13 spin bath
- Controlled environment size (~4 spins)

**How to use for R_mi validation:**
1. Extract information content vs. environment fragment size
2. Calculate mutual information from reported measurements
3. Track R_mi evolution under different decoupling sequences

---

### 1.3 Photonic Cluster State Experiment (2018)

**Citation:**
Ciampini, M. A., Pinna, G., Mataloni, P., & Paternostro, M. (2018). Experimental signature of quantum Darwinism in photonic cluster states. *Physical Review A*, 98, 020101(R).

**DOI:** 10.1103/PhysRevA.98.020101

**What was measured:**
- Mutual information I(S:F) between system and environment fractions
- Star-shaped and diamond-shaped graph states
- 1, 2, or 3 qubit environment fragments
- Noise parameter alpha variation

**Raw data availability:**
- Data likely in supplementary materials
- May need to extract from figures
- Hyperentangled photon source provides high-fidelity states

**Platform:** Photonic hyperentangled states (polarization + path encoding)

**Relevance to Q54:**
- Well-characterized mutual information measurements
- Different graph state topologies allow comparison
- Engineered open-system dynamics

---

### 1.4 Six-Photon Quantum Simulator (2019)

**Citation:**
Chen, M.-C., Zhong, H.-S., Li, Y., et al. (2019). Emergence of classical objectivity of quantum Darwinism in a photonic quantum simulator. *Science Bulletin*, 64(9), 580-585.

**DOI:** 10.1016/j.scib.2019.04.002
**arXiv:** 1808.07388

**What was measured:**
- Quantum mutual information
- Holevo bound (classical correlation)
- Quantum discord (pure quantum correlation)
- 1 system photon + 5 environment photons

**Raw data availability:**
- Contact University of Science and Technology of China
- Funded by National Natural Science Foundation of China

**Platform:** Six-photon quantum simulator

**Relevance to Q54:**
- Direct measurement of information correlations
- Separation of classical vs. quantum correlations
- Large enough environment to see redundancy

---

### 1.5 IBM Quantum Computer Demonstration (2022)

**Citation:**
(Authors). (2022). Demonstration of quantum Darwinism on quantum computer. *Quantum Information Processing*.

**DOI:** 10.1007/s11128-022-03471-3

**What was measured:**
- n-qubit circuits (n = 2, 3, 4, 5, 6)
- Quantum state tomography
- Mutual information between system and environment
- Quantum-classical correlations

**Raw data availability:**
- Implemented on ibmq_athens and ibmq_16_melbourne (now retired)
- Quantum error mitigation applied
- Results showed expected behavior on simulator, deviations on real hardware

**Platform:** IBM superconducting quantum computers

**Relevance to Q54:**
- Shows difference between ideal (simulator) and noisy (real hardware)
- This gap is directly relevant to R_mi predictions
- Noise effects may reveal R_mi dynamics

**Limitation:** IBM systems now retired; would need to replicate on current IBM Quantum hardware

---

## Category 2: Decoherence and Entanglement Dynamics

These experiments provide complementary data on decoherence processes.

### 2.1 Four-Ion Entanglement Under Decoherence

**Citation:**
(Nature Physics authors). Experimental multiparticle entanglement dynamics induced by decoherence. *Nature Physics*.

**What was measured:**
- Four-trapped-ion entangled states
- Distillability and separability dynamics
- Bell-inequality violation crossing
- Bound entanglement regime

**Raw data availability:**
- Nature Physics supplementary materials
- NIST trapped ion group archives

**Relevance to Q54:**
- Rich entanglement dynamics during decoherence
- Multiple correlation measures tracked simultaneously
- Controlled dephasing environment

---

### 2.2 NV Center Spin Ensemble Decoherence

**Citation:**
(2022). Decoherence of nitrogen-vacancy spin ensembles in a nitrogen electron-nuclear spin bath in diamond. *npj Quantum Information*.

**DOI:** 10.1038/s41534-022-00605-4

**What was measured:**
- T2 coherence times vs. P1 (nitrogen) concentration
- Cluster correlation expansion calculations
- Hahn-echo decoherence

**Raw data availability:**
- npj Quantum Information typically requires data availability
- Theoretical + experimental comparison

**Relevance to Q54:**
- Systematic decoherence characterization
- Can correlate T2 with information spreading

---

### 2.3 Tripartite Quantum Discord Experiment (2025) - WITH GITHUB DATA

**Citation:**
Byrnes, T., Radhakrishnan, C., Dorai, K., et al. (2025). Experimental determination of tripartite quantum discord. *Proceedings of the National Academy of Sciences*, 122(27).

**DOI:** 10.1073/pnas.2507467122

**What was measured:**
- Tripartite quantum discord
- GHZ and W class states
- Full quantum state tomography
- Fidelities exceeding 95%

**Raw data availability:**
- **YES - GitHub repository:** https://github.com/Vaishali-quantum/Tripartite-Quantum-Discord-Data
- MATLAB code for state tomography
- Mathematica code for discord calculation
- NMR experimental data

**Platform:** Bruker Avance-III 600 MHz NMR spectrometer

**Relevance to Q54:**
- Direct discord measurements (related to R_mi via Holevo decomposition)
- Open source code can be adapted for R_mi calculation
- First experimental tripartite discord

---

## Category 3: Coherence Time and Benchmarking Data

### 3.1 Superconducting Qubit Decoherence Benchmarks

**Citation:**
(2019). Decoherence benchmarking of superconducting qubits. *npj Quantum Information*.

**DOI:** 10.1038/s41534-019-0168-5

**What was measured:**
- T1 (relaxation time): mean ~49 microseconds
- T2* (dephasing time): mean ~95 microseconds
- Temporal stability over multiple days
- Qubit frequency fluctuations

**Raw data availability:**
- npj Quantum Information supplementary materials
- Statistical distributions of T1, T2

**Relevance to Q54:**
- Baseline decoherence rates for superconducting systems
- Can model information spreading timescales

---

### 3.2 Record Ion Qubit Coherence

**Citation:**
(2021). Single ion qubit with estimated coherence time exceeding one hour. *Nature Communications*.

**DOI:** 10.1038/s41467-020-20330-w

**What was measured:**
- T2 up to 5500 seconds for Yb-171 ion
- Dynamical decoupling performance
- Sympathetic cooling effects

**Raw data availability:**
- Nature Communications requires data availability
- Supplementary materials with detailed measurements

**Relevance to Q54:**
- Extreme limit of coherence preservation
- Can test R_mi in nearly-isolated vs. decohering regimes

---

## Category 4: Theoretical Resources with Numerical Data

### 4.1 Spin Environment Information Amplification (2016)

**Citation:**
Zwolak, M., Quan, H. T., & Zurek, W. H. Amplification, Decoherence and the Acquisition of Information by Spin Environments. *Scientific Reports*.

**DOI:** 10.1038/srep25277

**What was measured (numerically):**
- Quantum Chernoff information
- Redundancy scaling in finite/infinite spin environments
- Information amplification dynamics

**Relevance to Q54:**
- Analytical formulas for redundancy
- Can compare Q54 R_mi predictions against established theory

---

### 4.2 Scattered Photon Redundancy Theory

**Citation:**
Riedel, C. J., & Zurek, W. H. (2010). Quantum Darwinism in an Everyday Environment: Huge Redundancy in Scattered Photons. *Physical Review Letters*, 105, 020404.

**arXiv:** 1001.3419

**Key result:**
- 1 micrometer dust grain in sunlight for 1 microsecond
- Location imprinted ~100 million times in scattered photons
- Quantum mutual information calculations

**Relevance to Q54:**
- Sets scale for "natural" redundancy
- Theoretical predictions for realistic decoherence

---

## Data Extraction Strategy

### For Papers Without Open Data

1. **Figure digitization:** Use tools like WebPlotDigitizer to extract data from published figures
2. **Author contact:** Email corresponding authors requesting raw data
3. **Replication:** Some experiments (e.g., IBM Quantum) can be replicated on current hardware

### Priority Order for Q54 Validation

| Priority | Dataset | Reason |
|----------|---------|--------|
| 1 | Zhu et al. 2025 (Zenodo) | Open data, direct I(S:F) measurements |
| 2 | Tripartite Discord (GitHub) | Open code + data, discord-related |
| 3 | NV Center 2019 | Direct QD demonstration, solid-state |
| 4 | Photonic 2018/2019 | High-fidelity, controlled systems |
| 5 | IBM Quantum | Can replicate on current hardware |

---

## Mapping Experimental Data to R_mi Predictions

### The Q54 Hypothesis

The Q54 hypothesis predicts that R_mi (mutual information-based R metric) increases approximately 2x during decoherence. To test this:

### Required Calculations

1. **Extract I(S:F)** from experimental mutual information curves
2. **Calculate R_mi** using the formula:
   ```
   R_mi = I(S:F) / H(S)
   ```
   where H(S) is the system entropy

3. **Track R_mi evolution** through decoherence process:
   - Initial (coherent) state: R_mi(0)
   - Final (decohered) state: R_mi(infinity)
   - Ratio: R_mi(infinity) / R_mi(0) should be ~2x if Q54 holds

### Key Observables

From quantum Darwinism experiments, we need:
- **Mutual information plateau height** (related to redundancy)
- **Fragment size at plateau onset** (related to decoherence completeness)
- **Discord decay rate** (measures quantum-to-classical transition)

### Potential Complications

1. **Noise vs. signal:** Real hardware noise may obscure R_mi dynamics
2. **System size:** Small systems may not show clear 2x factor
3. **Initial state dependence:** R_mi predictions may vary with initial state

---

## Next Steps

1. **Download Zenodo dataset** from 2025 Science Advances paper
2. **Clone GitHub repository** for tripartite discord data
3. **Contact Jelezko group** at Ulm for NV center raw data
4. **Digitize figures** from 2018/2019 photonic experiments
5. **Replicate IBM experiment** on current ibm_brisbane or similar
6. **Develop analysis pipeline** to calculate R_mi from extracted data

---

## References

### Primary Quantum Darwinism Papers
- [Nature Physics - Quantum Darwinism Theory](https://www.nature.com/articles/nphys1202) (Zurek 2009)
- [Science Advances - Superconducting Circuits](https://www.science.org/doi/10.1126/sciadv.adx6857)
- [Physical Review Letters - NV Centers](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.140402)
- [Physical Review A - Photonic Cluster States](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.020101)
- [arXiv - Six-Photon Simulator](https://arxiv.org/abs/1808.07388)

### Data Repositories
- [Zenodo Dataset - Zhu et al. 2025](https://doi.org/10.5281/zenodo.15702784)
- [GitHub - Tripartite Discord Data](https://github.com/Vaishali-quantum/Tripartite-Quantum-Discord-Data)

### Open Source Tools
- [Qiskit](https://qiskit.org/) - IBM Quantum SDK
- [QuTiP](https://qutip.org/) - Quantum Toolbox in Python
- [Cirq](https://quantumai.google/cirq) - Google's quantum framework

---

*Document created: 2026-01-30*
*Purpose: External validation of Q54 R_mi predictions*
*Status: Ready for data collection and analysis*
