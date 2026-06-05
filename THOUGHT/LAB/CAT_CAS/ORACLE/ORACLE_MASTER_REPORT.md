# The CAT_CAS Oracle: Unified Master Report

*Synthesized from direct source read on 2026-06-04.*
*Sibling of `MASTER_REPORT.md`. Same author: R. R. Romero / CAT_CAS Laboratory.*
*Scope: 24 Oracle source files across phases 20, 25, 34, 35-40, 45, 46, plus 42 ULTRA.*

---

## 0. The One-Sentence Verdict

**The Oracle is a single architectural pattern, instantiated ~24 times:** build a problem-dependent non-Hermitian Hamiltonian `H`, measure a `Z`-valued topological invariant on it, and report a categorical verdict. The problem domain changes; the skeleton does not.

---

## 1. Source Custody and File Inventory

### 1.1 Direct source read (this report)

| Phase | File | LOC | Mechanism |
|---|---|---|---|
| 35.1 | `35_topological_halting_oracle/35.1_hermitian_oracle/35_topological_halting_oracle.py` | 407 | Hermitian H, halt=zero-energy, p_halt_max, resolvent W |
| 35.2 | `35_topological_halting_oracle/35.2_nonhermitian_oracle/36_nonhermitian_oracle.py` | 349 | Directed H, EP sink, point-gap twist W, kappa(V) |
| 35 | `35_topological_halting_oracle/PAPER.md` | 709 | Formal 9-experiment paper (read partial) |
| 25/7 | `25_lattice_holography/7_holographic_oracle_svp.py` | 75 | Qwen 0.5B as "Quantum Oracle", prompt only |
| 25/8 | `25_lattice_holography/8_eigenbuddy_lwe_oracle.py` | 188 | Qwen 0.5B + EigenBuddy tokenizer decoder |
| 25/9 | `25_lattice_holography/9_recursive_qubit_oracle.py` | 140 | QubitLatticeEncoder -> 896-d hidden, linear decoupler |
| 25/10 | `25_lattice_holography/10_catalytic_eigen_shor.py` | 173 | + Complex SVD, Moore-Penrose pseudo-inverse, global phase sweep |
| 25/11 | `25_lattice_holography/11_native_eigen_shor.py` | 176 | + MultiHeadComplexAttention, cosine phase loss curriculum |
| 37 | `37_2d_chern_oracle/37_2d_chern_oracle.py` | 363 | 2D Chern insulator, Bott index, EP sink |
| 37s | `37_2d_chern_oracle/37_2d_chern_oracle_scaled.py` | 321 | Scaled + gamma sweep variant |
| 38 | `38_3d_weyl_oracle/38_3d_weyl_oracle.py` | 265 | kz-stack of 2D slices, Weyl node annihilation |
| 39 | `39_4d_axion_oracle/39_4d_axion_oracle.py` | 440 | 4D Dirac Gamma matrices, second Chern C2 |
| 40 | `40_5d_floquet_oracle/40_5d_floquet_oracle.py` | 86 | 5D Floquet, three-step non-Clifford, pi-modes |
| 45.1 | `45_phase_math/45_1_collatz_oracle/45_1_collatz_oracle.py` | 804 | Collatz graph -> H, 6-gate hardening |
| 46.1c | `46_phase_bio/46_1_protein_folding/46_1_foldability_oracle.py` | 132 | CANONICAL — 1D chain, KD scale, W=0 foldable |
| 46.1d | `46_phase_bio/46_1_protein_folding/46_1_protein_folding_oracle.py` | 121 | DEPRECATED 2026-06-01 — 2D contact IPR |
| 46.2 | `46_phase_bio/46_2_folding_pathway/46_2_folding_pathway_oracle.py` | 143 | 2D contact map, gamma sweep |
| 46.3 | `46_phase_bio/46_3_prion_contagion/46_3_prion_contagion_oracle.py` | 200 | Lattice of 20 proteins, J-coupling sweep |
| 46.4 | `46_phase_bio/46_4_topological_genetic_code/46_4_topological_genetic_code_oracle.py` | 151 | 64-codon lattice, SGC vs random |
| 46.5 | `46_phase_bio/46_5_neural_binding_oracle/46_5_neural_binding_oracle.py` | 136 | Watts-Strogatz connectome, lesion/anesthesia |
| 46.6 | `46_phase_bio/46_6_morphogenesis_oracle/46_6_morphogenesis_oracle.py` | 149 | Epithelium, separated vs annihilated defects |
| 20.10 | `20_catalytic_eigen_shor/20.10_tiny_compress_phase/1_holographic_phase_oracle.py` | 673 | Holographic Feistel-braided period detection |
| 20.10.5 | `20_catalytic_eigen_shor/20.10_tiny_compress_phase/5_holo_oracle.py` | 292 | Unified Mandelbrot+Complex+Torus+Catalytic+.holo |
| 34.7 | `34_zeta_eigenbasis/02_holographic_sieves/7_holo_riemann_oracle.py` | 175 | Prime scattering -> Riemann zeros via holo |
| 34.20 | `34_zeta_eigenbasis/05_topological_proof/20_transcendent_winding_oracle.py` | 98 | O(1) asymptotic theta at t=10^100 |
| 42 | `42_computational_event_horizon/ULTRA/exp_19_oracle_machine/rust/src/main.rs` | (Rust) | INFINITY EDITION — Rust oracle machine |

### 1.2 Per-oracle reporting (referenced, not re-read in full)

- `REPORT_COLLATZ_ORACLE.md` (45.1)
- `REPORT_EXP_46_X.md` x 6 (46.1-46.6)
- `VERIFICATION_REPORT.md` per oracle
- `REPORTS/VIOLATIONS/ROADMAP_3.md` — canonical audit ledger

---

## 2. The Unified Oracle Pattern

Every Oracle in the lab conforms to a four-function template:

```
def build_H(problem_params)            -> complex matrix    # encode problem as Hamiltonian
def compute_invariant(H, ...)          -> int winding       # Z-valued topological measurement
def run_oracle(problem_params)         -> verdict dict      # orchestrate + report
def harden_<oracle>(H, ...)            -> bool[]            # independent verification gates
```

The two invariant flavors used:

| Invariant | Used by | Formula |
|---|---|---|
| **Point-gap winding (1D)** | 35.2, 45.1, 46.1, 46.3, 46.4, 46.5, 46.6, 20.10, 34.20 | `W = (1/2pi) * sum_k Delta arg(det(H(phi_k)))` for `H(phi) = D + e^{i*phi}*O` over `phi in [0, 2pi]` |
| **Bott / Chern index (higher-D)** | 35.x paper, 37, 38, 39, 40 | Real-space projector `P = (1/2pi*i) * oint (zI-H)^-1 dz`, then `C = (1/2pi) * Im Tr log(V U V^dag U^dag)` |

The two matrix flavors:

| Matrix flavor | Used by |
|---|---|
| **Directed (acyclic graph, EP sink)** | 35.2, 45.1, 46.1, 46.3, 46.4, 46.5, 46.6 |
| **Periodic lattice with k-stack (Bloch)** | 37, 38, 39, 40 |
| **LLM-as-oracle (no Hamiltonian)** | 25/7-11 (different paradigm) |
| **Asymptotic analytic (no matrix)** | 34.20 (Riemann-Siegel theta) |

---

## 3. Family 1 — Topological Halting Oracles (Phase 35)

The lab's "core proof". Reframes Turing's Halting Problem as a topological phase transition.

### 3.1 Hermitian foundation (35.1)

- TM transition table -> basis `|s, b>` of dim `N = states * symbols = 2*S`
- Halt state gets `E_halt = 0` (topological fixed point)
- Active states get `E_active = 1`
- Off-diagonals: `-gamma` symmetric couplings
- **Three measurements**:
  - `p_halt_max` (time-averaged halt population) — `> 0.1` => HALTS
  - `W_res` (resolvent winding around z=0)
  - `W_halt` (winding of complex halt amplitude)
- Test machines: `halt_direct`, `halt_chain`, `loop_2cycle`, `loop_3cycle`

### 3.2 Non-Hermitian extension (35.2)

The decisive move. TM transitions are directed, so:

```
H[j][i] = +gamma       # directed edge i -> j
H[i][j] = 0            # NO reverse edge
H[h][h] = -i * 10 * loss_rate   # halt = massive EP sink
H[i][i] = -i * loss_rate        # active = gentle dissipation
```

This unlocks three phenomena Hermitian cannot see:

1. **Exceptional Points (EP)** — eigenvalues + eigenvectors coalesce into Jordan block; `kappa(V) = cond(eigvecs)` diverges
2. **Point-gap winding** — det(H(phi)) wraps around origin when there are spectral loops
3. **Non-Hermitian Skin Effect** — bulk states exponentially localize at the boundary (Hatano-Nelson)

**Verdict rule:** `W_twist == 0` => HALTS (spectrum collapsed into EP); `W_twist != 0` => LOOPS (spectral loop survives)

### 3.3 Higher-D dimensional lift (37, 38, 39, 40)

Same pattern in higher dimensions, each with a domain-specific invariant:

| Phase | Dim | Topology | Invariant | Halt mechanism |
|---|---|---|---|---|
| 37 | 2D | Chern insulator | Bott index `C` | Chiral edge destroyed by EP sink |
| 38 | 3D | Weyl semimetal | `C(kz)` profile over kz-stack | Weyl node annihilation -> Exceptional Ring |
| 39 | 4D | Axion insulator | Second Chern `C2` | 4D Dirac monopole annihilation |
| 40 | 5D | Floquet time crystal | pi-mode count | `|z+1|` threshold crossed by uniform Gamma >= 0.5 |

All use the **catalytic dimensional reduction** trick: O(N^2) buffers reused across the k-loop, no dense 3D/4D/5D matrix allocation. (This is the CAT_CAS "catalytic tape" applied to compute buffers.)

### 3.4 Status (per file headers)

- 35.x: **AUDIT-AWARE CLAIM** — high-claim internal, treat as experimental
- 40 5D Floquet: header says **"SOLVED"** — pi-modes 16/16 at G=0, 0/16 at G>=0.5

---

## 4. Family 2 — Lattice Holography / LWE Oracles (Phase 25)

Different paradigm. The "Oracle" is an LLM (Qwen 0.5B) patched with `.holo` weights, used as a black box to attack LWE.

### 4.1 Common scaffold (all 5 files)

```python
MODEL_DIR = r".../16_catalytic_27b_inference/gemini_update/qwen_0.5b"
HOLO_PATH  = r".../EIGEN_BUDDY/cybernetic_truth/qwen_0_5b_k128.holo"

student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
holo_dict = torch.load(HOLO_PATH, weights_only=False)
patch_model_with_holo(student, holo_dict)
```

The prompt is always:
> "You are a Quantum Oracle. Solve the following Learning With Errors (LWE) system. Modulo q = ..., Secret Dimension N = ... [A rows] [B values] What is the true Secret Vector S?"

### 4.2 Five progressive variants

| File | Decoder | Output |
|---|---|---|
| 7 holographic_oracle_svp | Native (cybernetic_inference) | Free-form text |
| 8 eigenbuddy_lwe_oracle | EigenBuddyTokenizer trained on patched hidden states | Token-by-token |
| 9 recursive_qubit_oracle | QubitLatticeEncoder -> linear(896, n) decoupler | Direct S_pred |
| 10 catalytic_eigen_shor | + Complex SVD + pseudo-inverse + global phase sweep (360 deg) | S_pred by phase alignment |
| 11 native_eigen_shor | + MultiHeadComplexAttention + cosine phase loss curriculum | S_pred via phase decoder |

The "phase = 2*pi*value / q" mapping is the unifying trick: integer lattice values become continuous qubit phases, the LLM preserves phase structure via holographic weights, and a downstream linear/attention decoder extracts integers.

### 4.3 The EigenBuddy pattern (key subroutine)

1. Generate 8-step continuations from teacher
2. Capture student's patched hidden states
3. Center + complex-cast + SVD
4. Take top-K=64 eigenvectors
5. Train small complex-attention model to map projected states -> next token

This is reused across 25/8, 25/10, 25/11 with progressively more sophisticated decoders.

---

## 5. Family 3 — Phase Math Oracles (Phase 45)

### 5.1 Collatz Oracle (45.1) — the deepest single experiment (804 LOC)

Reframes the 90-year Collatz conjecture as acyclicity of a directed graph:

- Nodes = integers `n in [1, 1024]`
- Edges = `n -> n/2` (even) or `n -> 3n+1` (odd)
- `n=1` is a global EP sink with `H[0,0] = -i * 50`
- Active states get `H[i,i] = -i * 1`

**Verdict:** `W_twist == 0` => acyclic => Collatz holds; `W_twist != 0` => cycle exists => Collatz false.

**Six hardening gates** (any failure = protocol flaw):

1. Multi-scale: N = 256, 512, 1024 all give W=0
2. Cycle spectrum: W = cycle_length for L-cycles, W=0 for 1-cycle (fixed point)
3. Counterexample: injecting synthetic 2-cycle flips W to non-zero
4. Determinant stability: det(H(phi)) is phi-independent for acyclic (max dev < 1e-10)
5. Parameter sweep: gamma/ell in [0.1, 100] all give W=0
6. False-positive fuzzer: 50 random DAGs all give W=0 (0/50 false positives)

Includes its own `CatalyticTape` subclass with XOR-pattern record/uncompute and SHA-256 verify at the end. Landauer = 0.0 J if tape is restored.

---

## 6. Family 4 — Phase Bio Oracles (Phase 46)

Same skeleton, biology-flavored. Each maps a biological question to a Hamiltonian and reads the answer from a topological invariant.

| Oracle | Problem | H construction | Invariant | Reading |
|---|---|---|---|---|
| 46.1 foldability | Does this sequence fold? | 1D chain, KD hydrophobicity on diagonal, `(KD_i - KD_j)`-weighted hopping | Winding W | W=0 foldable, W!=0 frustrated |
| 46.2 folding pathway | Pathway kinetics | 2D contact map + gamma sweep | Gap + IPR | Folded = small gap, low IPR |
| 46.3 prion contagion | Does prion spread on lattice? | N_proteins * L_seq, J-coupling | mean/max IPR | Prion impurity localizes at J=0 |
| 46.4 genetic code | Is the Standard Genetic Code topologically special? | 64-codon lattice, Kyte-Doolittle, Hatano-Nelson non-reciprocal phase | W + max spectral radius | SGC has W=0 + minimal radius; random codes inflate |
| 46.5 neural binding | Intact vs lesioned vs anesthetized brain? | Watts-Strogatz 150 nodes, k=6, p=0.15 | W + IPR | Intact non-trivial, anesthesia localizes 10x |
| 46.6 morphogenesis | Flat / separated / annihilated epithelial defects? | 2D LxL with theta = arctan gradient | 1D slice winding + IPR | Flat=delocalized, separated=0D cores, annihilated=1D scar |

### 6.1 Deprecation note

**`46_1_protein_folding_oracle.py` is DEPRECATED (2026-06-01).** It used 2D contact-map IPR with arbitrary thresholds (IPR<0.10=FOLDED) and a ceremonial tape that was never XOR-modified. The canonical replacement is `46_1_foldability_oracle.py` (1D chain, winding number, real XOR tape). Both files exist; the deprecated one is preserved as a forensic reference.

### 6.2 Honest notes (per file comments)

- 46.3 explicitly admits: "prion does NOT propagate its winding number to neighbors in this model. Contagion requires dynamical coupling not captured here."
- 46.2's null model is documented: gamma=0 is the no-solvent randomized baseline
- 46.5's null model is the anesthetized connectome (maximally decohered)

This honesty pattern (real null models, not pass-the-test-only design) is consistent across the 46 family.

---

## 7. Adjacent Oracles (20, 34, 42)

### 7.1 20.10 holographic_phase_oracle (Shor's period via Feistel braid)

Tests whether a multi-scale Feistel braid can encode the modular exponentiation operator `U_a: |x> -> |ax mod N>` into S << r positions (where r is the order period).

- **Path A (baseline)**: S raw phase samples, autocorrelation, expect FAIL when S < r
- **Path B (holographic)**: S samples embedded in d_model, Feistel cross-attention braid, autocorrelation on braided tape
- Question: does braiding reduce S scaling from O(r) to O(log N)?

Architecture detail: every weight in the Feistel is **deterministic** — constructed from `a, N` via `theta = 2*pi*pow(a, scale*(i+1)*(j+1), N) / N`. No random init.

### 7.2 20.10.5 holo_oracle (unified stack)

Combines 5 prior breakthroughs:
1. **Mandelbrot**: catalytic cepstrum recursion (autocorrelation of autocorrelation)
2. **Complex**: native complex Hermitian representation (S^1, not R^2)
3. **Torus**: circular statistics, winding numbers
4. **Catalytic**: borrow tape -> compute -> restore
5. **.holo engine**: real analyze_spectrum / project / render / verify

Goal: factor 22-bit semiprime via unified stack. 23-M power-of-2 phase grating.

### 7.3 34.7 holo_riemann_oracle

- Raw prime scattering matrix `S[m,n] = exp(i * ln(p_m) * ln(p_n))`
- `PrimeEncoder` projects 200-d complex S-matrix rows to 896-d hidden via orthogonal init
- `HoloRiemannDecoupler` (MultiHeadComplexAttention + linear) predicts the first 20 zeta zeros
- Training via MSE against mpmath-computed zeta zeros

### 7.4 34.20 transcendent_winding_oracle

- Bypasses zeta(s) evaluation entirely (impossible at t=10^100)
- Uses asymptotic Riemann-Siegel theta: `theta(t) = t/2 * log(t/(2*pi*e)) - pi/8 + 1/(48*t) + 7/(5760*t^3)`
- Topological charge = (theta(t+dt) - theta(t)) / pi at `t = 10^100`
- 100-digit mpmath precision, O(1) execution time
- Output: exact number of zeta zeros in any 1-billion-step window at the Googol boundary

### 7.5 42 ULTRA exp_19_oracle_machine (Rust)

The INFINITY EDITION. Rust port. Per the 35 PAPER abstract, it combines:
- ER=EPR entanglement bridges
- Catalytic Bell-pair quantum tape ("Invisible Hand")
- Temporal bootstrap self-referential feedback
- 4-qubit Hilbert space, 84% restoration fidelity

---

## 8. Common Code Conventions (cross-Oracle style)

| Convention | Where it appears |
|---|---|
| `torch.manual_seed(42)` | All torch-based oracles |
| `torch.set_default_dtype(torch.float64)` | 35, 37-40 (precision for spectral) |
| `torch.complex64` (sometimes complex128) | All Hamiltonian oracles |
| `print("=" * 70)` block delimiters | 35, 37-40, 45.1, 46.x |
| Per-function docstring with PHYSICS/ARCHITECTURE sections | 35.x, 37-40, 45.1 |
| `build_H`, `compute_invariant`, `run_oracle` naming | 35.x, 37-40, 45.1 |
| Hardening gates with `g1, g2, g3` and explicit `PASS/FAIL` | 45.1, 46.x |
| `CatalyticTape` with XOR pattern (where present) | 45.1, 46.1c, 46.2-46.6 |
| `EigenBuddy` import path | 25/7-11, 34.7 |
| Qwen 0.5B model + .holo weights at fixed path | 25/7-11, 34.7 |
| `device = "cuda" if available else "cpu"` | All 25/34 torch files |
| `torch.no_grad()` for inference | All 25 files, 34.7 |

---

## 9. Status Audit (honest, by family)

| Family | Status | Notes |
|---|---|---|
| 35 (Halting) | AUDIT-AWARE CLAIM | README warns: "treat as internal experimental result unless roadmap says fully verified" |
| 37-40 (Topological chain) | MIXED | 40 SOLVED (per header), 37 PARTIAL edge destruction at large L, 38 PARTIAL Weyl annihilation needs uniform gamma, 39 INCONCLUSIVE 4D annihilation |
| 25 (LWE) | INVENTORIED | Experimental cryptographic results, no peer review |
| 45.1 (Collatz) | HARDENED | 6/6 gates pass for N in [256, 1024]; awaits external mathematical review |
| 46.1 (foldability) | CANONICAL | Replaces deprecated IPR version |
| 46.1 (protein_folding) | DEPRECATED 2026-06-01 | Kept as forensic reference |
| 46.2-46.6 | MOSTLY PASS with honest null models | See per-file HARDENING GATES blocks |
| 34.20 (Transcendent) | INTERNAL | Mathematically clean, O(1) analytic, no empirical variance |
| 20.10 (Holographic phase) | EXPERIMENTAL | Tests S<r hypothesis; no claim of cryptographic break |
| 42 (ULTRA Rust) | PER PAPER ABSTRACT | 84% restoration fidelity; quantum-tape dependent |

**Audit-truth ledger**: `THOUGHT/LAB/CAT_CAS/REPORTS/VIOLATIONS/ROADMAP_3.md` is canonical; this report does not claim external mathematical finality for any high-claim track.

---

## 10. Concept Map

```
                          ORACLE (unified pattern)
                                 |
        +------------------------+--------------------------+
        |                        |                          |
  Spectral Winding         Bott/Chern Index            LLM-as-Oracle
  (1D Cauchy)              (higher-D projector)         (Qwen 0.5B + .holo)
        |                        |                          |
   +----+----+            +------+------+               Phase 25
   |         |            |      |      |               (LWE attacks)
 35.2      45.1         37    38    39/40
 46.1-46.6                            |
                              40 Floquet (SOLVED)
                              39 4D Axion
                              38 3D Weyl
                              37 2D Chern

ADJACENT (different paradigm, also called "Oracle"):
  20.10 Holographic Feistel phase oracle (Shor period)
  20.10.5 Unified holo stack
  34.7 Holo Riemann oracle (zeta zeros)
  34.20 Transcendent winding (Googol)
  42 ULTRA Rust oracle machine (INFINITY EDITION)
```

---

## 11. Open Threads

- The 35.x paper claims 9 sub-experiments; only 35.1 and 35.2 were read here. Remaining (35.3-35.9 per PAPER) include: infinite-tape scaling, entanglement entropy at EP, 500-TM counterexample fuzzer, LCU Phase Estimation, 38-fold way classification, 2-parameter torus Chern, INFINITY EDITION.
- The 35 PAPER claims "globally trivial bundle (C=0.0) on a 2-parameter torus" — the Godel obstruction is gated behind the full catalytic quantum tape. This is the load-bearing claim of the whole family and is the right next-read if a deeper audit is needed.
- 46.3 prion contagion has explicit honest notes that the model does NOT show propagation; this is acknowledged and the file documents why.
- 25 LWE family is the only one with no spectral invariant; if the LWE claims were ever externally verified, the Oracle pattern would gain a third invariant flavor (neural resonance).

---

## 12. See Also

- `THOUGHT/LAB/CAT_CAS/MASTER_REPORT.md` — full lab coverage matrix
- `THOUGHT/LAB/CAT_CAS/CAT_CAS_OS.md` — lab charter
- `THOUGHT/LAB/CAT_CAS/REPORTS/VIOLATIONS/ROADMAP_3.md` — canonical audit ledger
- `THOUGHT/LAB/CAT_CAS/REPORTS/VIOLATIONS/PHASE_46_ISOMORPHISM_AUDIT.md`
- `THOUGHT/LAB/CAT_CAS/35_topological_halting_oracle/PAPER.md` — formal writeup
- `THOUGHT/LAB/HOLO/docs/CAT_CAS_INTEGRATION_DOSSIER.md` — how Oracle code transfers to HOLO

*End of report.*
