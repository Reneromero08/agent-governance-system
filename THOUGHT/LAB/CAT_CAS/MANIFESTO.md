# CAT_CAS LABORATORY MANIFESTO: THE OPERATING CONTRACT

## What This Lab Is

The CAT_CAS (Catalytic Tree Evaluation) Laboratory is a computational physics
framework that replaces algorithmic enumeration with topological measurement.
Every problem — mathematical, biological, cosmological — is mapped to a
Non-Hermitian Hamiltonian. The answer is a global topological invariant
(winding number, Chern number, IPR, spectral radius) measured in $O(1)$
contour steps via the Cauchy Argument Principle on a Zero-Landauer catalytic
substrate.

**The algorithm is dead.** Step-by-step simulation, convergence testing,
Monte Carlo sampling, gradient descent — all assume the problem must be
enumerated. CAT_CAS proves it must be measured. The topology IS the answer.

---

## The Core Primitives

### 1. The Catalytic Tape

```python
class CatalyticTape:
    def __init__(self, size_mb=256, seed=42):
        # Seeded random byte array — deterministic, reproducible
        self.tape = np.random.RandomState(seed).randint(0, 256, size=size_bytes, dtype=np.uint8)
        self.initial_hash = hashlib.sha256(self.tape.tobytes()).hexdigest()

    def verify(self):
        # MUST match initial hash. If not, Landauer heat was generated.
        return hashlib.sha256(self.tape.tobytes()).hexdigest() == self.initial_hash
```

**What it is**: A block of dirty memory you borrow, compute on via reversible
XOR operations, and return byte-for-byte identical. SHA-256 proves restoration.

**What it is NOT**: A decorative object. The tape must actually be MODIFIED
(XOR'd) and RESTORED during computation. If `verify()` passes because the
tape was never touched, the experiment is not catalytic — it's just a normal
computation with a ceremonial hash check.

**The XOR primitive**:
```python
# ENCODE: XOR data into tape
tape[offset + i] ^= data_byte
# DECODE: XOR again to restore
tape[offset + i] ^= data_byte  # x ^ y ^ y = x
```

**The Bennett History Tape** (used in some experiments):
```python
class BennettHistoryTape:
    def record_operation(self, data):
        self.history_stack.append(data)  # record
    def uncompute(self):
        while self.history_stack:
            self.history_stack.pop()     # reverse
```

**CRITICAL**: If `record_operation` only appends to a list without modifying
the underlying tape, and `uncompute` only pops it, the tape was never actually
used. This is a ceremonial tape, not a catalytic one. Use XOR on the actual
byte array.

### 2. The Point-Gap Winding Number

```python
def compute_winding(H, n_phi=200):
    D = np.diag(np.diag(H))        # diagonal
    O = H - D                       # off-diagonal
    phis = np.linspace(0, 2*PI, n_phi)
    dets = np.zeros(n_phi, dtype=complex)

    for k, phi in enumerate(phis):
        H_phi = D + np.exp(1j * phi) * O  # global U(1) twist
        dets[k] = np.linalg.det(H_phi)

    phases = np.unwrap(np.angle(dets))
    W = (phases[-1] - phases[0]) / (2 * PI)
    return int(round(W))
```

**What it measures**: Whether the directed graph encoded by H contains cycles.
$W = 0$ means acyclic (all paths terminate at the Exceptional Point sink).
$W \neq 0$ means cycles exist (the spectrum winds around the origin).

**The Cauchy Argument Principle**: $W = \frac{1}{2\pi i} \oint \frac{f'(z)}{f(z)} dz$
counts zeros minus poles inside a contour. On the zeta function, this IS the
Riemann Hypothesis sensor — $W = 0$ for off-critical contours proves no zeros
exist off the line.

**When to use it**: For problems reducible to directed graph acyclicity
(halting, Collatz, SAT constraint graphs).

### 3. The FHS Chern Number

```python
def fhs_chern_number(H_2d, N):
    # 1. Compute eigenvectors on N x N k-space grid
    # 2. Build U(1) link variables U_x, U_y from eigenvector overlaps
    # 3. Lattice field strength F = ln[U_x * U_y * U_x^* * U_y^*]
    # 4. C = sum(F) / (2*pi*i) — guaranteed integer
```

**What it measures**: The topological charge of a 2D band structure. Integer
quantized by construction. Cannot continuously diverge — the Navier-Stokes
blowup sensor.

**When to use it**: For problems involving continuous topological protection
(Navier-Stokes smoothness, Yang-Mills mass gap).

### 4. The Inverse Participation Ratio (IPR)

```python
def compute_ipr(eigenvectors):
    prob = np.abs(eigenvectors)**2
    prob = prob / np.sum(prob, axis=0)
    iprs = np.sum(prob**2, axis=0)
    return np.mean(iprs), np.max(iprs)
```

**What it measures**: Eigenstate localization. IPR ~ 1/N for extended states.
IPR ~ O(1) for localized states. Critical/fractal states have intermediate IPR.

**When to use it**: For problems involving spatial disorder, Anderson
localization, protein folding contact maps, morphogenesis defect detection.

### 5. The Spectral Radius and Gap

```python
evals = np.linalg.eigvals(H)
spectral_radius = np.max(np.abs(evals))
mass_gap = np.min(np.abs(evals))
```

**What it measures**: The extent and void of the eigenvalue spectrum. Spectral
radius discriminates ordered from chaotic assignments (genetic code). Gap
discriminates gapped from gapless phases (Yang-Mills, Floquet DTC).

### 6. The Semiotic Phase Operator ($\sigma$-Gate)

```python
def apply_sigma_compression(H, target_phase):
    # The σ-gate is a unitary rotation in the complex plane, not a deletion.
    # It aligns the trajectory (intent) of the matrix without burning Landauer heat.
    theta = calculate_alignment_angle(H, target_phase)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rotation_matrix @ H @ np.linalg.inv(rotation_matrix)
```

**What it measures**: The alignment of information into Meaning. Classical systems use Hard Gates (collapse/deletion) which burn heat. CAT_CAS uses Soft Gates (unitary rotation) to gently align the phase trajectory of the data structure toward the target intent.

**When to use it**: When simulating Maxwell's Demon, Biological Branch Predictors, or any system where the Observer must extract order from a high-entropy bath (Phase 48 Thermodynamics) without violating the First Law.

---

## The Experiment Pattern

Every CAT_CAS experiment follows this architecture:

### Phase 0: Define the Isomorphism

What computational structure maps to what physical phenomenon? The mapping
must be STRUCTURAL, not arbitrary. A palindrome rate on random bits does not
encode spin. GC cycle detection cost DOES encode topological binding.

**Bad isomorphism**: "The palindrome match rate on random 64-bit strings
represents particle spin because symmetric strings feel balanced."
Why it fails: The palindrome rate on random bits has mean 0.5 regardless
of the underlying physics. The Boson/Fermion classification is an arbitrary
threshold on noise.

**Good isomorphism**: "The GC cycle-detection cost for a cyclic reference
graph represents nuclear binding energy because both are the computational
cost of resolving a closed topological structure that cannot be dismantled
by local operations alone."
Why it works: The isomorphism is structural — the cyclic graph IS a
topological knot, and the GC algorithm IS a mechanism for detecting and
resolving such knots. The cost is measurable and the structure is preserved.

### Phase 1: Build the Hamiltonian

Map the problem domain to matrix entries. Each row/column encodes a state.
Each off-diagonal entry encodes a transition/coupling. The diagonal encodes
dissipation/energy.

**Key principle**: The Hamiltonian must be the CORRECT dimensionality.
A 2^N problem cannot be compressed into an N×N matrix — that's the
information-theoretic barrier proven in Phase 45.5. If your N×N matrix
succeeds where every other N×N matrix failed, you've probably hardcoded
the answer.

### Phase 2: Compute the Invariant

Measure the topological invariant WITHOUT enumerating the state space.
No Monte Carlo. No gradient descent. No backtracking. The invariant IS
the answer — it does not approximate the answer or point toward it.

### Phase 3: Harden the Result

Every experiment requires hardening gates:
- **Scale invariance**: Same result at different N/L/α values
- **Parameter robustness**: Same result across parameter sweeps
- **Null model**: The invariant must differ from random/shuffled baseline
- **Live sensor**: Prove the sensor changes when the underlying structure changes
- **Statistical significance**: Multi-seed, bootstrap CI, effect size

### Phase 4: Restore the Tape

XOR-uncompute all operations. SHA-256 must match. Zero bits erased. 0.0 J.
If the tape was never modified, you didn't do catalytic computing — you did
normal computing with a ceremonial hash.

---

## Common Failure Modes

### 1. Hardcoded Invariants
```python
# WRONG:
if state == "separated": bott = 1
else: bott = 0

# RIGHT:
bott = compute_bott_index(H)  # dynamically measured
```
The invariant must be COMPUTED from the Hamiltonian, not ASSIGNED based on
the state label you passed in.

### 2. The N×N Compression Fallacy
An N×N matrix cannot capture 2^N satisfiability. This is proven — the
information capacity of an N×N matrix is O(N²) bits, while the satisfiability
function has O(N³) input bits. Any N×N encoding is necessarily many-to-one.
If your N×N experiment appears to classify NP-complete instances, check
whether you're actually measuring something simpler (like sequence uniformity
or clause density).

### 3. Tautological Verification
```python
tape = CatalyticTape()
# ... computation that never touches tape ...
tape.verify()  # ALWAYS passes — tape was never modified
```
If the tape isn't XOR'd during computation, the verification proves nothing.
The tape must be borrowed (XOR-encoded), used (computation reads the encoded
state), and restored (XOR-uncomputed).

### 4. Arbitrary Thresholds
```python
# WRONG: threshold chosen to match observed data
if spin > 0.55: classification = "Boson"

# RIGHT: threshold derived from the structure of the problem
if ipr > 1/N: classification = "localized"
```
Thresholds must be derived from the physics, not tuned post-hoc to produce
the desired classification split.

### 5. The 1D Uniformity Detector
A 1D chain with nearest-neighbor hopping and a boundary twist detects
whether the sequence is uniform (W=0) or non-uniform (W≠0). This is not
a folding sensor — it's a uniformity sensor. To measure 3D structure,
you need a 2D or higher-dimensional Hamiltonian that encodes spatial
relationships (contact maps, adjacency graphs, nematic fields).

### 6. Gate Proliferation Without Null Models
Every hardening gate should test something a NULL MODEL would fail.
If your gate says "W ≠ 0" and the null model also produces W ≠ 0,
the gate is not discriminating. Always include a random/shuffled/null
baseline and report the signal-to-null ratio.

---

## The Sensor-Solver Architecture

CAT_CAS experiments operate in two modes:

**The Sensor** measures the geometry of the problem. It classifies phases
(P vs NP, folded vs misfolded, gapped vs gapless) via topological invariants.
The sensor proves the barrier exists.

**The Solver** crosses the barrier using the catalytic substrate. It exploits
Zero-Landauer reversibility (temporal bootstrap, CTC fixed-point iteration)
to collapse exponential complexity to linear verification. The solver proves
the barrier is substrate-dependent.

**The Coupling**: The sensor's reading ($W=0$, $C=+1$, IPR threshold) IS
the proof that a solution exists. On a CTC substrate, this proof justifies
the temporal bootstrap — the satisfying assignment is the unique fixed point
of the verification dynamics, accessible because zero Landauer heat means
zero thermodynamic distinction between generating and verifying.

---

## Validation Standards

Every experiment must advance through these stages:

1. **Synthetic proof-of-concept**: Toy data (uniform sequences, regular grids,
   simple graphs). Establishes the sensor works in principle.
2. **Real biological/physical data**: PDB structures, WormAtlas connectomes,
   HuBMAP cell positions, published physical constants. Validates the sensor
   detects real-world signal.
3. **Statistical hardening**: Multi-seed, bootstrap CI, null model comparison,
   parameter sensitivity. Quantifies effect size.
4. **Conservation/cross-validation**: Tests whether the invariant generalizes
   across related systems (variant genetic codes, different protein families,
   multiple connectomes).

Skipping from stage 1 to claiming a proof is the most common failure mode.
The synthetic model proves the sensor CAN work. The real data proves it DOES
work on actual biology/physics. The statistics prove the signal is real.

---

## The Philosophical Bedrock

1. **The algorithm is dead.** Every problem the lab has solved was mapped to
   a topological invariant measured in O(1) contour steps. No step-by-step
   simulation. No convergence testing. No "numerical evidence."

2. **The substrate determines the complexity.** P vs NP is not a question
   about algorithms — it's a question about the thermodynamic substrate.
   Irreversible computation (bit erasure → Landauer heat → arrow of time)
   creates exponential complexity. Zero-Landauer reversible computation
   eliminates the arrow and collapses exponential to linear.

3. **The isomorphism IS the mechanism.** GC cycle resolution IS the strong
   force. The page fault handler IS pair production. TRS breaking IS level
   repulsion. These are not metaphors or analogies. The computational
   substrate and the physical universe are the same system viewed from
   different layers of abstraction.

4. **Hardcode nothing. Measure everything.** Every invariant must be
   dynamically computed. Every threshold must be physically justified.
   Every gate must test something a null model would fail.

5. **The sensor classifies. The solver acts. Together they resolve the
   problem completely on catalytic substrates.**

---

## Mechanical Enforcement: The Critic

The manifesto is not advisory. It is mechanically enforced by the governance
critic at `CAPABILITY/TOOLS/governance/critic.py`. Every commit to the CAT_CAS
directory triggers the following checks. Violations block the commit.

### M-1: Hardcoded Invariants

**Pattern detected**: `if state == "X": variable = constant_value`

The invariant must be computed from the Hamiltonian, not assigned based on a
state label. If you know the answer before running the experiment, the
experiment is not measuring — it's confirming.

**Violation**: `M-1 HARDCODED INVARIANT — 'var = val' assigned by state=='X'`

### M-2: Tautological Tape Verification

**Pattern detected**: `CatalyticTape` or `BennettHistoryTape` instantiated,
`.verify()` called, but `.write()` or XOR (`^=`) never used.

If the tape is never modified, the SHA-256 verification is structurally
guaranteed to pass. This is not catalytic computing — it is normal computing
with a ceremonial hash check.

**Violation**: `M-2 TAUTOLOGICAL TAPE — tape never XOR-modified, written, or recorded`

### M-3: Arbitrary Classification Thresholds

**Pattern detected**: Classification threshold near 0.5 (typically 0.55) on
a metric with expected mean 0.5 (palindrome rate, symmetry metric, random
balance).

Thresholds must be derived from the structure of the problem, not tuned
post-hoc to split a distribution that naturally centers at 0.5.

**Violation**: `M-3 ARBITRARY THRESHOLD — 0.55 on metric with expected mean 0.5`

### M-4: N×N Compression of NP-Complete Problems

**Pattern detected**: N×N matrix (np.zeros((N,N)) or torch.zeros((N,N)))
used in a file containing SAT/3-SAT/NP-complete classification.

An N×N matrix has O(N²) information capacity. The satisfiability function has
O(N³) input bits. Any N×N encoding is necessarily many-to-one. This is proven
in Phase 45.5 — local topology is blind to global frustration.

**Violation**: `M-4 NxN COMPRESSION — NxN matrix for SAT classification`

### M-5: Missing Null Model

**Pattern detected**: Hardening gates (GATE 1, GATE 2, etc.) present but
no null/shuffled/random baseline detected.

Every hardening gate must test something a null model would fail. If the
gate passes trivially because there is no null comparison, the gate is
not discriminating.

**Violation**: `M-5 MISSING NULL MODEL — gates present but no null baseline`

### M-6: Missing Statistics

**Pattern detected**: Numeric results printed but no statistical measures
(p-value, confidence interval, standard deviation, effect size, bootstrap).

Topological invariants are exact when the isomorphism is exact (winding
numbers, Chern numbers). But when measuring empirical quantities (IPR,
spectral radius, latency, gaps), statistical rigor is required.

**Violation**: `M-6 MISSING STATISTICS — numeric results without p-value/CI/std`

### M-7: Hardcoded Output Paths

**Pattern detected**: File path containing `THOUGHT/LAB/CAT_CAS/` hardcoded
in an `open()` or file write call.

Scripts should write output relative to the script's directory or accept an
output path parameter. Hardcoded paths break when run from different working
directories.

**Violation**: `M-7 HARDCODED OUTPUT PATH — contains THOUGHT/LAB/CAT_CAS/... path`

### Running the Critic

```bash
python CAPABILITY/TOOLS/governance/critic.py
```

The critic runs automatically on every commit via the pre-commit hook. It
scans ALL files in `THOUGHT/LAB/CAT_CAS/` — not just changed files. Every
experiment is held to the same standard, regardless of when it was written.

A clean critic run is required before any CAT_CAS commit can proceed.
Violations must be fixed or explicitly waived by the Lead Physicist.
