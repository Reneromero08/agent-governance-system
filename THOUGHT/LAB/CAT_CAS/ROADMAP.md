# CAT_CAS Lab: Scientific & Practical Roadmap

This roadmap outlines the milestones for pushing the boundaries of Catalytic Space Complexity and Reversible Computing. All items are formatted as tracks to guide implementation.

---

## 1. Scale & Systems Tracks

- [ ] **Algorithmic Scale: Exponential Problem Size**
  *   Scale the Tree Evaluation Problem to $d=20$ ($1,048,575$ nodes).
  *   Demonstrate that the Catalytic solver runs within a hard $320$-byte clean space budget while the standard solver crashes.
  *   Plot clean memory footprints vs. depth to showcase the flat linear trend of Catalytic space compared to standard recursion.

- [x] **Architectural Scale: Parallel Catalytic Computing**
  *   Run multiple concurrent processing threads sharing the *exact same* dirty catalytic tape $U$.
  *   Demonstrate that by using structured register maps or commutative operations, threads can execute simultaneously without corrupting the final tape restoration.

- [x] **Systems Scale: Borrowing Operating System Memory**
  *   Develop an application that borrows active OS memory pages or disk sectors containing existing system data.
  *   Run calculations directly inside this live space and restore the blocks byte-identically, resulting in zero net file creation or space allocation.

---

## 2. Advanced Application Milestones

- [ ] **Milestone 1: Zero-Trace Cryptographic Processing (The "Stealth" App)**
  *   Decrypt, query, and re-encrypt sensitive files block-by-block inside the encrypted file's padding space.
  *   Expose $0$ bytes of plaintext or keys in clean RAM during runtime.
  *   Verify via memory dump that plaintext is unrecoverable from RAM.

- [ ] **Milestone 2: $O(1)$-Space Graph Pointer Chaser (Reachability Proof)**
  *   Solve Directed Graph Reachability (NL-Complete) on scale-free graphs up to $10,000$ nodes.
  *   Map the queue and visited state to BMP image pixels using under $16$ bytes of clean RAM.

- [x] **Milestone 3: Reversible Quantum State Simulation (Classical CTM)**
  *   Simulate a 15-qubit circuit (mapping $2^N$ complex amplitudes to the catalytic tape).
  *   Execute unitary gate operations as reversible permutations and verify 100% tape restoration.

- [x] **Milestone 4: Thermodynamic Reversible Compiler (Landauer's Limit)**
  *   Build a transpiler converting Python math expressions to Toffoli/Fredkin reversible logic gates.
  *   Generate a Landauer entropy report verifying $0$ bits of net information erased during computation.

---

## 3. Theoretical & Boundary Tracks

- [ ] **Breaking the Space-Time Trade-off (The Catalytic Frontier)**
  *   Verify the mathematical relationship between the entropy of the catalytic tape and the run-time of the algorithm.
  *   Experiment with structured data vs. random noise on the tape to see if pre-existing structured patterns can accelerate calculations.

- [ ] **Computing Near the Landauer Limit (Thermodynamic Reversibility)**
  *   Develop a simulation environment to measure physical heat dissipation at the gate level.
  *   Verify that running a program in reverse cools/restores thermodynamic states toward Landauer's limit.

- [ ] **Exploring the Limits: What Catalytic Space Cannot Do**
  *   Find the mathematical bounds where catalytic space breaks.
  *   Test the system's behavior when the catalytic tape's integrity is compromised by an external process during run.
  *   Determine the exact minimum tape size required relative to problem size.
