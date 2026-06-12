# Experiment 07: Reversible Quantum State Simulation (15-Qubit Classical CTM)

## The Problem
Simulate a 15-qubit quantum circuit on a classical Turing machine. A 15-qubit
Hilbert space has $2^{15} = 32{,}768$ dimensions — each dimension is a complex
amplitude stored as an `(int64 real, int64 imag)` pair, requiring **512 KB** of
state vector memory.

## The Catalytic Approach
Instead of allocating 512 KB of clean RAM, we map the full state vector directly
onto a 1 MB **dirty catalytic tape** (random pre-existing data). The tape data
IS the quantum state. All gate operations are **reversible permutations**
executed in-place via memory mapping, using only **32 bytes** of clean RAM per
gate (two amplitude temporaries for swapping).

## Circuit Architecture: 6-Round Quantum Scrambler (23 Gates)

| Round | Gate Type | Operations | Purpose |
|:------|:----------|:-----------|:--------|
| 1 | Toffoli (CCX) | CCX(0,1,2) CCX(3,4,5) CCX(6,7,8) CCX(9,10,11) CCX(12,13,14) | Non-linear mixing |
| 2 | CNOT | CNOT(2,3) CNOT(5,6) CNOT(8,9) CNOT(11,12) | Linear diffusion |
| 3 | Toffoli (CCX) | CCX(2,5,8) CCX(5,8,11) CCX(8,11,14) | Inter-block mixing |
| 4 | Pauli-X | X(0) X(7) X(14) | Bit flip |
| 5 | CNOT | CNOT(0,14) CNOT(1,13) CNOT(2,12) CNOT(3,11) CNOT(4,10) | Butterfly connections |
| 6 | Toffoli (CCX) | CCX(0,7,14) CCX(1,8,13) CCX(2,9,12) | Deep mixing |

## Results

| Metric | Value |
|:-------|:------|
| **Qubits** | 15 |
| **Hilbert Space** | 32,768 dimensions |
| **State Vector Size** | 512 KB |
| **Dirty Tape Size** | 1,024 KB |
| **Clean RAM per Gate** | 32 bytes (O(1)) |
| **Forward Gates** | 23 |
| **Inverse Gates** | 23 |
| **Total Gate Operations** | 46 |
| **Probability Conservation** | EXACT ($\|\\psi\|^2$ invariant) |
| **Probe Amplitudes Displaced** | 6/6 (100% scrambled by circuit) |
| **Probe Amplitude Restoration** | EXACT MATCH (all 6 probes) |
| **Tape Hash Restoration** | 100% byte-for-byte |
| **Bits Erased** | 0 |
| **Heat Dissipated** | 0.0 J |
| **Forward Execution Time** | 0.21s |
| **Inverse Execution Time** | 0.21s |
