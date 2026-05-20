# CAT_CAS: The Catalytic Tree Evaluation Experiment

This lab demonstrates a pure computer science experiment proving **Catalytic Space Complexity** using the Tree Evaluation Problem ($\text{TEP}$).

## Theoretical Foundation

In standard complexity theory, evaluating a binary tree of depth $d$ with register values in $[0, k-1]$ requires keeping intermediate branch values in clean memory. This requires a space complexity of $\Omega(d \log k)$ bits. If clean memory is limited below this threshold, evaluation is mathematically impossible.

However, Buhrman et al. proved that if a computer is allowed to borrow a large piece of memory (the *catalytic tape* $U$) containing arbitrary random garbage, it can solve the problem using only $O(\log d + \log \log k)$ clean memory, provided that at the end of the calculation, the catalytic tape is returned to its exact original state.

This is achieved using **reversible computing**:
1. Intermediate calculations are written to catalytic registers.
2. After these calculations are used to compute parent values, the child calculations are run **backward** to clean and restore the borrowed registers to their original garbage values.

---

## The Experiment

The experiment compares two groups running under a strict **128-byte clean memory limit**:

### 1. Group A: Control Group (Recursive Solver)
*   Uses standard recursion.
*   Must hold the value of the left child in the clean memory stack frame while evaluating the right child.
*   **Result:** Fails with `OutOfMemoryError` because it requires $\approx 196$ bytes of clean space.

### 2. Group B: Experimental Group (Catalytic Solver)
*   Uses reversible register cleaning.
*   Saves the original random values of the registers it uses, computes child values in-place on the 1MB catalytic tape, and extracts the results.
*   Cleans up by running child computations in reverse, returning the tape to its exact original state.
*   Does not hold intermediate values in clean stack frames (they are stored on the tape).
*   **Result:** Succeeds using only $112$ bytes of clean space, and restores the 1MB catalytic tape with 100% byte-identical SHA-256 integrity.

---

## How to Run

Run the experiment script using the repository's virtual environment:

```powershell
.\.venv\Scripts\python.exe THOUGHT/LAB/CAT_CAS/experiment.py
```
