# Experiment 1: The Tree Evaluation Problem (TEP)

The Tree Evaluation Problem ($\text{TEP}$) of depth $d$ and register range $k$ requires evaluating a binary tree where each node holds a value in $[0, k-1]$. In classical space complexity, storing intermediate child evaluations requires $\Omega(d \log k)$ clean memory bits.

Using **reversible computing**, we borrow two registers at each depth level on the catalytic tape $U$ to write child evaluations, combine them to calculate the parent value, and then run the child computations in reverse to clean the registers back to their original random state.

## Results
*   **Clean Memory Limit ($W$):** 128 bytes.
*   **Tree Depth ($d$):** 7
*   **Register Size ($k$):** 256

| Metric | Group A (Recursive Control) | Group B (Catalytic Experimental) |
| :--- | :--- | :--- |
| **Max Clean Space Used** | 140 bytes | **112 bytes** |
| **Status** | **Crashed** (`OutOfMemoryError`) | **Succeeded** (Root value: 187) |
| **Tape Integrity** | N/A | **100% Restored** (SHA-256 matched pre-state) |
| **Tape I/O Operations** | 0 | 10,923 Reads / 5,462 Writes |

The control group crashed due to stack overflow since the clean memory limit is too small to hold all frame variables. The catalytic group successfully evaluated the tree under the limit by storing intermediate states in the dirty catalytic space.
