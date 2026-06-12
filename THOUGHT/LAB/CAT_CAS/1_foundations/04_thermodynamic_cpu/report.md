# Experiment 4: Thermodynamic Reversible CPU

This experiment compares the thermodynamic efficiency of irreversible and reversible 8-bit ripple-carry addition.

## Physical Model: Landauer's Limit
Each logically irreversible erasure of 1 bit of information dissipates a minimum thermodynamic energy of:
$$E = k_B T \ln(2)$$
At room temperature ($29.15^\circ$C / $293.15$ K), this equates to approximately $2.8054 \times 10^{-21}$ Joules per bit erased. By executing addition strictly using reversible Toffoli gates and running the reverse pass to clean the intermediate carry and sum registers, we avoid all logical erasures.

## Results
Inputs: $A = 187$ ($0\text{b}10111011$), $B = 94$ ($0\text{b}10111100$). Expected 8-bit Sum: $25$.

| Metric | Group A (Irreversible Control) | Group B (Reversible Experimental) |
| :--- | :--- | :--- |
| **Computed Sum** | 25 | 25 |
| **Information Erased** | **31 bits** | **0 bits** |
| **Landauer Energy Dissipation** | **$8.6968 \times 10^{-20}$ J** | **$0.0$ J** |
| **Status** | Correct Sum (Lossy) | **Correct Sum (Zero-Heat)** |

Because the reversible CPU computes the sum using only bijective operations and unwinds all intermediate operations, it erases exactly zero bits of information, operating at the physical Landauer limit of zero thermodynamic heat dissipation.
