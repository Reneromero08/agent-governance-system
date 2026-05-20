# Experiment 5: Multi-Bit Reversible Arithmetic & Logic Compiler

To demonstrate compiler scaling, we built a fully dynamic compilation engine that translates high-level expressions involving 8-bit variables (like `X`, `Y`, `Z`, `W`) and operations (addition `+`, bitwise `^`, `&`, `|`, `~`) down to bit-level reversible gates.

## Carry & Register Reclamation
When compiling additions like `temp = U + V`, the compiler automatically inserts an un-computation block to clean up the intermediate carry bits `C_1..C_8` back to 0. 

A final global un-computation cleans up all intermediate multi-bit registers, ensuring that the only modified registers at the end of execution are the output registers (`OUT_0..OUT_7`) and the CPU's internal gate execution stack is fully rewound.

## Scaled Compiler Verification Results
Inputs: $X = 187$, $Y = 94$, $Z = 51$, $W = 12$.

| Expression | Expected Result | Compiled Gates | Irreversible Bits Erased | Reversible Bits Erased | Reversible Energy (J) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `(X & Y) ^ ~Z` | 214 | 40 | 24 | **0** | **0.0 J** |
| `((X | Y) & Z) ^ W` | 63 | 48 | 24 | **0** | **0.0 J** |
| `~(X & Y & Z) ^ (W | X)` | 82 | 72 | 40 | **0** | **0.0 J** |
| `X + Y` | 25 | 88 | 16 | **0** | **0.0 J** |
| `(X + Y) & ~Z` | 8 | 112 | 32 | **0** | **0.0 J** |
| `((X + Y) ^ Z) & (W + X)` | 2 | 200 | 48 | **0** | **0.0 J** |

By dynamically nesting sub-uncomputation blocks (such as carry cleaning) and performing a global unwind of intermediate term registers, the compiler generates arbitrary logical and arithmetic circuits that run with **zero entropy change** and **zero net bits of information erased**.
