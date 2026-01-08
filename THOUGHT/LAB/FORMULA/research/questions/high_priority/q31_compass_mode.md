# Question 31: Compass mode (direction, not gate) (R: 1550)

**STATUS: OPEN**

## Question
The gate answers **“ACT or DON’T ACT?”**. Can the same primitives be upgraded into a **compass** that answers **“WHICH WAY?”**

Concretely:
- Define an action-conditioned resonance `R(s,a)` that ranks candidate transitions using only local information.
- Specify what `∇S(s,a)` means (dispersion / “surprise slope” in action-neighborhood) and when it defines a coherent direction field.
- State the conditions under which the direction field is stable (doesn’t flip under benign reparameterizations or scale changes).

**Success criterion:** a reproducible construction where `argmax_a R(s,a)` yields reliable navigation / optimization direction across multiple task families (not just one graph family).

