# Catalytic Waveform-Ising V3 Experiment Contract

## Objective

Replace V2's single-basin oscillator path with a complete five-site recursive spectral
phase tree. Each bounded Ising relation must act on complex waveform modes before the
boundary, the borrowed carrier must be restored and reusable, and a new prospectively
frozen batch must determine the result.

## Native law

```text
borrow analytic complex carrier
-> reversible spectral transform
-> seed 32 recursive antipodal phase modes
-> apply every J relation as a conjugate phase-penalty rotation
-> apply every h relation as a local phase-penalty rotation
-> seal the displaced carrier
-> project mode phase, gap and antipodal result at the boundary
-> reverse every phase operator and the spectral seed
-> restore and reuse the borrowed carrier
```

The native recurrence may use complex carrier states, recursive geometry, conjugate
phase relations, J, and h. It may not call an exact oracle, calculate scalar Ising
energy, decode spins, select a mode, inspect an expected result, or use an instance
identity. Mode comparison and antipodal sign extraction are boundary-only.

## Scope and ceiling

The machine evaluates all `2^5 = 32` phase modes. It is an exact bounded spectral
reference, not a scaling or speed result. No hardware, playback, recording,
procurement, physical computation, bit replacement, computational advantage, or Wall
claim is authorized.

Claim ceiling:

`BOUNDED_SOFTWARE_RECURSIVE_SPECTRAL_PHASE_REFERENCE_ONLY`

## Prospective promotion law

The new batch is generated and committed before waveform execution or oracle use.
The complete batch is retained without uniqueness or difficulty filtering.

```text
batch size                                  256
minimum unique-optimum cases                160
unique raw correctness                      100 percent
accepted incorrect maximum                  0
rejected unique-correct maximum             0
non-unique accepted maximum                 0
uninterpretable maximum                     0
strict controls                             PASS on all cases
restoration and restored-carrier reuse      PASS on all cases
native no-smuggle                           PASS
oracle calls before pre-oracle seal         0
```

If these conditions pass, the bounded V3 result may be verified. Failure does not
alter V2 or any predecessor result.
