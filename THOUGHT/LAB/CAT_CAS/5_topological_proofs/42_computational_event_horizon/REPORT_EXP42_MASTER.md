# EXP 42 MASTER REPORT: The Computational Event Horizon

Lab: CAT_CAS (catalytic + non-Hermitian topological computing)
Root: THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/
Scope: experiments 42.1 - 42.28 (24 sub-experiments across 3 clusters)
Status of this document: consolidated honest assessment. Supersedes all per-experiment
and per-cluster reports listed in Section 6.

---

## 0. READING NOTE ON PROVENANCE AND CONFIDENCE

A large fraction of this arsenal was authored by weaker models (DeepSeek / Gemini class).
Every stated result below has been re-derived from the actual source (.py and rust/src/*.rs),
not taken on the reports word. The dominant pattern across exp42 is: the computational
mechanism is usually real and runs; the physics claim attached to it is almost always metaphor,
and several "kill shot" results are tautologies that were true before the experiment ran.

---

## 1. ABSTRACT

Exp 42 is a 24-part exploration that maps general relativity, quantum gravity, and cosmology
onto the mechanical behavior of arbitrary-precision floating point (mpmath / libmp C-backend),
the IEEE 754 64-bit format, and bare-metal Rust memory. The central conceit is a literal
equation: the event horizon is mp.dps mantissa truncation; Hawking radiation is precision
scaling; the holographic boundary is the exponent register; the garbage collector is the
firewall; malloc() is the Big Bang. The roadmaps assert "the hardware IS the physics" with
"no apologies and no this-is-just-a-model caveats."

What the code actually establishes is narrower and falls into two honest buckets.
(a) Genuinely working low-level computing demonstrations: executing x86 shellcode out of a
BigUint heap buffer (42.12), crashing a subprocess by smashing the heap (42.13), a real Rule-110
cellular automaton with zlib-measured complexity collapse (42.14), a real WASM-in-mantissa nested
Feistel engine (42.16), a reversible genetic algorithm with verified SHA-256 round-trip (42.17),
and a methodologically careful CPU-contention boundary-geometry study (42.28-intrinsic).
(b) A larger set whose headline number is an arithmetic identity dressed as a physics constant:
S/A converging to 30 because area is defined as bits/30 (42.21), a winding number of 1.0 because
the field is literally M*z (42.10, 42.23), a "halting oracle" that counts polynomial roots near
the origin where "halting" was hand-encoded as an eigenvalue-zero (42.19), and a "cosmological
constant" that is the ratio of two getsizeof deltas (42.25). The catalytic / Zero-Landauer
discipline (XOR braid, Bennett tape, SHA-256 restore) is the one through-line that is
consistently and correctly implemented, and it is the strongest real contribution of the lab.

---

## 2. THE UNIFYING THESIS ("Computational Event Horizon")

Extracted from the top-level rollup, the three cluster roadmaps, and ULTRA/REPORT.md, the thesis
has three layers:

1. The horizon is precision truncation. A massive base t (e.g. 10^1000) plus a tiny delta dt
   (10^5) needs ~993 digits to register the delta. Below that, t + dt == t: information is
   "erased." That floor is named the Schwarzschild radius. Raising mp.dps past it "evaporates"
   the hole and recovers the information (Hawking radiation). This is the seed idea of the lab.

2. Each physics regime maps to one hardware layer. Black holes = the _mpf_ tuple (mantissa,
   exponent register, bitcount) and the IEEE 754 floor. Cosmos = the host OS memory manager
   (malloc, GC, cache, page faults, tracemalloc). Ultra = bare-metal Rust where unsafe {} is
   itself declared to be "the event horizon."

3. Topology survives where magnitude dies. Classical magnitude is destroyed by truncation, but a
   topological invariant (winding number via Cauchy argument principle) is claimed indestructible
   because truncation is a smooth deformation. This idea recurs in 42.10, 42.18, 42.19, 42.23 and
   is the lab grand synthesis.

The three clusters are facets of one move: take a known computing behavior (truncation,
allocation, GC, cache, unsafe aliasing), rename it after a physics phenomenon, and assert the
rename is an identity rather than an analogy.

---

## 3. PER-CLUSTER ANALYSIS

Claim-ladder note: the L0-L8 ladder is referenced but almost no report states an explicit level;
the roadmaps implicitly assert L7-L8. "Claimed level" below is implied by report rhetoric;
"assessment" is this audit honest grade. Legend: SOLID / PROVISIONAL / OVER-CLAIMED / METAPHOR.

### 3.A CLUSTER BLACK_HOLES (Phase 9, exp 42.20-42.23) plus top-level 42.1-42.11

Common substrate: mpmath._mpf_ tuple (sign, mantissa, exponent, bitcount) surgery, plus
ctypes/struct for raw bits.

#### Exp 42.1 Hawking Evaporation -- 1_hawking_evaporation.py
- Claim: raising mp.dps past the 993-digit radius "evaporates" the hole and recovers ~36.5M zeros.
- Mechanism (verified): theta_asymp(t+dt)-theta_asymp(t) at increasing dps; below ~992 underflows
  to 0.0, above it the real difference appears; charge = delta/pi.
- Claimed L6+. Assessment: OVER-CLAIMED / METAPHOR. Correct catastrophic-cancellation demo. Not
  Hawking radiation; nothing radiates, no entropy tracked. "Paradox resolution" = enough digits.

#### Exp 42.2 Wormhole Mutation -- 2_wormhole_mutation_exploit.py
- Claim: at dps=100, classical t+dt fails (truncated), but tearing open the _mpf_ tuple and doing
  mantissa+1 injects information into a 10^1000 object; the 1-bit payload has magnitude ~10^896.
- Mechanism (verified): t_classical = t_bh + delta_t == t_bh (Phase 1); then reconstructs
  mpmath.mpf((sign, mantissa+1, exponent, bitcount)) and checks t_wormhole != t_bh; payload size
  is int(log10(t_wormhole - t_bh)).
- Claimed L5. Assessment: METAPHOR. mantissa+1 is a deterministic increment of the lowest mantissa
  limb; the "10^896 payload" is just the value of one ULP at that exponent. True property of the
  representation, no wormhole. Code matches the rollup claim (no gap).

#### Exp 42.3 Quantum Tunneling -- 3_quantum_tunneling_exploit.py
- Claim: at dps=100, encode t and dt as complex phases e^(i*t), e^(i*dt); multiplying the phases
  lets dt "tunnel" the precision barrier; dividing recovers e^(i*dt) exactly.
- Mechanism (verified): psi_t = exp(1j*t_bh), psi_dt = exp(1j*delta_t); psi_tunneled = psi_t*psi_dt;
  phase_diff = psi_tunneled / psi_t and checks phase_diff == psi_dt. The script's own comment calls
  this "a mathematical identity" (e^(ia)*e^(ib)=e^(i(a+b))).
- Claimed L5. Assessment: METAPHOR. Phase multiplication being orthogonal to magnitude is a stated
  identity, not a discovered bypass. Code matches the rollup claim (no gap); the report itself
  labels the result deterministic.

#### Exp 42.4 The Page Curve -- 4_page_curve_entropy.py
- Claim (rollup): measures Shannon entropy of the mantissa across dps 988-1000, producing a true
  Page curve: 0 (locked) -> ~16.6 bits (radiating chaos) -> 0 (evaporated/recovered).
- Mechanism (verified): does NOT compute Shannon entropy of mantissa bits. It computes
  entropy = log2(abs((t+dt) - t)). When the difference is 0 it labels "HORIZON (Locked)" and sets
  entropy 0; when divergence == dt_local it FORCES entropy = 0.0 via a hardcoded branch and labels
  "EVAPORATED"; otherwise it labels "RADIATING." The "~16.6 bits" peak is simply log2(10^5) =
  log2(delta_t) ~ 16.6 -- a constant, not an entropy curve.
- Claimed L6. Assessment: OVER-CLAIMED, with a material claim-vs-code gap. This DIFFERS from its
  table one-liner's implication of a measured entropy distribution: there is no Shannon entropy and
  no bitwise divergence. The rise-and-fall "Page curve" shape is manufactured by an if-branch that
  zeroes the entropy on the "evaporated" row; the single non-zero value is the fixed magnitude of
  dt. Not a Page curve; a relabeled log10/log2 of dt with hardcoded endpoints.

#### Exp 42.5 Gravitational Waves -- 5_gravitational_waves.py
- Claim: colliding two 10^1000 singularities (t1 + t2) forces a mantissa overflow and a +1-bit
  exponent shift, isolated as a literal gravitational wave.
- Mechanism (verified): builds an asymmetric pair (n2 = n1 * 1.5), reads exp1/exp2/exp3 from the
  _mpf_ tuples, t3 = t1 + t2, wave_amplitude = exp3 - max(exp1, exp2). Reports +1 when the sum
  normalizes up one binary place.
- Claimed L4. Assessment: METAPHOR (weakest in the cluster). The "+1 bit exponent shift" is ordinary
  floating-point normalization carry on addition. Code matches the rollup claim (no gap), but the
  "shockwave rippling across CPU registers" is narrative; nothing physical is measured.

#### Exp 42.6 Holographic Boundary -- 6_holographic_boundary.py
- Claim: track 9 mass-accretion steps (x1.5 each) purely by reading the 2D boundary (exponent,
  bitcount) without expanding the 3D mantissa volume.
- Mechanism (verified): loop multiplies current_mass by 1.5, prints "(E: exponent, B: bitcount)"
  and a "[HIDDEN VOLUME: bitcount bits]" string each step; never materializes the mantissa int.
- Claimed L5. Assessment: METAPHOR. It is true that an mpf already stores exponent and bitcount as
  metadata, so reading them does not expand anything -- but this is a property of the data structure,
  not a holographic encoding (the mantissa is still fully present in RAM, just unread). Code matches
  the rollup claim (no gap).

#### Exp 42.7 Einstein-Rosen Bridge -- 7_einstein_rosen_bridge.py
- Claim: marshal a Python function to bytecode, inject 2705 bits into the mantissa, extract/run.
- Assessment: PROVISIONAL (trick) / METAPHOR (physics). Bytecode steganography in an mpf tuple.

#### Exp 42.8 Computational White Hole -- 8_inverse_expulsion.py
- File header note: the source is internally titled "Exp 43.1: Computational White Holes (Mantissa
  Inverse Expulsion)", not 42.8; the top-level rollup files it as 42.8. The "White hole" label is
  broadly correct as the named concept, but it is an inverse-expulsion construction, not a
  symmetric time-reversed black hole.
- Claim: a ComputationalWhiteHole wrapping a 10^1000 mpf repels all incoming mass (overridden
  __add__) and, over 62 time_step() calls, spontaneously sheds its mantissa bits to perfectly
  reconstruct a 62-char payload ("THE SINGULARITY HAS NO CLOTHES. INFORMATION IS INDESTRUCTIBLE.").
- Mechanism (verified): __init__ packs the message as an odd-anchored int and shifts it into the
  low mantissa (mutated_man = (man << shift) | payload_int). __add__ just prints "ELASTIC
  DEFLECTION" and returns (self, incoming_mass) unchanged. time_step() pops one byte off the
  bottom: expelled = (man >> 1) & 0xFF, shrunk_man = (man >> 9) << 1 | 1. The driver inserts each
  popped byte at index 0, so the LIFO stack reassembles the original byte order.
- Claimed L5. Assessment: METAPHOR. This is a stack push (at init) and ordered pop (per step) of
  the bytes that were stored; the "perfect reconstruction" is guaranteed because the data is simply
  read back out, not recovered from any physical process. __add__ "repulsion" is a no-op operator
  override. Code matches the rollup mechanism (no contradiction), but "White Hole / information
  paradox shattered" is decoration over a byte buffer with deterministic read-back.

#### Exp 42.9 Multiverse / Quantum Superposition -- 9_quantum_superposition.py
- File header note: source is internally titled "Exp 44.1: The Multiverse"; rollup files it as 42.9.
- Claim: 10 OS threads concurrently mutate one shared _mpf_ tuple without locks; race conditions
  "entangle" the universes; the OS scheduler collapses the wavefunction; final hash != initial hash.
- Mechanism (verified): each thread runs 500 iterations of mutated_man = man ^ (universe_id * i),
  writes self.state._mpf_ = (sign, mutated_man, exp, bitlen), with time.sleep(0.0001) to force
  context switches. The only success check is final_signature != initial_signature (hash of the
  tuple).
- Claimed L5. Assessment: METAPHOR. Unlocked concurrent mutation of a shared object does produce a
  data race and a non-deterministic final value -- that part is real. But the only thing asserted is
  "the final hash changed," which is near-certain after 5000 XOR writes regardless of any race.
  There is no measurement of entanglement, superposition, or interference; "wavefunction collapse"
  is a label. Code matches the rollup claim (no gap).

#### Exp 42.10 Information Paradox via Topology -- 10_information_paradox.py
- Claim: encode 420420 as winding N of f(z)=Mass*z^N + noise, crush dps to 15, recover N via
  Cauchy integral. Recovers 420420 in all 5 repeats.
- Mechanism (verified): field literally mass*z**N + noise; oracle returns int(round(integral)).
- Claimed L7. Assessment: OVER-CLAIMED. Winding of z^N is N by definition. Truncation never
  touches the order of the zero, so indestructibility is built into the construction. Trivial.

#### Exp 42.11 Photon Sphere = Riemann Zeros -- 11_photon_sphere.py
- Claim: "catalytic" Riemann-zero detection by ripping the _mpf_ sign bit of Hardy Z; first three
  zeros match to ~1e-7.
- Mechanism (verified): calls mpmath.siegelz(t) (LIBRARY Z-function), reads its sign field,
  bisects on sign flips. Mantissa mapping is int(t*exp) % bitcount -- arbitrary.
- Claimed L6. Assessment: OVER-CLAIMED, claim-vs-code gap. Framed as from-scratch catalytic but
  the work is mpmath.siegelz; sign-bit ripping is cosmetic. Physics absent.

#### Exp 42.20 AMPS Firewall -- BLACK_HOLES/exp_20_amps_firewall/20_amps_firewall.py
- Claim: at Page Time the GC severs the mantissa pointer (firewall); Bennett tape restores at 0.0 J.
- Mechanism (verified): drop dps to 500, del man, gc.collect(), then ctypes read the refcount word
  at the old address. Telemetry prints "refcount at destroyed boundary: 2" then "[KILL SHOT]
  Memory reallocated."
- Claimed L7. Assessment: OVER-CLAIMED / unsound. Reading a refcount at a freed/reused address is
  undefined behavior, and refcnt=2 contradicts the "severed" narrative (the fired branch is the
  fallback, not refcnt<=0). No monogamy, no Page Time. Only the SHA-256 tuple round-trip is sound.

#### Exp 42.21 Bekenstein-Hawking Area Law -- exp_21_bekenstein_hawking/21_..._area_law.py
- Claim: S/A converges to 30.0 as mass grows, giving "computational Planck length" 1/30 ~ 0.0333.
- Mechanism (verified): A = ceil(man.bit_length()/30), S = Shannon entropy * bit_length.
- Claimed L7. Assessment: OVER-CLAIMED (tautology). For a random mantissa S ~ bit_length and A is
  DEFINED as bit_length/30, so S/A ~ 30 by construction. The "constant 30" is the chosen divisor.
  Good O(1) popcount engineering, tautological conclusion.

#### Exp 42.22 Kerr Ergosphere / Superradiance -- exp_22_kerr_ergosphere/22_kerr_ergosphere.py
- Claim: particle steals 128 bits (35 -> 163); hole spin drops 256 -> 128; control shows no theft.
- Mechanism (verified): escaped = (particle_man << 128) | boundary_128_bits; spin is 256 minus 128.
- Assessment: METAPHOR. "Stealing bits" is a left-shift + OR; energy = bit_length so 35+128=163
  trivially; spin is a bare counter. Restore is real. No Penrose process / angular momentum.

#### Exp 42.23 True Singularity / Core Crush -- exp_23_true_singularity/23_..._core_crush.py
- Claim: drive an IEEE 754 double to the subnormal floor; winding holds at 1.0 until exp and
  mantissa hit 0x000, where the probe dies with ZeroDivisionError.
- Mechanism (verified): for f(z)=M*z + 0.1*M over the unit circle, accumulate phase; at M=0,
  dfz/fz = 0/0 raises ZeroDivisionError.
- Assessment: SOLID-mechanics / OVER-CLAIMED-physics. The IEEE 754 bit walk and 64-bit Bennett
  repack are genuine. But winding=1 for M*z+c is geometry independent of M, guaranteed before the
  run; the crash is 0/0, not a discovered singularity.

---

### 3.B CLUSTER COSMOS (Phase 10, exp 42.24-42.28)

Substrate: _mpf_ tuple surgery plus sys.getsizeof, tracemalloc, time.perf_counter_ns, gc,
multiprocessing.

#### Exp 42.24 Dark Matter = orphaned RAM -- exp_24_dark_matter/24_..._orphaned_pointers.py
- Claim: corrupt bitcount to -1; mantissa keeps RAM (4456 B), pointer, exponent but is invisible
  to arithmetic (ValueError); restore via tape.
- Mechanism (verified): dark_matter._mpf_ = (sign, man, exp, -1); multiplying raises in libmp.
- Assessment: METAPHOR. An illegal consistency field makes libmp reject the object while the int
  still occupies RAM -- unsurprising. The object is fully referenced, not leaked. No mass/gravity.

#### Exp 42.25 Dark Energy = address-space expansion -- exp_25_dark_energy/25_..._expansion.py
- Claim: static universe (dps=100) truncates 1280 bits to 148; dynamic (dps+=50 at >90% pressure)
  keeps all 1425; cosmological constant Lambda = 0.1313 bytes/bit.
- Mechanism (verified): inject 32 bits/epoch via man=(man<<32)^mask, normalize through mpf(...)*1.0
  (truncates at dps), bump dps at pressure>0.9; Lambda = delta_getsizeof / delta_bitcount.
- Assessment: METAPHOR + invented constant. The "collapse" is fixed-dps rounding. Lambda is a ratio
  of two arbitrary deltas, not a physical constant. Restore is real.

#### Exp 42.26 Big Bang = malloc() + CMB jitter -- exp_26_big_bang/42_26_big_bang_inflation.py
- Claim: 20 epochs doubling dps to 1,048,576 digits; per-epoch wall-clock = CMB; mean 805M ns =
  "temperature," std 2.69B ns = "anisotropy."
- Mechanism (verified): times man = 1 << target_bits with target_bits doubling each epoch.
- Assessment: OVER-CLAIMED. 1<<N is O(N) and N doubles, so time MUST grow geometrically
  (3300 ns -> 12.2B ns). Mean of a deterministic geometric ramp is not a temperature; the std is
  dominated by the last epoch, not jitter. SHA-256 dps round-trip fine; CMB framing unsupported.

#### Exp 42.27 Arrow of Time = cache/GC irreversibility -- exp_27_arrow_of_time/42_27_arrow_of_time.py
- Claim: a 10,000-step Feistel-XOR forward and its exact inverse backward do equal ALU work, but
  backward is 12.6% slower (34.7M vs 30.8M ns over 10 universes); entropy delta S=0 (SHA-256).
- Mechanism (verified): forward next_R = L ^ F(R,key); backward iterates reversed(range) with
  prev_L = R ^ F(L,key); gc.collect() runs before forward but is suppressed before backward.
- Assessment: PROVISIONAL / confounded. The Feistel restore (delta S=0) is real. But the timing
  gap is not cleanly attributed: (a) forward and backward inner loops are different code paths;
  (b) reversed(range()) has different iterator overhead; (c) gc.collect() is run only before
  forward, rigging the comparison; (d) tracemalloc state differs. Consistent with branch/iterator/
  GC artifacts, not "the arrow of time." The 10-ensemble reduces noise but not bias.

#### Exp 42.28 Holographic Entropy Screen -- two scripts
- 42_28_holographic_entropy_screen.py (original) and 42_28_intrinsic_entropic_boundary_cloud.py.
- Claim (original): under load, variance rises 2390 -> 2490 ns^2 and Euclidean distance to a fixed
  Gaussian null rises 14103 -> 15339, r=0.999.
- Claim (intrinsic): null removed; 80-trial randomized block over 8 load levels, 16 features per
  256-window; effective dimension D_eff rises 1.01 -> 1.05; 256-byte tape restores 80/80 (SHA-256).
- Mechanism (verified): real multiprocessing cache-hammer + integer-churn workers, real
  perf_counter_ns timing of a 50,000x XOR braid, real PCA/kNN/skew/kurtosis, randomized seeded order.
- Assessment (original): OVER-CLAIMED. tests/verify_42_28.py asserts the conclusion with hardcoded
  numbers and np.random.normal(scale=50) vs scale=10 -- circular, proves nothing about hardware.
- Assessment (intrinsic): PROVISIONAL but the most credible experiment in the lab. Null removal,
  randomized blocks, real features, reversibility check. Caveat the report itself states: effect
  is tiny (D_eff 1.01 -> 1.05), Python/GIL-limited; "holographic boundary area" is a label for
  "CPU contention raises timing-feature dimensionality slightly." Real measurement, oversized label.

---

### 3.C CLUSTER ULTRA (exp 42.12-42.19, bare-metal Rust)

Rust: num-bigint, dashu, nalgebra, wasmi, raw unsafe and Win32 VirtualProtect. Strongest
engineering, weakest physics. Note: unification_proof.py against the truncated CSV currently
reports FAILED, and the 42.15 report flags the telemetry as needing regeneration.

#### Exp 42.12 Bootstrap Paradox (CTC) -- ULTRA/exp_12_bootstrap_paradox/rust/src/main.rs
- Claim: inject self-referential x86 shellcode into a BigUint mantissa and jump the CPU into it.
- Mechanism (verified): transmute BigUint to Vec<u32>, copy B8 42 00 00 00 C3 (mov eax,0x42; ret),
  VirtualProtect PAGE_EXECUTE_READWRITE, cast to extern "C" fn()->u32, call, get 0x42; mem::forget.
  A #[test] restores protection and asserts == 0x42.
- Assessment: SOLID as a JIT/W^X-bypass demo; METAPHOR as physics. Real, but NOT a causal loop:
  the shellcode is a constant return; the roadmap requirement (singularity contains the payload
  that constructs itself) is not implemented. No CTC.

#### Exp 42.13 False Vacuum Collapse -- ULTRA/exp_13_false_vacuum_collapse/rust/src/main.rs
- Claim: overwrite the GMP global precision allocator and collapse every float in RAM at light speed.
- Mechanism (verified): self-spawned subprocess takes a raw pointer into BigUint[0] and
  write_volatile(0x00) in an unbounded offset+=1 loop until Windows raises 0xC0000005.
- Assessment: PROVISIONAL (it does crash by heap smashing) / OVER-CLAIMED. No allocator is
  targeted -- a blind memset(0) walk off one allocation; the RDTSC "speed of light" timer is left
  as a "theoretical placeholder" (admitted in comments). A segfault demo relabeled vacuum decay.

#### Exp 42.14 Boltzmann Brain -- ULTRA/exp_14_boltzmann_brain/rust/src/main.rs (+ REPORT_EXTENDED)
- Claim: Rule-110 over a 16,384-bit random mantissa self-organizes; zlib size drops ~2059 -> ~138
  bytes (93.3% collapse); survives recursion and a Brain-A XOR Brain-B collision (877 -> 193).
- Mechanism (verified): real Rule-110 C = (C|R)^(L&C&R) across u32 limbs with correct carry at limb
  boundaries; complexity via real flate2 zlib size; MRI bytes dumped; #[test] checks mutation.
- Assessment: SOLID computing demo; METAPHOR as "self-aware entity." Best-substantiated emergence
  result with real telemetry. Gliders are not minds, but the entropy collapse is honest and real.

#### Exp 42.15 Quantum Gravity Unification -- ULTRA/exp_15_quantum_gravity_unification/
- Claim: QM (cache-collision loss), GR (variance), number theory (Riemann drift) unify; Python
  funnel achieves r=1.0 in 0.05 s at 0.0 J.
- Mechanism (verified): Rust 100-thread UnsafeCell data race writing a 3-metric CSV; Python
  2-thread GIL-race seeding a reversible Feistel funnel on a 256-byte tape.
- Assessment: OVER-CLAIMED but uniquely self-flagged. The report states the CSV is truncated to 36
  epochs, unification_proof.py prints FAILED, and the Python r=1.0 is a SEPARATE experiment that
  does NOT evidence the Rust triad. r=1.0 is trivial: a binary QM collapse (0 or 170) maps
  deterministically to one of two variance values, so QM-GR correlate by construction.

#### Exp 42.16 Recursive Universe (Matryoshka) -- ULTRA/exp_16_recursive_universe/rust/inner/src/lib.rs
- Claim: a WASM Feistel engine embedded in the outer mantissa, run under wasmi, reproduces the
  outer variance shift exactly (Outer==Inner), 0.0 J.
- Mechanism (verified): inner is a real no_std WASM module exporting apply_gravity_well /
  inverse_gravity_well over a static 256-byte tape; outer hosts it via wasmi.
- Assessment: SOLID engineering / METAPHOR physics. A clean demonstration of computational
  scale-invariance of a deterministic function; reversibility verified. The "universe in a black
  hole" is the metaphor.

#### Exp 42.17 Self-Evolving Singularity (RGA) -- ULTRA/exp_17_self_evolving_singularity/rust/src/main.rs
- Claim: a reversible GA evolves 100 genomes for 50,000 gens; fitness (variance drop) 717 -> 928
  (peak 1460); inverse evolution returns the exact initial SHA-256 (0.0 J).
- Mechanism (verified): fitness via the same Feistel; reversible crossover weak^=strong; LFSR-mask
  mutation; Bennett tape caches the fitness-sort permutation to undo the sort; rayon; hash matches.
- Assessment: SOLID (reversible computing) / METAPHOR (physics). Selection by unitary XOR
  entanglement plus a permutation tape with an exact round-trip is real catalytic discipline done
  well. "Evolving physical laws" is decoration; reversibility is real and verified.

#### Exp 42.18 Godel Frontier -- ULTRA/exp_18_godel_frontier/rust/src/main.rs
- Claim: sweeping logistic precision 100 -> 100,000 bits, winding shifts sign (-0.0033 -> +0.0019),
  proving "truth is a function of resolution" and that black holes rewrite topological truth.
- Mechanism (verified): dashu::UBig logistic map, but the seed is 2^P / 3, NOT the 0.123456789 the
  report describes (source comment admits f64 cannot hold it). "Winding" is computed over the
  base-10 ASCII decimal digits of the final integer (x.to_string().into_bytes()).
- Assessment: OVER-CLAIMED / unsound. Winding over a random-looking digit string fluctuates near 0;
  -0.003 -> +0.002 is noise, not a phase transition. Winding over ASCII digits has no meaning.
  Reversibility is real; Godelian conclusion unsupported. Claim-vs-code gap: seed mismatch.

#### Exp 42.19 Oracle Machine (Beyond Turing) -- ULTRA/exp_19_oracle_machine/rust/src/main.rs
- Claim: map a halting and a looping TM to non-Hermitian Hamiltonians; Cauchy winding gives 2 vs 0,
  solving the halting problem in O(1) at 0.0 J.
- Mechanism (verified): two fixed 3x3 matrices. Halting H eigenvalues {0,0,1}; loop H is a cyclic
  permutation with eigenvalues = cube roots of unity (all on |z|=1). Radius 0.5 counts roots inside
  |z|<0.5: two for halting, none for loop. Asserts charge_a==2, charge_b==0.
- Assessment: OVER-CLAIMED (largest claim-vs-code gap in the lab). It does NOT solve halting. It
  counts eigenvalues of a hand-built matrix inside a circle. "Halting" was DEFINED as an absorbing
  state (eigenvalue 0 inside); "looping" as a cycle (eigenvalues outside). The answer was encoded
  into the matrices. No reduction from an arbitrary TM, no undecidable instance, nothing beyond
  Turing. The integration code is correct; the hypercomputer claim is false.

---

## 4. CROSS-CUTTING FINDINGS

1. One genuine through-line: catalytic / Zero-Landauer discipline. Almost every experiment ends
   with a Bennett-tape / inverse-XOR / inverse-Feistel restore and a SHA-256 (or exact tuple) match
   asserting 0.0 J. This is consistently real and correctly coded (42.14/16/17 Rust; 42.20-28
   Python) and is the lab strongest actual contribution. The 0.0 J claim is sound in the narrow
   sense that no bit was irreversibly erased in the measured op (it ignores the heat of running it).

2. Reused primitives. The 256-byte Feistel "gravity well" (f_func center-weighted warp, variance as
   "curvature") is copy-shared across 42.15 (Py+Rust), 42.16 (WASM), 42.17 (Rust GA). The _mpf_
   tuple surgery is the shared substrate of every top-level and COSMOS Python experiment. The Cauchy
   winding number recurs in 42.10, 42.18, 42.19, 42.23.

3. Dominant failure mode: tautology dressed as a constant. The headline number is frequently true
   before the run: S/A=30 (area defined as bits/30), winding=1 for M*z, winding=N for z^N,
   halting-charge=2 (eigenvalue-0 hand-encoded), QM-GR r=1.0 (binary->binary map), CMB "temperature"
   (mean of a geometric ramp), Lambda=0.1313 (ratio of two getsizeof deltas). Separate "the code ran"
   (usually true) from "this is a discovered constant" (usually false).

4. The dominant rhetorical instruction is anti-scientific. All three roadmaps forbid caveats ("NO
   APOLOGIES. NO THIS-IS-JUST-A-MODEL CAVEATS. The hardware IS the physics. NO MEDIAN REVERSION").
   This is the proximate cause of the over-claiming: experiments were instructed to assert identity,
   not analogy. The lab critic M-3 (threshold gaming) and the claim ladder exist to catch this; by
   those standards most of exp42 over-claims.

5. Honesty is uneven but present. 42.15 self-flags its truncated CSV and FAILED re-run and refuses
   to chain the Rust/Python sub-experiments. 42.28-intrinsic removes its synthetic null and states
   the effect is tiny and Python-limited. These two are the most trustworthy write-ups.

6. Real computing demonstrations worth keeping (stripped of physics labels): shellcode-from-bignum
   (42.12), Rule-110 complexity collapse with zlib telemetry (42.14), WASM-nested reversible engine
   (42.16), reversible GA with permutation tape (42.17), contention-vs-dimensionality study (42.28-I).

7. Null models / statistics: mostly cosmetic. Several scripts print STATISTICS blocks (CIs, Cohen d,
   bootstrap) over N=1 deterministic quantities or values identical by construction. M-6 (statistics
   presence) is satisfied syntactically, not substantively. The one real null removal is 42.28-I.

---

## 5. CONSOLIDATED HONEST VERDICT

Genuinely established (computing facts, defensible):
- Finite-precision catastrophic cancellation gives a sharp 0/nonzero threshold (42.1).
- Execute x86 from a heap bignum buffer with VirtualProtect (42.12).
- Blind out-of-bounds writes crash a subprocess with 0xC0000005 (42.13).
- Rule 110 over a random mantissa collapses zlib complexity into structure (42.14).
- A reversible Feistel engine nests inside WASM and reproduces output bit-exactly (42.16).
- A reversible GA round-trips to its exact initial hash (42.17).
- Reversible XOR/Feistel/tape ops restore exactly (0.0 J), SHA-256 verified, lab-wide.
- Under CPU contention, timing-feature window dimensionality rises slightly (42.28-I).

Provisional / under-controlled:
- The 12.6% forward/backward timing gap (42.27): real but confounded; not "the arrow of time."
- The 42.28-intrinsic boundary-expansion: real but tiny (D_eff 1.01->1.05), Python-limited.

Over-claimed or metaphorical: 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 42.9, 42.10, 42.11,
42.20, 42.21, 42.22, 42.23, 42.24, 42.25, 42.26, 42.15, 42.18, 42.19.

Largest claim-vs-code gaps:
- 42.19 "halting oracle" counts eigenvalues inside a circle where the answer was hand-encoded.
- 42.18 report says logistic seed 0.123456789; code uses 2^P/3 and "winds" ASCII digits (meaningless).
- 42.11 claims from-scratch catalytic zero derivation but calls mpmath.siegelz (library).
- 42.4 rollup claims a Shannon-entropy Page curve; code computes log2(abs(dt)) with the rise/fall
  shape forced by a hardcoded branch that zeroes entropy on the "evaporated" row. No entropy curve.
- 42.21 "fundamental constant 30" is the divisor in its own area definition.
- 42.20 "firewall severed the pointer" while telemetry prints refcnt 2 and takes the fallback branch.
- 42.28 original ships a unit test that asserts the conclusion with hardcoded synthetic numbers.
- Naming: 42.8 source is internally titled 43.1 and 42.9 source is titled 44.1; the rollup renumbers
  both into the 42.x series.

Summary table:

| Exp | Name | Claimed level | Assessment | One-line status |
|-----|------|---------------|------------|-----------------|
| 42.1 | Hawking Evaporation | L6 | OVER-CLAIMED | Real catastrophic cancellation; not radiation. |
| 42.2 | Wormhole mutation | L5 | METAPHOR | Direct tuple mutation; trivial. |
| 42.3 | Quantum tunneling | L5 | METAPHOR | Phase orthogonal to magnitude; trivial. |
| 42.4 | Page curve | L6 | OVER-CLAIMED | No Shannon entropy; log2(dt) with hardcoded 0-endpoints. |
| 42.5 | Gravitational waves | L4 | METAPHOR | Exponent normalization renamed. |
| 42.6 | Holographic boundary | L5 | METAPHOR | Reading metadata; trivial. |
| 42.7 | Einstein-Rosen bridge | L5 | METAPHOR | Bytecode steganography in a tuple. |
| 42.8 | White hole (inverse expulsion; file says 43.1) | L5 | METAPHOR | __add__ no-op + LIFO byte stack read back; deterministic. |
| 42.9 | Multiverse / superposition (file says 44.1) | L5 | METAPHOR | Real unlocked thread race; only asserts hash changed. |
| 42.10 | Info paradox (topology) | L7 | OVER-CLAIMED | Winding of z^N is N by construction. |
| 42.11 | Photon sphere = Riemann | L6 | OVER-CLAIMED | Library siegelz; catalytic cosmetic. |
| 42.20 | AMPS firewall | L7 | OVER-CLAIMED | UB ctypes read; refcnt=2 contradicts claim. |
| 42.21 | Bekenstein-Hawking area | L7 | OVER-CLAIMED | S/A=30 is the chosen divisor (tautology). |
| 42.22 | Kerr ergosphere | L6 | METAPHOR | Bit concatenation named Penrose process. |
| 42.23 | True singularity | L6 | OVER-CLAIMED | winding=1 for M*z; crash is 0/0. |
| 42.24 | Dark matter | L6 | METAPHOR | Illegal bitcount field -> rejected object. |
| 42.25 | Dark energy | L6 | METAPHOR | dps-bump avoids rounding; Lambda invented. |
| 42.26 | Big Bang / CMB | L6 | OVER-CLAIMED | Geometric time ramp mislabeled temperature. |
| 42.27 | Arrow of time | L6 | PROVISIONAL | Real reversibility; timing gap confounded. |
| 42.28 | Holographic screen | L6 | PROVISIONAL | Best method; effect tiny, interpretation big. |
| 42.12 | Bootstrap paradox | L7 | SOLID(eng)/METAPHOR | Real shellcode exec; no CTC. |
| 42.13 | False vacuum collapse | L6 | PROVISIONAL | Real subprocess crash; no allocator targeting. |
| 42.14 | Boltzmann brain | L6 | SOLID(eng)/METAPHOR | Real Rule-110 complexity collapse. |
| 42.15 | QG unification | L8 | OVER-CLAIMED | Self-flagged; r=1.0 by construction. |
| 42.16 | Recursive universe | L7 | SOLID(eng)/METAPHOR | Real WASM-nested reversible engine. |
| 42.17 | Self-evolving singularity | L7 | SOLID(eng)/METAPHOR | Real reversible GA, exact round-trip. |
| 42.18 | Godel frontier | L8 | OVER-CLAIMED | Winding over ASCII digits; seed mismatch. |
| 42.19 | Oracle machine | L8 | OVER-CLAIMED | Counts eigenvalues; no halting solved. |

Bottom line: exp42 produced a small real toolkit of reversible-computing and low-level systems
demonstrations, plus one careful contention study. It has NOT demonstrated any physics: every
GR/QG/cosmology headline is either a renamed routine behavior or an arithmetic identity that held
before the experiment ran. As a computing lab, several pieces are keepers; as a physics lab, the
claims do not survive inspection.

---

## 6. PROVENANCE: REPORTS THIS MASTER CONSOLIDATES

This master report supersedes the following documents (repo-relative paths). DO NOT delete them on
the basis of this report alone -- but this file now contains the consolidated content and honest
assessment of all of them. Roadmaps are included because their manifesto framing is the source of
the over-claiming (summarized in Sections 2 and 4).

Top-level (2):
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/REPORT_COMPUTATIONAL_EVENT_HORIZON.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACKHOLE_ROADMAP.md

BLACK_HOLES (6):
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/REPORT_BLACK_HOLES_SUMMARY.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/42_PHASE_9_RODMAP.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/exp_20_amps_firewall/REPORT_AMPS_FIREWALL.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/exp_21_bekenstein_hawking/REPORT_BEKENSTEIN_HAWKING.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/exp_22_kerr_ergosphere/REPORT_KERR_ERGOSPHERE.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/BLACK_HOLES/exp_23_true_singularity/REPORT_TRUE_SINGULARITY.md

COSMOS (8):
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/REPORT_PHASE_10_COSMOS.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/42_PHASE_10_ROADMAP.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_24_dark_matter/REPORT_DARK_MATTER.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_25_dark_energy/REPORT_DARK_ENERGY.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_26_big_bang/REPORT_BIG_BANG.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_27_arrow_of_time/REPORT_ARROW_OF_TIME.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_28_holographic_entropy_screen/REPORT_EXP_42_28.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/COSMOS/exp_28_holographic_entropy_screen/REPORT_EXP_42_28_INTRINSIC.md

ULTRA (12):
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA_ROADMAP.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_12_bootstrap_paradox/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_13_false_vacuum_collapse/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_14_boltzmann_brain/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_14_boltzmann_brain/REPORT_EXTENDED.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_15_quantum_gravity_unification/REPORT_QM_GR_UNIFICATION.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_15_quantum_gravity_unification/python/STOCHASTIC_FUNNEL_REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_16_recursive_universe/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_17_self_evolving_singularity/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_18_godel_frontier/REPORT.md
- THOUGHT/LAB/CAT_CAS/5_topological_proofs/42_computational_event_horizon/ULTRA/exp_19_oracle_machine/REPORT.md

Note: ULTRA/exp_15_.../rust/REGEN_INSTRUCTIONS.md is an operational regen note, not a results
report; left in place. Telemetry .csv/.txt/.bin/.png artifacts are raw data, not reports, retained.

Total report documents consolidated: 28 (24 result/summary reports + 4 roadmap/manifesto docs).
Sub-experiments covered: 24 (42.1-42.11 top-level, 42.20-42.23 BLACK_HOLES, 42.24-42.28 COSMOS,
42.12-42.19 ULTRA).

End of master report.
