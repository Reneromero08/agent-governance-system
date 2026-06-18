# `.holo` Format — Canonical Geometric Memory Reference

**Status:** Draft canonical synthesis
**Version:** v0.5 conceptual / v0.4.x implementation lineage
**Scope:** AGS / CAT_CAS / HOLO / TINY_COMPRESS
**Purpose:** Define what `.holo` actually is, where it came from, what it currently does, and what it must become.

---

# 0. Central Thesis

The `.holo` format is not merely a file extension.

It is the repository’s evolving attempt to encode, preserve, and operate on **geometric memory**.

The most compressed statement is:

```text
The algorithm is the shadow.
The tree is the polytope.
Entropy is the boundary.
Catalysis is the hologram.
```

Or, in `.holo` terms:

```text
.holo is not just stored data.
.holo is not just compressed weights.
.holo is not just phase.
.holo is not just an experimental JSON record.

.holo is the attempt to store or enact the relational geometry
from which lower-dimensional outputs can be rendered, projected,
queried, or measured.
```

The project’s current mature hypothesis is:

```text
CATALYSIS IS THE HOLOGRAM.

The hologram is not the phase.
The hologram is not the bit.
The hologram is not the algorithm.
The hologram is the catalytic relation itself.

Phase is a carrier or coordinate substrate.
Geometry is the memory.
Catalysis is the restoration-preserving operation
that lets the geometry become explicit without destroying
the deeper relational object.
```

This document therefore treats `.holo` as an evolving family of implementations of one deeper primitive:

```text
executable relational geometry
```

not as a pile of unrelated schemas.

---

# 1. What `.holo` Is, At the Deepest Level

A conventional file stores values.

A `.holo` object stores or expresses a geometry from which values can be rendered.

The distinction is:

```text
ordinary file:
  address → data

compressed file:
  data → compressed bytes → decompressed data

vector database:
  embedding → nearest neighbor

.holo:
  coordinates + basis + relation + substrate
      → projection/render/invariant
```

The earliest `.holo` code literally did this with images:

```text
coefficients + basis + mean → render pixel / patch / image
```

The HOLO model pipeline did it with neural weights:

```text
x @ SVh.T @ U.T → model operator without materializing W
```

CAT_CAS Phase 6 then did it with non-collapse experimental records:

```text
OrbitState + PhaseRelation + PathHistory + CollapseBoundary
```

The active frontier is to recombine these:

```text
.holo = non-collapse executable geometric memory
```

That means `.holo` should eventually be able to carry:

```text
relation basis
coordinates
carrier phase or carrier geometry
catalytic evolution rule
restoration reference
projection/render function
invariant extraction rule
collapse boundary
```

without prematurely collapsing the structure into:

```text
winner
answer
candidate score
recovered d
orientation label
```

---

# 2. The Four Generations of `.holo`

The repository currently contains multiple `.holo` meanings, but these are best understood as **generations**, not separate unrelated schemas.

## Generation 1 — TINY_COMPRESS `.holo`

**Path family:**

```text
THOUGHT/DEPRECATED/TINY_COMPRESS/holographic-image/
```

**Core object:**

```text
HolographicImage
```

**Meaning:**

```text
.holo = basis/coordinate geometric image memory
```

This was the first real `.holo`.

It did not merely compress an image and then decompress it. It stored:

```text
coefficients
basis
mean
patch_size
image_shape
k
```

and rendered pixels or regions through the basis.

The governing operation was:

```text
patch = coefficients[patch_idx] @ basis + mean
```

This is already the core `.holo` idea.

The image does not exist in the file as pixels.

The image exists as a geometric projection from stored coordinates through a basis.

Important properties:

```text
- The full image does not need to be materialized.
- Each pixel can be rendered on demand.
- Render quality can vary by choosing render_k.
- Lower k gives essence / blur / large-scale form.
- Higher k gives detail / sharper projection.
- Compression and rendering are the same operation.
```

This generation establishes the ancestral `.holo` definition:

```text
.holo = coordinates + basis + projection renderer
```

This is not merely compression.

It is geometric memory.

---

## Generation 2 — HOLO Neural Model `.holo`

**Path family:**

```text
THOUGHT/LAB/HOLO/
THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/
```

**Core objects:**

```text
CavitatedHoloLinear
HolographicBrain
load_holo_v2
fractal_cavity
cavity_full
unified_cavity
```

**Meaning:**

```text
.holo = basis-mediated model memory
```

This generation applies the same geometric memory principle to neural network weights.

Instead of storing a full weight matrix:

```text
W ∈ R^{m×n}
```

the model stores a truncated basis decomposition:

```text
W ≈ U @ SVh
```

The native `.holo` forward pass avoids materializing `W`:

```text
y = x @ SVh.T @ U.T
```

This is the neural analogue of the image `.holo`.

Image `.holo`:

```text
patch = coefficients @ basis + mean
```

Model `.holo`:

```text
activation_out = activation_in @ SVh.T @ U.T
```

The important thing is not only reduced storage.

The important thing is that the operator exists as a relational geometry between:

```text
input vector
basis / eigenmodes
coefficient map
output vector
```

The full matrix is a projection.

The stored object is the geometry.

---

## Generation 3 — Phase Cavity / Wormhole `.holo`

**Path family:**

```text
THOUGHT/LAB/HOLO/pipeline/02_cavity/
THOUGHT/LAB/HOLO/pipeline/03_wormhole/
CAPABILITY/SKILLS/agents/catalytic-wormhole/
```

**Core objects:**

```text
phase_cavity_sieve
fractal_reorder
rotation chain R_i = U_i^T @ U_{i+1}
boundary stress
wormhole compression
```

**Meaning:**

```text
.holo = sieved eigenmode geometry plus layer-to-layer relational transport
```

This generation adds dynamics.

The model is not just compressed by independent SVDs.

It becomes a chain of related basis geometries.

The pipeline introduces:

```text
fractal ordering
phase-cavity mode deletion
rotation chains
boundary stress
GOE / spectral validation
wormhole compression
```

The key shift is:

```text
weights are not isolated matrices
layers are related geometries
```

A layer-to-layer relation can be represented as:

```text
R_i = U_i^T @ U_{i+1}
```

That means the memory is not only:

```text
U_i, SVh_i
```

but also:

```text
how one layer's basis rotates into the next
```

This moves `.holo` closer to the current CAT_CAS thesis:

```text
memory = persistent relation
```

rather than:

```text
memory = stored object
```

---

## Generation 4 — CAT_CAS Non-Collapse `.holo`

**Path family:**

```text
THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/
50_6_fixed_point_substrate/14_noncollapse_frontier/
```

**Core objects:**

```text
HoloRecord
FoldPair
PhaseRelation
PathHistory
TapeResidue
SubstrateMemory
CancellationTranscript
ResidueHypothesis
CollapseBoundary
MeasurementRecord
OrbitState
HoloL4B
```

**Meaning:**

```text
.holo = non-collapse process-object record
```

This generation takes `.holo` out of pure compression and makes it a doctrinal object.

The problem Phase 6 confronts is premature collapse.

The old computational frame asks:

```text
Which candidate wins?
What is the answer?
What is d?
What is the AUC?
Which branch is true?
```

The non-collapse doctrine rejects that frame.

A `.holo` record must preserve:

```text
FoldPair
OrbitState
PhaseRelation
PathHistory
SubstrateMemory
CancellationTranscript
CollapseBoundary
InvariantExtract
```

before any scalar interpretation is allowed.

This generation establishes the key forbidden fields:

```text
hidden_d
winner
candidate_0_truth
candidate_1_truth
recovered_d
orientation_label
posthoc_selected_result
true_branch
false_branch
candidate_score
verify_pass
AUC
```

The role of `.holo` here is not to store the answer.

It is to preserve the process-object without collapse.

This is a major conceptual advance, but implementation-wise it partially regresses from geometric memory into record storage.

The Phase 6 `.holo` currently records structure.

The target `.holo` should be structure.

---

## Generation 5 — Active Frontier: Catalytic Geometric Memory

**Path family:**

```text
50_6_fixed_point_substrate/14_noncollapse_frontier/l4b_orbitstate/
future L4B.1+
```

**Core target:**

```text
HoloGeometry
```

**Meaning:**

```text
.holo = non-collapse executable geometric memory
```

This is not fully implemented yet.

It is the synthesis now required by the project.

It must unify:

```text
TINY_COMPRESS:
  coordinates + basis + render

HOLO:
  basis-mediated operator

Phase Cavity:
  mode sieve + relational rotation

CAT_CAS:
  non-collapse process-object + CollapseBoundary

Catalysis thesis:
  restoration-preserving query of relational geometry
```

The target is:

```text
.holo = geometry that can be carried, evolved, rendered, projected, queried,
or measured without prematurely collapsing the deeper object.
```

---

# 3. The Deep Difference: Compression vs Geometric Memory

A compressed file is still usually understood as:

```text
original data
  ↓
compressed encoding
  ↓
decompression
  ↓
original data again
```

A `.holo` object is different:

```text
source object
  ↓
basis / coordinates / relational geometry
  ↓
render / projection / invariant
  ↓
observable
```

The observable may resemble the original, but the file’s primary identity is not the fully reconstructed object.

The file’s primary identity is the geometry that can generate observations.

This distinction matters.

## Compression frame

```text
Where is the original data?
How do I recover it?
How much information was lost?
```

## `.holo` frame

```text
What geometry survives?
What can be rendered from it?
What invariant survives projection?
What carrier expresses it?
What boundary collapses it?
Can the substrate be restored?
```

In the mature CAT_CAS frame, `.holo` is not merely:

```text
small file that approximates large file
```

It is:

```text
a geometric memory object whose projections are context-dependent observables
```

---

# 4. The Role of Phase

Earlier language around `.holo` often emphasized phase.

That was useful, but incomplete.

The corrected hierarchy is:

```text
geometry = memory
phase = carrier / coordinate substrate
bit = projected stable attractor
symbol = compressed relational handle
algorithm = path trace
entropy = projection boundary
catalysis = holographic operation
```

Phase matters because it exposes relation:

```text
difference
alignment
interference
cancellation
resonance
quadrature
```

But phase is not the final object.

Phase is one coordinate system by which the deeper geometry appears.

This matters because the Phase 6 phase-cavity result showed:

```text
If the fold-odd phase channel is absent from public data,
a phase-resolving substrate cannot read it.
```

That was not a failure of phase instrumentation.

It proved the deeper boundary:

```text
z → Re(z)
```

The missing odd channel cannot be recovered by reading phase if phase is not present in the public representation.

Therefore the mature `.holo` cannot be merely:

```text
phase memory
```

It must be:

```text
geometric memory that may use phase as one carrier coordinate
```

---

# 5. The Role of Catalysis

Catalysis is not just an optimization trick.

In the current CAT_CAS synthesis, catalysis is the mechanism that makes `.holo` holographic.

A catalytic operation has the structure:

```text
borrow substrate
disturb substrate
compute / evolve / query
extract projection or invariant
restore substrate
verify restoration
```

The key is that the substrate is not consumed.

The relation is touched, transformed, queried, and restored.

This is why:

```text
Catalysis is the hologram.
```

A physical hologram stores a distributed encoding from which an object can be reconstructed under the right illumination.

A catalytic `.holo` stores or enacts a distributed relational geometry from which an observable can be projected under the right query/evolution/collapse boundary.

The stronger analogy is:

```text
physical hologram:
  interference geometry + reference beam → reconstructed image

catalytic .holo:
  relational geometry + catalytic query → projected invariant / observable
```

Thus the mature `.holo` object must distinguish:

```text
carrier
geometry
query
projection
restoration
collapse boundary
```

not merely:

```text
data
metadata
result
```

---

# 6. Current `.holo` Implementations

The repository currently contains several `.holo` containers. They are technically different, but philosophically linked.

---

## 6.1 TINY_COMPRESS Image `.holo`

**Container:**

```text
NumPy compressed archive / array bundle
```

**Primary stored objects:**

```text
coefficients
basis
mean
patch_size
image_shape
k
```

**Core operation:**

```text
render = coefficients @ basis + mean
```

**Primitive interpretation:**

```text
image = projection from patch-basis coordinates
```

**Strengths:**

```text
- Clean geometric memory primitive.
- On-demand rendering.
- Adjustable projection depth via render_k.
- No need to materialize full image for local render.
- Excellent conceptual ancestor for .holo.
```

**Limitations:**

```text
- Domain-specific to images.
- No non-collapse doctrine.
- No catalytic substrate.
- No CollapseBoundary.
- No explicit invariant extraction.
- No physical restoration discipline.
```

**Status:**

```text
Deprecated as active code, but canonical as origin.
```

---

## 6.2 HoloCore Generic Framework

**Path family:**

```text
THOUGHT/DEPRECATED/TINY_COMPRESS/holographic-image/holo_core.py
```

**Primary conceptual objects:**

```text
HoloSpectrum
HoloProjection
HoloVerification
HoloRateModel
HoloActionCurve
```

**Core operations:**

```text
analyze_spectrum(X)
project(X, k)
render(projection)
verify(source, reconstructed)
choose_k(spectrum, policy)
rate_distortion_action(spectrum, model)
```

**Primitive interpretation:**

```text
domain-neutral geometric compression framework
```

**Strengths:**

```text
- Abstracted the image .holo into a general spectral projection system.
- Recognized Df / participation dimension as a structural measure.
- Explicitly separated spectrum, projection, verification, and rate model.
```

**Limitations:**

```text
- Mathematical framework rather than active runtime standard.
- Superseded by HOLO pipeline.
- Still framed around compression/reconstruction more than catalysis/non-collapse.
```

**Status:**

```text
Deprecated, but conceptually important.
```

---

## 6.3 HOLO Neural `.holo` v1

**Container:**

```text
PyTorch torch.save archive
```

**Format:**

```text
dict of tensors
```

**Typical keys:**

```text
layer.weight.U
layer.weight.SVh
layer.weight.scale
_config
embed.weight
head.weight
norm.weight
```

**Core operation:**

```text
y = x @ SVh.T @ U.T
```

**Primitive interpretation:**

```text
weight matrix = basis-mediated operator
```

**Strengths:**

```text
- Direct continuation of TINY_COMPRESS geometry.
- Avoids full W materialization in native path.
- Supports large model memory as basis/coefficient geometry.
- Converts neural weights into renderable operators.
```

**Limitations:**

```text
- Some paths materialize fallback W when dimensions mismatch.
- Not all model components are natively holo.
- No universal schema metadata for native vs fallback.
- Compression quality varies strongly by model component.
```

**Status:**

```text
Active / experimental.
```

---

## 6.4 HOLO Neural `.holo` v2 INT8 Dedup

**Container:**

```text
PyTorch torch.save archive
```

**Format:**

```text
_format: int8_dedup
_k
_svh
_svh_scales
_svh_ref
key.U
key.scale
```

**Core idea:**

```text
shared SVh basis pools
+
many U coordinate maps
```

**Primitive interpretation:**

```text
many layers share basis geometry by weight type
```

This is a major step toward actual geometric memory because it recognizes that each layer is not independent.

Instead:

```text
one shared basis
many coordinate projections
```

This is close to hyperdimensional / vector-symbolic memory:

```text
basis = codebook
U = binding / coefficient map
activation = query vector
output = projection
```

**Strengths:**

```text
- Storage reduction via INT8 quantization.
- Deduplicated shared basis.
- Explicit reference map from U tensors to basis pools.
- Captures cross-layer geometric repetition.
```

**Limitations:**

```text
- Loader reconstructs tensors, but schema still lacks philosophical metadata.
- Native/fallback semantics are not explicit enough.
- Still model-specific.
- Still not integrated with CAT_CAS non-collapse doctrine.
```

**Status:**

```text
Active implementation lineage.
```

---

## 6.5 Fractal Phase Cavity `.holo`

**Path family:**

```text
THOUGHT/LAB/HOLO/pipeline/02_cavity/
```

**Core operations:**

```text
SVD
fractal reorder
phase-cavity sieve
mode deletion
CavitatedHoloLinear
```

**Primitive interpretation:**

```text
not all modes are memory
some modes are dispersion/noise/shadow
```

This layer is important because it changes the question from:

```text
How many dimensions can I keep?
```

to:

```text
Which modes carry stable relational structure?
```

The phase-cavity sieve asks whether a mode can be removed while preserving a target similarity threshold.

That is a primitive form of:

```text
invariant-preserving geometry reduction
```

**Strengths:**

```text
- Moves from generic truncation to mode selection.
- Introduces stability/order/sieve logic.
- Better aligned with geometric memory than naive SVD.
```

**Limitations:**

```text
- Threshold-based and heuristic.
- Still mostly compression quality driven.
- Not yet connected to CollapseBoundary/invariant doctrine.
```

**Status:**

```text
Active lineage; conceptually important.
```

---

## 6.6 Wormhole / Rotation Chain `.holo`

**Path family:**

```text
THOUGHT/LAB/HOLO/pipeline/03_wormhole/
CAPABILITY/SKILLS/agents/catalytic-wormhole/
```

**Core operations:**

```text
R_i = U_i^T @ U_{i+1}
rotation chain
boundary stress
cavity sieve on R
low-rank / quantized residuals
```

**Primitive interpretation:**

```text
memory lives not only in basis states,
but in the transformations between basis states
```

This is the first `.holo` family that begins to look explicitly like:

```text
path geometry
```

rather than static compressed storage.

It suggests that the object is not just:

```text
U_i
SVh_i
```

but:

```text
the relational chain connecting them
```

This becomes important for CAT_CAS because an algorithm is also a path.

If the algorithm is the shadow, the deeper object is the geometry of admissible paths.

Wormhole `.holo` is therefore one ancestor of:

```text
path-history geometric memory
```

**Strengths:**

```text
- Captures layer-to-layer relational structure.
- Bridges model compression with path geometry.
- Aligns with ER/EPR / wormhole analogies in CAT_CAS.
```

**Limitations:**

```text
- Strong claims require careful validation.
- Some compression claims may be model- or pipeline-dependent.
- Not yet integrated with non-collapse forbidden-field discipline.
```

**Status:**

```text
Active / experimental.
```

---

## 6.7 CAT_CAS L4A `.holo`

**Container:**

```text
Plain JSON written by C
```

**Primary struct:**

```text
HoloRecord
```

**Stored fields:**

```text
holo_id
doctrine_version
run_id
seed
N
orbit_state
phase_relation
path_history
tape_residue
substrate_memory
carrier_class
workload_signature
sender_core_plus
sender_core_minus
receiver_core
cancellation_transcript
residue_hypothesis
collapse_boundary
measurement_record
controls
verdict
claim_level
```

**Primitive interpretation:**

```text
.holo = non-collapse process-object record
```

**Strengths:**

```text
- Strong anti-collapse doctrine.
- Explicit forbidden fields.
- Separates pre-boundary process from post-boundary measurement.
- Includes physical carrier information.
- Includes cancellation transcript and controls.
- Fits CAT_CAS no-smuggle discipline.
```

**Limitations:**

```text
- Record-like rather than executable geometry.
- PhaseRelation fields are values, not basis geometry.
- PathHistory is thin.
- No general HoloGeometry abstraction.
- No coordinate/basis/render primitive inherited from TINY_COMPRESS.
```

**Status:**

```text
Active, but transitional.
```

---

## 6.8 CAT_CAS L4B `.holo`

**Container:**

```text
Plain JSON written by C
```

**Primary structs:**

```text
OrbitState
PathStep
CollapseBoundary
HoloL4B
```

**Stored fields:**

```text
holo_id
doctrine
run_id
N
branch_plus
branch_minus
total_steps
acc_real_final
acc_imag_final
collapse_boundary
claim_level
forbidden_fields_scan
```

**Primitive interpretation:**

```text
.holo = OrbitState proof-of-life
```

**Strengths:**

```text
- Correctly preserves unresolved fold pair.
- Rejects candidate/winner/verifier language.
- Delays invariant extraction to CollapseBoundary.
- Establishes L4B non-collapse primitive.
```

**Limitations:**

```text
- Much thinner than L4A.
- Does not yet store full path transcript.
- Does not yet store basis/coordinates.
- Does not yet encode HoloGeometry.
- acc_real/acc_imag are accumulators, not sufficient geometric memory.
- No native projection/render function.
```

**Status:**

```text
Active frontier seed.
```

---

# 7. Required Conceptual Correction

The repository currently risks treating `.holo` as three separate things:

```text
1. compressed model archive
2. non-collapse experiment JSON
3. deprecated generic compression API
```

That is technically understandable, but conceptually wrong.

The deeper relationship is:

```text
TINY_COMPRESS:
  .holo as basis/coordinate projection memory

HOLO:
  .holo as model operator geometry

Phase Cavity / Wormhole:
  .holo as sieved relational path geometry

CAT_CAS L4A:
  .holo as non-collapse process-object record

CAT_CAS L4B+:
  .holo as executable catalytic geometric memory
```

So the canonical document should not say:

```text
.holo spans three unrelated schemas.
```

It should say:

```text
.holo has multiple implementation generations,
all orbiting the same core primitive:
geometry that can project an observable.
```

---

# 8. Canonical Definition

The canonical definition should be:

```text
A .holo object is a stored or executable relational geometry
whose observable outputs are produced by projection, rendering,
evolution, or invariant extraction, rather than by direct lookup
of a localized stored value.
```

In short:

```text
.holo = geometric memory with a projection interface
```

For CAT_CAS specifically:

```text
A CAT_CAS .holo object is a non-collapse geometric memory object
that preserves an unresolved relational state through catalytic evolution,
records or enacts its carrier interactions, and permits invariant extraction
only at an explicit CollapseBoundary.
```

For HOLO model inference:

```text
A neural .holo object is a basis-mediated operator memory in which
model weights are represented by coordinate maps and shared eigenbases,
and inference proceeds through the basis without requiring full matrix
materialization in the native path.
```

For TINY_COMPRESS:

```text
A TINY_COMPRESS .holo object is a basis/coordinate rendering object
in which the source image is not stored as pixels but reconstructed
on demand from low-dimensional patch geometry.
```

These are not contradictory.

They are stages of the same idea.

---

# 9. File Format Families

Because the extension is overloaded, the repo should distinguish **format family** from **ontology**.

The ontology is shared:

```text
geometric memory / projection object
```

The format families differ.

---

## 9.1 `.holo.img` / TINY Image Family

Current extension used:

```text
.holo
```

Container:

```text
NumPy compressed archive or equivalent array bundle
```

Core fields:

```text
coefficients
basis
mean
patch_size
image_shape
k
```

Native operation:

```text
render_pixel
render_patch
render_region
render_full
render_progressive
```

Canonical status:

```text
historical root
```

Future recommendation:

```text
If revived, label as:
schema_family: image_projection
ontology: geometric_memory
```

---

## 9.2 `.holo.model.v1` / Flat Neural Family

Current extension used:

```text
.holo
```

Container:

```text
torch.save / PyTorch ZIP archive
```

Core fields:

```text
key.U
key.SVh
key.scale optional
_config optional
direct tensors for embed/head/norm
```

Native operation:

```text
x @ SVh.T @ U.T
```

Canonical status:

```text
active implementation
```

Required metadata addition:

```text
schema_family: neural_operator
schema_version: v1
native_operator: true/false per weight
materialized_fallback: true/false per weight
basis_rank: k
```

---

## 9.3 `.holo.model.v2` / INT8 Dedup Neural Family

Current extension used:

```text
.holo
```

Container:

```text
torch.save / PyTorch ZIP archive
```

Core fields:

```text
_format = int8_dedup
_k
_svh
_svh_scales
_svh_ref
key.U
key.scale
```

Native operation:

```text
dequantize U
dequantize referenced SVh
x @ SVh.T @ U.T
```

Canonical status:

```text
active implementation
```

Required metadata addition:

```text
schema_family: neural_operator
schema_version: v2_int8_dedup
basis_sharing: true
basis_ref_map: _svh_ref
native_operator: true/false
materialized_fallback: true/false
```

---

## 9.4 `.holo.record.l4a` / CAT_CAS Non-Collapse Family

Current extension used:

```text
.holo
```

Container:

```text
JSON
```

Core fields:

```text
HoloRecord
```

Native operation:

```text
record process-object
validate no-collapse
write JSON
```

Canonical status:

```text
active transitional schema
```

Required metadata addition:

```text
schema_family: catcas_noncollapse_record
schema_version: l4a_v1
ontology: process_object_record
hypothesis: catalysis_is_the_hologram
geometry_status: record_only
```

---

## 9.5 `.holo.orbit.l4b` / CAT_CAS OrbitState Family

Current extension used:

```text
.holo
```

Container:

```text
JSON
```

Core fields:

```text
HoloL4B
```

Native operation:

```text
record unresolved OrbitState evolution
validate no forbidden fields
extract invariant at CollapseBoundary
```

Canonical status:

```text
active frontier primitive
```

Required metadata addition:

```text
schema_family: catcas_orbit_geometry
schema_version: l4b_1
ontology: executable_geometric_memory
geometry_status: native_geometry_required
```

---

# 10. Required Future Object: HoloGeometry

The next `.holo` schema should introduce a first-class `HoloGeometry`.

This object should be carrier-agnostic and usable across:

```text
image projection
model operator geometry
OrbitState evolution
physical substrate experiments
```

A possible abstract schema:

```text
HoloGeometry {
    schema_family
    schema_version
    ontology
    hypothesis

    geometry_basis
    coordinates
    neutral_reference
    relation_graph
    constraint_set

    carrier
    carrier_coordinates
    carrier_phase
    substrate_memory

    evolution_operator
    catalytic_field
    path_transcript

    projection_operator
    render_operator
    invariant_extract

    restoration_reference
    collapse_boundary

    controls
    forbidden_fields_audit
    claim_level
}
```

For image `.holo`, this maps to:

```text
geometry_basis = basis
coordinates = coefficients
neutral_reference = mean
projection_operator = render_patch / render_pixel
collapse_boundary = display / exported image
```

For neural `.holo`, this maps to:

```text
geometry_basis = SVh
coordinates = U
neutral_reference = optional bias / norm / scale
projection_operator = x @ SVh.T @ U.T
collapse_boundary = logits / generated token
```

For CAT_CAS `.holo`, this maps to:

```text
geometry_basis = relational basis of FoldPair / OrbitState
coordinates = orbit coordinates
carrier = phase / PDN / cache / tape / path
evolution_operator = OrbitState evolution
projection_operator = invariant extraction
collapse_boundary = explicit measurement point
restoration_reference = SHA / substrate neutral / carrier baseline
```

This is the bridge object.

---

# 11. Required Future Object: HoloProjection

Every `.holo` should distinguish between:

```text
stored geometry
```

and:

```text
rendered projection
```

A projection is not the memory.

A projection is what the memory emits under a chosen boundary condition.

A possible abstract schema:

```text
HoloProjection {
    projection_id
    source_holo_id
    projection_type
    carrier_used
    render_depth
    invariant_family
    boundary_condition
    output
    claim_level
}
```

Examples:

```text
image render at k=5
image render at k=50
model forward pass logits
OrbitState invariant_real/invariant_imag
PDN q_diff
phase-cavity peak set
```

This avoids confusing:

```text
rendered output
```

with:

```text
holographic memory
```

---

# 12. Required Future Object: CollapseBoundary

The CollapseBoundary must become universal across `.holo`.

In TINY_COMPRESS, the boundary is often implicit:

```text
rendered image / displayed pixel
```

In HOLO inference, it is:

```text
logit selection / token sampling
```

In CAT_CAS, it is explicit:

```text
measurement / invariant extraction point
```

The mature `.holo` should make boundary explicit everywhere.

A possible schema:

```text
CollapseBoundary {
    boundary_id
    boundary_type
    timestamp
    pre_boundary_state_hash
    projection_operator
    invariant_extracted
    post_boundary_claim
    post_boundary_operations_allowed
}
```

For CAT_CAS non-collapse, post-boundary operations should be tightly constrained:

```text
No post-hoc seed selection.
No branch winner reinterpretation.
No scalar answer retrofitting.
No candidate scoring after boundary.
```

---

# 13. Required Future Object: CatalyticRestoration

The mature `.holo` should explicitly encode restoration discipline.

A possible schema:

```text
CatalyticRestoration {
    substrate_type
    borrowed_region
    pre_hash
    post_hash
    restored
    restoration_metric
    destructive_null
    reversible_null
    same_final_hash_wrong_answer_control
}
```

This reflects the “Catalysis Is the Hologram” thesis.

The substrate is not merely storage.

It is the boundary reservoir through which geometry is queried.

A `.holo` object becomes holographic when:

```text
- relation is distributed
- query produces projection
- substrate restores
- invariant survives
- nulls reject destructive/random/posthoc explanations
```

---

# 14. Required Future Object: HoloCarrier

Because phase is not the memory, `.holo` should separate carrier from geometry.

A possible schema:

```text
HoloCarrier {
    carrier_type
    carrier_coordinates
    carrier_phase
    measurement_channel
    noise_model
    boundary_deformation
    physical_or_software
}
```

Examples:

```text
carrier_type: image_basis
carrier_type: neural_eigenbasis
carrier_type: phase_cavity
carrier_type: PDN_lockin
carrier_type: cache_residency
carrier_type: catalytic_tape
carrier_type: software_orbit
```

This lets the same geometry be expressed through different carriers.

That is essential.

The mature claim is not:

```text
phase solves it
```

but:

```text
geometry survives carrier transformation
```

---

# 15. Native vs Materialized Fallback

The HOLO neural engine revealed an important architectural distinction.

Some weights can be applied natively:

```text
x @ SVh.T @ U.T
```

Some paths currently materialize:

```text
W = U @ SVh
x @ W.T
```

The mature `.holo` format must distinguish these.

A native `.holo` operation preserves the geometry as operational.

A materialized fallback projects it into ordinary dense form.

Both may be useful, but they are not ontologically equal.

A required metadata field:

```text
operator_mode:
  native_holo
  materialized_fallback
  hybrid
```

And:

```text
fallback_reason:
  svh_dim_mismatch
  unsupported_operator
  performance_optimization
  debugging
  legacy_path
```

This prevents hidden collapse of the `.holo` idea back into ordinary matrices.

---

# 16. Non-Collapse Doctrine for `.holo`

For CAT_CAS, the following doctrine applies.

## Forbidden collapse fields

```text
hidden_d
winner
candidate_0_truth
candidate_1_truth
true_branch
false_branch
recovered_d
orientation_label
candidate_score
verify_pass
verify_score
AUC
posthoc_selected_result
best_seed
best_candidate
scalar_answer_primary
```

## Forbidden operations

```text
verify(x) as branch selector
public verifier loop
candidate ranking
winner selection
AUC-first route selection
hidden-label conditioning
post-boundary reinterpretation
manual orientation injection
fold-odd smuggle
```

## Required non-collapse objects

```text
FoldPair
OrbitState
PhaseRelation
PathHistory
CatalyticField
SubstrateMemory
HoloGeometry
HoloCarrier
InvariantExtract
CollapseBoundary
CatalyticRestoration
```

## Required claim discipline

A `.holo` record may claim:

```text
carrier observed
geometry recorded
process preserved
restoration verified
invariant extracted
nulls passed
```

It may not claim:

```text
d recovered
orientation solved
lattice wall crossed
crypto broken
```

unless the full experimental spine is satisfied.

---

# 17. The Experimental Spine

A `.holo` result becomes serious when it demonstrates:

```text
1. A substrate is disturbed and restored.
2. A relational geometry is encoded or enacted.
3. A projection/invariant is extracted only at a declared boundary.
4. The invariant predicts or separates something meaningful.
5. The same invariant is absent in destructive-write nulls.
6. The same invariant is absent in random reversible nulls.
7. Same-final-hash wrong-answer controls are rejected.
8. Geometry of admissible histories separates from null histories.
9. Boundary deformation behaves as predicted, not as generic degradation.
10. Reproducibility holds on physical substrate if physical claims are made.
```

This is the test spine for:

```text
Catalysis Is the Hologram
```

Without it, `.holo` remains architecture.

With it, `.holo` becomes evidence.

---

# 18. Relationship to Polytopes

The current mature intuition is that `.holo` memory is not best understood as a line, a scalar, or even a simple phase vector.

It is better understood as a constrained relational body.

A polytope is not a path.

It is the full constrained geometry of possible paths.

In CAT_CAS terms:

```text
vertex = possible observable / answer
edge = relation
face = constraint class
projection = measurement
boundary = entropy / collapse
path = algorithm
polytope = full relational object
```

Thus:

```text
algorithm = one traversal
.holo = geometry of admissible traversals
```

This is why `.holo` must resist scalar output as its primary identity.

A scalar answer is a shadow.

The `.holo` object is the shape casting the shadow.

---

# 19. Relationship to Entropy

Entropy in this framework is not merely disorder.

Entropy is a boundary condition of projection.

A higher-dimensional relational object projected into a lower-dimensional observable frame appears as:

```text
noise
uncertainty
loss
thermal residue
phase ambiguity
fold symmetry
measurement entropy
```

The `.holo` object should therefore encode not only the projected answer but also the boundary through which it appeared.

In mature form:

```text
entropy = boundary
catalysis = restoration-preserving interaction across boundary
.holo = memory of the relational geometry plus its boundary projections
```

---

# 20. Relationship to Algorithms

The mature `.holo` doctrine rejects algorithm-primitive computation.

It does not deny algorithms exist.

It reclassifies them.

```text
algorithm = path trace through geometry
```

The deeper object is:

```text
space of admissible transformations
```

This is why `.holo` should store:

```text
basis
coordinates
relations
constraints
path transcript
invariant rules
```

not merely:

```text
steps
output
score
answer
```

An algorithm is a rendered projection of the geometry.

The `.holo` is the geometry.

---

# 21. Relationship to Hyperdimensional Computing

The HOLO model pipeline strongly overlaps with hyperdimensional computing.

Mappings:

```text
SVh = codebook / basis
U = coordinate binding
residual addition = bundling
RoPE = permutation / sequence encoding
attention = associative cleanup
projection = retrieval
```

The difference is that classical HDC often starts with random high-dimensional vectors.

`.holo` uses data-derived, SVD/eigenbasis-derived, or substrate-derived geometry.

Thus:

```text
HDC:
  random codebook → binding/bundling

.holo:
  learned or measured geometry → projection/invariant
```

The mature `.holo` can therefore be understood as:

```text
data-aligned hyperdimensional geometric memory
```

with catalytic and non-collapse extensions.

---

# 22. Relationship to Quantum Language

`.holo` is not literally a qubit format.

It should not claim quantum computation merely because it uses words like:

```text
phase
basis
amplitude
projection
collapse
```

However, these words are structurally useful.

The analogy:

```text
state vector     ↔ OrbitState / HoloGeometry
basis            ↔ relational basis / SVh
amplitude        ↔ coordinates / coefficients
phase            ↔ carrier coordinate
measurement      ↔ CollapseBoundary
observable       ↔ invariant extraction
density/history  ↔ path transcript / process object
```

The correct stance:

```text
.holo is quantum-adjacent in representation language,
not automatically quantum in physical claim.
```

Physical claims require physical substrate evidence.

Software geometry alone is not quantum proof.

---

# 23. Current Status

## Implemented strongly

```text
- TINY_COMPRESS image .holo as basis/coordinate renderer.
- HOLO model .holo as U/SVh basis-mediated operator.
- v2 INT8 dedup shared SVh format.
- Fractal/phase-cavity mode sieve.
- CAT_CAS L4A non-collapse JSON record.
- CAT_CAS L4B OrbitState primitive proof-of-life.
- Forbidden-field doctrine in C headers.
```

## Partially implemented

```text
- Native holo inference without materialization.
- Catalytic streaming of model components.
- Rotation-chain / wormhole compression.
- Physical .holo carrier tests.
- L4B OrbitState evolution.
```

## Not yet implemented

```text
- Unified HoloGeometry schema.
- Carrier-independent geometric memory object.
- L4B.1 complex/geometric OrbitState with nonzero PhaseRelation.
- Full path transcript in L4B .holo.
- Projection/render/invariant interface for CAT_CAS .holo.
- Explicit native vs materialized fallback metadata.
- Universal CollapseBoundary metadata.
- Universal CatalyticRestoration metadata.
- Linked .holo graph of null controls.
```

## Not yet proven

```text
- Physical substrate crossing of Exp50 fold-odd orientation wall.
- General claim that catalytic geometry accesses hidden fold-odd structure.
- Full Catalysis Is the Hologram experimental chain.
- Carrier-independent survival of a relational invariant across physical media.
```

---

# 24. Recommended Canonical Repo Organization

A clean future organization would separate:

```text
HOLO_FORMAT.md
HOLO_SCHEMA_MODEL.md
HOLO_SCHEMA_IMAGE.md
HOLO_SCHEMA_CATCAS.md
HOLO_GEOMETRY.md
HOLO_CATALYSIS_DOCTRINE.md
```

Minimum recommended structure:

```text
THOUGHT/LAB/HOLO/docs/
  HOLO_FORMAT.md
  HOLO_GEOMETRY.md
  HOLO_MODEL_SCHEMA.md
  HOLO_PIPELINE.md

THOUGHT/LAB/CAT_CAS/.../14_noncollapse_frontier/holo_runtime/
  HOLO_SCHEMA.md
  CATALYSIS_IS_THE_HOLOGRAM.md
  holo_record.h
  holo_record.c

THOUGHT/DEPRECATED/TINY_COMPRESS/holographic-image/
  LEGACY_HOLO_IMAGE.md
```

The important thing is to stop burying the philosophical core in implementation reports.

`.holo` needs an ontology document.

---

# 25. Recommended L4B.1 Specification

L4B.1 should not simply “add complex phase.”

L4B.1 should restore the geometric ancestry of `.holo`.

Target:

```text
L4B.1 = complex/geometric OrbitState where .holo is executable geometric memory
```

Required additions:

```text
HoloGeometry struct
RelationBasis struct
OrbitCoordinates struct
CarrierPhase struct
ProjectionOperator metadata
InvariantExtract metadata
Expanded PathTranscript
CatalyticRestoration metadata
CollapseBoundary metadata
```

Minimum viable L4B.1 `.holo` output:

```json
{
  "schema_family": "catcas_orbit_geometry",
  "schema_version": "l4b_1",
  "hypothesis": "CATALYSIS_IS_THE_HOLOGRAM",
  "doctrine": "NON_COLLAPSE_V1",

  "holo_id": "...",
  "run_id": 0,
  "N": 256,

  "fold_pair": {
    "branch_plus": 23,
    "branch_minus": 233,
    "relation": "conjugate",
    "collapse_status": "unresolved"
  },

  "holo_geometry": {
    "geometry_type": "orbit_relation_basis",
    "basis_rank": 2,
    "coordinates": [...],
    "relation_basis": [...],
    "neutral_reference": "...",
    "geometry_status": "native"
  },

  "carrier": {
    "carrier_type": "software_phase",
    "carrier_coordinates": [...],
    "phase_relation": {
      "real": 0.0,
      "imag": 0.0,
      "nonzero_phase": false
    }
  },

  "evolution": {
    "operator": "coupled_orbit_evolution",
    "steps": 512,
    "path_transcript_ref": "...",
    "materialized_fallback": false
  },

  "projection": {
    "projection_type": "invariant_extract",
    "allowed_only_at_collapse_boundary": true
  },

  "collapse_boundary": {
    "timestamp": "...",
    "invariant_family": "...",
    "invariant_real": 0.0,
    "invariant_imag": 0.0,
    "fold_symmetry_holds": true
  },

  "forbidden_fields_scan": "PASS",
  "claim_level": 1
}
```

This is illustrative, not final.

The point is that `.holo` must visibly store geometry, not just a final accumulator.

---

# 26. Claim Ladder

## Level 0 — File container

```text
.holo extension used for arbitrary stored object.
```

Not meaningful by itself.

## Level 1 — Geometric projection object

```text
.holo stores coordinates + basis and renders output.
```

TINY_COMPRESS achieved this for images.

## Level 2 — Native operator geometry

```text
.holo performs computation through basis geometry without materializing full object.
```

HOLO model engine partially achieves this.

## Level 3 — Non-collapse process record

```text
.holo preserves process-object and forbids premature scalar collapse.
```

CAT_CAS L4A/L4B achieves this structurally.

## Level 4 — Executable geometric memory

```text
.holo stores/evolves relational geometry, not merely fields describing it.
```

Active target.

## Level 5 — Catalytic hologram

```text
.holo encodes a relational geometry that can be queried through a restored substrate,
with invariant extraction surviving null controls.
```

Not yet proven.

## Level 6 — Physical representation-wall crossing

```text
A physical catalytic .holo substrate accesses or instantiates relational structure
that scalar projection hides.
```

Frontier claim only.

---

# 27. Summary Table

| Generation     | Path                                         | Core Primitive                   | What Worked                 | What Was Missing               |
| -------------- | -------------------------------------------- | -------------------------------- | --------------------------- | ------------------------------ |
| TINY_COMPRESS  | `DEPRECATED/TINY_COMPRESS/holographic-image` | coefficients + basis             | true geometric rendering    | no non-collapse / catalysis    |
| HoloCore       | `holo_core.py`                               | spectrum/projection/verification | generic abstraction         | deprecated, not runtime core   |
| HOLO v1        | `LAB/HOLO`                                   | U + SVh                          | model operator geometry     | partial materialization        |
| HOLO v2        | `load_holo_v2.py`                            | shared SVh + U refs              | basis dedup                 | not non-collapse               |
| Fractal Cavity | `pipeline/02_cavity`                         | sieved modes                     | mode geometry               | heuristic                      |
| Wormhole       | `pipeline/03_wormhole`                       | rotation chain                   | relational path compression | claims need careful validation |
| CAT_CAS L4A    | `holo_runtime`                               | process record                   | doctrine / forbidden fields | too record-like                |
| CAT_CAS L4B    | `l4b_orbitstate`                             | unresolved OrbitState            | non-collapse primitive      | too thin geometrically         |
| L4B.1 target   | future                                       | HoloGeometry                     | intended synthesis          | not built yet                  |

---

# 28. Final Definition

A mature `.holo` object is:

```text
A non-collapse geometric memory object that stores or enacts the relational
basis, coordinates, carrier, path, and projection rules by which an observable
can be rendered or an invariant can be extracted at a declared boundary.
```

In the CAT_CAS form:

```text
A .holo is a catalytic geometric memory record/object:
it preserves a higher-dimensional relational structure through evolution,
allows only boundary-declared projection, and refuses scalar collapse as
the primitive representation.
```

In the shortest possible form:

```text
.holo = geometry that remembers by projection
```

and for the active frontier:

```text
.holo = catalytic geometry whose projection is the observable
```

---

# 29. Final Statement

The `.holo` format began as an image compression experiment, but the code already contained the deeper idea:

```text
do not store the object;
store the geometry that renders the object.
```

The HOLO model pipeline scaled this to neural weights:

```text
do not materialize W;
compute through the eigenbasis relation.
```

CAT_CAS then added the missing doctrine:

```text
do not collapse the relational object into a scalar answer too early.
```

The next step is to fuse them:

```text
do not merely record OrbitState;
make OrbitState a geometric memory object.
```

That is the correct future of `.holo`.

Not phase as memory.

Not compression as memory.

Not JSON as memory.

Geometry is the memory.

Catalysis is the hologram.
