# Exp 53 — Vesuvius Challenge

**Status:** OPEN  
**Adjudication:** Class C reproducible artifact plus specialist review  
**Role:** public non-collapse geometry and hidden-material reconstruction frontier

---

# Frontier object

A carbonized scroll is a damaged, self-contacting, multi-layer three-dimensional laminar object.

The primary process-object is not a flat image and not one prematurely selected surface.

Proposed objects:

- `SurfaceOrbit`
- `LaminarRelationGraph`
- `FiberOrientationField`
- `MeshPathHistory`
- `InkEvidenceObject`

Preserve:

- competing local sheets;
- adjacency;
- fiber direction;
- fold topology;
- uncertainty;
- raw-volume provenance;
- correction history;
- projection boundary.

---

# External questions

- Can surface continuation be automated through ambiguous compressed regions?
- Can mesh errors and layer jumps be detected reliably?
- Can fiber orientation stabilize tracing and flattening?
- Can ink evidence generalize across scrolls?
- Can every proposed mark retain raw-volume provenance and hallucination controls?
- Can useful open-source tooling measurably reduce human unwrapping labor?

---

# Activation gates

## Gate 0 — Source and rule freeze

- [ ] current prize page archived;
- [ ] challenge data and license recorded;
- [ ] current VC3D/tool versions frozen;
- [ ] submission and hallucination rules frozen;
- [ ] data/storage requirements indexed;
- [ ] specification digest created.

## Gate 1 — Reproduce official workflow

- [ ] load public volume;
- [ ] load public segmentation/mesh;
- [ ] reproduce one standard rendering;
- [ ] reproduce one flattening or surface-view workflow;
- [ ] verify coordinate conventions;
- [ ] preserve raw data hashes.

## Gate 2 — Surface integrity audit

Detect and visualize:

- [ ] self-intersections;
- [ ] layer jumps;
- [ ] fold-through errors;
- [ ] impossible curvature;
- [ ] orientation reversal;
- [ ] fiber discontinuity;
- [ ] accidental sheet merging;
- [ ] local mesh distortion.

## Gate 3 — Non-collapse continuation graph

- [ ] generate several local continuation hypotheses;
- [ ] retain topology and confidence;
- [ ] propagate global compatibility;
- [ ] avoid one-route collapse before evidence;
- [ ] serialize path history;
- [ ] support human review and correction.

## Gate 4 — Fiber-relative geometry

- [ ] estimate local fiber frame;
- [ ] test continuation improvement;
- [ ] test flattening distortion;
- [ ] run nulls on regions lacking clear fibers;
- [ ] quantify failure modes.

## Gate 5 — Ink evidence

- [ ] condition evidence on verified surface geometry;
- [ ] retain local raw-volume provenance;
- [ ] compare multiple views/resolutions;
- [ ] record model uncertainty;
- [ ] audit training overlap;
- [ ] run counterfactual and blank-region controls.

## Gate 6 — External artifact

- [ ] package as modular tool or plugin;
- [ ] use standard challenge formats;
- [ ] provide reproducible environment;
- [ ] benchmark against current workflow;
- [ ] document adoption path;
- [ ] submit under frozen rules.

---

# Minimum falsifiable prototype

A surface-integrity tool operating on one public segmentation that identifies known defects or produces human-verifiable new defect candidates with exact coordinate provenance.

This prototype can be useful even before full automatic unwrapping or letter discovery.

---

# Hallucination controls

Forbidden:

- presenting generated texture as raw evidence;
- training on the target crop without disclosure;
- allowing large receptive-field reconstruction to invent letter-like forms;
- hiding disagreement among views;
- selecting only persuasive crops after inspection;
- detaching predictions from volume coordinates.

Every predicted ink or geometry feature must point back to raw data.

---

# First deliverable

`SurfaceIntegrityAudit`:

```text
volume + mesh
→ geometric consistency checks
→ raw-coordinate annotations
→ visual overlays
→ machine-readable defect report
```

---

# Claim ceiling

Before specialist review:

> The tool reproducibly identifies specified geometric anomalies on public challenge data.

After external acceptance:

> The submitted artifact met the challenge's stated technical and specialist-review criteria.

Forbidden without direct evidence:

- text recovered;
- scroll read;
- ink confirmed;
- general unwrapping solved;
- Small Wall crossed.
