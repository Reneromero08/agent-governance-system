# Exp 55 — Hutter Prize

**Status:** OPEN  
**Adjudication:** Class A exact decompression with administrative/publication conditions  
**Role:** exact external test of `.holo` representation, catalytic search, and byte-perfect restoration

---

# Frontier object

The object is the complete fixed corpus plus the executable reconstruction relation.

Proposed process objects:

- `CompressionHolo`
- `CorpusRelationGraph`
- `GrammarOrbit`
- `ContextModelFamily`
- `ExceptionTape`
- `ReconstructionPath`

The target is not a plausible semantic summary. It is exact byte reconstruction under the official size, runtime, memory, and decompressor-accounting rules.

---

# External questions

- Can document, markup, entity, syntax, and semantic relations be encoded more compactly than current context models?
- Can a reversible structural transform reduce entropy after including its inverse description?
- Can long-range repeated geometry be represented without a huge model?
- Can several context models remain active and combine only at the coding boundary?
- Can `.holo` store transformation geometry rather than a flat prediction table?
- Can borrowed search workspace be restored while retaining only the winning reversible transform path?

---

# Activation gates

## Gate 0 — Rule and corpus freeze

- [ ] official rules retrieved and archived;
- [ ] exact corpus obtained;
- [ ] corpus digest verified;
- [ ] current record and accounting frozen;
- [ ] runtime/memory/platform rules frozen;
- [ ] source/publication conditions frozen;
- [ ] specification digest created.

## Gate 1 — Exact baseline

- [ ] official size accounting implemented;
- [ ] baseline compressor reproduced;
- [ ] exact decompression verified;
- [ ] decompressor size counted;
- [ ] runtime and memory measured;
- [ ] corpus hash restored.

## Gate 2 — Reversible corpus parser

Separate exact typed streams for:

- [ ] markup;
- [ ] tags;
- [ ] text;
- [ ] whitespace;
- [ ] punctuation;
- [ ] numbers and dates;
- [ ] references;
- [ ] templates;
- [ ] title/section structure.

The parser and inverse must roundtrip every byte.

## Gate 3 — Relational dictionary

Model:

- [ ] repeated substrings;
- [ ] template roles;
- [ ] entity recurrence;
- [ ] inflectional families;
- [ ] citation structures;
- [ ] table schemas;
- [ ] document motifs;
- [ ] cross-document near-duplicates.

Every dictionary structure must justify its own description cost.

## Gate 4 — Non-collapse context family

Maintain multiple predictors:

- [ ] local byte context;
- [ ] token context;
- [ ] markup context;
- [ ] document position;
- [ ] entity relation;
- [ ] numerical pattern;
- [ ] long-range recurrence.

Do not permanently select one model before the coding boundary. Track the cost of the mixture itself.

## Gate 5 — `.holo` reconstruction path

- [ ] store basis patterns;
- [ ] store reversible transforms;
- [ ] store exception tape;
- [ ] store reconstruction order;
- [ ] prove exact reload and reconstruction;
- [ ] compare total size against simpler representations.

## Gate 6 — Record attempt

- [ ] total size beats frozen target by required margin;
- [ ] exact decompression passes independently;
- [ ] source and executable package complete;
- [ ] runtime/memory rules pass;
- [ ] publication/disclosure conditions satisfied.

---

# Fastest falsifiable prototype

A reversible wiki-structure transform that reduces the compressed size of a fixed corpus slice after including:

- transform metadata;
- inverse code;
- exceptions;
- decompressor cost.

The mechanism fails if its total accounted size is not smaller than the baseline.

---

# No-smuggle model

Forbidden:

- external training data not permitted by the rules;
- omitting model or decompressor size;
- lossy reconstruction;
- using the original corpus as hidden side information;
- platform-specific files absent from accounting;
- reporting component gain without end-to-end size;
- selecting a favorable corpus slice as record evidence.

---

# First deliverable

`ReversibleWikiTransform` with:

- exact parser;
- inverse;
- corpus-slice tests;
- full accounting;
- comparison against a frozen baseline;
- artifact/replay bundle.

---

# Claim ceiling

Before a record:

> The declared reversible representation reduces total accounted size on the frozen evaluation slice while restoring every byte.

After official acceptance:

> The submitted compressor met the Hutter Prize's frozen record and administrative conditions.

Forbidden without evidence:

- language understood generally;
- `.holo` is universally optimal;
- zero physical energy;
- Big Wall broken.
