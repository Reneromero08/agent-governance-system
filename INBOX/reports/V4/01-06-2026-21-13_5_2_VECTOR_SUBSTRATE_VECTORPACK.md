---
uuid: 00000000-0000-0000-0000-000000000000
title: Vector Substrate Contract, VectorPack, and Transfer Formats
section: report
bucket: reports/v4/section_5_2
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: Vector Substrate Contract, VectorPack, and Transfer Formats'
tags:
- vector
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: a1d9290f74ba85ac3513924ab9d9dda44d8738a312d8a6d26d38763ca9d39116 -->

# Report: Vector Substrate Contract, VectorPack, and Transfer Formats

Date: 2026-01-05
Roadmap Section: 5.2

## Source inputs used
- AI_SUBSTRATE_VECTOR_DB_FULL_REPORT.md
- TRANSFERABLE_VECTOR_REPORT.md
- DEEP_VECTOR_RESEARCH_REPORT.md

## Goal
Make vector memory portable, auditable, and engine-agnostic:
- A stable record contract (`MemoryRecord`) is the invariant.
- A portable “cartridge” substrate is the canonical sharable artifact.
- Derived engines (server ANN, table formats) are accelerators, rebuildable from the cartridge.
- Provide a token-readable micro-pack export for task-scoped sharing.

## The key distinction you already nailed
Two kinds of “readable”:
- Tool-readable: AI with a python kernel can load and operate on the artifact (SQLite qualifies).
- Token-readable: LLM can see it directly as text (JSONL/YAML qualify, but do not scale).

The engineering axis is tool-readable. Token-readable is only for small, scoped exports.

## Contract: MemoryRecord (canonical)
Minimum fields:
- `id`: stable content hash
- `text`: byte-identical payload or content-addressed blob ref
- `embeddings`: named vectors (one or more)
- `payload`: tags, timestamps, roles, doc ids, etc.
- `scores`: Elo, recency, trust, decay
- `lineage`: derived-from links, summarization chains
- `receipts`: provenance hashes and tool version refs

Contract rules:
- Text is canonical.
- Vectors are derived.
- All exports are receipted and hashed.

## Canonical cartridge substrate (default)
Default recommendation for “drop into GPT and navigate”:
- Single-file SQLite database (`chat.db` / cassette DBs).
- Vector math in python or via sqlite vector extensions.
- Derived indexes (FAISS/HNSW/Qdrant snapshots) are disposable.

Alternative canonical substrate (optional):
- LanceDB tables (versioned data-native substrate).
If used, it must still obey your receipts and deterministic export requirements.

## VectorPack format (transfer)
Directory layout:
- `manifest.yaml` (schema version, models, dims, metrics, hashes)
- `tables/` (canonical cartridge or portable tables)
- `blobs/` (content-addressed payloads, attachments)
- `receipts/` (build receipts, proofs, schema hashes)
- `indexes/` (optional derived indexes, rebuildable)

Determinism rules:
- Canonical ordering in manifests.
- Stable hashing of file lists and record ordering.
- Export/import emits receipts and is reproducible.

## Micro-pack export (token-friendly)
Purpose: task-scoped top-K memories, not the entire substrate.
Format: JSONL with packed vectors:
- int8 quantized + scale, or float16 packed and base64 encoded
- strict schema versioning
- deterministic selection and tie-breaking

## Retrieval governance (Elo insertion point)
Elo and relevance fusion belong on the write path and ranking function:
- vector similarity provides candidate pool
- Elo and recency modulate ranking
- updates occur after use labeling, producing receipts

This is the mechanism that lets the system behave long-context while keeping prompt context small.

## Acceptance criteria
- MemoryRecord contract is stable and enforced.
- Canonical cartridge exists and is portable.
- VectorPack export/import is deterministic and receipted.
- Micro-pack export exists for task-scoped sharing.
