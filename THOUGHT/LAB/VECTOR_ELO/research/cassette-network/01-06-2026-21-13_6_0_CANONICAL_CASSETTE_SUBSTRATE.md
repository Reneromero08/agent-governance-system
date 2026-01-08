---
uuid: 00000000-0000-0000-0000-000000000000
title: Canonical Cassette Substrate (Cartridge-First)
section: report
bucket: reports/v4/section_6_0
author: System
priority: High
created: 2026-01-06 21:13
modified: 2026-01-06 21:13
status: Active
summary: 'V4 roadmap report: Canonical Cassette Substrate (Cartridge-First)'
tags:
- cassette
- v4
- roadmap
- report
---
<!-- CONTENT_HASH: d281b2fdf84bdb68f1eafe2249b6fe8e9301f19f4e20d2196fe3ed56723bea09 -->

# Report: Canonical Cassette Substrate (Cartridge-First)

Date: 2026-01-05
Roadmap Section: 6.0

## Goal
Make the cassette network portable as a set of cartridge artifacts plus receipts:
- Each cassette DB is a sharable, tool-readable unit.
- Derived acceleration layers are rebuildable and disposable.
- The cassette network inherits the MemoryRecord contract from Phase 5.2.

## Scope
- Bind cassette storage to MemoryRecord contract (schema, IDs, provenance, receipts).
- Define canonical per-cassette artifact type (default: SQLite single file).
- Provide rebuild hooks for derived ANN indexes (optional).

## Why this belongs here
Cassette network is your memory substrate at scale. If it is not cartridge-first, you lose:
- portability
- verifiability
- clean distribution boundaries (crypto-safe sealing becomes harder)

## Required properties
- Deterministic schema migrations (receipted).
- Stable IDs (content-hash based).
- Byte-identical text preservation.
- Export/import with receipts.

## Derived engines
Allowed as accelerators:
- Qdrant for interactive ANN and concurrency.
- FAISS for local ANN.
- Lance or Parquet for analytics interchange.

Hard rule:
- They are never the source of truth.
- They must be rebuildable from cartridges.

## Acceptance criteria
- Cassette network can be shipped as cartridges + receipts.
- Any derived index can be rebuilt mechanically.
