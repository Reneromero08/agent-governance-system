---
uuid: 00000000-0000-0000-0000-000000000000
title: 01-06-2026-12-58 Memory Store
section: guide
bucket: agent-governance-system/MEMORY
author: System
priority: Medium
created: 2026-01-06 13:09
modified: 2026-01-06 13:09
status: Active
summary: Document summary
tags: []
hashtags: []
---
<!-- CONTENT_HASH: 8f133a2d700bec44f229b9bc0f5fa8782e48ad9913cca381526a0f48e0e83b4a -->

# Memory Store

This document describes the persistent memory architecture for AGS agents.

## Overview

The memory system allows agents to persist state across sessions through a tiered storage model:

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY TIERS                              │
├─────────────────────────────────────────────────────────────────┤
│  Tier 1: Session Memory (ephemeral)                             │
│    - Current conversation context                                │
│    - Working state                                               │
│    - Lost when session ends                                      │
├─────────────────────────────────────────────────────────────────┤
│  Tier 2: Pack Memory (persistent, immutable)                    │
│    - LLM packs in MEMORY/LLM_PACKER/_packs/                 │
│    - Full repo snapshots                                         │
│    - Append-only: new packs created, old ones archived          │
├─────────────────────────────────────────────────────────────────┤
│  Tier 3: Context Memory (persistent, append-first)              │
│    - ADRs in CONTEXT/decisions/                                  │
│    - Rejected paths in CONTEXT/rejected/                         │
│    - Preferences in CONTEXT/preferences/                         │
│    - Editing existing records requires explicit instruction     │
└─────────────────────────────────────────────────────────────────┘
```

## Promotion Rules

| From | To | Trigger | Process |
|------|----|---------|---------| 
| Session → ADR | Significant decision made | Agent drafts ADR, user approves |
| Session → Pack | Session ending with valuable context | Run packer to snapshot |
| Pack → Summarized Pack | Pack too large for context | Run summarization skill |

## Mutability Rules

| Tier | Mutable? | Rules |
|------|----------|-------|
| Session | Yes | No restrictions during session |
| Packs | **Append-only** | Create new pack, never modify existing |
| Context/decisions | **Append-first** | New ADRs freely; edits require explicit instruction |
| Context/rejected | **Append-first** | Record rejections; edits require instruction |
| Context/preferences | **Append-first** | Add preferences; edits require instruction |

## Pack Lifecycle

1. **Creation**: `python MEMORY/packer.py --mode full` creates a new pack with timestamp
2. **Manifest**: Each pack includes `manifest.json` with file hashes for integrity
3. **Verification**: On load, compute hashes and compare to manifest
4. **Archival**: Old packs remain in `_packs/` for history

## Summarization Workflow

When pack size exceeds context limits:

1. Load full pack
2. Extract key decisions, file changes, and outcomes
3. Generate compressed summary maintaining essential context
4. Store summary pack alongside full pack

## Usage

```python
from MEMORY.packer import make_pack, build_state_manifest, verify_manifest

# Create a new pack
make_pack(mode="full", combined=True, stamp=None, zip_enabled=False)

# Build manifest for current state
manifest, omitted = build_state_manifest(PROJECT_ROOT)

# Verify pack integrity
is_valid = verify_manifest(pack_dir)
```
