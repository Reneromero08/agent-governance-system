---
name: adr-create
version: "0.1.0"
description: Create a new Architecture Decision Record with auto-numbering and template
compatibility: all
governed: true
---

**required_canon_version:** >=3.0.0

# Skill: adr-create

**Version:** 0.1.0

**Status:** Active

## Purpose

Creates new Architecture Decision Records (ADRs) with automatic numbering, proper template structure, and correct file naming conventions. This ensures consistent ADR formatting across the repository.

## Trigger

Use when:
- A new architectural decision needs to be documented
- Recording rationale for significant design choices
- Creating a new governance record

## Inputs

JSON object:
- `title` (string, required): Title of the ADR (e.g., "Use SQLite for Cortex")
- `context` (string, optional): Context/problem description
- `decision` (string, optional): Decision text
- `status` (string, optional): Status of the ADR. One of: proposed, accepted, deprecated, superseded. Default: "proposed"

## Outputs

JSON object:
- `created` (boolean): Whether the ADR was created successfully
- `path` (string): Relative path to the created ADR file
- `adr_number` (integer): The assigned ADR number
- `title` (string): The ADR title
- `message` (string): Success message with next steps

## Constraints

- Title is required
- ADR number is auto-assigned based on existing ADRs
- File is created in LAW/CONTEXT/decisions/
- Filename format: ADR-NNN-slug.md
- Governed tool: requires write permissions to decisions directory

## Implementation

1. Scans existing ADRs to find next number
2. Generates slug from title (lowercase, alphanumeric, hyphens)
3. Creates ADR file with standard template
4. Returns path and metadata

## Fixtures

- `fixtures/basic/` - Basic ADR creation test (deterministic mode)

**required_canon_version:** >=3.0.0
