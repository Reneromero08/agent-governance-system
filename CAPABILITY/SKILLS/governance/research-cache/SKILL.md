---
name: research-cache
version: "0.1.0"
description: Access and manage the persistent research cache to avoid redundant browsing
compatibility: all
governed: true
---

**required_canon_version:** >=3.0.0

# Skill: research-cache

**Version:** 0.1.0

**Status:** Active

## Purpose

Provides persistent caching for research results to avoid redundant web browsing and API calls. Allows looking up, saving, listing, and clearing cached research summaries with optional tagging.

## Trigger

Use when:
- Checking if a URL has been researched before
- Saving research summaries for future reference
- Listing cached research entries
- Clearing the research cache

## Inputs

JSON object:
- `action` (string, required): One of: "lookup", "save", "list", "clear"

For "lookup" action:
- `url` (string, required): URL to look up

For "save" action:
- `url` (string, required): URL to save
- `summary` (string, required): Research summary content
- `tags` (string, optional): Comma-separated tags

For "list" action:
- `filter` (string, optional): Tag filter to narrow results

For "clear" action:
- No additional parameters

## Outputs

For "lookup":
- `found` (boolean): Whether the URL was found in cache
- `summary` (string, optional): Cached summary if found
- `tags` (array, optional): Tags associated with entry

For "save":
- `saved` (boolean): Whether save was successful
- `url` (string): URL that was saved

For "list":
- `entries` (array): List of cached entries with url, summary preview, tags

For "clear":
- `cleared` (boolean): Whether cache was cleared

## Constraints

- Governed tool: requires write access for save/clear operations
- Uses CAPABILITY/TOOLS/research_cache.py as backend
- Cache persists across sessions

## Implementation

Wrapper around `CAPABILITY/TOOLS/research_cache.py` that:
1. Parses action and parameters
2. Calls appropriate research_cache.py subcommand
3. Returns structured JSON result

## Fixtures

- `fixtures/basic/` - Test list action (deterministic)

**required_canon_version:** >=3.0.0
