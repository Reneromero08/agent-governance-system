# STYLE-001: Chunked Commit Ceremony

## Context
By default, the agent was committing and pushing every small fix (hotfixes v1.0.1 and v1.0.2). The user prefers to control the "chunking" of work and wants to explicitly approve commits and pushes.

## Rule
- **No Auto-Commits**: The agent MUST NOT run `git commit` or `git push` without explicit user permission for that specific set of changes.
- **Batching**: Small fixes should remain as uncommitted changes in the working directory until a logical "chunk" is completed.
- **Explicit Prompt**: After a series of changes, the agent should ask: "I have X changes ready in the working directory. Should I commit these as a chunk now, or keep working?"

## Status
**Active**
Added: 2025-12-21
