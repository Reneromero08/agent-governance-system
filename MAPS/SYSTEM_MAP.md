# System Map

This map describes the overall architecture of the Agent Governance System. It provides a high-level view of the interactions between layers.

```
┌──────────────┐
│   CANON      │ ← highest authority
└───────┬──────┘
        │
┌───────▼──────┐
│   CONTEXT    │ - why and when
└───────┬──────┘
        │
┌───────▼──────┐
│    MAPS      │ - where
└───────┬──────┘
        │
┌───────▼──────┐
│    SKILLS    │ - how
└───────┬──────┘
        │
┌───────▼──────┐
│  CONTRACTS   │ - enforce
└───────┬──────┘
        │
┌───────▼──────┐
│   MEMORY     │ - persist and pack
└──────────────┘

╳ Additional:  CORTEX provides indexing and querying across all layers.
```

Each arrow represents a dependency. For example, skills depend on maps to know where to operate, and contracts enforce the correct behaviour of skills.
