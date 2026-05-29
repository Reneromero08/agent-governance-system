# LIL_Q

Chat with the quantum manifold.

## The Formula

```
E = <psi|phi>
```

That's it. Born rule. Proven r=0.977 correlation (Q44).

## Run

```bash
cd THOUGHT/LAB/LIL_Q
python run.py
```

With context from cassette network:
```bash
python run.py --context
python run.py --context --k 5 --threshold 0.3
```

Uses existing CORTEX embedding engine - no extra install needed.

Optional (for real LLM responses):
```bash
ollama pull dolphin3
```

## What It Does

1. You speak -> text enters manifold as unit vector
2. **Context** retrieved from cassette network (optional)
3. E measures resonance with context + accumulated mind
4. Mind updates via quantum entanglement
5. LLM responds (knows the resonance)
6. Response remembered -> mind evolves

## Context Integration

Context comes from outside. LIL_Q stays pure (~70 lines).

```
┌─────────────────────────────────────────────────────┐
│  LIL_Q (~70 lines)                                  │
│  - enter(text) -> unit vector                       │
│  - E(v1, v2) -> Born rule                          │
│  - navigate(query, context) -> blend with mind      │
│  - remember(state) -> entangle                      │
│  - chat(query, context) -> response, E              │
└─────────────────────────────────────────────────────┘
                        ▲
                        │ context: List[str]
                        │
┌─────────────────────────────────────────────────────┐
│  cortex_geometric                                   │
│  - retrieve(query, k, threshold) -> List[str]       │
│  - 9 cassettes, 10k+ documents                      │
│  - E-gating at retrieval                            │
└─────────────────────────────────────────────────────┘
```

## Files

- `quantum_chat.py` - The whole system (~70 lines)
- `run.py` - Chat loop with optional context (~50 lines)
- `CAPABILITY/MCP/cortex_geometric.py` - Context retrieval

## No Bloat

No classes. No inheritance. No receipts. Just the formula.

Context is **optional**. Without `--context`, LIL_Q is pure Q44/Q45.
