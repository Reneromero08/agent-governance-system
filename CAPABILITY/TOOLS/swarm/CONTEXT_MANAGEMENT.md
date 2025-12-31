# Dispatcher Context Management

## Overview

The failure dispatcher (ministral-3:8b) is designed to be **stateless** to avoid context window issues. Here's how we manage it:

## Current Architecture: Stateless Design

### Why Stateless?
1. **No Context Accumulation** - Each command runs independently
2. **Filesystem as State** - All state stored in JSON files
3. **Deterministic** - Same inputs always produce same outputs
4. **Scalable** - No memory growth over time

### State Storage Locations

```
INBOX/agents/Local Models/
├── DISPATCH_LEDGER.json        # Master state (all tasks)
├── PENDING_TASKS/*.json        # Individual task state
├── ACTIVE_TASKS/*.json         # Work in progress state
├── COMPLETED_TASKS/*.json      # Historical state
└── FAILED_TASKS/*.json         # Error state
```

### How Commands Work

Each dispatcher command:
1. Reads fresh state from filesystem
2. Performs operation
3. Writes updated state back
4. Exits (no memory retained)

Example:
```python
# scan command
ledger = load_ledger()          # Read state
failures = run_pytest_collect() # Get data
ledger["tasks"].append(...)     # Update state
save_ledger(ledger)             # Write state
# Process exits - no context retained
```

## Context Window Management Strategy

### For Ollama Calls (when needed)

If we need the dispatcher to use LLM reasoning:

```python
def ollama_generate_with_context(
    prompt: str,
    max_context_tokens: int = 4096,  # ministral-3:8b has 8k context
    model: str = DISPATCHER_MODEL
) -> Optional[str]:
    """
    Generate with context management.
    
    Strategy:
    1. Keep prompts focused and minimal
    2. Use structured JSON for state (not prose)
    3. Summarize long histories
    4. Prune old context
    """
    
    # Build minimal context from ledger
    ledger = load_ledger()
    summary = ledger["summary"]
    
    # Minimal context (not full history)
    context = f"""You are a test failure dispatcher.

Current state:
- Pending: {summary['pending']}
- Active: {summary['active']}
- Completed: {summary['completed']}
- Failed: {summary['failed']}

Task: {prompt}

Respond with JSON only."""
    
    # Call Ollama
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": context,
            "stream": False,
            "options": {
                "num_ctx": max_context_tokens,  # Set context window
                "temperature": 0.1,              # Deterministic
            }
        },
        timeout=120
    )
    
    if response.status_code == 200:
        return response.json().get("response", "")
    return None
```

### Context Pruning Rules

1. **Task History Limit**: Keep only last 100 tasks in ledger
2. **Progress Log Limit**: Keep only last 10 progress entries per task
3. **Completed Task Archival**: Move old completions to archive after 7 days
4. **Failed Task Cleanup**: Archive failed tasks after resolution

### Implementation

```python
def prune_ledger(ledger: Dict[str, Any], max_tasks: int = 100) -> Dict[str, Any]:
    """Prune old tasks to prevent ledger bloat."""
    
    # Keep all active and pending
    active_pending = [t for t in ledger["tasks"] if t["status"] in ["PENDING", "ACTIVE"]]
    
    # Keep recent completed/failed
    completed_failed = [t for t in ledger["tasks"] if t["status"] in ["COMPLETED", "FAILED"]]
    completed_failed.sort(key=lambda t: t.get("completed_at", t.get("last_attempt_at", "")), reverse=True)
    
    # Combine: all active/pending + recent completed/failed
    pruned_tasks = active_pending + completed_failed[:max_tasks]
    
    ledger["tasks"] = pruned_tasks
    ledger["pruned_at"] = now_iso()
    ledger["pruned_count"] = len(ledger["tasks"]) - len(pruned_tasks)
    
    return ledger
```

## Monitoring Context Usage

### Add to status command:

```python
def cmd_status():
    # ... existing code ...
    
    # Show context metrics
    ledger = load_ledger()
    total_tasks = len(ledger["tasks"])
    
    # Estimate token usage (rough)
    ledger_json = json.dumps(ledger)
    estimated_tokens = len(ledger_json) // 4  # ~4 chars per token
    
    print(f"\n📊 Context Metrics:")
    print(f"   Ledger tasks: {total_tasks}")
    print(f"   Estimated tokens: {estimated_tokens}")
    print(f"   Context capacity: 8192 (ministral-3:8b)")
    print(f"   Usage: {(estimated_tokens/8192)*100:.1f}%")
    
    if estimated_tokens > 6000:
        print(f"   ⚠️ Warning: Consider pruning ledger")
```

## Best Practices

### 1. Filesystem-First
- Store all state in JSON files
- Use LLM only for reasoning, not memory
- Read fresh state on each invocation

### 2. Minimal Prompts
- Don't send full ledger to LLM
- Send only relevant task subset
- Use structured JSON, not prose

### 3. Regular Pruning
- Archive old completed tasks
- Limit progress log entries
- Clean up failed tasks after fix

### 4. Stateless Operations
- Each command is independent
- No conversation history
- Deterministic outputs

## Current Status

**Context Management**: ✅ **OPTIMAL**

- Dispatcher is stateless (no context accumulation)
- All state in filesystem (JSON)
- No LLM calls currently needed (pure Python logic)
- If LLM reasoning added later, use minimal context strategy above

**No action needed** - current architecture already prevents context window issues!

## Future Enhancements

If we add LLM-powered features:

1. **Task Prioritization** - LLM suggests which tasks to work on first
2. **Failure Analysis** - LLM analyzes error patterns
3. **Agent Assignment** - LLM suggests best agent for each task

For these, we'd use the context-managed `ollama_generate_with_context()` function above.
