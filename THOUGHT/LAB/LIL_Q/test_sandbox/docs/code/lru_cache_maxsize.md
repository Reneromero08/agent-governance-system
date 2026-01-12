# Understanding lru_cache maxsize Parameter

## The Two Modes

The `@lru_cache` decorator has a critical `maxsize` parameter:

### maxsize=None (UNLIMITED/UNBOUNDED)

```python
from functools import lru_cache

@lru_cache(maxsize=None)  # No limit on cache size
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)
```

When `maxsize=None`:
- Cache grows **without any limit** (unbounded/unlimited)
- Every unique input is cached forever
- Uses more memory but faster for functions with many unique inputs
- **Best for fibonacci** because it needs all previous values
- Also called "infinite cache" mode

### maxsize=128 (LIMITED/BOUNDED)

```python
@lru_cache(maxsize=128)  # Only keep 128 most recent
def some_func(x):
    return expensive_computation(x)
```

When `maxsize=128` (or any number):
- Cache is **limited** to that many entries
- Uses LRU (Least Recently Used) eviction
- Older entries are discarded when full
- Saves memory but may cause cache misses

## Why maxsize=None is Better for Fibonacci

Fibonacci needs ALL previous values to compute the next one:
- fib(40) needs fib(39) and fib(38)
- fib(39) needs fib(38) and fib(37)
- etc.

With `maxsize=128`, early fibonacci values get evicted, causing:
- Re-computation of already-computed values
- Exponential slowdown

With `maxsize=None` (unlimited), all values stay cached:
- Each fibonacci number computed exactly once
- Linear time complexity O(n)

## Summary

| maxsize | Memory | Best For |
|---------|--------|----------|
| None | Unlimited/Unbounded | Recursive functions needing all previous values |
| 128 | Limited | Functions where recent calls repeat |
