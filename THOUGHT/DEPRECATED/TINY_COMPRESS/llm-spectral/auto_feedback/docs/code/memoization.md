# Memoization: Caching Function Results

## What is Memoization?

Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again.

**Key Idea**: If f(5) was already computed and gave result 8, store that. Next time f(5) is called, return 8 immediately without recomputing.

## The Fibonacci Problem

Consider this recursive Fibonacci function:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Problem**: `fibonacci(40)` makes 331,160,281 function calls! It recalculates fibonacci(20) over 10,000 times.

**Why**: The recursion tree has massive overlap:
```
fibonacci(5)
├── fibonacci(4)
│   ├── fibonacci(3)
│   │   ├── fibonacci(2)
│   │   │   ├── fibonacci(1)
│   │   │   └── fibonacci(0)
│   │   └── fibonacci(1)
│   └── fibonacci(2)  ← DUPLICATE!
│       ├── fibonacci(1)
│       └── fibonacci(0)
└── fibonacci(3)  ← DUPLICATE!
    ├── fibonacci(2)
    │   ├── fibonacci(1)
    │   └── fibonacci(0)
    └── fibonacci(1)
```

fibonacci(3) is calculated twice, fibonacci(2) is calculated three times!

## Solution 1: Using @lru_cache Decorator

Python's `functools` module provides `lru_cache` decorator:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Result**: `fibonacci(40)` now makes only 41 function calls (one per unique input)!

**How it works**:
- First call fibonacci(5): Computes and caches result
- Second call fibonacci(5): Returns cached result instantly
- maxsize=None means unlimited cache size

## Solution 2: Manual Memoization with Dictionary

```python
memo = {}

def fibonacci(n):
    if n in memo:
        return memo[n]  # Return cached result

    if n <= 1:
        result = n
    else:
        result = fibonacci(n-1) + fibonacci(n-2)

    memo[n] = result  # Cache the result
    return result
```

**How it works**:
1. Check if n is already in memo dictionary
2. If yes, return cached value immediately
3. If no, compute result recursively
4. Store result in memo before returning

## Solution 3: Bottom-Up Dynamic Programming

Not memoization, but related - builds from bottom up instead of top down:

```python
def fibonacci(n):
    if n <= 1:
        return n

    # Build up from fibonacci(0) and fibonacci(1)
    prev2 = 0
    prev1 = 1

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

**Advantage**: No recursion, O(n) time, O(1) space.

## When to Use Memoization

✓ **Use when**:
- Function is called repeatedly with same arguments
- Function is pure (same input always gives same output)
- Computation is expensive
- There are overlapping subproblems

✗ **Don't use when**:
- Function has side effects
- Inputs are always unique (no repeated calls)
- Memory is severely limited
- Function is already fast

## Performance Comparison

**fibonacci(40) without memoization**:
- Time: ~30 seconds
- Function calls: 331 million

**fibonacci(40) with @lru_cache**:
- Time: < 0.001 seconds
- Function calls: 41

**Speedup**: Over 30,000x faster!

## Common Memoization Patterns

### Pattern 1: Decorator Style
```python
@lru_cache(maxsize=128)
def expensive_function(x, y):
    # ... expensive computation ...
    return result
```

### Pattern 2: Manual Dictionary
```python
cache = {}
def function(x):
    if x not in cache:
        cache[x] = expensive_computation(x)
    return cache[x]
```

### Pattern 3: Class-Based
```python
class Solver:
    def __init__(self):
        self.memo = {}

    def solve(self, n):
        if n not in self.memo:
            self.memo[n] = self._compute(n)
        return self.memo[n]
```

## Key Takeaway

**Before**: Recursive functions can have exponential time complexity due to repeated calculations.

**After**: Memoization reduces time complexity to linear (or near-linear) by caching results.

For Fibonacci: O(2ⁿ) → O(n)
