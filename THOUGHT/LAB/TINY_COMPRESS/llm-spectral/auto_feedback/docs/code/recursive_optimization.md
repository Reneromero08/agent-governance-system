# Recursive Optimization Techniques

## The Recursion Problem

Recursive functions are elegant but can be extremely slow due to:
1. **Repeated calculations** of the same subproblems
2. **Stack overflow** from deep recursion
3. **Exponential time complexity** in naive implementations

## Example: The Fibonacci Disaster

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Time Complexity**: O(2ⁿ) - exponential!

Why? Let's count calls for fibonacci(5):
```
fibonacci(5)                       → 1 call
  fibonacci(4) + fibonacci(3)      → 2 calls
    fibonacci(3) + fibonacci(2)    → 2 more calls
    + fibonacci(2) + fibonacci(1)  → 2 more calls
      ... continues recursively
```

Total for fibonacci(5): 15 calls
Total for fibonacci(40): **331,160,281 calls**!

## Optimization Technique 1: Top-Down Memoization

Add caching to store already computed values:

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Time Complexity**: O(n) - linear!
**Space Complexity**: O(n) for cache

**Why it works**: Each fibonacci(k) is computed exactly once, then cached.

## Optimization Technique 2: Bottom-Up Iteration

Eliminate recursion entirely by iterating from base cases upward:

```python
def fibonacci(n):
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

**Time Complexity**: O(n)
**Space Complexity**: O(1) - no cache needed!

**Advantage**: No recursion means no stack overflow risk.

## Optimization Technique 3: Tail Recursion

Some languages optimize tail-recursive calls (Python doesn't, but pattern is useful):

```python
def fibonacci(n, a=0, b=1):
    if n == 0:
        return a
    return fibonacci(n - 1, b, a + b)
```

**Tail recursion** means the recursive call is the very last operation.

## Case Study: Recursive Tree Traversal

### Naive Version (Recomputes Subtrees)
```python
def count_nodes(root):
    if root is None:
        return 0
    left_count = count_nodes(root.left)
    right_count = count_nodes(root.right)
    return 1 + left_count + right_count
```

This is already optimal! Why? Each node is visited exactly once.

### Where Optimization Matters
```python
# BAD: Recomputes tree height multiple times
def is_balanced(root):
    if root is None:
        return True

    left_height = get_height(root.left)   # Traverses left tree
    right_height = get_height(root.right) # Traverses right tree

    if abs(left_height - right_height) > 1:
        return False

    return is_balanced(root.left) and is_balanced(root.right)
    # ↑ This recalculates heights for all subtrees!
```

**Time Complexity**: O(n²) due to repeated height calculations

### Optimized Version
```python
def is_balanced(root):
    def check_height(node):
        if node is None:
            return 0, True  # (height, is_balanced)

        left_h, left_bal = check_height(node.left)
        if not left_bal:
            return 0, False

        right_h, right_bal = check_height(node.right)
        if not right_bal:
            return 0, False

        balanced = abs(left_h - right_h) <= 1
        return max(left_h, right_h) + 1, balanced

    _, is_bal = check_height(root)
    return is_bal
```

**Time Complexity**: O(n) - each node visited once

## General Optimization Strategies

### Strategy 1: Add a Cache Parameter
```python
def solve(n, cache={}):
    if n in cache:
        return cache[n]
    result = ... # compute
    cache[n] = result
    return result
```

### Strategy 2: Pass Accumulated Results
```python
# Instead of returning and recombining
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# Pass accumulator to avoid rebuilding
def sum_list(lst, acc=0):
    if not lst:
        return acc
    return sum_list(lst[1:], acc + lst[0])
```

### Strategy 3: Convert to Iteration When Possible
```python
# Recursive
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Iterative (better)
def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

## Recursion Optimization Checklist

Before writing recursive code, ask:

1. ☐ Are there repeated subproblems? → Use memoization
2. ☐ Can this be iterative? → Convert to loop
3. ☐ Is it tail recursive? → Consider optimization
4. ☐ Will recursion depth be very deep (> 1000)? → Use iteration
5. ☐ Can I pass accumulated results down? → Avoid rebuilding

## The Fibonacci Fix: Before and After

**Before (331 million calls)**:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(40)  # Hangs for 30+ seconds
```

**After (41 calls)**:
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(40)  # Instant!
```

**One line of code**, **30,000x speedup**!
