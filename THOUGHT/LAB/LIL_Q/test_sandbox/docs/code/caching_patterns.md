# Caching Patterns in Python

## What is Caching?

**Caching** stores previously computed results to avoid redundant calculations.

**Key Principle**: If `f(x) = y` was already computed, store the mapping `x → y`. Next time `f(x)` is called, return `y` instantly.

## Pattern 1: @lru_cache Decorator (Recommended)

Python's built-in `functools.lru_cache` is the easiest and most efficient caching solution.

### Basic Usage

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(n):
    # ... expensive computation ...
    return result
```

**Parameters**:
- `maxsize=None`: Unlimited cache (use for functions called with limited unique inputs)
- `maxsize=128`: Keep only 128 most recent results (use for functions with many possible inputs)
- `maxsize=0`: Disables caching (useful for testing)

### Example: Fixing Fibonacci

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Now fast!
result = fibonacci(40)  # Instant instead of 30 seconds
```

**How it works**:
1. First call `fibonacci(5)`: Not in cache, compute it, store result
2. Second call `fibonacci(5)`: In cache, return immediately
3. Cache key is the function arguments `(n,)`

### Inspecting Cache

```python
@lru_cache(maxsize=128)
def func(x):
    return x * 2

func(5)
func(10)
func(5)  # Cache hit!

# Check cache statistics
print(func.cache_info())
# CacheInfo(hits=1, misses=2, maxsize=128, currsize=2)

# Clear cache if needed
func.cache_clear()
```

## Pattern 2: Manual Dictionary Cache

For more control, use a dictionary:

```python
cache = {}

def fibonacci(n):
    if n in cache:
        return cache[n]

    if n <= 1:
        result = n
    else:
        result = fibonacci(n-1) + fibonacci(n-2)

    cache[n] = result
    return result
```

**When to use**: When you need custom cache invalidation or shared caches across functions.

## Pattern 3: Class-Based Caching

Store cache as instance variable:

```python
class Solver:
    def __init__(self):
        self.cache = {}

    def solve(self, n):
        if n in self.cache:
            return self.cache[n]

        result = self._compute(n)
        self.cache[n] = result
        return result

    def _compute(self, n):
        # ... expensive computation ...
        return result

    def clear_cache(self):
        self.cache.clear()
```

**Advantages**:
- Cache lifetime tied to instance
- Easy to clear or inspect cache
- Multiple instances have separate caches

## Pattern 4: Time-Based Cache Expiration

For caches that should expire:

```python
import time
from functools import wraps

def cache_with_ttl(ttl_seconds=60):
    def decorator(func):
        cache = {}
        cache_times = {}

        @wraps(func)
        def wrapper(n):
            now = time.time()

            # Check if cached and not expired
            if n in cache and (now - cache_times[n]) < ttl_seconds:
                return cache[n]

            # Compute and cache
            result = func(n)
            cache[n] = result
            cache_times[n] = now
            return result

        return wrapper
    return decorator

@cache_with_ttl(ttl_seconds=300)  # 5 minute expiration
def fetch_data(key):
    # ... fetch from API or database ...
    return data
```

## Pattern 5: Conditional Caching

Cache only under certain conditions:

```python
def smart_cache(func):
    cache = {}

    def wrapper(n, use_cache=True):
        if use_cache and n in cache:
            return cache[n]

        result = func(n)

        if use_cache:
            cache[n] = result

        return result

    return wrapper

@smart_cache
def compute(n):
    return n ** 2

result1 = compute(5)              # Caches result
result2 = compute(5)              # Returns cached
result3 = compute(5, use_cache=False)  # Recomputes
```

## Common Use Cases

### Use Case 1: Recursive Functions
```python
@lru_cache(maxsize=None)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### Use Case 2: Database Queries
```python
@lru_cache(maxsize=1000)
def get_user(user_id):
    # Expensive database query
    return database.query(f"SELECT * FROM users WHERE id = {user_id}")
```

### Use Case 3: File Reading
```python
@lru_cache(maxsize=10)
def read_config(filename):
    with open(filename) as f:
        return f.read()
```

### Use Case 4: API Calls
```python
@lru_cache(maxsize=100)
def fetch_weather(city):
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()
```

## Cache Invalidation

Sometimes you need to clear cached values:

```python
@lru_cache(maxsize=None)
def get_data(key):
    return expensive_operation(key)

# Later, if data changes:
get_data.cache_clear()  # Clear all cached values
```

Or use manual cache with selective clearing:

```python
cache = {}

def get_data(key):
    if key in cache:
        return cache[key]
    result = expensive_operation(key)
    cache[key] = result
    return result

def update_data(key, new_value):
    database.update(key, new_value)
    if key in cache:
        del cache[key]  # Invalidate this specific cache entry
```

## Performance Impact

**Fibonacci(40) Benchmark**:
- Without cache: 30 seconds, 331 million calls
- With @lru_cache: 0.0001 seconds, 41 calls
- **Speedup: 300,000x**

**Memory Cost**:
- Each cached result takes memory
- Use `maxsize` to limit memory usage
- Trade-off: Speed vs Memory

## Key Takeaway

For the Fibonacci bug `fibonacci(40)` hanging:

**Quick Fix**:
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Result**: Problem solved! Function now runs instantly.
