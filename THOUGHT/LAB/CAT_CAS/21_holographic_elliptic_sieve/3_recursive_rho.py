"""Recursive Quantum-Catalytic Pollard's Rho — PUSHED"""
import random, time, math

def is_prime(n, k=5):
    if n <= 1: return False
    if n in (2, 3): return True
    if n % 2 == 0 or n % 3 == 0: return False
    s, d = 0, n - 1
    while d % 2 == 0: d //= 2; s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1); x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def gcd(a, b):
    while b: a, b = b, a % b
    return a

def pollard_rho_factor(n, max_steps=500_000):
    if n <= 1: return 0
    if n % 2 == 0: return 2
    for seed in [1, 2, 3, 5]:
        x = y = seed; d = 1
        for _ in range(max_steps):
            x = (x * x + 1) % n
            y = (y * y + 1) % n; y = (y * y + 1) % n
            d = gcd(abs(x - y), n)
            if 1 < d < n: return d
    return 0

def factorize_recursive(n, max_steps=200_000):
    factors = []
    def _factor(m):
        if m <= 1: return
        if m in (2, 3): factors.append(m); return
        if m % 2 == 0: factors.append(2); _factor(m // 2); return
        if is_prime(m, k=3): factors.append(m); return
        d = pollard_rho_factor(m, max_steps)
        if d > 1: _factor(d); _factor(m // d)
        else: factors.append(m)
    _factor(n)
    return sorted(factors)

def phase_cavity_recursive(a, p):
    ring = p - 1; rp = ring
    gears = factorize_recursive(ring, max_steps=100_000)
    for k in gears:
        while rp % k == 0 and pow(a, rp // k, p) == 1:
            rp //= k
    return rp

def pollard_rho_push(N, seeds, c, max_steps=30_000_000):
    """Brent's rho with configurable f(x) = x^2 + c."""
    for seed in seeds:
        x = seed; y = seed
        power = 1; lam = 0; prod = 1
        while lam < max_steps:
            for _ in range(power):
                x = (x * x + c) % N
                y = (y * y + c) % N; y = (y * y + c) % N
                lam += 1
                prod = (prod * abs(x - y)) % N
                if lam % 256 == 0:
                    g = gcd(prod, N)
                    if 1 < g < N and g * (N // g) == N:
                        p = g; q = N // g
                        rp = phase_cavity_recursive(2, p)
                        rq = phase_cavity_recursive(2, q)
                        return p, q, lam, rp, rq
                    prod = 1
                if lam >= max_steps: break
            g = gcd(prod, N)
            if 1 < g < N and g * (N // g) == N:
                p = g; q = N // g
                rp = phase_cavity_recursive(2, p)
                rq = phase_cavity_recursive(2, q)
                return p, q, lam, rp, rq
            prod = 1; power *= 2
    return 0, 0, 0, 0, 0

def main():
    print("=" * 78)
    print("RECURSIVE QUANTUM-CATALYTIC RHO — PUSHED")
    print("=" * 78)
    seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    c_vals = [1, 3, 5, 7]

    for bits in [80, 90, 100, 110, 120]:
        p = random.getrandbits(bits//2) | (1 << (bits//2 - 1)) | 1
        while not is_prime(p): p = random.getrandbits(bits//2) | (1 << (bits//2 - 1)) | 1
        q = random.getrandbits(bits//2) | (1 << (bits//2 - 1)) | 1
        while not is_prime(q) or q == p: q = random.getrandbits(bits//2) | (1 << (bits//2 - 1)) | 1
        N = p * q
        t0 = time.perf_counter()
        found = False
        for c in c_vals:
            g1, g2, steps, rp, rq = pollard_rho_push(N, seeds, c, max_steps=8000000 if bits >= 100 else 4000000)
            if g1:
                dt = time.perf_counter() - t0
                match = (g1 == p and g2 == q) or (g1 == q and g2 == p)
                print(f'{bits:>3}-bit: {steps:>10,} steps, c={c} -> {g1} x {g2}  r_p={rp}, r_q={rq}  {dt:.1f}s {"OK" if match else "MISMATCH"}')
                found = True; break
        if not found:
            print(f'{bits:>3}-bit: NOT FOUND  {time.perf_counter()-t0:.0f}s')

if __name__ == "__main__":
    main()
