import numpy as np
import math

def spacing_ratio(ev):
    angles = np.sort(np.angle(ev))
    if len(angles) < 4: return 0.0
    sp = np.diff(angles)
    sp = sp[sp > 1e-15]
    if len(sp) < 2: return 0.0
    ms = sp.mean()
    if ms < 1e-15: return 0.0
    uf = sp / ms
    r = []
    for i in range(len(uf)-1):
        a, b = min(uf[i], uf[i+1]), max(uf[i], uf[i+1])
        if b > 1e-15: r.append(a/b)
    return np.mean(r) if r else 0.0

def primes_upto_N_count(N):
    est = int(N * (math.log(N) + math.log(math.log(N)))) if N > 6 else 20
    sieve = np.ones(est, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(est**0.5) + 1):
        if sieve[i]: sieve[i*i:est:i] = False
    return np.where(sieve)[0][:N].astype(np.float64)

def test_matrix(N, mode="primes"):
    if mode == "primes":
        vals = np.log(primes_upto_N_count(N))
    elif mode == "integers":
        vals = np.log(np.arange(2, N + 2))
    elif mode == "random":
        vals = np.random.rand(N) * 10
        
    S = np.zeros((N, N), dtype=np.complex128)
    for m in range(N):
        for n in range(N):
            S[m, n] = np.exp(1j * vals[m] * vals[n])
            
    U, _, Vh = np.linalg.svd(S)
    S_unitary = U @ Vh
    return spacing_ratio(np.linalg.eigvals(S_unitary))

print("N=400 Spacing Ratios:")
print(f"Primes:   {test_matrix(400, 'primes'):.4f}")
print(f"Integers: {test_matrix(400, 'integers'):.4f}")
print(f"Random:   {test_matrix(400, 'random'):.4f}")
