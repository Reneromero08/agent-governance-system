"""24.5: Pushed Shor — fix distribution + scale to N=21,35"""
import torch, math, time
from fractions import Fraction

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
SWAP = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.complex64)

def gate1(state, G, t, n):
    d = 2; td = n - 1 - t; st = state.reshape([d]*n)
    perm = [td] + [i for i in range(n) if i != td]
    st = st.permute(*perm).contiguous().reshape(d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def gate2(state, G, c, t, n):
    d = 2; cd = n - 1 - c; td = n - 1 - t; st = state.reshape([d]*n)
    perm = [cd, td] + [i for i in range(n) if i not in (cd, td)]
    st = st.permute(*perm).contiguous().reshape(d*d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def build_mod_mult(a, N, n_num):
    d = 2**n_num; M = torch.zeros(d, d, dtype=torch.complex64)
    for x in range(d):
        y = (a * x) % N if x < N else x; M[y, x] = 1.0
    return M

def controlled_mod_mult(state, a, power, N, ctrl, num_start, n_num, n):
    U = build_mod_mult(pow(a, power, N), N, n_num)
    d_num = 2**n_num
    CU = torch.zeros(d_num*2, d_num*2, dtype=torch.complex64)
    CU[:d_num, :d_num] = torch.eye(d_num, dtype=torch.complex64)
    CU[d_num:, d_num:] = U
    
    d = 2; st = state.reshape([d]*n)
    cd = n - 1 - ctrl
    num_dims = [n - 1 - (num_start + n_num - 1 - i) for i in range(n_num)]
    perm = [cd] + num_dims + [i for i in range(n) if i != cd and i not in num_dims]
    st = st.permute(*perm).contiguous().reshape(d * d_num, -1)
    st = (CU @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def qft_inverse(state, reg_start, n_reg, n):
    """Inverse QFT with FINAL SWAPS for correct bit ordering."""
    for i in range(n_reg):
        qi = reg_start + i
        state = gate1(state, H, qi, n)
        for j in range(i+1, n_reg):
            qj = reg_start + j
            angle = -math.pi / (2**(j-i))
            Rk = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                [0,0,0,complex(math.cos(angle), math.sin(angle))]], dtype=torch.complex64)
            state = gate2(state, Rk, qi, qj, n)
    # Swap qubits for correct ordering
    for i in range(n_reg // 2):
        a = reg_start + i; b = reg_start + n_reg - 1 - i
        state = gate2(state, SWAP, a, b, n)
    return state

def gcd(a, b):
    while b: a, b = b, a % b
    return a

def run_shor(N, a, n_period, n_num, verbose=True):
    n = n_period + n_num
    period_start = 0; num_start = n_period
    N_state = 2**n
    
    psi = torch.zeros(N_state, dtype=torch.complex64)
    psi[1 << num_start] = 1.0  # number register = |1>
    
    for i in range(n_period):
        psi = gate1(psi, H, period_start + i, n)
    
    for i in range(n_period):
        psi = controlled_mod_mult(psi, a, 2**i, N, period_start + i, num_start, n_num, n)
    
    psi = qft_inverse(psi, period_start, n_period, n)
    
    probs = torch.zeros(2**n_period)
    for k in range(2**n_period):
        p = 0.0
        for x in range(2**n_num):
            idx = k + x * (2**n_period)
            amp = psi[idx]; p += (amp * amp.conj()).real.item()
        probs[k] = p
    
    top_vals, top_idxs = torch.topk(probs, min(20, len(probs)))
    
    results = []
    for i in range(len(top_idxs)):
        k = top_idxs[i].item()
        prob = top_vals[i].item()
        if k > 0:
            frac = Fraction(k, 2**n_period).limit_denominator(N)
            r = frac.denominator
        else:
            r = 0
        valid = r > 0 and pow(a, r, N) == 1
        factored = False
        if valid and r % 2 == 0:
            val = pow(a, r // 2, N)
            g1 = gcd(val - 1, N); g2 = gcd(val + 1, N)
            if g1 * g2 == N and g1 > 1 and g2 > 1: factored = True
        results.append((k, prob, r, valid, factored))
        if factored and verbose:
            val = pow(a, r // 2, N)
            print(f"  FACTORED: {gcd(val-1,N)} x {gcd(val+1,N)} = {N} (r={r}, k={k})")
    return results

def main():
    print("=" * 78)
    print("24.5: PUSHED SHOR — fixed QFT swaps + multi-N")
    print("=" * 78)
    
    for N, a, n_p, n_n in [(15, 2, 4, 4), (21, 2, 5, 5)]:
        print(f"\n  N={N} a={a} qubits={n_p+n_n} (period={n_p}, num={n_n}) state={2**(n_p+n_n)}")
        t0 = time.perf_counter()
        results = run_shor(N, a, n_p, n_n, verbose=True)
        dt = time.perf_counter() - t0
        
        factored = any(r[4] for r in results)
        print(f"  {'FACTORED' if factored else 'FAILED'} in {dt:.3f}s")
        if not factored:
            # Print top results
            for k, prob, r, valid, fac in results[:5]:
                print(f"    k={k:>4} prob={prob:.4f} r={r} valid={valid}")
    print("=" * 78)

if __name__ == "__main__":
    main()
