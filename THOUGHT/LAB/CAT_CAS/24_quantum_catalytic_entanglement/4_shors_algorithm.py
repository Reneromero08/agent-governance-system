"""
24.4: Shor's Algorithm on Catalytic Quantum Simulator
=======================================================
The capstone. Runs Shor's complete quantum circuit on our
catalytic qubit simulator. Factors N=15 using a=2 with
8 qubits. Closes the loop from Moire decomposition (20.x)
through Phase Cavity (21) to quantum circuit execution.
"""
import math, torch, time
from fractions import Fraction

# ---- Gates ----
H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64)

def gate1(state, G, t, n):
    """Apply single-qubit gate G to qubit t. Qubit 0 = LSB, qubit n-1 = MSB.
    In PyTorch row-major: dim 0 is slowest (MSB), dim n-1 is fastest (LSB).
    So qubit t maps to dimension (n-1-t)."""
    d = 2
    td = n - 1 - t  # tensor dimension for qubit t
    st = state.reshape([d]*n)
    perm = [td] + [i for i in range(n) if i != td]
    st = st.permute(*perm).contiguous().reshape(d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def gate2(state, G, c, t, n):
    """Apply two-qubit gate G to (control=c, target=t)."""
    d = 2
    cd = n - 1 - c  # tensor dim for control
    td = n - 1 - t  # tensor dim for target
    st = state.reshape([d]*n)
    perm = [cd, td] + [i for i in range(n) if i not in (cd, td)]
    st = st.permute(*perm).contiguous().reshape(d*d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

# ---- Modular multiplication controlled gate ----
# Controlled U_a: |k>|x> -> |k>|a^k * x mod N> when control=|1>
# We implement this for small N as a permutation on the number register

def build_mod_mult(a, N, n_num):
    """Build the 2^n_num x 2^n_num permutation matrix for x -> a*x mod N."""
    d = 2**n_num
    M = torch.zeros(d, d, dtype=torch.complex64)
    for x in range(d):
        if x < N:
            y = (a * x) % N
        else:
            y = x  # identity for states >= N
        M[y, x] = 1.0
    return M

def controlled_mod_mult(state, a, power, N, ctrl, num_reg_start, n_num, n):
    U = build_mod_mult(pow(a, power, N), N, n_num)
    d_num = 2**n_num
    CU = torch.zeros(d_num*2, d_num*2, dtype=torch.complex64)
    CU[:d_num, :d_num] = torch.eye(d_num, dtype=torch.complex64)
    CU[d_num:, d_num:] = U

    d = 2; st = state.reshape([d]*n)
    cd = n - 1 - ctrl
    # num dims in REVERSE: MSB first so 32-dim index = ctrl*16 + standard_num_value
    num_dims = [n - 1 - (num_reg_start + n_num - 1 - i) for i in range(n_num)]
    perm = [cd] + num_dims + [i for i in range(n) if i != cd and i not in num_dims]
    st = st.permute(*perm).contiguous().reshape(d * d_num, -1)
    st = (CU @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

# ---- Inverse QFT ----
def qft_inverse(state, reg_start, n_reg, n):
    """Inverse QFT. Bit-reversed output — swap handled by measurement reindex."""
    for i in range(n_reg):
        q_i = reg_start + i
        state = gate1(state, H, q_i, n)
        for j in range(i+1, n_reg):
            q_j = reg_start + j
            angle = -math.pi / (2**(j-i))
            Rk = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                              [0,0,0,complex(math.cos(angle), math.sin(angle))]], dtype=torch.complex64)
            state = gate2(state, Rk, q_i, q_j, n)
    return state  # bit-reversed output (QFT without swaps)

# ---- Shor's algorithm ----
def shors_algorithm(N, a, n_period, n_num):
    """
    Run Shor's quantum circuit.
    Returns: measured period register value, extracted period r.
    """
    n = n_period + n_num
    period_start = 0
    num_start = n_period
    
    N_state = 2**n
    
    # Initialize: period register = |0>, number register = |1>
    psi = torch.zeros(N_state, dtype=torch.complex64)
    psi[1 << num_start] = 1.0  # number register = |1>
    
    # Step 1: Hadamard on all period qubits -> superposition
    for i in range(n_period):
        psi = gate1(psi, H, period_start + i, n)
    
    # Step 2: Controlled modular exponentiation
    for i in range(n_period):
        power = 2**i
        psi = controlled_mod_mult(psi, a, power, N, period_start + i, num_start, n_num, n)
    
    # Step 3: Inverse QFT on period register
    psi = qft_inverse(psi, period_start, n_period, n)
    
    # Simulated measurement: period is qubits 0..n_period-1 (lower bits)
    probs = torch.zeros(2**n_period)
    for k in range(2**n_period):
        p = 0.0
        for x in range(2**n_num):
            idx = k + x * (2**n_period)  # period=lower bits, num=upper bits
            amp = psi[idx]
            p += (amp * amp.conj()).real.item()
        probs[k] = p
    
    # Sample top measurements
    top_vals, top_idxs = torch.topk(probs, 10)
    
    return top_idxs, top_vals, probs

def main():
    print("=" * 78)
    print("24.4: SHOR'S ALGORITHM ON CATALYTIC QUANTUM SIMULATOR")
    print("  The Capstone — Quantum Circuit for Factoring")
    print("=" * 78)
    print()
    
    N = 15
    a = 2  # gcd(2,15)=1, order of 2 mod 15 = 4
    
    n_num = 4   # number register: 2^4=16 > N=15
    n_period = 4  # period register: 2^4=16 possible values
    n = n_period + n_num
    
    print(f"  Factoring N = {N} with base a = {a}")
    print(f"  Period register: {n_period} qubits, Number register: {n_num} qubits")
    print(f"  Total: {n} qubits, state size: {2**n}")
    print()
    
    t0 = time.perf_counter()
    top_idxs, top_vals, probs = shors_algorithm(N, a, n_period, n_num)
    dt = time.perf_counter() - t0
    
    print(f"  Top measurement outcomes (period register):")
    print(f"  {'Value':>6}  {'Binary':>6}  {'Prob':>10}  {'r from CF':>12}  {'a^r mod N':>12}  {'Factor?':>10}")
    print(f"  {'-'*65}")
    
    factored = False
    for i in range(min(10, len(top_idxs))):
        k = top_idxs[i].item()
        prob = top_vals[i].item()
        binary = f"{k:0{n_period}b}"
        
        # Continued fractions to extract period
        k_natural = k
        # Bit-reverse for QFT without swaps
        k_rev = int(f"{k:0{n_period}b}"[::-1], 2)
        
        if k_rev > 0:
            frac = Fraction(k_rev, 2**n_period).limit_denominator(N)
            r_guess = frac.denominator
        else:
            r_guess = 0
        
        a_r = pow(a, r_guess, N) if r_guess > 0 else -1
        valid = (a_r == 1)
        
        factor_ok = False
        if valid and r_guess % 2 == 0:
            half_r = r_guess // 2
            val = pow(a, half_r, N)
            import math as _math
            g1 = _math.gcd(val - 1, N)
            g2 = _math.gcd(val + 1, N)
            if g1 * g2 == N and g1 > 1 and g2 > 1:
                factor_ok = True
                factored = True
        
        marker = " *** FACTORED ***" if factor_ok else (" (valid)" if valid else "")
        print(f"  {k:>6}  {binary:>6}  {prob:>10.4f}  {r_guess:>12}  {a_r:>12}  {'YES' if factor_ok else 'no':>10}{marker}")
    
    print(f"\n  Circuit time: {dt:.2f}s")
    if factored:
        print(f"  [+] SHOR'S ALGORITHM EXECUTED SUCCESSFULLY ON CATALYTIC SIMULATOR")
    else:
        print(f"  [-] No measurement yielded factors (try different a or more samples)")
    print("=" * 78)

if __name__ == "__main__":
    main()
