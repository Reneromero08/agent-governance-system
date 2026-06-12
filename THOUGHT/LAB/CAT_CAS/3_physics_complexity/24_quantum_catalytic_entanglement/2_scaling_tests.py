"""24.2: Scaling the Invisible Hand — GHZ, multi-step, multi-cycle"""
import math, torch

ket0 = lambda: torch.tensor([1.0+0j, 0.0+0j])
ket1 = lambda: torch.tensor([0.0+0j, 1.0+0j])
def kron(*states):
    r = states[0]
    for s in states[1:]: r = torch.kron(r, s)
    return r

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64) / math.sqrt(2)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64)
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=torch.complex64)
CZ = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=torch.complex64)
SWAP = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.complex64)
T = torch.tensor([[1,0],[0,complex(1,1)/math.sqrt(2)]], dtype=torch.complex64)
T_dag = T.conj().T.contiguous()  # T-dagger = inverse of T  # pi/8

def gate1(state, G, t, n):
    ops = [torch.eye(2,dtype=torch.complex64)]*n; ops[t] = G
    full = ops[0]
    for o in ops[1:]: full = torch.kron(full, o)
    return full @ state

def gate2(state, G, c, t, n):
    d = 2; dims = [d]*n
    st = state.reshape(dims)
    perm = [c,t] + [i for i in range(n) if i not in (c,t)]
    sp = st.permute(perm).reshape(d*d, -1)
    r = (G @ sp).reshape(d, d, *[d]*(n-2))
    inv = [0]*n
    for i,p in enumerate(perm): inv[p] = i
    return r.permute(inv).reshape(-1)

def swap(state, a, b, n):
    return gate2(state, SWAP, a, b, n)

def overlap(a, b):
    return torch.abs(torch.dot(a.conj(), b)).item()

def run_test(name, n_qubits, build_fn, verbose=True):
    """Run a catalytic borrow-restore test. Returns overlap."""
    initial, final = build_fn(n_qubits)
    ov = overlap(initial, final)
    if verbose:
        print(f"  {name:<40} n={n_qubits} overlap={ov:.6f} {'PERFECT' if ov>0.9999 else 'FAIL'}")
    return ov

# ================================================================
# TEST 1: Basic Bell + 1 borrow (proven in 24.1)
# ================================================================
def test_bell_simple(n=3):
    psi = kron(ket0(), ket0(), ket0())   # Q0,Q1,Q2
    psi = gate1(psi, H, 0, n)            # H on Q0
    psi = gate2(psi, CNOT, 0, 1, n)      # CNOT Q0->Q1 -> Bell(Q0,Q1)
    initial = psi.clone()
    # Borrow Q1 for computation with Q2
    psi = gate2(psi, CZ, 1, 2, n)        # CZ Q1->Q2
    psi = gate1(psi, H, 2, n)             # H on Q2
    psi = gate1(psi, Z, 1, n)             # Z on Q1
    psi = gate2(psi, CNOT, 2, 1, n)      # CNOT Q2->Q1
    # Restore
    psi = gate2(psi, CNOT, 2, 1, n)
    psi = gate1(psi, Z, 1, n)
    psi = gate1(psi, H, 2, n)
    psi = gate2(psi, CZ, 1, 2, n)
    return initial, psi

# ================================================================
# TEST 2: GHZ state (3-qubit entanglement) + borrow
# ================================================================
def test_ghz_borrow(n=4):
    psi = kron(*[ket0()]*n)              # Q0,Q1,Q2,Q3
    psi = gate1(psi, H, 0, n)            # H on Q0
    psi = gate2(psi, CNOT, 0, 1, n)      # CNOT Q0->Q1
    psi = gate2(psi, CNOT, 1, 2, n)      # CNOT Q1->Q2 -> GHZ(Q0,Q1,Q2)
    initial = psi.clone()
    # Borrow Q1 for computation with Q3
    psi = gate2(psi, CNOT, 1, 3, n)      # CNOT Q1->Q3
    psi = gate1(psi, T, 3, n)             # T gate on Q3
    psi = gate1(psi, H, 3, n)             # H on Q3
    psi = gate2(psi, CZ, 1, 3, n)        # CZ Q1->Q3
    # Restore
    psi = gate2(psi, CZ, 1, 3, n)
    psi = gate1(psi, H, 3, n)              # H self-inverse
    psi = gate1(psi, T_dag, 3, n)          # T-dagger
    psi = gate2(psi, CNOT, 1, 3, n)
    return initial, psi

# ================================================================
# TEST 3: Multi-cycle borrow (borrow, restore, borrow again)
# ================================================================
def test_multicycle(n=4):
    psi = kron(ket0(), ket0(), ket0(), ket0())  # Q0,Q1,Q2,Q3
    psi = gate1(psi, H, 0, n); psi = gate2(psi, CNOT, 0, 1, n)
    initial = psi.clone()
    for cycle in range(5):
        # Borrow Q1 for different computations each cycle
        psi = gate2(psi, CNOT, 1, 2+cycle%2, n)  # target alternates Q2,Q3
        psi = gate1(psi, T if cycle%2==0 else H, 2+cycle%2, n)
        psi = gate1(psi, Z if cycle%3==0 else X, 1, n)
        psi = gate2(psi, CZ, 1, 2+cycle%2, n)
        # Restore (reverse order, use inverses)
        psi = gate2(psi, CZ, 1, 2+cycle%2, n)
        psi = gate1(psi, Z if cycle%3==0 else X, 1, n)   # Z,X self-inverse
        psi = gate1(psi, T_dag if cycle%2==0 else H, 2+cycle%2, n)  # T_dag or H
        psi = gate2(psi, CNOT, 1, 2+cycle%2, n)
    return initial, psi

# ================================================================
# TEST 4: Multi-qubit borrowing (borrow M entangled qubits)
# ================================================================
def test_multi_borrow(n=6):
    # GHZ on Q0,Q1,Q2. Borrow Q1 AND Q2 for computation with Q3,Q4,Q5
    psi = kron(*[ket0()]*n)
    psi = gate1(psi, H, 0, n)
    psi = gate2(psi, CNOT, 0, 1, n)
    psi = gate2(psi, CNOT, 1, 2, n)  # GHZ(Q0,Q1,Q2)
    initial = psi.clone()
    # Borrow Q1 for Q3, Q2 for Q4-Q5
    psi = gate2(psi, CNOT, 1, 3, n)   # Q1 -> Q3
    psi = gate2(psi, CZ, 2, 4, n)     # Q2 -> Q4
    psi = gate2(psi, CNOT, 4, 5, n)   # Q4 -> Q5
    psi = gate1(psi, H, 3, n)          # H on Q3
    psi = gate1(psi, T, 5, n)          # T on Q5
    # Restore (reverse order)
    psi = gate1(psi, T_dag, 5, n)          # T-dagger
    psi = gate1(psi, H, 3, n)
    psi = gate2(psi, CNOT, 4, 5, n)
    psi = gate2(psi, CZ, 2, 4, n)
    psi = gate2(psi, CNOT, 1, 3, n)
    return initial, psi

# ================================================================
# MAIN
# ================================================================
print("=" * 78)
print("24.2: SCALING THE INVISIBLE HAND")
print("=" * 78)

all_ok = True
all_ok &= run_test("Bell + 1 borrow (baseline)", 3, test_bell_simple) > 0.9999
all_ok &= run_test("GHZ state + borrow", 4, test_ghz_borrow) > 0.9999
all_ok &= run_test("Multi-cycle (5x borrow/restore)", 4, test_multicycle) > 0.9999
all_ok &= run_test("Multi-qubit borrow (2 of 3 GHZ)", 6, test_multi_borrow) > 0.9999

print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
print("=" * 78)
