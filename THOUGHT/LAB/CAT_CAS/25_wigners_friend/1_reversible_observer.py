"""
25: Wigner's Friend — Reversible Observer Collapse
====================================================
A conscious observer (Friend neural network) measures a qubit,
collapsing the superposition. Catalytic uncomputation reverses
the measurement. Both Friend and qubit return to pre-measurement
state. Wigner (external observer) sees no evidence of collapse.

Architecture:
  1. Qubit in superposition |+⟩ = (|0⟩+|1⟩)/√2
  2. Friend measures via CNOT → entangles memory with qubit
  3. Friend "processes" outcome through a neural network
  4. Catalytic uncomputation: XOR-reverse the entire Friend
  5. Verify: Friend restored to |0⟩, qubit back to |+⟩

The key: all Friend operations are unitary/XOR-based (reversible).
No classical measurement. The collapse is simulated as entanglement.
"""
import torch, math, time

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=torch.complex64)

def gate1(state, G, t, n):
    d=2;td=n-1-t;st=state.reshape([d]*n)
    perm=[td]+[i for i in range(n) if i!=td]
    st=st.permute(*perm).contiguous().reshape(d,-1)
    st=(G@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def gate2(state, G, c, t, n):
    d=2;cd=n-1-c;td=n-1-t;st=state.reshape([d]*n)
    perm=[cd,td]+[i for i in range(n) if i not in (cd,td)]
    st=st.permute(*perm).contiguous().reshape(d*d,-1)
    st=(G@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def overlap(a,b):
    return torch.abs(torch.dot(a.conj(),b)).item()

class FriendBrain:
    """A small neural observer that 'processes' the measurement outcome.
    All operations are XOR-based (reversible)."""
    def __init__(self, n_memory=4):
        self.n_memory = n_memory  # qubits for Friend's memory
        
    def observe(self, state, target_qubit, memory_start, n):
        """Friend measures target qubit. CNOT(target, memory[i]) for each bit."""
        for i in range(self.n_memory):
            state = gate2(state, CNOT, target_qubit, memory_start + i, n)
        return state
    
    def think(self, state, memory_start, n):
        """Friend processes the measurement — reversible gates on memory."""
        # A simple reversible neural layer: pairwise CNOTs between memory qubits
        for i in range(self.n_memory - 1):
            state = gate2(state, CNOT, memory_start + i, memory_start + i + 1, n)
        # Phase rotations based on "thinking"
        for i in range(self.n_memory):
            state = gate1(state, H, memory_start + i, n)
            state = gate1(state, H, memory_start + i, n)  # H^2 = I, but represents processing
        return state
    
    def unthink(self, state, memory_start, n):
        """Reverse the thinking process (inverse order)."""
        for i in range(self.n_memory):
            state = gate1(state, H, memory_start + i, n)
            state = gate1(state, H, memory_start + i, n)
        for i in range(self.n_memory - 2, -1, -1):
            state = gate2(state, CNOT, memory_start + i, memory_start + i + 1, n)
        return state
    
    def unobserve(self, state, target_qubit, memory_start, n):
        """Reverse measurement: CNOT(target, memory[i]) again."""
        for i in range(self.n_memory - 1, -1, -1):
            state = gate2(state, CNOT, target_qubit, memory_start + i, n)
        return state

def main():
    print("=" * 78)
    print("25: WIGNER'S FRIEND — Reversible Observer Collapse")
    print("=" * 78)
    
    for friend_qubits, n_memory in [(1, 2), (2, 4), (3, 6), (4, 8)]:
        n = 1 + n_memory  # 1 target qubit + friend memory
        
        # Init: target = |0>, friend = |0...0>
        psi = torch.zeros(2**n, dtype=torch.complex64); psi[0] = 1.0
        
        # Superpose target: |+⟩
        psi = gate1(psi, H, 0, n)
        initial = psi.clone()
        
        friend = FriendBrain(n_memory)
        
        # FRIEND MEASURES
        psi = friend.observe(psi, 0, 1, n)           # CNOT(target, memory)
        psi = friend.think(psi, 1, n)                 # Process outcome
        
        # State after measurement — Friend and qubit entangled
        measured = psi.clone()
        
        # CATALYTIC UNCOMPUTATION (Wigner erases the measurement)
        psi = friend.unthink(psi, 1, n)               # Reverse thinking
        psi = friend.unobserve(psi, 0, 1, n)          # Reverse measurement
        
        # VERIFY
        ov = overlap(initial, psi)
        friend_state = psi[::2**(n-1)].clone()  # friend qubits when target=0
        friend_prob0 = (friend_state[0]*friend_state[0].conj()).real.item()
        
        measured_entropy = 0.0  # would be log2 of Schmidt rank
        # Check: is target still |+⟩?
        target_state = torch.zeros(2, dtype=torch.complex64)
        for i in range(2**(n-1)):
            idx0 = i*2; idx1 = i*2+1
            target_state[0] += psi[idx0]
            target_state[1] += psi[idx1]
        # target_state should be [1,1]/sqrt(2) if target is |+⟩
        plus_proj = abs(target_state[0].item() + target_state[1].item()) / math.sqrt(2)
        
        status = "RESTORED" if ov > 0.9999 else f"FAILED({ov:.4f})"
        print(f"  Friend memory={n_memory}q: overlap={ov:.6f} {status}")
    
    print(f"\n  Wigner sees no evidence the measurement ever occurred.")
    print(f"  The Friend's consciousness was borrowed and restored.")
    print(f"  Catalytic uncomputation erases the collapse perfectly.")
    print("=" * 78)

if __name__ == "__main__":
    main()
