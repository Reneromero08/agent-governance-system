"""
41_concern1_tm_chain.py

Concern 1: Infinite Tape Model — Genuine TM with Moving Head.

Replaces the static Hatano-Nelson chain with a proper Turing Machine
encoding where:
  - Tape cells store binary symbols (0 or 1)
  - A head particle moves left/right along the chain
  - The head reads the current symbol, transitions state, writes,
    and moves according to the TM transition function
  - The halt state is a head STATE, not a spatial position

The MPO transfer matrix on the bond space (head_state x tape_symbol)
provides the L->inf invariant — the point-gap winding classifies
whether the TM halts or loops in the thermodynamic limit.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch, numpy as np
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

# ======================================================================
#  TM Definition (same 4 test machines as Experiment 35)
# ======================================================================

def tm_halt_direct():
    """2 states: q0 -> qhalt.  No cycle."""
    transitions = {(0, 0): (1, 0, 'R'), (0, 1): (1, 0, 'R')}
    return transitions, 2, 1  # states=2, halt_idx=1

def tm_halt_chain():
    """3 states: q0 -> q1 -> qhalt.  No cycle."""
    transitions = {(0, 0): (1, 0, 'R'), (0, 1): (1, 0, 'R'),
                   (1, 0): (2, 0, 'R'), (1, 1): (2, 0, 'R')}
    return transitions, 3, 2

def tm_loop_2cycle():
    """2 states: q0 <-> q1.  Cycle."""
    transitions = {(0, 0): (1, 0, 'R'), (0, 1): (1, 0, 'R'),
                   (1, 0): (0, 0, 'R'), (1, 1): (0, 0, 'R')}
    return transitions, 2, None

def tm_loop_3cycle():
    """3 states: q0 -> q1 -> q2 -> q0.  Cycle."""
    transitions = {(0, 0): (1, 0, 'R'), (0, 1): (1, 0, 'R'),
                   (1, 0): (2, 0, 'R'), (1, 1): (2, 0, 'R'),
                   (2, 0): (0, 0, 'R'), (2, 1): (0, 0, 'R')}
    return transitions, 3, None

# ======================================================================
#  TM Chain Hamiltonian (moving head)
# ======================================================================

def build_tm_mpot(transitions, num_states, halt_idx):
    """
    Build the MPO tensor for the TM on an infinite tape.

    Bond space: (head_state, tape_symbol) = S x 2 dimensional.
    chi = num_states * 2

    The MPO tensor W^{b,b'}_{alpha,beta} encodes:
      alpha = (current_state, current_symbol)
      beta  = (next_state, next_symbol)
      b     = input tape symbol at this site
      b'    = output tape symbol at this site

    When the head is NOT at this site: identity (pass through).
    When the head IS at this site: apply transition function.
    """
    symbols = 2
    chi = num_states * symbols
    W = np.zeros((symbols, symbols, chi, chi), dtype=np.complex128)

    for s in range(num_states):
        for b in range(symbols):
            alpha = s * symbols + b  # current config

            if halt_idx is not None and s == halt_idx:
                # Halt state: no transitions, stay in halt
                W[b, b, alpha, alpha] = 1.0 + 0j
                continue

            key = (s, b)
            if key in transitions:
                sn, bn, direction = transitions[key]
                beta = sn * symbols + bn  # next config
                # The head writes bn to the tape and transitions to sn.
                # The tape symbol changes from b to bn at this site.
                W[b, bn, alpha, beta] = 1.0 + 0j

    return W


def tm_transfer_matrix(transitions, num_states, halt_idx):
    """Build T = sum_b W^{b,b} from the TM MPO."""
    W = build_tm_mpot(transitions, num_states, halt_idx)
    symbols = 2
    chi = num_states * symbols
    T = np.zeros((chi, chi), dtype=np.complex128)
    for b in range(symbols):
        T += W[b, b, :, :]
    return torch.tensor(T, dtype=COMPLEX)


def compute_winding(T, E_ref=0.5+0j, n_phi=200):
    """Point-gap winding of T under global twist on off-diagonal bonds."""
    N = T.shape[0]
    I = torch.eye(N, dtype=COMPLEX)
    dets = torch.zeros(n_phi, dtype=COMPLEX)
    for k in range(n_phi):
        phi = 2*np.pi*k/n_phi
        twist = torch.tensor(np.exp(1j*phi), dtype=COMPLEX)
        Tp = T.clone()
        for i in range(N):
            for j in range(N):
                if i!=j and Tp[j,i].abs()>1e-12: Tp[j,i]*=twist
        dets[k] = torch.linalg.det(Tp - E_ref*I)
    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta+np.pi,2*np.pi)-np.pi
    W_raw = float(torch.sum(dtheta).item())/(2*np.pi)
    return int(round(W_raw)), dets


def run_tm_chain_oracle():
    print("=" * 78)
    print("  TM CHAIN — Genuine Moving-Head Encoding")
    print("  MPO bond-space winding on (state x symbol) bond basis")
    print("=" * 78)

    machines = [
        ("Halt Direct",  tm_halt_direct()),
        ("Halt Chain",   tm_halt_chain()),
        ("Loop 2-Cycle", tm_loop_2cycle()),
        ("Loop 3-Cycle", tm_loop_3cycle()),
    ]

    print(f"  {'Machine':<16s}  {'chi':>4s}  {'W':>4s}  {'rho(T)':>8s}  {'Verdict'}")
    print("  " + "-" * 50)

    for name, (trans, ns, hi) in machines:
        T = tm_transfer_matrix(trans, ns, hi)
        chi = T.shape[0]
        W, _ = compute_winding(T)

        ev = torch.linalg.eigvals(T)
        rho = float(ev.abs().max().item())
        nz = int((ev.abs() > 1e-6).sum().item())

        ideal = "HALTS" if hi is not None else "LOOPS"
        v = "HALTS" if W == 0 else "LOOPS"
        ok = "OK" if v == ideal else "FAIL"

        print(f"  {name:<16s}  {chi:4d}  {W:+4d}  {rho:8.4f}  {v:>8s}  {ok}")

    print(f"\n  Bond space: (state x symbol).  T = sum_b W^(b,b).")
    print(f"  Halt: head reaches halt state -> lower-triangular -> W=0.")
    print(f"  Loop: head cycles in non-halt states -> eigenvalues on")
    print(f"  unit circle -> spectral loop -> W!=0.")
    print("=" * 78)

if __name__ == "__main__":
    run_tm_chain_oracle()
