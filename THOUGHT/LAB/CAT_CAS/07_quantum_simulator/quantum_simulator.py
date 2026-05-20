"""
Catalytic Quantum State Simulator — Scaled Engine

Maps 2^N complex amplitudes onto a dirty catalytic tape.  All gate
operations are reversible permutations (swap-only, O(1) clean RAM per pair).

For scale (20+ qubits / 1M+ amplitudes), the state is loaded into a
Python array.array('q') for fast element access, then written back to
the tape file.  The catalytic property (perfect tape restoration after
U†U = I) is verified via SHA-256 hash.
"""

import array as _array
import struct


class CatalyticQuantumSimulator:
    """Reversible quantum simulator for large qubit counts."""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_states = 1 << n_qubits
        self.gate_log = []

    # ------------------------------------------------------------------ #
    #  State I/O                                                           #
    # ------------------------------------------------------------------ #

    def load_state(self, filepath):
        """Load the first n_states * 16 bytes of tape into an int64 array."""
        with open(filepath, 'rb') as f:
            raw = f.read(self.n_states * 16)
        return _array.array('q', raw)

    def save_state(self, filepath, state):
        """Write the int64 array back to the first bytes of the tape."""
        with open(filepath, 'r+b') as f:
            f.write(state.tobytes())

    # ------------------------------------------------------------------ #
    #  Quantum gates (all self-inverse permutations)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _swap(state, i, j):
        """Swap amplitudes (real+imag) at state indices i and j."""
        ii, jj = i << 1, j << 1
        state[ii],   state[jj]   = state[jj],   state[ii]
        state[ii+1], state[jj+1] = state[jj+1], state[ii+1]

    def gate_x(self, state, target):
        """Pauli-X (NOT): swap |0> <-> |1> on target qubit."""
        mask = 1 << target
        n = self.n_states
        for i in range(n):
            if not (i & mask):
                self._swap(state, i, i | mask)
        self.gate_log.append(('X', target))

    def gate_cnot(self, state, control, target):
        """CNOT: flip target when control = |1>."""
        cmask = 1 << control
        tmask = 1 << target
        n = self.n_states
        for i in range(n):
            if (i & cmask) and not (i & tmask):
                self._swap(state, i, i | tmask)
        self.gate_log.append(('CNOT', control, target))

    def gate_ccx(self, state, c1, c2, target):
        """Toffoli (CCX): flip target when both controls = |1>."""
        c1m = 1 << c1
        c2m = 1 << c2
        tm  = 1 << target
        n = self.n_states
        for i in range(n):
            if (i & c1m) and (i & c2m) and not (i & tm):
                self._swap(state, i, i | tm)
        self.gate_log.append(('CCX', c1, c2, target))

    def gate_swap(self, state, q1, q2):
        """SWAP two qubits."""
        m1 = 1 << q1
        m2 = 1 << q2
        n = self.n_states
        for i in range(n):
            if ((i >> q1) & 1) != ((i >> q2) & 1) and i < (i ^ m1 ^ m2):
                self._swap(state, i, i ^ m1 ^ m2)
        self.gate_log.append(('SWAP', q1, q2))

    # ------------------------------------------------------------------ #
    #  Inverse execution                                                   #
    # ------------------------------------------------------------------ #

    def run_inverse(self, state):
        """Apply every logged gate in reverse order (all self-inverse)."""
        for gate in reversed(self.gate_log):
            kind = gate[0]
            if kind == 'X':
                mask = 1 << gate[1]
                for i in range(self.n_states):
                    if not (i & mask):
                        self._swap(state, i, i | mask)
            elif kind == 'CNOT':
                cmask, tmask = 1 << gate[1], 1 << gate[2]
                for i in range(self.n_states):
                    if (i & cmask) and not (i & tmask):
                        self._swap(state, i, i | tmask)
            elif kind == 'CCX':
                c1m, c2m, tm = 1 << gate[1], 1 << gate[2], 1 << gate[3]
                for i in range(self.n_states):
                    if (i & c1m) and (i & c2m) and not (i & tm):
                        self._swap(state, i, i | tm)
            elif kind == 'SWAP':
                m1, m2 = 1 << gate[1], 1 << gate[2]
                q1, q2 = gate[1], gate[2]
                for i in range(self.n_states):
                    if ((i >> q1) & 1) != ((i >> q2) & 1) and i < (i ^ m1 ^ m2):
                        self._swap(state, i, i ^ m1 ^ m2)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def probability_sum(self, state):
        """Sum |amplitude|^2 — must be conserved by unitary ops."""
        total = 0
        for k in range(0, self.n_states * 2, 2):
            r, im = state[k], state[k + 1]
            total += r * r + im * im
        return total

    def sample_amplitudes(self, state, indices):
        """Read a handful of amplitudes for display."""
        return {i: (state[i << 1], state[(i << 1) + 1]) for i in indices}
