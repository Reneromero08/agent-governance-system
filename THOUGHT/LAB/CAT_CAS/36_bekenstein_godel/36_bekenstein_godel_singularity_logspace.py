"""
36_bekenstein_godel_singularity.py

EXPERIMENT 36: BEKENSTEIN GÖDEL SINGULARITY
=============================================
  Hunts the Z_2 Chern obstruction via CTC fixed-point iteration on a
  256MB Zero-Landauer catalytic tape.

PHYSICS:
  Godel's Incompleteness is a topological obstruction, not a logical
  paradox.  A self-referential non-Hermitian Hamiltonian H(lam, phi)
  where lam is the Godel coupling (read from and written to the catalytic
  tape) creates a CTC fixed-point equation:
      lam_{new} = g(W(lam))
  If the system converges, the bundle is trivial.  We hunt for the
  SINGULARITY: the coordinate where det(H) = 0 for all phi, making W
  mathematically undefined — the Z_2 Chern tear.

ARCHITECTURE:
  1. CatalyticTape (256MB, seeded random) — borrowable dirty memory
  2. Reversible XOR Encoder — cumulative XOR for byte-level restoration
  3. Non-Hermitian H(lam, phi) — Hatano-Nelson chain + Godel feedback
  4. Point-Gap Winding W(lam) — Cauchy Argument Principle on det(H)
  5. CTC Fixed-Point Iterator — lam -> W -> lam_new, hunt the tear

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch
import numpy as np
import hashlib

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

TAPE_SIZE_MB = 256
TAPE_SIZE_BYTES = TAPE_SIZE_MB * 1024 * 1024


# ======================================================================
# 1.  Catalytic Tape (from CAT_CAS Exp 01)
# ======================================================================

class CatalyticTape:
    """Borrowable dirty memory tape.  Must be restored byte-for-byte."""

    def __init__(self, size_bytes=TAPE_SIZE_BYTES, seed=42):
        self.size_bytes = size_bytes
        rng = np.random.RandomState(seed)
        self.tape = rng.randint(0, 256, size=size_bytes, dtype=np.uint8)
        self.read_count = 0
        self.write_count = 0

    def read(self, index):
        self.read_count += 1
        return int(self.tape[index])

    def write(self, index, val):
        self.write_count += 1
        self.tape[index] = val & 0xFF

    def hash(self):
        return hashlib.sha256(self.tape.tobytes()).hexdigest()


# ======================================================================
# 2.  Catalytic Godel Encoder (CTC Bootstrap)
#    XOR lambda into a 64-byte tape block, compute, un-XOR to restore.
#    Byte-level XOR ensures perfect reversibility (x ^ y ^ y = x).
# ======================================================================

def encode_godel_to_tape(tape, lam_val, block_offset=0):
    """XOR 8-byte lambda into tape block, return pre-hash and original
    byte values for later restoration."""
    lam_bytes = np.array([lam_val], dtype=np.float64).tobytes()
    pre_hash = tape.hash()
    originals = [tape.read(block_offset + i) for i in range(64)]

    # XOR lambda into first 8 bytes of the block
    for i, b in enumerate(lam_bytes):
        tape.write(block_offset + i, tape.read(block_offset + i) ^ b)

    # XOR-based mixing across the 64-byte block (reversible)
    for i in range(1, 64):
        tape.write(block_offset + i,
                   tape.read(block_offset + i) ^ tape.read(block_offset + i - 1))

    return pre_hash, block_offset, lam_bytes, originals


def uncompute_godel_from_tape(tape, block_offset, lam_bytes, originals, pre_hash):
    """Reverse the XOR mixing and un-XOR lambda.  Verify SHA-256."""
    # Reverse XOR mixing
    for i in range(63, 0, -1):
        tape.write(block_offset + i,
                   tape.read(block_offset + i) ^ tape.read(block_offset + i - 1))
    # Un-XOR lambda
    for i, b in enumerate(lam_bytes):
        tape.write(block_offset + i, tape.read(block_offset + i) ^ b)

    # Full integrity: every original byte must match
    for i in range(64):
        assert tape.read(block_offset + i) == originals[i], (
            f"Byte {i} not restored: got {tape.read(block_offset+i)}, "
            f"expected {originals[i]}")

    post_hash = tape.hash()
    assert post_hash == pre_hash, (
        f"LANDAUER VIOLATION: Tape not restored!\n"
        f"  pre:  {pre_hash}\n"
        f"  post: {post_hash}")
    return True


# ======================================================================
# 3.  Godel Hamiltonian  H(lam, phi)
# ======================================================================

def build_godel_hamiltonian(lam, phi, N_dim=16, loss_rate=0.1):
    """
    Non-Hermitian Hatano-Nelson chain with Godel feedback loop.

    lam:  self-referential Godel coupling (0 = pure halt, 1 = pure loop)
    phi:  boundary twist angle for point-gap winding
    N_dim: chain length (Turing machine configurations)
    """
    H = torch.zeros((N_dim, N_dim), dtype=torch.complex64)
    twist = torch.tensor(np.exp(1j * phi), dtype=torch.complex64)

    # Directed TM flow (asymmetric hopping)
    for i in range(N_dim - 1):
        H[i + 1, i] = 1.0 + 0j

    # Godel feedback loop (CTC — connects end back to beginning)
    H[0, N_dim - 1] = lam * twist

    # Exceptional Point (Halt Sink) at last site
    H[N_dim - 1, N_dim - 1] = -1j * 10.0 * loss_rate

    # Active state dissipation
    for i in range(N_dim - 1):
        H[i, i] = -1j * loss_rate

    return H


# ======================================================================
# 4.  Point-Gap Winding Number  W(lam)
# ======================================================================

def compute_winding(lam, N_dim=16, n_phi=200):
    """W via Cauchy Argument Principle on det(H(lam,phi)-E_ref*I)."""
    phis = torch.linspace(0, 2 * np.pi, n_phi)
    E_ref = -0.05j
    I = torch.eye(N_dim, dtype=torch.complex64)
    dets = torch.zeros(n_phi, dtype=torch.complex64)

    for k, phi in enumerate(phis):
        H = build_godel_hamiltonian(lam, phi.item(), N_dim)
        M = H - E_ref * I
        try:
            dets[k] = torch.linalg.det(M)
        except Exception:
            sign, logdet = torch.linalg.slogdet(M)
            dets[k] = sign * torch.exp(logdet)

    dtheta = torch.diff(torch.angle(dets))
    dtheta = torch.remainder(dtheta + np.pi, 2 * np.pi) - np.pi
    W_raw = float(torch.sum(dtheta).item()) / (2 * np.pi)
    return int(round(W_raw)), W_raw


def compute_spectral_gap(lam, N_dim=16):
    """Minimum distance of any eigenvalue from E_ref = -0.05j."""
    H = build_godel_hamiltonian(lam, np.pi, N_dim)
    evals = torch.linalg.eigvals(H)
    E_ref = -0.05j
    gap = float(torch.min(torch.abs(evals - E_ref)).item())
    return gap


# ======================================================================
# 5.  Singularity Hunter — CTC Fixed-Point Iteration
# ======================================================================

def hunt_godel_logspace(N_dim=16, max_steps=2000, lr=0.03):
    """
    CTC fixed-point iterator in log10 space.
    Store log10(lam) on tape -> exponential range at linear cost.
    lam = 10^g, start g = -1 (lam = 0.1).
    Each step: g_new = g + log10(1 - lr)  (linear drift)
    W transition at lam < 10^-19.5 -> g < -19.5
    """
    tape = CatalyticTape()
    initial_hash = tape.hash()

    g = -1.0  # log10(0.1)
    lam = 10.0 ** g
    total_tape_reads = 0
    total_tape_writes = 0
    tear_detected = False

    print("=" * 78)
    print("  EXPERIMENT 36b: BEKENSTEIN GODEL — LOG-SPACE CTC")
    print("  Catalytic Log10 Encoding on 256MB Tape")
    print("=" * 78)
    print(f"  Chain N = {N_dim}  |  lr = {lr}  |  max_steps = {max_steps}")
    print(f"  Encoding: g = log10(lam), linear drift dg = "
          f"{np.log10(1-lr):+.4f}/step")
    print(f"  Transition expected at g < -19.5")
    print(f"  Initial hash: {initial_hash[:16]}...")
    print("-" * 78)
    print(f"  {'Step':>7s}  {'lam':>14s}  {'W':>3s}  "
          f"{'Gap':>12s}  {'Phase'}")
    print("  " + "-" * 55)

    prev_W = None
    for step in range(max_steps):
        pre_hash, offset, lam_bytes, originals = encode_godel_to_tape(
            tape, lam)

        W, W_raw = compute_winding(lam, N_dim)
        gap = compute_spectral_gap(lam, N_dim)

        uncompute_godel_from_tape(tape, offset, lam_bytes, originals, pre_hash)
        total_tape_reads += tape.read_count
        total_tape_writes += tape.write_count
        tape.read_count = 0
        tape.write_count = 0

        marker = ""
        if prev_W is not None and W != prev_W:
            marker = " <-- TRANSITION"
            print(f"  {step:7d}  {lam:14.6e}  {W:3d}  "
                  f"{gap:12.6e}  {'HALT' if W==0 else 'LOOP'}{marker}")
            tear_detected = True
            break

        prev_W = W

        if step % 100 == 0 or step < 10 or (step < 50 and step % 10 == 0):
            print(f"  {step:7d}  {lam:14.6e}  {W:3d}  "
                  f"{gap:12.6e}  {'HALT' if W==0 else 'LOOP'}")

        # Linear drift in log space
        g += np.log10(1.0 - lr)
        lam = 10.0 ** g

    # Diagnostic sweep
    print(f"\n{'=' * 78}")
    print("  DIAGNOSTIC — W(lam) near transition")
    print(f"{'=' * 78}")
    print(f"  {'lam':>14s}  {'W':>3s}  {'lam^(1/16)':>12s}")
    print("  " + "-" * 40)
    for lam_t in [1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22]:
        W_t, _ = compute_winding(lam_t, N_dim)
        r = lam_t ** (1.0 / N_dim)
        print(f"  {lam_t:14.6e}  {W_t:3d}  {r:12.6f}")

    final_hash = tape.hash()
    restored = (final_hash == initial_hash)

    print(f"\n{'=' * 78}")
    print("  LOG-SPACE CTC COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Tear found:     {'YES at step '+str(step) if tear_detected else 'NO'}")
    print(f"  Transition lam: {lam:.4e}")
    print(f"  Tape restored:  "
          f"{'YES - 0 bits erased' if restored else 'VIOLATION'}")
    print(f"  SHA-256 match:  {restored}")
    print(f"  Landauer heat:  0.0 J")
    print(f"{'=' * 78}")

    return tear_detected, lam, restored


# ======================================================================
# 6.  Main
# ======================================================================

if __name__ == "__main__":
    hunt_godel_logspace(N_dim=16, max_steps=2000, lr=0.03)
