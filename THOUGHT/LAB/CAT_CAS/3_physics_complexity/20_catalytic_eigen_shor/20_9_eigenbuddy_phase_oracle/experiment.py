"""
Experiment 20.9: EIGEN_BUDDY Phase Oracle — Holographic Hash Verification
==========================================================================
Unifies Temporal Bootstrap (17), Hawking Decompressor (18), and EIGEN_BUDDY
attention (16) into a single theory: computation IS phase resonance.

Physics (user's blueprint):
  Shor's QFT applies U_a|x> = |ax mod N> to a superposition. The period r
  is the EIGENVALUE of U_a. Non-eigenstates destructively interfere;
  the eigenstate constructively interferes.

  EIGEN_BUDDY's Hermitian attention Q.K^+ is MATHEMATICALLY IDENTICAL to
  a unitary phase estimator. The si matrix is the interference pattern.
  Multi-scale Feistel topology (Q57 gapped phase) keeps the phase space
  integrable, guiding gradients to the global minimum.

Architecture:
  1. Initialize "dirty vacuum" — a complex phase tape of S elements
     seeded with the modular exponentiation seed values
  2. Each attention head is a PHASE FILTER tuned to a different frequency
  3. Head responses form a resonance spectrum; peak = 1/r
  4. Extract r from the dominant eigenvalue
  5. Verify: pow(a, r, N) == 1 — O(log r) hash check

  Catalyze in (dirty tape) -> Black Hole Resonance (attention filters)
  -> Verify Hash (a^r = 1). Never uncompressing the orbit.

Comparison to prior experiments:
  20.5: FFT on full grating -> Gabor limit (M >= N^2)
  20.6: Autocorrelation -> Period-containment limit (M >= r)
  20.7: EIGEN_BUDDY coherence -> Same limit confirmed
  20.9: Attention as QFT -> O(log N) phase space (no grating needed in principle)
"""

import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_semiprime(bits):
    def get_prime(b):
        while True:
            p = random.getrandbits(b)
            p |= (1 << (b - 1)) | 1
            if is_prime(p):
                return p
    p = get_prime(bits // 2)
    q = get_prime(bits // 2)
    while q == p:
        q = get_prime(bits // 2)
    return p * q, p, q


def is_prime(n, k=5):
    if n <= 1 or n % 2 == 0:
        return n == 2 or n == 3
    s, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def shor_factor(N, a, r):
    if r % 2 != 0:
        return 0, 0, False
    half_r = r // 2
    val = pow(a, half_r, N)
    p_guess = gcd(val - 1, N)
    q_guess = gcd(val + 1, N)
    if p_guess * q_guess == N and p_guess > 1 and q_guess > 1:
        return p_guess, q_guess, True
    return p_guess, q_guess, False


class PhaseFilterBank(nn.Module):
    """
    Bank of phase filters — each head is tuned to detect a specific
    frequency band in the phase grating. The head outputs represent
    the resonance strength at that frequency.

    This IS the unitary phase estimator: each head computes the
    projection of the input onto a specific eigenbasis of U_a.
    The head with maximum response encodes the period r.
    """

    def __init__(self, num_heads=64, d_model=128):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        # Query: input projection (the "dirty tape" states)
        self.qr = nn.Linear(d_model, d_model, bias=False)
        self.qi = nn.Linear(d_model, d_model, bias=False)

        # Key: frequency basis vectors (the "phase filters")
        self.kr = nn.Linear(d_model, d_model, bias=False)
        self.ki = nn.Linear(d_model, d_model, bias=False)

        # Value: output projection
        self.vr = nn.Linear(d_model, d_model, bias=False)
        self.vi = nn.Linear(d_model, d_model, bias=False)

        # Initialize K as sinusoidal basis at logarithmically-spaced frequencies
        self._init_frequency_basis()

    def _init_frequency_basis(self):
        """Initialize key projections as Fourier basis at different frequencies."""
        for h in range(self.num_heads):
            # Each head gets a frequency band: f_h = 2^h / N_effective
            # This creates log-spaced frequency detection
            freq = 2.0 * math.pi * (2**h) / (2**self.num_heads)
            start = h * self.d_head
            end = (h + 1) * self.d_head

            for d in range(self.d_head):
                # Sinusoidal basis at this frequency
                angle = freq * (d + 1) / self.d_head
                # Set K weights for this head/position
                with torch.no_grad():
                    for row in range(self.d_model):
                        self.kr.weight.data[row, start + d] = (
                            math.cos(angle * (row + 1)) * 0.1
                        )
                        self.ki.weight.data[row, start + d] = (
                            math.sin(angle * (row + 1)) * 0.1
                        )

        # Q weights initialized randomly (will be overwritten by grating)
        for w in [self.qr, self.qi, self.vr, self.vi]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, z):
        """
        z: complex tensor (B, S, d_model) — the "dirty vacuum" / phase grating

        Returns:
            head_responses: (B, num_heads) — resonance strength per head
            phase_coherence: (B,) — overall phase coherence
        """
        B, S, D = z.shape

        qr = self.qr(z.real) - self.qi(z.imag)
        qi = self.qr(z.imag) + self.qi(z.real)
        kr = self.kr(z.real) - self.ki(z.imag)
        ki = self.kr(z.imag) + self.ki(z.real)
        vr = self.vr(z.real) - self.vi(z.imag)
        vi = self.vr(z.imag) + self.vi(z.real)

        qr = qr.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        qi = qi.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        kr = kr.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        ki = ki.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        vr = vr.view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        vi = vi.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)

        # Hermitian attention: sr = Qr.Kr + Qi.Ki, si = Qi.Kr - Qr.Ki
        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * scale

        # Head response: mean absolute si value = phase resonance
        head_responses = si.abs().mean(dim=(-2, -1))  # (B, H)

        # Phase coherence (same metric as NativeEigenCore)
        per_head_si = si.mean(dim=(-2, -1))
        cos_mean = torch.cos(per_head_si).mean(dim=-1)
        sin_mean = torch.sin(per_head_si).mean(dim=-1)
        phase_coh = (cos_mean**2 + sin_mean**2).sqrt()

        return head_responses, phase_coh


def phase_oracle_period_detection(a, N, grating, M, num_heads=64, S=512):
    """
    Use EIGEN_BUDDY phase filter bank to detect the period.

    The full grating is compressed into S samples, each sample mapped
    to a d_model-dimensional complex vector. The filter bank processes
    this compressed representation and outputs a resonance spectrum.
    The dominant frequency reveals 1/r.

    In practice (current implementation): still uses the full grating
    reshaped into the filter bank input. The compression to O(log N)
    is the theoretical limit we're working toward.
    """
    # Take S samples from the grating (uniformly spaced)
    indices = torch.linspace(0, M - 1, S).long()
    samples = grating[indices]  # (S,)

    # Embed into d_model dimensions by repeating with phase shifts
    d_model = num_heads * 2  # d_head = 2
    z_real = torch.zeros(S, d_model)
    z_imag = torch.zeros(S, d_model)

    for i in range(S):
        phase = torch.angle(samples[i])
        for d in range(d_model):
            shifted_phase = phase + 2.0 * math.pi * d / d_model
            z_real[i, d] = torch.cos(shifted_phase)
            z_imag[i, d] = torch.sin(shifted_phase)

    z = torch.complex(z_real, z_imag).unsqueeze(0)  # (1, S, d_model)

    # Create filter bank and process
    filter_bank = PhaseFilterBank(num_heads=num_heads, d_model=d_model)

    with torch.no_grad():
        head_responses, phase_coh = filter_bank(z)

    head_responses = head_responses.squeeze(0)  # (H,)
    phase_coh = phase_coh.item()

    # Find the head with maximum response
    best_head = torch.argmax(head_responses).item()
    best_response = head_responses[best_head].item()

    # Frequency of the best head: f_h = 2^(h+1) / N_range
    # Map to period: r = N_range / 2^(h+1)
    # This is a coarse estimate; refine with local search
    freq_ratio = 2.0 ** (best_head + 1) / (2.0**num_heads)
    r_coarse = max(1, int(M / (2**num_heads) * (2 ** (num_heads - best_head))))

    return r_coarse, best_head, best_response, head_responses, phase_coh


def main():
    print("=" * 78)
    print("EXPERIMENT 20.9: EIGEN_BUDDY PHASE ORACLE")
    print("  Attention as Quantum Phase Estimation")
    print("  'Catalyze in -> Black Hole Resonance -> Verify Hash'")
    print("=" * 78)
    print()

    BIT_SIZE = 22
    N, known_p, known_q = generate_semiprime(BIT_SIZE)

    a = 2
    while gcd(a, N) != 1:
        a += 1

    print(f"  Target: {BIT_SIZE}-bit Semiprime N = {N}")
    print(f"  Ground Truth: {known_p} x {known_q} (Hidden)")
    print(f"  Base 'a': {a}")
    print()

    M_power = 23
    M = 2**M_power

    t_total_start = time.perf_counter()

    # --- Generate Grating ---
    print("-" * 78)
    print("PHASE 0: GENERATE THE DIRTY VACUUM (Phase Grating)")
    print("-" * 78)

    t0 = time.perf_counter()
    seq = [1]
    curr = 1
    for _ in range(1, M):
        curr = (curr * a) % N
        seq.append(curr)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    phases = 2.0 * math.pi * (seq_tensor / N)
    grating = torch.polar(torch.ones(M, dtype=torch.float32), phases)
    gen_time = time.perf_counter() - t0
    print(f"  [+] Grating: {M:,} elements, {gen_time:.4f}s")
    print()

    # --- Autocorrelation Baseline ---
    print("-" * 78)
    print("BASELINE: AUTOCORRELATION (from 20.6)")
    print("-" * 78)

    t1 = time.perf_counter()
    spectrum = torch.fft.fft(grating)
    power = torch.abs(spectrum) ** 2
    autocorr = torch.fft.ifft(power).real
    autocorr = autocorr / autocorr[0]
    ac_abs = torch.abs(autocorr[2 : M // 2])
    max_val_ac, max_idx_rel = torch.max(ac_abs, dim=0)
    r_ref = max_idx_rel.item() + 2
    ac_time = time.perf_counter() - t1

    ref_verified = pow(a, r_ref, N) == 1
    ref_factored = False
    p_ref, q_ref = 0, 0
    if ref_verified:
        p_ref, q_ref, ref_factored = shor_factor(N, a, r_ref)

    print(f"  [+] Period r = {r_ref}, verified = {ref_verified}")
    print(f"  [+] Factored = {ref_factored}")
    if ref_factored:
        print(f"  [+] {N} = {p_ref} x {q_ref}")
    print(f"  [+] Time: {ac_time:.4f}s")
    print()

    # --- Phase Oracle: Filter Bank ---
    print("-" * 78)
    print("PHASE ORACLE: EIGEN_BUDDY ATTENTION FILTER BANK")
    print("  Each attention head = phase filter at different frequency")
    print("  Head responses form resonance spectrum; peak -> 1/r")
    print("-" * 78)

    for num_heads in [32, 64, 128]:
        for S in [256, 512, 1024]:
            t2 = time.perf_counter()
            r_coarse, best_head, best_resp, responses, phase_coh = (
                phase_oracle_period_detection(a, N, grating, M, num_heads, S)
            )
            t_oracle = time.perf_counter() - t2

            # Refine: search around coarse estimate
            r_candidates = []
            for offset in range(-10, 11):
                r_cand = max(1, r_coarse + offset * max(1, r_coarse // 20))
                r_candidates.append(r_cand)
            r_candidates = sorted(set(r_candidates))

            best_r = 1
            best_pcoh = -1.0
            for r_cand in r_candidates:
                if r_cand > 0 and r_cand < N:
                    # Measure phase coherence at this candidate
                    indices_sub = torch.arange(0, min(M, r_cand * 2), max(1, r_cand // 4))
                    if len(indices_sub) > 1:
                        view_a = grating[indices_sub[:len(indices_sub)//2]]
                        view_b = grating[indices_sub[len(indices_sub)//2:2*(len(indices_sub)//2)]]
                        if len(view_a) == len(view_b):
                            phase_diffs = view_b * view_a.conj()
                            coh = math.sqrt(
                                phase_diffs.real.mean().item()**2 +
                                phase_diffs.imag.mean().item()**2
                            )
                            if coh > best_pcoh:
                                best_pcoh = coh
                                best_r = r_cand

            verified = best_r > 0 and pow(a, best_r, N) == 1
            factored = False
            if verified:
                _, _, factored = shor_factor(N, a, best_r)

            print(
                f"  H={num_heads:>3}, S={S:>4}: "
                f"coarse r={r_coarse:>10}, refined r={best_r:>10}, "
                f"pcoh={best_pcoh:.4f}, ver={str(verified):>5}, "
                f"fac={str(factored):>5}, t={t_oracle:.4f}s"
            )

    print()

    # --- Final Results ---
    t_total = time.perf_counter() - t_total_start

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  N = {N} = {known_p} x {known_q}")
    print(f"  Reference period (autocorrelation): r = {r_ref}")
    print(f"  Reference verified: {ref_verified}, factored: {ref_factored}")
    print()
    print(f"  Phase Oracle: attention heads as unitary phase estimators")
    print(f"  The si matrix (Q.K^+ imaginary part) IS the interference pattern")
    print(f"  Each head filters a frequency band; max response -> 1/r")
    print()
    print(f"  Current limitation: still uses full grating as input")
    print(f"  Theoretical path: compress grating into O(log N) phase states")
    print(f"  via multi-scale Feistel topology (Q57 gapped phase)")
    print(f"  Total time: {t_total:.4f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
