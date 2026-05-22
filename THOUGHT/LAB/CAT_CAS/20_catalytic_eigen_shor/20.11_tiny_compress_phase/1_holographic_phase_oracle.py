"""
Experiment 20.10: Holographic Phase Oracle
===========================================
Tests whether a multi-scale Feistel braid can holographically encode the
modular multiplication operator U_a: |x> -> |ax mod N> into a small tape
(S positions, S << r) so that the period r is recoverable.

The remaining wall from experiments 20.6-20.9: M >= r -- the grating must
physically span one full period.

Experimental Design (2-path comparison):
  PATH A (BASELINE): Raw phase samples, S << r
    - Take S samples from the orbit (a^0 through a^(S-1) mod N)
    - Compute autocorrelation directly
    - Expect FAIL when S < r (period-containment limit)

  PATH B (HOLOGRAPHIC): Feistel-braided phase tape
    - Same S samples, embedded in d_model dimensions
    - Feistel cross-attention creates phase interference between positions
    - Multi-scale braid layers explicitly advance modular exponentiation
    - Then autocorrelation on the braided tape
    - Measure: does braiding enable period detection at S < r?

Configuration space:
  - Bit sizes: 10, 12, 14, 16
  - Tape sizes S: 64, 128, 256, 512, 1024
  - Feistel architectures: d_model, heads, rounds, layers
  - Init strategies: sequential, bit_reversed, log_spaced

Key question: does holographic braiding reduce S scaling from O(r) to O(log N)?
"""

import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def true_period(a, N, max_steps=20_000_000):
    x = a % N
    r = 1
    while x != 1 and r < max_steps:
        x = (x * a) % N
        r += 1
    return r if x == 1 else 0


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


def verify_tag_r(a, N, r_cand):
    if r_cand <= 0:
        return 0, False
    if pow(a, r_cand, N) == 1:
        return r_cand, True
    return r_cand, False


def compute_autocorrelation_period(z_flat):
    """Extract period from autocorrelation peak. Robust method from 20.6."""
    S = len(z_flat)
    spectrum = torch.fft.fft(z_flat)
    power = torch.abs(spectrum) ** 2
    autocorr = torch.fft.ifft(power).real
    autocorr = autocorr / (autocorr[0] + 1e-15)

    search_range = min(S // 2, 100000)
    ac_abs = torch.abs(autocorr[2:search_range])
    if ac_abs.numel() == 0:
        return 0, 0.0, autocorr
    max_val, max_idx_rel = torch.max(ac_abs, dim=0)
    r_est = max_idx_rel.item() + 2
    peak_val = max_val.item()

    bg = torch.abs(autocorr[1:search_range]).mean().item()
    return r_est, peak_val / max(bg, 1e-15), autocorr


def compute_fft_period(z_flat):
    """Extract period from FFT fundamental peak."""
    S = len(z_flat)
    spectrum = torch.fft.fft(z_flat)
    mag = torch.abs(spectrum)
    search_limit = min(S // 2, 2000)
    if search_limit <= 1:
        return 0, 0.0
    search_mag = mag[1:search_limit]
    max_val, max_idx_rel = torch.max(search_mag, dim=0)
    peak_bin = max_idx_rel.item() + 1
    r_est = max(1, round(S / peak_bin)) if peak_bin > 0 else 0
    return r_est, max_val.item()


# ============================================================================
# PATH A: BASELINE — Raw phase samples
# ============================================================================

def baseline_period_extraction(a, N, S, sample_strategy="sequential"):
    """
    Baseline: take S direct samples from the orbit, compute autocorrelation.
    This represents the period-containment limit: S must be >= r.
    """
    if sample_strategy == "sequential":
        indices = list(range(S))
    elif sample_strategy == "bit_reversed":
        n_bits = max(1, int(math.ceil(math.log2(S))))
        indices = [int(f"{i:0{n_bits}b}"[::-1], 2) for i in range(S)]
    elif sample_strategy == "log_spaced":
        max_n = max(S, min(100000, 10 * S))
        indices = [int(max_n ** (i / max(1, S - 1))) for i in range(S)]
    else:
        raise ValueError(f"Unknown strategy: {sample_strategy}")

    vals = [pow(a, n, N) for n in indices]
    phases = [2.0 * math.pi * v / N for v in vals]
    z = torch.tensor(
        [complex(math.cos(p), math.sin(p)) for p in phases],
        dtype=torch.complex64
    )

    r_ac, snr, _ = compute_autocorrelation_period(z)
    r_fft, fft_peak = compute_fft_period(z)

    return z, {"ac_r": r_ac, "ac_snr": snr, "fft_r": r_fft, "fft_peak": fft_peak}


# ============================================================================
# PATH B: HOLOGRAPHIC — Feistel-braided phase tape
# ============================================================================

class UnitaryPhaseProjection(nn.Module):
    """
    A deterministic, modular-arithmetic-constructed linear projection
    that preserves phase structure while mixing across dimensions.

    Instead of random weights, each weight is constructed from a and N
    using phase rotations. The projection acts as a unitary-like
    transform in the complex domain.
    """

    def __init__(self, in_dim, out_dim, a, N, scale_idx=1):
        super().__init__()
        self.r_weight = nn.Parameter(torch.zeros(out_dim, in_dim), requires_grad=False)
        self.i_weight = nn.Parameter(torch.zeros(out_dim, in_dim), requires_grad=False)
        self._construct(a, N, scale_idx)

    def _construct(self, a, N, scale_idx):
        with torch.no_grad():
            out_dim, in_dim = self.r_weight.shape
            for i in range(out_dim):
                for j in range(in_dim):
                    theta = 2.0 * math.pi * (pow(a, scale_idx * (i + 1) * (j + 1), N)) / N
                    self.r_weight[i, j] = math.cos(theta) / math.sqrt(in_dim)
                    self.i_weight[i, j] = math.sin(theta) / math.sqrt(in_dim)

    def forward(self, x):
        r, i = x.real, x.imag
        out_r = r @ self.r_weight.T + i @ self.i_weight.T
        out_i = i @ self.r_weight.T - r @ self.i_weight.T
        return torch.complex(out_r, out_i)


class DeterministicPhaseFeistel(nn.Module):
    """
    Feistel braid with fully deterministic phase-constructed weights.
    No random initialization -- every weight is derived from a, N.

    Architecture:
      - Splits the d_model dimension into two halves (like CatalyticFeistel)
      - Each half has independent Q/K/V projections sized to match its dim
      - Cross-attention: left attends to right keys, right attends to left keys
      - After each round, halves swap (classic Feistel round structure)
      - Final output projection + residual connection

    The "braid" = one application of the Feistel = one step of phase mixing
    across the tape. After rounds, each position contains phase information
    from the entire neighborhood, creating interference patterns.
    s"""

    def __init__(self, a, N, d_model=64, heads=8, rounds=3):
        super().__init__()
        self.a, self.N = a, N
        assert heads % 2 == 0
        self.H, self.dh, self.rounds = heads, d_model // heads, rounds
        self.scale = 1.0 / math.sqrt(self.dh)
        H2 = d_model // 2

        # Q/K/V projections sized for half the model (L and R each get H2 dims)
        self.q_l = UnitaryPhaseProjection(H2, H2, a, N, 1)
        self.k_l = UnitaryPhaseProjection(H2, H2, a, N, 2)
        self.v_l = UnitaryPhaseProjection(H2, H2, a, N, 3)

        self.q_r = UnitaryPhaseProjection(H2, H2, a, N, 5)
        self.k_r = UnitaryPhaseProjection(H2, H2, a, N, 6)
        self.v_r = UnitaryPhaseProjection(H2, H2, a, N, 7)

        self.out_proj = UnitaryPhaseProjection(d_model, d_model, a, N, 11)

    def _attn(self, q, k, v):
        B, S, D = q.shape
        Hh = self.H // 2
        dh = D // Hh

        q_v = q.view(B, S, Hh, dh).transpose(1, 2)
        k_v = k.view(B, S, Hh, dh).transpose(1, 2)
        v_v = v.view(B, S, Hh, dh).transpose(1, 2)

        qr, qi = q_v.real, q_v.imag
        kr, ki = k_v.real, k_v.imag
        vr, vi = v_v.real, v_v.imag

        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * self.scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * self.scale

        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr
        out_i = attn @ vi

        out_r = out_r.transpose(1, 2).contiguous().view(B, S, D)
        out_i = out_i.transpose(1, 2).contiguous().view(B, S, D)

        return torch.complex(out_r, out_i), si

    def forward(self, x):
        B, S, D = x.shape
        H2 = D // 2
        x_l = x[:, :, :H2]
        x_r = x[:, :, H2:]

        q_l, k_l, v_l = self.q_l(x_l), self.k_l(x_l), self.v_l(x_l)
        q_r, k_r, v_r = self.q_r(x_r), self.k_r(x_r), self.v_r(x_r)

        total_si = 0
        for _ in range(self.rounds):
            out_l, si_l = self._attn(q_l, k_r, v_l)
            out_r, si_r = self._attn(q_r, k_l, v_r)
            total_si = total_si + si_l + si_r
            q_l, q_r = q_r, q_l
            k_l, k_r = k_r, k_l
            v_l, v_r = v_r, v_l

        out = torch.cat([out_l, out_r], dim=-1)
        out = self.out_proj(out)

        # Residual connection: preserve 50% of original signal
        out = out + 0.5 * x

        return out, total_si


class HolographicBraidEngine(nn.Module):
    """
    Multi-scale holographic braiding engine.

    Layers operate at different scales of modular exponentiation:
      Layer 0: phase shift by a^(2^0) = a
      Layer 1: phase shift by a^(2^1) = a^2
      Layer 2: phase shift by a^(2^2) = a^4
      ...

    Between layers, explicit modular arithmetic phase shifts advance the
    exponent. The Feistel cross-attention spreads phase correlations across
    positions, creating holographic interference patterns.
    """

    def __init__(self, a, N, d_model=64, heads=8, rounds=3, num_layers=4):
        super().__init__()
        self.a, self.N = a, N
        self.num_layers = num_layers
        self.feistels = nn.ModuleList([
            DeterministicPhaseFeistel(a, N, d_model, heads, rounds)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        current = x
        all_si = []
        all_layer_outputs = []

        for k, feistel in enumerate(self.feistels):
            current, si = feistel(current)
            all_si.append(si)

            power = pow(self.a, 2 ** k, self.N)
            phase_shift = 2.0 * math.pi * power / self.N
            rot = complex(math.cos(phase_shift), math.sin(phase_shift))
            current = current * rot
            all_layer_outputs.append(current)

        return current, all_si, all_layer_outputs


# ============================================================================
# TAPE INITIALIZATION (shared between paths)
# ============================================================================

def init_phase_tape(a, N, S, d_model, init_mode="sequential"):
    """
    Initialize S tape positions with phase-encoded modular exponentiation
    values, embedded into d_model dimensions.
    """
    if init_mode == "sequential":
        indices = list(range(S))
    elif init_mode == "bit_reversed":
        n_bits = max(1, int(math.ceil(math.log2(S))))
        indices = [int(f"{i:0{n_bits}b}"[::-1], 2) for i in range(S)]
    elif init_mode == "log_spaced":
        if S <= 1:
            indices = [0]
        else:
            max_n = max(S, min(100000, 10 * S))
            indices = [int(max_n ** (i / max(1, S - 1))) for i in range(S)]
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    tape = torch.zeros(S, d_model, dtype=torch.complex64)
    for s, n in enumerate(indices):
        val = pow(a, n, N)
        base_phase = 2.0 * math.pi * val / N
        for d in range(d_model):
            harmonic = (d + 1) * base_phase
            tape[s, d] = complex(math.cos(harmonic), math.sin(harmonic))

    return tape.unsqueeze(0), indices


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment_pair(a, N, S, d_model, heads, rounds, num_layers, init_mode):
    """
    Run both PATH A (baseline) and PATH B (holographic) for one configuration.
    Returns comparison results.
    """
    t0 = time.perf_counter()

    # PATH A: Baseline
    bl_z, bl_results = baseline_period_extraction(a, N, S, init_mode)
    bl_ac_ok = verify_tag_r(a, N, bl_results["ac_r"])[1]
    bl_fft_ok = verify_tag_r(a, N, bl_results["fft_r"])[1]
    t_bl = time.perf_counter()

    # PATH B: Holographic
    tape, indices = init_phase_tape(a, N, S, d_model, init_mode)
    engine = HolographicBraidEngine(a, N, d_model=d_model, heads=heads,
                                      rounds=rounds, num_layers=num_layers)
    with torch.no_grad():
        tape_out, all_si, layer_outputs = engine(tape)
    z_braided = tape_out.mean(dim=-1).squeeze(0)

    hg_ac_r, hg_ac_snr, _ = compute_autocorrelation_period(z_braided)
    hg_fft_r, hg_fft_peak = compute_fft_period(z_braided)
    hg_ac_ok = verify_tag_r(a, N, hg_ac_r)[1]
    hg_fft_ok = verify_tag_r(a, N, hg_fft_r)[1]

    # Phase coherence at each layer
    layer_coherences = []
    for lo in layer_outputs:
        zlo = lo.mean(dim=-1).squeeze(0)
        mag = zlo.abs().mean().item()
        layer_coherences.append(mag)

    si_mean = sum(s.abs().mean().item() for s in all_si) / max(len(all_si), 1)

    t_hg = time.perf_counter()

    return {
        "S": S,
        "baseline": {
            "ac_r": bl_results["ac_r"], "ac_ok": bl_ac_ok,
            "fft_r": bl_results["fft_r"], "fft_ok": bl_fft_ok,
            "ac_snr": bl_results["ac_snr"],
            "any_ok": bl_ac_ok or bl_fft_ok,
        },
        "holographic": {
            "ac_r": hg_ac_r, "ac_ok": hg_ac_ok,
            "fft_r": hg_fft_r, "fft_ok": hg_fft_ok,
            "ac_snr": hg_ac_snr,
            "any_ok": hg_ac_ok or hg_fft_ok,
        },
        "timing": {
            "t_baseline": t_bl - t0,
            "t_holographic": t_hg - t_bl,
            "t_total": t_hg - t0,
        },
        "diagnostics": {
            "si_mean": si_mean,
            "layer_coherences": layer_coherences,
            "z_braided_mean_mag": z_braided.abs().mean().item(),
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 20.10: HOLOGRAPHIC PHASE ORACLE")
    print("  Can a multi-scale Feistel braid encode U_a in S << r positions?")
    print("=" * 80)
    print()
    print("  PATH A (Baseline):  Raw S samples -> autocorrelation")
    print("  PATH B (Holographic): S samples -> Feistel braid -> autocorrelation")
    print()
    print("  If PATH B succeeds when PATH A fails and S < r:")
    print("    >> Holographic encoding beats the period-containment limit.")
    print("    Otherwise: period-containment limit HOLDS.")
    print()

    BIT_SIZES = [10, 12, 14, 16]
    S_VALUES = [64, 128, 256, 512, 1024, 2048]
    D_MODEL = 64
    HEADS = 8
    ROUNDS = 3
    NUM_LAYERS = 4

    print("-" * 80)
    print(f"  Feistel config: d_model={D_MODEL}, heads={HEADS}, "
          f"rounds/layer={ROUNDS}, layers={NUM_LAYERS}")
    print(f"  All weights: deterministic (constructed from a, N via phase rotation)")
    print("-" * 80)
    print()

    all_rows = []

    for BIT_SIZE in BIT_SIZES:
        N, p_known, q_known = generate_semiprime(BIT_SIZE)
        a = 2
        while gcd(a, N) != 1:
            a += 1
        r_true = true_period(a, N, max_steps=5_000_000)

        print("=" * 80)
        print(f"  BIT SIZE: {BIT_SIZE}  N = {N} ({p_known} x {q_known})")
        print(f"  a = {a},  TRUE PERIOD r = {r_true}")
        print("=" * 80)
        print(f"  {'S':>6} {'S>=r':>6} | {'BL ac_r':>10} {'ok':>5} {'SNR':>8} | {'HG ac_r':>10} {'ok':>5} {'SNR':>8} | {'t(s)':>7} {'si':>8}")
        print(f"  {'-'*95}")

        for S in S_VALUES:
            if r_true == 0:
                break
            if S > 4 * r_true:
                continue

            result = run_experiment_pair(
                a, N, S, D_MODEL, HEADS, ROUNDS, NUM_LAYERS, "sequential"
            )

            bl = result["baseline"]
            hg = result["holographic"]
            t = result["timing"]
            d = result["diagnostics"]

            S_ge_r = "YES" if S >= r_true else "no"

            print(
                f"  {S:>6} {S_ge_r:>6} | "
                f"{bl['ac_r']:>10} {str(bl['ac_ok']):>5} {bl['ac_snr']:>8.2f} | "
                f"{hg['ac_r']:>10} {str(hg['ac_ok']):>5} {hg['ac_snr']:>8.2f} | "
                f"{t['t_total']:>7.3f} {d['si_mean']:>8.4f}"
            )

            all_rows.append({
                "bits": BIT_SIZE, "N": N, "r_true": r_true, "S": S,
                "S_ge_r": S_ge_r,
                "bl_ac_r": bl["ac_r"], "bl_ac_ok": bl["ac_ok"],
                "bl_snr": bl["ac_snr"], "bl_any_ok": bl["any_ok"],
                "hg_ac_r": hg["ac_r"], "hg_ac_ok": hg["ac_ok"],
                "hg_snr": hg["ac_snr"], "hg_any_ok": hg["any_ok"],
                "si_mean": d["si_mean"],
                "hg_layer_coh_0": d["layer_coherences"][0] if d["layer_coherences"] else 0,
                "hg_layer_coh_final": d["layer_coherences"][-1] if d["layer_coherences"] else 0,
                "z_mag": d["z_braided_mean_mag"],
            })

        print()

    # ====================================================================
    # SCALING ANALYSIS
    # ====================================================================
    print("=" * 80)
    print("SCALING ANALYSIS: Baseline vs. Holographic")
    print("=" * 80)
    print()
    print(f"  {'Bits':>6} {'r':>10} {'S':>6} {'S>=r':>6} {'BL ok':>6} {'HG ok':>6} {'HG>BL':>6} {'BL SNR':>8} {'HG SNR':>8} {'si':>8}")
    print(f"  {'-'*90}")

    for row in all_rows:
        hg_beats_bl = "***" if (row["hg_any_ok"] and not row["bl_any_ok"]) else ""
        print(
            f"  {row['bits']:>6} {row['r_true']:>10} {row['S']:>6} {row['S_ge_r']:>6} "
            f"{str(row['bl_any_ok']):>6} {str(row['hg_any_ok']):>6} {hg_beats_bl:>6} "
            f"{row['bl_snr']:>8.2f} {row['hg_snr']:>8.2f} {row['si_mean']:>8.4f}"
        )

    print()
    print("  (***) = Holographic succeeds where Baseline fails")
    print()

    # ====================================================================
    # VERDICT
    # ====================================================================
    print("-" * 80)
    print("VERDICT:")
    print("-" * 80)

    bl_successes = {}
    hg_successes = {}
    hg_only_successes = {}

    for row in all_rows:
        key = (row["bits"], row["r_true"])
        if row["bl_any_ok"]:
            if key not in bl_successes:
                bl_successes[key] = float('inf')
            bl_successes[key] = min(bl_successes[key], row["S"])
        if row["hg_any_ok"]:
            if key not in hg_successes:
                hg_successes[key] = float('inf')
            hg_successes[key] = min(hg_successes[key], row["S"])
        if row["hg_any_ok"] and not row["bl_any_ok"]:
            if key not in hg_only_successes:
                hg_only_successes[key] = float('inf')
            hg_only_successes[key] = min(hg_only_successes[key], row["S"])

    for key in sorted(set(list(bl_successes.keys()) + list(hg_successes.keys()))):
        bits, r = key
        bl_min = bl_successes.get(key, None)
        hg_min = hg_successes.get(key, None)
        ratio_bl = f"{bl_min/max(1,r):.4f}" if bl_min else "---"
        ratio_hg = f"{hg_min/max(1,r):.4f}" if hg_min else "---"
        improvement = ""
        if hg_min is not None and bl_min is None:
            improvement = " <- HOLOGRAPHIC BEATS LIMIT"
        elif hg_min is not None and bl_min is not None and hg_min < bl_min:
            improvement = " <- HG improves on BL"
        print(f"  {bits}-bit (r={r}): BL min S={bl_min or 'N/A'} (ratio={ratio_bl}), "
              f"HG min S={hg_min or 'N/A'} (ratio={ratio_hg}){improvement}")

    if hg_only_successes:
        print()
        print("  HOLOGRAPHIC-ONLY SUCCESSES (Baseline failed, Holographic succeeded):")
        for key in sorted(hg_only_successes):
            bits, r = key
            s = hg_only_successes[key]
            print(f"    {bits}-bit, r={r}: HG succeeded at S={s} (S/r={s/max(1,r):.4f})")
        print()
        print("  *** The holographic Feistel braid enables period detection at S < r ***")
        print("  *** where the baseline (raw autocorrelation) fails. ***")
    else:
        print()
        print("  No holographic-only successes detected.")
        print("  The Feistel braid does not improve period extraction over baseline.")
        print("  Period-containment limit (S >= r) HOLDS for this architecture.")

    # ====================================================================
    # DETAILED DIAGNOSTICS
    # ====================================================================
    print()
    print("=" * 80)
    print("DETAILED LAYER-BY-LAYER DIAGNOSTICS")
    print("=" * 80)

    diag_bits = 12
    diag_N, diag_p, diag_q = generate_semiprime(diag_bits)
    diag_a = 2
    while gcd(diag_a, diag_N) != 1:
        diag_a += 1
    diag_r = true_period(diag_a, diag_N, max_steps=5_000_000)
    print(f"\n  N = {diag_N}, a = {diag_a}, r = {diag_r}")

    for diag_S in [128, 256, 512, 1024]:
        tape, indices = init_phase_tape(diag_a, diag_N, diag_S, D_MODEL, "sequential")
        engine = HolographicBraidEngine(diag_a, diag_N, D_MODEL, HEADS, ROUNDS, NUM_LAYERS)
        with torch.no_grad():
            tape_out, all_si, layer_outputs = engine(tape)

        print(f"\n  --- S = {diag_S} (r = {diag_r}, S/r = {diag_S/max(1,diag_r):.4f}) ---")
        print(f"  Layer outputs (mean abs magnitude per layer):")
        for k, lo in enumerate(layer_outputs):
            zlo = lo.mean(dim=-1).squeeze(0)
            mag = zlo.abs().mean().item()
            r_ac, snr, _ = compute_autocorrelation_period(zlo)
            ac_ok = verify_tag_r(diag_a, diag_N, r_ac)[1]
            print(f"    Layer {k}: mean|z|={mag:.4f}, AC r={r_ac}, SNR={snr:.2f}, "
                  f"verified={ac_ok}")

        z_final = tape_out.mean(dim=-1).squeeze(0)
        r_final, snr_final, _ = compute_autocorrelation_period(z_final)
        print(f"  Final output: mean|z|={z_final.abs().mean().item():.4f}, "
              f"AC r={r_final}, SNR={snr_final:.2f}, "
              f"verified={verify_tag_r(diag_a, diag_N, r_final)[1]}")

        # Baseline comparison
        bl_z, bl_results = baseline_period_extraction(diag_a, diag_N, diag_S, "sequential")
        print(f"  Baseline:     AC r={bl_results['ac_r']}, SNR={bl_results['ac_snr']:.2f}, "
              f"verified={verify_tag_r(diag_a, diag_N, bl_results['ac_r'])[1]}")

    print()
    print("=" * 80)
    print("EXPERIMENT 20.10 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()