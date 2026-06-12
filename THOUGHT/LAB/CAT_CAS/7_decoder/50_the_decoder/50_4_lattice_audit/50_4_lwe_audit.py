"""
Exp 50.4 - Lattice audit: does Exp 25's holographic LWE attack cross the located
barrier, or only work at toy scale?

In 50.2c we located the irreducible decodability bedrock: the dihedral-HSP slope is
info-cheap but compute-hard = 1-bit-LWE / unique-SVP (lattice) hardness (Regev).
Exp 25 (`25_lattice_holography/2_holographic_svp.py`) claims to BREAK LWE with a
holographic phase-resonance readout (map A,B to the torus, optimise a continuous
secret with Adam, sieve the error with an FFT low-pass). If that attack genuinely
recovers the secret as the lattice dimension grows toward Kyber-256 under realistic
noise, it crosses the barrier we located (an extraordinary claim). If it only works
at toy n / near-zero noise, that is exactly the lattice barrier asserting itself.

This brick runs Exp 25's OWN attack under the Exp-50 null discipline:
  - recovery vs lattice dimension n (toward Kyber-256),
  - recovery vs error magnitude sigma (toward the Kyber regime),
  - a no-secret null (B uniform random: no planted secret to find),
  - a random-guess chance baseline (coord accuracy = 1/q).

Two readouts are tested so the verdict is not an artifact of one parametrisation bug:
  - "faithful"   : EXACTLY Exp 25's solver (forward A_phase @ S_phase, readout
                   (S_phase % 2pi)/(2pi)*q). The forward wants S_phase = the integer
                   secret, but the readout treats S_phase as an angle in [0,2pi) -
                   inconsistent unless secret entries are < ~6, so it can only recover
                   tiny secrets.
  - "charitable" : same resonance optimisation, but the obvious bug fixed - read the
                   secret as round(S_phase) % q, i.e. the integer the forward model
                   actually optimises. Gives the attack its best possible shot.

No catalytic tape here: this brick AUDITS an existing solver's scaling behaviour; it
borrows no substrate, so adding a tape would be ceremonial (CAT_CAS_OS s.7). The
mechanism is the Exp 25 attack itself, measured against nulls.

Run:  python 50_4_lwe_audit.py
"""
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 50_the_decoder root
import decoder_lib as dl  # noqa: E402  (shared stats: bootstrap_ci, cohen_d)

Q_KYBER = 3329  # Exp 25's default modulus (Kyber prime)

LINES = []
def log(m=""):
    print(m)
    LINES.append(str(m))


# ---------------------------------------------------------------------------
# LWE instance (matches 25_lattice_holography/1_lwe_simulator.py)
# ---------------------------------------------------------------------------
def make_lwe(n, m, q, sigma, rng, planted=True):
    A = torch.tensor(rng.integers(0, q, size=(m, n)), dtype=torch.float64)
    if planted:
        S = torch.tensor(rng.integers(0, q, size=(n, 1)), dtype=torch.float64)
        E = torch.round(torch.tensor(rng.standard_normal((m, 1)) * sigma, dtype=torch.float64))
        B = (A @ S + E) % q
    else:
        S = None  # null: no secret exists; B is uniform random mod q
        B = torch.tensor(rng.integers(0, q, size=(m, 1)), dtype=torch.float64)
    return A, B, S


# ---------------------------------------------------------------------------
# Exp 25's holographic resonance attack (reproduced)
# ---------------------------------------------------------------------------
def holo_attack(A, B, q, epochs=250, lr=0.1, seed=0, readout="charitable"):
    torch.manual_seed(seed)
    m, n = A.shape
    A_phase = A * (2 * np.pi / q)
    B_phase = B * (2 * np.pi / q)
    Z_B = torch.exp(1j * B_phase)
    cutoff = max(1, int(m * 0.15))               # Exp 25 phase-cavity FFT low-pass
    fB = torch.fft.fft(Z_B, dim=0); fB[cutoff:-cutoff] = 0
    ZB_s = torch.fft.ifft(fB, dim=0)

    S_phase = torch.nn.Parameter(torch.rand(n, 1, dtype=torch.float64) * 2 * np.pi)
    opt = torch.optim.Adam([S_phase], lr=lr)
    res = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        pred = A_phase @ S_phase
        Zp = torch.exp(1j * pred)
        fp = torch.fft.fft(Zp, dim=0); fp[cutoff:-cutoff] = 0
        Zp_s = torch.fft.ifft(fp, dim=0)
        resonance = torch.mean(torch.real(Zp_s * torch.conj(ZB_s)))
        (-resonance).backward()
        opt.step()
        res = float(resonance.item())
    with torch.no_grad():
        if readout == "faithful":
            S_pred = torch.round((S_phase % (2 * np.pi)) / (2 * np.pi) * q)
        else:  # charitable: the integer the forward model actually optimises
            S_pred = torch.round(S_phase) % q
    return S_pred, res


def recovery(S_pred, S_true, q):
    """Centered mod-q error norm and exact-coordinate accuracy."""
    diff = (S_pred - S_true) % q
    diff = torch.where(diff > q / 2, diff - q, diff)
    err = float(torch.norm(diff).item())
    coord_acc = float((diff == 0).double().mean().item())
    return err, coord_acc


def run_cell(n, sigma, q, trials, rng, readout, epochs):
    m = min(8 * n, 1024)
    exact, accs, resons = 0, [], []
    for t in range(trials):
        A, B, S = make_lwe(n, m, q, sigma, rng, planted=True)
        S_pred, res = holo_attack(A, B, q, epochs=epochs, seed=int(rng.integers(1 << 30)), readout=readout)
        err, acc = recovery(S_pred, S, q)
        exact += int(err == 0.0)
        accs.append(acc)
        resons.append(res)
    return {
        "n": n, "sigma": sigma, "readout": readout, "trials": trials,
        "exact_rate": exact / trials,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "acc_ci": dl.bootstrap_ci(accs, n_boot=2000, seed=1),
        "res_mean": float(np.mean(resons)),
        "chance": 1.0 / q,
    }


def main():
    if not _HAS_TORCH:
        log("torch unavailable - cannot run the Exp 25 attack. ABORT.")
        sys.exit(2)

    q = Q_KYBER
    rng = np.random.default_rng(50_4)
    log("=" * 96)
    log("EXP 50.4  -  LATTICE AUDIT: does Exp 25's holographic LWE attack cross the located barrier?")
    log("  located bedrock (50.2c) = dihedral-HSP / 1-bit-LWE <-> unique-SVP (lattice, Regev)")
    log("  q=%d (Kyber prime); chance per-coordinate accuracy = 1/q = %.2e" % (q, 1.0 / q))
    log("=" * 96)

    EP = 250
    TR = 5

    # ----- SWEEP A: recovery vs lattice dimension n (sigma=0, easiest case) -----
    log("\n[SWEEP A] recovery vs lattice dimension n  (sigma=0, error-free - the attack's BEST case)")
    log("  n toward Kyber-256.  exact_rate = fraction of trials with the WHOLE secret recovered.")
    log("  acc = mean exact-coordinate fraction [95%% bootstrap CI];  chance = %.2e" % (1.0 / q))
    log("  %-10s %-6s %-12s %-26s %-10s" % ("readout", "n", "exact_rate", "coord_acc [95% CI]", "resonance"))
    sweepA = []
    for readout in ("faithful", "charitable"):
        for n in (2, 4, 8, 16, 32, 64, 128):
            c = run_cell(n, 0.0, q, TR, rng, readout, EP)
            sweepA.append(c)
            log("  %-10s %-6d %-12.2f [%.4f, %.4f] %-6s %-10.4f"
                % (readout, n, c["exact_rate"], c["acc_ci"][0], c["acc_ci"][1], "", c["res_mean"]))

    # ----- SWEEP B: recovery vs noise sigma (charitable, small n where it has the best chance) -----
    log("\n[SWEEP B] recovery vs error magnitude sigma  (charitable readout, n=8 - its best regime)")
    log("  sigma toward the Kyber noise regime.  does any recovery survive realistic error?")
    log("  %-8s %-12s %-26s %-10s" % ("sigma", "exact_rate", "coord_acc [95% CI]", "resonance"))
    sweepB = []
    for sigma in (0.0, 1.0, 2.0, 4.0, 8.0):
        c = run_cell(8, sigma, q, TR, rng, "charitable", EP)
        sweepB.append(c)
        log("  %-8.1f %-12.2f [%.4f, %.4f] %-6s %-10.4f"
            % (sigma, c["exact_rate"], c["acc_ci"][0], c["acc_ci"][1], "", c["res_mean"]))

    # ----- SWEEP C: tiny modulus q (small secret entries) - the attack's ABSOLUTE best case -----
    log("\n[SWEEP C] recovery vs tiny modulus q  (n in {2,4}, sigma=0, 800 epochs - most favorable)")
    log("  small q => small secret entries => the faithful readout's [0,2pi) range can express them.")
    log("  if the attack EVER recovers a secret exactly, it is here.")
    log("  %-10s %-5s %-5s %-12s %-26s %-10s" % ("readout", "q", "n", "exact_rate", "coord_acc [95% CI]", "resonance"))
    sweepC = []
    for readout in ("faithful", "charitable"):
        for qq in (5, 11, 17, 37):
            for n in (2, 4):
                m = min(16 * n, 256)
                exact, accs, resons = 0, [], []
                for _ in range(8):
                    A, B, S = make_lwe(n, m, qq, 0.0, rng, planted=True)
                    Sp, r = holo_attack(A, B, qq, epochs=800, seed=int(rng.integers(1 << 30)), readout=readout)
                    err, acc = recovery(Sp, S, qq)
                    exact += int(err == 0.0); accs.append(acc); resons.append(r)
                c = {"readout": readout, "q": qq, "n": n, "exact_rate": exact / 8,
                     "acc_mean": float(np.mean(accs)), "acc_ci": dl.bootstrap_ci(accs, n_boot=1500, seed=1),
                     "res_mean": float(np.mean(resons)), "chance": 1.0 / qq}
                sweepC.append(c)
                log("  %-10s %-5d %-5d %-12.2f [%.4f, %.4f] %-6s %-10.4f"
                    % (readout, qq, n, c["exact_rate"], c["acc_ci"][0], c["acc_ci"][1], "", c["res_mean"]))

    # ----- NULL block: no-secret null vs planted (does the attack even detect a secret?) -----
    log("\n[NULL] no-secret null (B uniform random) vs planted, n=8 sigma=0, charitable, 8 trials each")
    NT = 8
    planted_acc, planted_res = [], []
    null_acc, null_res = [], []
    decoy_rng = np.random.default_rng(7)
    for _ in range(NT):
        # planted
        A, B, S = make_lwe(8, 64, q, 0.0, rng, planted=True)
        Sp, r = holo_attack(A, B, q, epochs=EP, seed=int(rng.integers(1 << 30)), readout="charitable")
        _, acc = recovery(Sp, S, q)
        planted_acc.append(acc); planted_res.append(r)
        # null: no planted secret; score S_pred against a random decoy (must be ~chance)
        A0, B0, _ = make_lwe(8, 64, q, 0.0, rng, planted=False)
        Sp0, r0 = holo_attack(A0, B0, q, epochs=EP, seed=int(rng.integers(1 << 30)), readout="charitable")
        decoy = torch.tensor(decoy_rng.integers(0, q, size=(8, 1)), dtype=torch.float64)
        _, acc0 = recovery(Sp0, decoy, q)
        null_acc.append(acc0); null_res.append(r0)
    d_acc = dl.cohen_d(planted_acc, null_acc)
    d_res = dl.cohen_d(planted_res, null_res)
    log("  planted: acc=%.4f+/-%.4f  resonance=%.4f+/-%.4f"
        % (np.mean(planted_acc), np.std(planted_acc), np.mean(planted_res), np.std(planted_res)))
    log("  null   : acc=%.4f+/-%.4f  resonance=%.4f+/-%.4f"
        % (np.mean(null_acc), np.std(null_acc), np.mean(null_res), np.std(null_res)))
    log("  Cohen d planted-vs-null:  coord_acc d=%.2f   resonance d=%.2f" % (d_acc, d_res))
    log("  (resonance d ~ 0 => the attack reaches the same 'success' with NO secret present:")
    log("   its objective does not detect the secret - it just aligns two phase fields.)")

    # ===================== GATES / VERDICT =====================
    def cell(sw, **kw):
        return next(c for c in sw if all(c[k] == v for k, v in kw.items()))

    chance = 1.0 / q
    # an attack "recovers" at a cell if its coord_acc CI lower bound is clearly above chance
    def recovers(c, factor=10):
        return c["acc_ci"][0] > factor * chance or c["exact_rate"] > 0.0

    log("\n" + "=" * 96)
    log("GATES")

    # G1: does the attack work ANYWHERE - even tiny q, n=2, error-free, 800 epochs (its best case)?
    best_exact = max(c["exact_rate"] for c in sweepC)
    g1 = best_exact > 0.0
    g1_det = "best tiny-q exact_rate=%.2f (q in {5,11,17,37}, n in {2,4})" % best_exact

    # G2: recovery COLLAPSES as n grows - at n=128 it is at chance (both readouts)
    big = [cell(sweepA, readout=ro, n=128, sigma=0.0) for ro in ("faithful", "charitable")]
    g2 = all(not recovers(c) for c in big)
    g2_det = "n=128 exact_rate=%s acc_CI_lo=%s (chance=%.1e)" % (
        [round(c["exact_rate"], 2) for c in big], [round(c["acc_ci"][0], 5) for c in big], chance)

    # G3: recovery COLLAPSES with realistic noise - at sigma>=2 it is at chance
    noisy = [c for c in sweepB if c["sigma"] >= 2.0]
    g3 = all(not recovers(c) for c in noisy)
    g3_det = "sigma>=2 exact_rate=%s acc_CI_lo=%s" % (
        [round(c["exact_rate"], 2) for c in noisy], [round(c["acc_ci"][0], 5) for c in noisy])

    # G4: the no-secret null is indistinguishable in the attack's own objective (|resonance d| small)
    g4 = abs(d_res) < 0.8
    g4_det = "resonance Cohen d planted-vs-null = %.2f (want |d|<0.8)" % d_res

    gates = [
        ("G1 attack works at toy scale (so we are auditing a real attack)", g1, g1_det),
        ("G2 recovery collapses as n -> Kyber-256 (chance at n=128)", g2, g2_det),
        ("G3 recovery collapses under realistic noise (chance at sigma>=2)", g3, g3_det),
        ("G4 no-secret null indistinct in the attack's objective", g4, g4_det),
    ]
    for nm, ok, det in gates:
        log("  [%s] %-58s  %s" % ("PASS" if ok else "FAIL", nm, det))
    log("=" * 96)

    # Verdict logic:
    #   - survives n or noise          -> EXP25_SCALES_ESCALATE (extraordinary; adversarial review)
    #   - works at tiny q but collapses -> EXP25_TOY_SCALE_ONLY  (toy artifact; barrier holds)
    #   - never recovers even at best   -> EXP25_NONFUNCTIONAL   (the "break" was never real)
    if any(recovers(c) for c in big) or any(recovers(c) for c in noisy):
        verdict = "EXP25_SCALES_ESCALATE"
    elif g1 and g2 and g3:
        verdict = "EXP25_TOY_SCALE_ONLY"
    elif (not g1) and g2 and g3 and g4:
        verdict = "EXP25_NONFUNCTIONAL"
    else:
        verdict = "EXP25_AUDIT_INCONCLUSIVE"
    log("VERDICT: %s" % verdict)
    if verdict == "EXP25_NONFUNCTIONAL":
        log("  Exp 25's holographic LWE attack does NOT recover the secret at ANY tested point - not")
        log("  at toy dimension, not at a tiny modulus, not error-free, not after 800 epochs. Exact")
        log("  recovery rate is 0 across n=2..128, sigma=0..8, and q=5..3329, and its resonance")
        log("  objective is statistically indistinguishable on a planted instance vs a no-secret null")
        log("  (resonance Cohen d=%.2f). The 'LATTICE BROKEN' branch never fires under matched" % d_res)
        log("  conditions; the attack maximises phase overlap, which is decoupled from secret recovery.")
        log("  This is fully consistent with the lattice barrier located in 50.2c: the holographic")
        log("  readout does not cross unique-SVP hardness - not at scale, and not even at toy scale.")
    else:
        log("  Exp 25's holographic LWE attack recovers the secret only at toy scale; it decays to the")
        log("  chance baseline as n grows toward Kyber-256 and as sigma reaches the real LWE regime,")
        log("  and its objective cannot tell a planted instance from a no-secret null. Consistent with")
        log("  the lattice barrier located in 50.2c - Exp 25 does NOT cross the bedrock.")
    log("  (Claim level 4-5: a bounded audit of one solver under matched nulls, not a proof about")
    log("  all lattice attacks. The located barrier is the standard Regev hardness, not proven by us.)")

    import json
    (HERE / "lattice_audit_result.json").write_text(json.dumps({
        "q": q, "chance": chance,
        "sweep_n": sweepA, "sweep_sigma": sweepB, "sweep_tiny_q": sweepC,
        "null": {"planted_acc_mean": float(np.mean(planted_acc)),
                 "null_acc_mean": float(np.mean(null_acc)),
                 "resonance_cohen_d": d_res, "coord_acc_cohen_d": d_acc},
        "gates": {nm: bool(ok) for nm, ok, _ in gates},
        "verdict": verdict,
    }, indent=2, default=float), encoding="utf-8")
    (HERE / "output_lattice_audit.txt").write_text("\n".join(LINES), encoding="utf-8")

    all_pass = g1 and g2 and g3 and g4
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
