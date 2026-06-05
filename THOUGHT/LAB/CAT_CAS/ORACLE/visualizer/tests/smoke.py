"""Smoke tests for the engine.

Verifies that engine output matches the source's main() print.

Run from the visualizer/ directory:
    python -m tests.smoke
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VISUALIZER_DIR = os.path.dirname(HERE)
if VISUALIZER_DIR not in sys.path:
    sys.path.insert(0, VISUALIZER_DIR)

from engine import oracle_1d, oracle_2d, oracle_3d


def test_1d_all_machines():
    """For each of the 4 test machines, verify the winding W and verdict.

    Source reference: 36_nonhermitian_oracle.py:335-339.
      halt_direct  -> W=0  -> HALTS
      halt_chain   -> W=0  -> HALTS
      loop_2cycle  -> W=+1 -> LOOPS
      loop_3cycle  -> W=+1 -> LOOPS
    """
    expected = {
        "halt_direct": ("HALTS", 0),
        "halt_chain":  ("HALTS", 0),
        "loop_2cycle": ("LOOPS", 1),
        "loop_3cycle": ("LOOPS", 1),
    }

    print("  1D oracle (35.2) -- 4 test machines")
    print("  " + "-" * 50)
    for machine, (verdict, w) in expected.items():
        r = oracle_1d.run(machine=machine, n_phi=400)
        actual_w = r["winding"]["Wint"]
        actual_v = r["verdict"]
        assert actual_w == w, f"{machine}: W={actual_w}, expected {w}"
        assert actual_v == verdict, f"{machine}: verdict={actual_v}, expected {verdict}"
        print(f"  {machine:12s}  W={actual_w:+d}  verdict={actual_v:5s}  OK")


def test_1d_halt_sink_strength():
    """Vary halt_mult, confirm halt machine still halts (sink strength doesn't break it)."""
    for halt_mult in (2.0, 5.0, 10.0, 30.0):
        r = oracle_1d.run(machine="halt_direct", halt_mult=halt_mult)
        assert r["verdict"] == "HALTS", f"halt_mult={halt_mult}: not halting"
        assert r["winding"]["Wint"] == 0
    print("  halt_mult sweep (halt_direct): 2..30  all W=0  OK")


def test_1d_gamma_zero_loop():
    """For loop_2cycle, gamma=0 -> no off-diagonal -> H is diagonal -> det constant -> W=0.

    Hmm actually with gamma=0, H[0,2] = 0, so twist has no effect. Hmm but the loop
    is a 2-cycle. With gamma=0, the transitions are not realized. The system is fully
    decoupled. Det is constant. W=0.
    """
    r = oracle_1d.run(machine="loop_2cycle", gamma=0.0)
    # With gamma=0, there's no cycle-closing edge to twist, so W=0.
    assert r["winding"]["Wint"] == 0
    assert r["verdict"] == "HALTS"  # HALTS because W=0
    print("  gamma=0 (loop_2cycle):  W=0  verdict=HALTS  OK (decoupled)")


def test_1d_dim_matches_machine():
    """N = num_states * 2."""
    expected_dims = {"halt_direct": 4, "halt_chain": 6, "loop_2cycle": 4, "loop_3cycle": 6}
    for machine, n in expected_dims.items():
        r = oracle_1d.run(machine=machine)
        assert r["N"] == n, f"{machine}: N={r['N']}, expected {n}"
    print(f"  N = num_states * 2 for all 4 machines  OK")


# ---- 2D (Phase 1B) -----------------------------------------------------

def test_2d_loop_L8():
    """L=8, gamma_halt=0.0 -> chiral edge protected -> C=+1 -> LOOPS.
    Matches 37_2d_chern_oracle.py run_2d_oracle(L=8) default.
    """
    r = oracle_2d.run(L=8, gamma_halt=0.0)
    assert r["L"] == 8
    assert r["N"] == 64
    assert r["halt_pos"] == [4, 4]
    assert r["halt_site"] == 4 * 8 + 4
    assert r["bott"]["C"] == 1, f"expected C=1, got {r['bott']['C']}"
    assert r["verdict"] == "LOOPS"
    print(f"  L=8  loop case:  C={r['bott']['C']:+d}  verdict=LOOPS  OK")


def test_2d_halt_L8():
    """L=8, gamma_halt=10.0 -> EP sink destroys edge -> C=0 -> HALTS.
    Matches 37_2d_chern_oracle.py run_2d_oracle(L=8) halt case.
    """
    r = oracle_2d.run(L=8, gamma_halt=10.0)
    assert r["bott"]["C"] == 0, f"expected C=0, got {r['bott']['C']}"
    assert r["verdict"] == "HALTS"
    print(f"  L=8  halt case:  C={r['bott']['C']:+d}  verdict=HALTS  OK")


def test_2d_sink_strength_sweep():
    """gamma_halt sweep: increasing gamma_halt drives C from nonzero to 0.
    At L=8 with default t1=1, t2=0.5, phi=pi/4, the gap is robust.
    """
    cs = []
    for g in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        r = oracle_2d.run(L=8, gamma_halt=g)
        cs.append((g, r["bott"]["C"]))
    # At L=8 the topology is robust; should be C=1 for small g and 0 for large g.
    assert cs[0][1] == 1, f"g=0 expected C=1, got {cs[0][1]}"
    assert cs[-1][1] == 0, f"g=10 expected C=0, got {cs[-1][1]}"
    print(f"  L=8  gamma_halt sweep: 0->1, 10->0  OK  {cs}")


def test_2d_halt_site_imag():
    """At the halt site (center), Im(H_site_site) = -(loss + gamma_halt)."""
    r = oracle_2d.run(L=8, gamma_halt=10.0)
    site = r["halt_site"]
    H_site = r["H"][site][site]
    expected_im = -(0.05 + 10.0)
    assert abs(H_site["im"] - expected_im) < 1e-3, \
        f"H[halt][halt].im = {H_site['im']}, expected {expected_im}"
    print(f"  L=8  halt site: Im(H)={H_site['im']:.4f}  (expected {expected_im})  OK")


def test_2d_dim():
    """N = L*L."""
    for L in (4, 6, 8, 10):
        r = oracle_2d.run(L=L, gamma_halt=0.0, include_projector=False)
        assert r["N"] == L * L, f"L={L}: N={r['N']}, expected {L*L}"
    print(f"  N = L*L for L in 4,6,8,10  OK")


# ---- 3D (Phase 1C) -----------------------------------------------------

def test_3d_loop_L8():
    """L=8, n_kz=24, gamma_halt=0 -> nonzero C slices -> LOOPS.

    Matches 38_3d_weyl_oracle.py run_3d_weyl_oracle default:
      Non-zero slices: 14/24  Max |C| = 2  -> LOOPS
    """
    r = oracle_3d.run(L=8, n_kz=24, gamma_halt=0.0)
    assert r["L"] == 8
    assert r["n_kz"] == 24
    assert r["profile"]["max_abs_C"] == 2
    assert r["profile"]["nonzero"] == 14
    assert r["verdict"] == "LOOPS (Fermi arc exists)"
    # Weyl node positions for m0=0.5, tz=1.5
    expected_nodes = [1.2309594173407747, 5.052225889350412]
    assert abs(r["profile"]["weyl_nodes"][0] - expected_nodes[0]) < 1e-3
    assert abs(r["profile"]["weyl_nodes"][1] - expected_nodes[1]) < 1e-3
    print(f"  L=8  loop:  maxC={r['profile']['max_abs_C']}  nonzero={r['profile']['nonzero']}/24  LOOPS  OK")


def test_3d_halt_L8():
    """L=8, gamma_halt=15 -> all slices C=0 -> HALTS.

    Lab source's bott_index raises ValueError on NaN at high gamma;
    the engine wrapper catches it and records C=0.
    """
    r = oracle_3d.run(L=8, n_kz=24, gamma_halt=15.0)
    assert r["profile"]["max_abs_C"] == 0
    assert r["profile"]["nonzero"] == 0
    # nan_slices should be present and non-empty (source crashed on these)
    assert "nan_slices" in r["profile"]
    assert len(r["profile"]["nan_slices"]) > 0
    assert r["verdict"] == "HALTS (no Fermi arc)"
    print(f"  L=8  halt:  maxC=0  nonzero=0/24  nan_slices={len(r['profile']['nan_slices'])}  HALTS  OK")


def test_3d_gamma_sweep():
    """gamma_sweep L=8, n_kz=12. Low g -> nonzero; g=15 -> all 0."""
    gs = oracle_3d.gamma_sweep(L=8, n_kz=12, gammas=[0.0, 5.0, 15.0])
    assert gs["L"] == 8
    assert gs["n_kz"] == 12
    assert len(gs["results"]) == 3
    # g=0, 5 -> LOOPS
    assert gs["results"][0]["verdict"].startswith("LOOPS")
    assert gs["results"][1]["verdict"].startswith("LOOPS")
    # g=15 -> HALTS
    assert gs["results"][2]["verdict"].startswith("HALTS")
    print(f"  L=8  gamma_sweep: g=0,5 LOOPS; g=15 HALTS  OK")


def test_3d_slice_mass():
    """M(kz) = m0 - tz*cos(kz). For kz=0: M = m0 - tz = 0.5 - 1.5 = -1.0."""
    s = oracle_3d.build_slice(L=6, kz=0.0)
    assert abs(s["M_kz"] - (-1.0)) < 1e-6
    s = oracle_3d.build_slice(L=6, kz=float(__import__('numpy').pi))
    assert abs(s["M_kz"] - 2.0) < 1e-6  # M(pi) = 0.5 - 1.5*(-1) = 2.0
    print(f"  L=6  M(kz) = m0 - tz*cos(kz): kz=0->-1, kz=pi->+2  OK")


def test_3d_weyl_node_count():
    """For |m0/tz| < 1, there are exactly 2 Weyl nodes in [0, 2*pi)."""
    p = oracle_3d.c1_profile(L=8, n_kz=24, gamma_halt=0.0, m0=0.5, tz=1.5)
    assert len(p["weyl_nodes"]) == 2
    # For m0/tz > 1, no Weyl nodes (M never zero)
    p2 = oracle_3d.c1_profile(L=8, n_kz=12, gamma_halt=0.0, m0=5.0, tz=1.0)
    assert len(p2["weyl_nodes"]) == 0
    print(f"  Weyl node count: 2 when |m0/tz|<1, 0 when m0>tz  OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TESTS: 1D engine (Phase 1A)")
    print("=" * 60)
    test_1d_all_machines()
    test_1d_halt_sink_strength()
    test_1d_gamma_zero_loop()
    test_1d_dim_matches_machine()
    print()
    print("=" * 60)
    print("  SMOKE TESTS: 2D engine (Phase 1B)")
    print("=" * 60)
    test_2d_loop_L8()
    test_2d_halt_L8()
    test_2d_sink_strength_sweep()
    test_2d_halt_site_imag()
    test_2d_dim()
    print()
    print("=" * 60)
    print("  SMOKE TESTS: 3D engine (Phase 1C)")
    print("=" * 60)
    test_3d_loop_L8()
    test_3d_halt_L8()
    test_3d_gamma_sweep()
    test_3d_slice_mass()
    test_3d_weyl_node_count()
    print("=" * 60)
    print("  ALL PASSED")
