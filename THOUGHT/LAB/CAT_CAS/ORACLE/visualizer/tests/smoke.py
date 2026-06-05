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

from engine import oracle_1d, oracle_2d, oracle_3d, oracle_4d, oracle_5d


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


# ---- 4D (Phase 1D) -----------------------------------------------------

def test_4d_dim():
    """N = 4 * L * L (4-component spinor on LxL spatial lattice)."""
    for L in (4, 6):
        r = oracle_4d.run(L=L, n_k=4, gamma_halt=0.0)
        assert r["N"] == 4 * L * L, f"L={L}: N={r['N']}, expected {4*L*L}"
    print(f"  N = 4*L*L for L in 4,6  OK")


def test_4d_c2_quantized():
    """L=6, n_k=4, gamma=0 -> C2 is a nonzero integer (LOOPS).

    This is the source's scale-up: 4D Dirac monopoles are robust at L>=6.
    """
    r = oracle_4d.run(L=6, n_k=4, gamma_halt=0.0)
    assert r["L"] == 6
    assert r["N"] == 144
    assert r["n_k"] == 4
    assert r["grid"]["C2"] != 0, f"expected nonzero C2, got {r['grid']['C2']}"
    assert r["verdict"] == "LOOPS (4D Dirac monopoles protected)"
    print(f"  L=6  n_k=4 loop:  C2={r['grid']['C2']:+d}  nonzero={r['grid']['nonzero']}/{r['grid']['total']}  LOOPS  OK")


def test_4d_c1_grid_shape():
    """C1_grid is [n_k][n_k] integers, total = n_k*n_k."""
    g = oracle_4d.c1_grid(L=4, n_k=4, gamma_halt=0.0)
    assert len(g["C1_grid"]) == 4
    for row in g["C1_grid"]:
        assert len(row) == 4
    assert g["total"] == 16
    assert len(g["C1_profile"]) == 16
    print(f"  C1_grid shape: 4x4, C1_profile len=16  OK")


def test_4d_gamma_sweep():
    """4D gamma_sweep runs without error and produces structured output."""
    gs = oracle_4d.gamma_sweep(L=4, n_k=4, gammas=[0.0, 5.0, 15.0])
    assert len(gs["results"]) == 3
    for r in gs["results"]:
        assert "C2" in r
        assert "verdict" in r
        assert r["verdict"] in (
            "LOOPS (4D protected)",
            "HALTS (monopoles annihilated)",
        )
    print(f"  L=4  gamma_sweep: 3 gammas, all complete  OK")


def test_4d_mass_formula():
    """M(kz, kw) = m0 - tz*cos(kz) - tw*cos(kw).

    At kz=0, kw=0:  M = m0 - tz - tw = 1 - 1 - 1 = -1.
    At kz=pi, kw=pi:  M = m0 - tz*(-1) - tw*(-1) = 1 + 1 + 1 = 3.
    """
    s = oracle_4d.build_slice(L=4, kz=0.0, kw=0.0)
    assert abs(s["M_kw"] - (-1.0)) < 1e-6
    s = oracle_4d.build_slice(L=4, kz=float(__import__('numpy').pi),
                              kw=float(__import__('numpy').pi))
    assert abs(s["M_kw"] - 3.0) < 1e-6
    print(f"  M(kz,kw) = m0 - tz*cos(kz) - tw*cos(kw): corner checks OK")


def test_4d_halt_site_imag():
    """At the halt site, Im(H_site_site) = -(loss + gamma_halt)."""
    r = oracle_4d.run(L=4, n_k=2, gamma_halt=10.0)
    s = oracle_4d.build_slice(L=4, kz=0.0, kw=0.0, gamma_halt=10.0)
    H_site = s["H"][s["halt_site"]][s["halt_site"]]
    expected_im = -(0.05 + 10.0)
    assert abs(H_site["im"] - expected_im) < 1e-3
    print(f"  L=4  halt site: Im(H)={H_site['im']:.4f}  (expected {expected_im})  OK")


# ---- 5D (Phase 1E) -----------------------------------------------------

def test_5d_dim():
    """N = 4 * L * L (4-comp spinor on LxL spatial lattice)."""
    for L in (4, 6):
        r = oracle_5d.run(L=L, n_k=2, g=0.0, t1=0.0)
        assert r["N"] == 4 * L * L, f"L={L}: N={r['N']}, expected {4*L*L}"
    print(f"  N = 4*L*L for L in 4,6  OK")


def test_5d_loop_solved():
    """t1=0 (ideal Floquet), g=0 -> 32 pi-modes per slice, all 16 active.

    The SOLVED 5D protocol: G2*G1*G5 = diag(-i,+i,+i,-i) per site,
    so U_site = diag(+1,-1,-1,+1) and 2 pi-modes per site survive.
    """
    r = oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.0)
    assert r["grid"]["total"] == 512
    assert r["grid"]["active"] == 16
    assert r["verdict"] == "LOOPS (pi-modes robust)"
    print(f"  L=4  solved:  total=512  active=16/16  LOOPS  OK")


def test_5d_halt_melted():
    """t1=0, g=0.5 -> 0 pi-modes (uniform Gamma melts them)."""
    r = oracle_5d.run(L=4, n_k=4, g=0.5, t1=0.0)
    assert r["grid"]["total"] == 0
    assert r["grid"]["active"] == 0
    assert r["verdict"] == "HALTS (pi-modes melted)"
    print(f"  L=4  melt:    total=0    active=0/16   HALTS  OK")


def test_5d_hopping_survival():
    """t1=0.1, g=0 -> pi-modes still survive (small hopping is OK)."""
    r = oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.1)
    assert r["grid"]["total"] == 512
    assert r["grid"]["active"] == 16
    assert r["verdict"] == "LOOPS (pi-modes robust)"
    print(f"  L=4  t1=0.1:  total=512  active=16/16  LOOPS  OK (small hopping survives)")


def test_5d_strong_hopping_melts():
    """t1=1.0, g=0 -> strong hopping also melts (no topological protection)."""
    r = oracle_5d.run(L=4, n_k=4, g=0.0, t1=1.0)
    assert r["grid"]["total"] == 0
    assert r["verdict"] == "HALTS (pi-modes melted)"
    print(f"  L=4  t1=1.0:  total=0    active=0/16   HALTS  OK (strong hopping melts)")


def test_5d_gamma_sweep():
    """gamma_sweep: low g -> LOOPS, g >= 0.5 -> HALTS."""
    gs = oracle_5d.gamma_sweep(L=4, n_k=4, gammas=[0.0, 0.3, 0.5, 1.0])
    assert len(gs["results"]) == 4
    assert gs["results"][0]["verdict"].startswith("LOOPS")
    assert gs["results"][1]["verdict"].startswith("LOOPS")
    assert gs["results"][2]["verdict"].startswith("HALTS")
    assert gs["results"][3]["verdict"].startswith("HALTS")
    print(f"  L=4  gamma_sweep: g=0,0.3 LOOPS; g=0.5,1.0 HALTS  OK")


def test_5d_n_grid_uniform():
    """At ideal t1=0, every (kz, kw) slice gives the same pi-mode count."""
    g = oracle_5d.pi_mode_grid(L=4, n_k=4, g=0.0, t1=0.0)
    for row in g["n_grid"]:
        for n in row:
            assert n == 32, f"expected 32 pi-modes per slice, got {n}"
    print(f"  L=4  n_grid uniform: every slice has 32 pi-modes  OK")


def test_5d_count_pi_modes():
    """count_pi_modes on a known U gives the right number."""
    U_dict = oracle_5d.floquet_operator(L=4, kz=0.0, kw=0.0, t1=0.0, g=0.0)
    count = oracle_5d.count_pi_modes(U_dict["U"], threshold=0.3)
    assert count["n_pi_modes"] == 32
    print(f"  count_pi_modes(t1=0): 32 pi-modes  OK")


# ---- 1F (Cross-dimension, JSON, HTTP) ---------------------------------

import json
import time
from typing import Any, Dict, List, Tuple


def test_all_engines_canonical_run():
    """Run a canonical run() for each of the 5 engines.

    Verifies the full pipeline (build H/U -> measure topology -> verdict)
    for every dimension.  Uses default parameters that should yield
    a clean LOOPS for each (matching the source's default working case).
    """
    canonicals: List[Tuple[str, Dict[str, Any]]] = [
        ("1D halt_direct", oracle_1d.run(machine="halt_direct", n_phi=200)),
        ("1D loop_2cycle", oracle_1d.run(machine="loop_2cycle", n_phi=200)),
        ("2D loop L=8",    oracle_2d.run(L=8, gamma_halt=0.0, include_projector=False)),
        ("3D loop L=8",    oracle_3d.run(L=8, n_kz=24, gamma_halt=0.0)),
        ("4D loop L=6",    oracle_4d.run(L=6, n_k=4, gamma_halt=0.0)),
        ("5D solved",      oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.0)),
    ]
    for label, r in canonicals:
        assert "verdict" in r, f"{label}: no verdict in output"
        assert r["verdict"] in (
            "LOOPS", "HALTS",
            "LOOPS (chiral edge protected)", "HALTS (edge destroyed)",
            "LOOPS (Fermi arc exists)", "HALTS (no Fermi arc)",
            "LOOPS (4D Dirac monopoles protected)",
            "LOOPS (pi-modes robust)",
        ), f"{label}: unexpected verdict {r['verdict']!r}"
    print(f"  6 canonical runs (1D halt+loop, 2D, 3D, 4D, 5D) all return valid verdicts  OK")


def test_json_serializable():
    """Every engine output is JSON-serializable (frontend requirement)."""
    cases: List[Tuple[str, Any]] = [
        ("1D halt_direct",   oracle_1d.run(machine="halt_direct", n_phi=80)),
        ("1D loop_2cycle",   oracle_1d.run(machine="loop_2cycle", n_phi=80)),
        ("2D L=8 loop",      oracle_2d.run(L=8, gamma_halt=0.0, include_projector=True)),
        ("2D L=8 halt",      oracle_2d.run(L=8, gamma_halt=10.0, include_projector=True)),
        ("3D L=8 loop",      oracle_3d.run(L=8, n_kz=12, gamma_halt=0.0)),
        ("4D L=6 loop",      oracle_4d.run(L=6, n_k=4, gamma_halt=0.0)),
        ("5D L=4 solved",    oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.0)),
    ]
    for label, r in cases:
        try:
            s = json.dumps(r)
        except (TypeError, ValueError) as e:
            raise AssertionError(f"{label}: not JSON-serializable: {e}")
        # Also parse it back to confirm round-trip
        r2 = json.loads(s)
        assert r2["verdict"] == r["verdict"]
    print(f"  All 7 engine outputs are JSON-serializable (round-trip verified)  OK")


def test_against_lab_source_outputs():
    """Confirm the documented lab source outputs.

    Each value is the exact number the lab source's `run_*_oracle` or
    `gamma_annihilation_sweep` would print at the default settings.
    """
    # 1D: 36_nonhermitian_oracle.py:335-339
    r = oracle_1d.run(machine="halt_direct")
    assert r["winding"]["Wint"] == 0
    r = oracle_1d.run(machine="loop_2cycle")
    assert r["winding"]["Wint"] == 1
    # 2D: 37_2d_chern_oracle.py run_2d_oracle(L=8)
    r = oracle_2d.run(L=8, gamma_halt=0.0, include_projector=False)
    assert r["bott"]["C"] == 1
    r = oracle_2d.run(L=8, gamma_halt=10.0, include_projector=False)
    assert r["bott"]["C"] == 0
    # 3D: 38_3d_weyl_oracle.py run_3d_weyl_oracle default (L=8, n_kz=24)
    r = oracle_3d.run(L=8, n_kz=24, gamma_halt=0.0)
    assert r["profile"]["max_abs_C"] == 2
    assert r["profile"]["nonzero"] == 14
    # 4D: 39_4d_axion_oracle.py scale_up (L=6, n_k=6)
    r = oracle_4d.run(L=6, n_k=6, gamma_halt=0.0)
    assert r["grid"]["C2"] != 0  # C2 quantized nonzero at L=6
    # 5D: 40_5d_floquet_oracle.py SOLVED (t1=0, g=0)
    r = oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.0)
    assert r["grid"]["total"] == 512
    assert r["grid"]["active"] == 16
    print(f"  All 5 dimensions match documented lab source outputs  OK")


def test_http_endpoints():
    """Hit all 5 dimension endpoints via FastAPI TestClient (no server needed)."""
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, os.path.join(VISUALIZER_DIR))
    # Import server module (not __main__).
    import importlib
    if "server" in sys.modules:
        del sys.modules["server"]
    import server as _server
    client = TestClient(_server.app)

    # /api/health
    h = client.get("/api/health").json()
    assert h["status"] == "ok"

    # 1D
    r = client.get("/api/dim1/run", params={"machine": "halt_direct"}).json()
    assert r["verdict"] == "HALTS"
    r = client.get("/api/dim1/run", params={"machine": "loop_2cycle"}).json()
    assert r["verdict"] == "LOOPS"

    # 2D
    r = client.get("/api/dim2/run", params={"L": 8, "gamma_halt": 0.0,
                                            "include_projector": False}).json()
    assert r["bott"]["C"] == 1

    # 3D
    r = client.get("/api/dim3/run", params={"L": 8, "n_kz": 12,
                                            "gamma_halt": 0.0}).json()
    assert r["verdict"].startswith("LOOPS")

    # 4D
    r = client.get("/api/dim4/run", params={"L": 6, "n_k": 4,
                                            "gamma_halt": 0.0}).json()
    assert r["grid"]["C2"] != 0

    # 5D
    r = client.get("/api/dim5/run", params={"L": 4, "n_k": 4, "t1": 0.0,
                                            "g": 0.0}).json()
    assert r["grid"]["total"] == 512

    print(f"  HTTP: /api/health + 5 dimension endpoints all return expected data  OK")


def test_summary_table():
    """Print a unified summary table of all 5 dimensions (loop case)."""
    print()
    print("  " + "=" * 58)
    print("  CANONICAL LOOP CASE PER DIMENSION")
    print("  " + "=" * 58)
    print(f"  {'dim':<6} {'verdict':<35} {'topology metric':<20}")
    print("  " + "-" * 58)

    r = oracle_1d.run(machine="loop_2cycle", n_phi=200)
    print(f"  {'1D':<6} {r['verdict']:<35} W={r['winding']['Wint']:+d} (point gap)")

    r = oracle_2d.run(L=8, gamma_halt=0.0, include_projector=False)
    print(f"  {'2D':<6} {r['verdict']:<35} C={r['bott']['C']:+d} (Bott)")

    r = oracle_3d.run(L=8, n_kz=12, gamma_halt=0.0)
    print(f"  {'3D':<6} {r['verdict']:<35} max|C|={r['profile']['max_abs_C']} (Fermi arc)")

    r = oracle_4d.run(L=6, n_k=4, gamma_halt=0.0)
    print(f"  {'4D':<6} {r['verdict']:<35} C2={r['grid']['C2']:+d} (axion)")

    r = oracle_5d.run(L=4, n_k=4, g=0.0, t1=0.0)
    print(f"  {'5D':<6} {r['verdict']:<35} pi-modes={r['grid']['total']} (Floquet)")
    print("  " + "=" * 58)


# ---- Main entry point ---------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

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
    print()
    print("=" * 60)
    print("  SMOKE TESTS: 4D engine (Phase 1D)")
    print("=" * 60)
    test_4d_dim()
    test_4d_c2_quantized()
    test_4d_c1_grid_shape()
    test_4d_gamma_sweep()
    test_4d_mass_formula()
    test_4d_halt_site_imag()
    print()
    print("=" * 60)
    print("  SMOKE TESTS: 5D engine (Phase 1E)")
    print("=" * 60)
    test_5d_dim()
    test_5d_loop_solved()
    test_5d_halt_melted()
    test_5d_hopping_survival()
    test_5d_strong_hopping_melts()
    test_5d_gamma_sweep()
    test_5d_n_grid_uniform()
    test_5d_count_pi_modes()
    print()
    print("=" * 60)
    print("  SMOKE TESTS: 1F (cross-dimension)")
    print("=" * 60)
    test_all_engines_canonical_run()
    test_json_serializable()
    test_against_lab_source_outputs()
    test_http_endpoints()
    test_summary_table()

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"  ALL PASSED  ({elapsed:.2f}s)")
