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

from engine import oracle_1d


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


if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TESTS: 1D engine (Phase 1A)")
    print("=" * 60)
    test_1d_all_machines()
    test_1d_halt_sink_strength()
    test_1d_gamma_zero_loop()
    test_1d_dim_matches_machine()
    print("=" * 60)
    print("  ALL PASSED")
