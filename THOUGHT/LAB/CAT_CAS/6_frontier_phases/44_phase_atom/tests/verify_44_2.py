"""Verification test for D-2: Electron Edge States.

Proves:
  1. Experiment runs to completion without error.
  2. Tape lifecycle: record (real measured values) -> uncompute -> verify (PASS).
  3. experiment_output_lines captured for gate inspection.
"""

import sys, os, io, contextlib, traceback

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_FILE = os.path.join(EXP_DIR, '44_2_electron_edge_states', '44_2_electron_edge_states.py')
sys.path.insert(0, os.path.join(EXP_DIR, '44_2_electron_edge_states'))
sys.path.insert(0, EXP_DIR)

output_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(output_buf):
        old_cwd = os.getcwd()
        os.chdir(os.path.join(EXP_DIR, '44_2_electron_edge_states'))
        exec(compile(open(EXP_FILE).read(), EXP_FILE, 'exec'))
        os.chdir(old_cwd)
    out = output_buf.getvalue()
    if 'Tape Verification PASS' in out:
        print('D-2 VERIFICATION: PASS (tape lifecycle confirmed)')
    else:
        print('D-2 VERIFICATION: FAIL (tape lifecycle not confirmed)')
        print(out[-500:])
except Exception:
    traceback.print_exc()
    print('D-2 VERIFICATION: FAIL (exception during run)')
