"""Verification test for D-6: Quark Confinement.

Proves:
  1. Experiment runs to completion without error.
  2. Tape lifecycle: record -> uncompute -> verify (PASS).
"""

import sys, os, io, contextlib, traceback

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_FILE = os.path.join(EXP_DIR, '47_6_quark_confinement', '47_6_quark_confinement.py')
sys.path.insert(0, os.path.join(EXP_DIR, '47_6_quark_confinement'))
sys.path.insert(0, EXP_DIR)

output_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(output_buf):
        old_cwd = os.getcwd()
        os.chdir(os.path.join(EXP_DIR, '47_6_quark_confinement'))
        exec(compile(open(EXP_FILE).read(), EXP_FILE, 'exec'))
        os.chdir(old_cwd)
    out = output_buf.getvalue()
    if 'Tape Verification PASS' in out:
        print('D-6 VERIFICATION: PASS (tape lifecycle confirmed)')
    else:
        print('D-6 VERIFICATION: FAIL (tape lifecycle not confirmed)')
        print(out[-500:])
except Exception:
    traceback.print_exc()
    print('D-6 VERIFICATION: FAIL (exception during run)')
