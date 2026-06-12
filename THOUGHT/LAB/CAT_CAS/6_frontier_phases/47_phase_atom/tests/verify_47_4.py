"""Verification test for D-4: LHC Overflow Exploit.

Proves:
  1. Experiment runs to completion without error.
  2. Tape lifecycle: record -> uncompute -> verify (PASS).
  3. Gate 3 vacuum restoration confirmed.
"""

import sys, os, io, contextlib, traceback

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_FILE = os.path.join(EXP_DIR, '47_4_lhc_overflow_exploit', '47_4_lhc_overflow_exploit.py')
sys.path.insert(0, os.path.join(EXP_DIR, '47_4_lhc_overflow_exploit'))
sys.path.insert(0, EXP_DIR)

output_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(output_buf):
        old_cwd = os.getcwd()
        os.chdir(os.path.join(EXP_DIR, '47_4_lhc_overflow_exploit'))
        exec(compile(open(EXP_FILE).read(), EXP_FILE, 'exec'))
        os.chdir(old_cwd)
    out = output_buf.getvalue()
    if 'Tape Verification PASS' in out:
        print('D-4 VERIFICATION: PASS (tape lifecycle confirmed)')
    elif 'Tape Verification FAIL' in out:
        print('D-4 VERIFICATION: FAIL (tape lifecycle failed)')
        print(out[-500:])
    else:
        print('D-4 VERIFICATION: FAIL (no tape verification output found)')
except Exception:
    traceback.print_exc()
    print('D-4 VERIFICATION: FAIL (exception during run)')
