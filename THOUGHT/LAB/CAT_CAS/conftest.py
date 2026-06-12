import sys
from pathlib import Path

# Put the shared primitives package dir on sys.path so pytest-collected tests
# (e.g. 47_phase_atom/tests/) can `import catalytic_tape` / `from catalytic_engine ...`
# directly, without per-test ../ bootstraps.
_LIB = str(Path(__file__).parent / "_lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
