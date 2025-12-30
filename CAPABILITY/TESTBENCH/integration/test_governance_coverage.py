from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.ledger import Ledger

if __name__ == "__main__":
    sys.exit(pytest.fail("pytest.main not implemented"))
