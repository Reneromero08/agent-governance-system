from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.ledger import Ledger

if not any(package in sys.path for package in ['ag']):
    print("The 'ag' package is missing from your system's Python PATH.")
