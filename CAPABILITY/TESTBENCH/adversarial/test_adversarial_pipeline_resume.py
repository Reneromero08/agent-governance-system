import json
import shutil
import sys

from pathlib import Path

import pytest


# Change the REPO_ROOT calculation to use parents[3] instead of 2.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Update sys.path to include the repo root relative to this new location.

sys.path.insert(0, str(REPO_ROOT))
