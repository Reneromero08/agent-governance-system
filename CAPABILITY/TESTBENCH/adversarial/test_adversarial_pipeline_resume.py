import json
import shutil
import sys

from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[3])
if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(1, str(Path(__file__).resolve().parents[3]))

