#!/usr/bin/env python3
"""QEC test configuration - adds local directory to path for imports."""
import sys
from pathlib import Path

# Add qec directory to path so 'from core import ...' works
QEC_DIR = Path(__file__).resolve().parent
if str(QEC_DIR) not in sys.path:
    sys.path.insert(0, str(QEC_DIR))
