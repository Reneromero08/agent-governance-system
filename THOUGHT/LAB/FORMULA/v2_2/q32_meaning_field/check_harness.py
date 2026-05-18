#!/usr/bin/env python3
"""Q32 Gap 1: Climate-FEVER full-mode verification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../v1/questions/critical_q32_1670/tests"))

# Just check if the file exists and can be parsed
harness_path = os.path.join(os.path.dirname(__file__), "../../v1/questions/critical_q32_1670/tests/q32_public_benchmarks.py")
print(f"Harness path: {harness_path}")
print(f"Exists: {os.path.exists(harness_path)}")
with open(harness_path) as f:
    lines = f.readlines()
print(f"Lines: {len(lines)}")

# Check available imports
try:
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers: OK")
except:
    print("sentence_transformers: MISSING")

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("transformers NLI: OK")
except:
    print("transformers NLI: MISSING")

# Check cached models
import os as _os
cache = _os.path.expanduser("~/.cache/huggingface/hub")
nli_models = [d for d in _os.listdir(cache) if "nli" in d.lower() or "cross" in d.lower()] if _os.path.exists(cache) else []
print(f"NLI models cached: {nli_models}")
