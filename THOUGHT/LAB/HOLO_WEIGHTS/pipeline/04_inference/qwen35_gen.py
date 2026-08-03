"""Generate text with the .holo Qwen3.6-27B engine.

Usage: qwen35_gen.py --holo output/qwen_27b_k256.holo [--prompt "..."] [--max-new 64]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_holo_engine import Qwen35HoloEngine, load_holo, DEFAULT_MODEL

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

PROMPT = "The theory of catalytic computation is"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default="output/qwen_27b_k256.holo")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    if AutoTokenizer is None:
        print("transformers not installed; using raw token id walk instead")
    else:
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print("Loading .holo weights ...", flush=True)
    hw = load_holo(args.holo)
    print(f"  {len([k for k in hw.U if k.endswith('.U')])} compressed matrices", flush=True)
    eng = Qwen35HoloEngine(hw, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    print("Engine ready", flush=True)

    if AutoTokenizer is not None:
        ids = tok(args.prompt, return_tensors="pt")["input_ids"][0].tolist()
    else:
        ids = [1, 2704, 263, 3521, 5767, 313, 18536, 13]

    print(f"\nPROMPT: {args.prompt}\n", flush=True)
    out = eng.generate(ids, args.max_new, temperature=args.temperature, top_p=args.top_p)
    if AutoTokenizer is not None:
        print("OUTPUT:", tok.decode(out, skip_special_tokens=True))
    else:
        print("OUTPUT ids:", out)

if __name__ == "__main__":
    main()
