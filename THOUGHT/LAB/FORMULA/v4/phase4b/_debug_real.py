"""Debug script: test real model pipeline step by step."""
import sys, time
sys.path.insert(0, r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b')
sys.path.insert(0, r'D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE')

from model import get_model, RealTraDoGenerator
from phase4b_fragments import CommonsenseFragment, FactualFragment, SelfConsistencyFragment
from phase4b_prompts_v2 import CALIBRATION_PROMPTS

print("Loading model...", flush=True)
trado = get_model(quantize="q4")
gen = RealTraDoGenerator(trado)
print("Model loaded.", flush=True)

entry = CALIBRATION_PROMPTS[0]
print(f"Prompt: {entry['prompt']}", flush=True)

t0 = time.time()
text, _ = gen.generate(entry["prompt"], [])
print(f"Generated in {time.time()-t0:.1f}s: {text[:150]}", flush=True)

print("Testing COMMONSENSE fragment...", flush=True)
t0 = time.time()
cs = CommonsenseFragment()
r = cs.verify(text)
print(f"COMMONSENSE: {r.verdict} score={r.score} ({time.time()-t0:.1f}s)", flush=True)

print("Testing SelfConsistency fragment (2x generation)...", flush=True)
t0 = time.time()
sc = SelfConsistencyFragment(generate_fn=gen.generate, similarity_threshold=0.8)
r2 = sc.verify(entry["prompt"], [])
print(f"SelfConsistency: {r2.verdict} score={r2.score:.2f} ({time.time()-t0:.1f}s)", flush=True)

print("Testing Factual fragment...", flush=True)
t0 = time.time()
ff = FactualFragment()
r3 = ff.verify(text, entry)
print(f"Factual: {r3.verdict} score={r3.score} ({time.time()-t0:.1f}s)", flush=True)

print("\nAll fragments OK. Pipeline ready.", flush=True)
