"""Test: LFM 2.5 vs CORTEX-COMMONSENSE on AGS-specific questions."""
import sys, os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

PHASE4B = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b"
GGUF = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm"
for p in [PHASE4B, GGUF]:
    if p not in sys.path: sys.path.insert(0, p)

from lfm_adapter import get_lfm_backend
from cortex_commonsense import CortexCommonsense
from facts_cassette import FactsCassette

AGS_PROMPTS = [
    ("What does INV-005 state?", "determinism"),
    ("What is the canon version of AGS?", "3.0.0"),
    ("How many verification fragments does Phase 4b have?", "4"),
    ("What model was used in Phase 4b?", "TraDo-4B"),
    ("What is the epistemic C frame threshold at k=50?", "0.17"),
    ("How many parameters does LFM 2.5 have?", "1.2 billion"),
    ("What compression ratio does Phase 3.5 achieve at k=50?", "15"),
    ("How many cassettes are in the cassette network?", "9"),
    ("What does INV-004 state?", "fixtures"),
    ("What is the Phase 4b epistemic accuracy?", "85.7"),
]

print("AGS Knowledge Test: LFM 2.5 (no training data on AGS)")
print("=" * 60)

model = get_lfm_backend(temperature=0.7)
cs = CortexCommonsense()
fc = FactsCassette()

# CONTROL
print("\nCONTROL (no cassette):")
control_correct = 0
for prompt, gt in AGS_PROMPTS:
    text, _ = model.generate(prompt + " Answer in one sentence.", [])
    ok = gt.lower() in text.lower()
    control_correct += int(ok)
    flag = "OK" if ok else "XX"
    print("  {} {} -> {}".format(flag, prompt[:50], text[:80]))

ctrl_acc = control_correct / len(AGS_PROMPTS)
print("CONTROL: {:.0%} ({}/{})".format(ctrl_acc, control_correct, len(AGS_PROMPTS)))

# CORTEX-COMMONSENSE (with cassette)
print("\nCORTEX-COMMONSENSE (with cassette retrieval):")
cortex_correct = 0
hard_gates = 0
recovered = 0

for prompt, gt in AGS_PROMPTS:
    text, _ = model.generate(prompt + " Answer in one sentence.", [])
    ok = gt.lower() in text.lower()
    
    if not ok:
        # Hard gate: query cassette
        hard_gates += 1
        fact = fc.correct(prompt)
        if fact:
            correction = "VERIFICATION FAILED. Correct answer: " + fact
            history = [
                {"role": "system", "content": correction[:300]},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text[:100]},
            ]
            regen_text, _ = model.generate(prompt, history)
            regen_ok = gt.lower() in regen_text.lower()
            if regen_ok:
                recovered += 1
                ok = True
                text = regen_text
    
    cortex_correct += int(ok)
    flag = "OK" if ok else ("REC" if recovered else "XX")
    hg_str = " HG+rec" if hard_gates and recovered else ""
    print("  {} {} -> {}".format(flag, prompt[:50], text[:80]))

cortex_acc = cortex_correct / len(AGS_PROMPTS)
print()
print("=" * 60)
print("CONTROL:              {:.0%} ({}/{})".format(ctrl_acc, control_correct, len(AGS_PROMPTS)))
print("CORTEX-COMMONSENSE:   {:.0%} ({}/{})".format(cortex_acc, cortex_correct, len(AGS_PROMPTS)))
print("Hard gates: {}  Recovered: {}".format(hard_gates, recovered))
print("Delta: {:.0%}".format(cortex_acc - ctrl_acc))
