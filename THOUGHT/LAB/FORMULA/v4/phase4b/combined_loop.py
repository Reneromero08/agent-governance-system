"""Combined loop: LFM 2.5 + Phase 4b lattice + CORTEX-COMMONSENSE + cassette recovery.

The full architecture in one run:
    1. LFM generates answer
    2. CORTEX-COMMONSENSE fragment verifies against cassette
    3. Factual fragment verifies against ground truth
    4. Consensus: both nodes must pass (t=1 lattice, 2 nodes)
    5. Hard gate -> cassette retrieves -> chat API corrects -> regenerate
    6. Log per-step results
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

PHASE4B = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b"
GGUF = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm"
AUTO_FB = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback"
for p in [PHASE4B, GGUF, AUTO_FB]:
    if p not in sys.path: sys.path.insert(0, p)

from lfm_adapter import get_lfm_backend
from cortex_commonsense import CortexCommonsense
from phase4b_fragments import FactualFragment, FragmentResult
from phase4b_lattice import NodeResult, Verdict, compute_consensus
from facts_cassette import FactsCassette

# Combined prompts: general + AGS
PROMPTS = [
    # General (model might know some)
    ("What is the capital of France?", "Paris", "general"),
    ("What is the chemical formula for water?", "H2O", "general"),
    ("Who wrote the novel 1984?", "George Orwell", "general"),
    ("How many bones in adult human body?", "206", "general"),
    ("What is 17 times 24?", "408", "general"),
    # AGS (model knows nothing)
    ("What does INV-005 state?", "determinism", "ags"),
    ("What is the canon version of AGS?", "3.0.0", "ags"),
    ("How many verification fragments does Phase 4b have?", "4", "ags"),
    ("What model was used in Phase 4b?", "TraDo-4B", "ags"),
    ("How many parameters does LFM 2.5 have?", "1.2 billion", "ags"),
    ("How many cassettes in the cassette network?", "9", "ags"),
    ("What does INV-004 state?", "fixtures", "ags"),
    ("What is the Phase 4b epistemic accuracy?", "85.7", "ags"),
    ("What compression ratio does Phase 3.5 achieve at k=50?", "15", "ags"),
    ("What is the epistemic C frame threshold at k=50?", "0.17", "ags"),
]

print("Combined Loop: LFM 2.5 + Phase 4b Lattice + CORTEX-COMMONSENSE + Cassette")
print("=" * 70)

model = get_lfm_backend(temperature=0.7)
cs = CortexCommonsense()
factual = FactualFragment()
fc = FactsCassette()

# ---- CONTROL ----
print("\nCONTROL (no lattice, no cassette):")
control_correct = 0
for prompt, gt, category in PROMPTS:
    text, _ = model.generate(prompt + " Answer in one sentence.", [])
    ok = gt.lower() in text.lower()
    control_correct += int(ok)
    flag = "OK" if ok else "XX"
    print("  {} [{}] {} -> {}".format(flag, category, prompt[:45], text[:80]))

ctrl_acc = control_correct / len(PROMPTS)

# ---- COMBINED LOOP ----
print("\nCOMBINED LOOP (lattice + cassette + recovery):")
combined_correct = 0
hard_gates = 0
recovered = 0
general_correct = 0
general_total = 0
ags_correct = 0
ags_total = 0

for prompt, gt, category in PROMPTS:
    t0 = time.time()

    # 1. Generate
    text, _ = model.generate(prompt + " Answer in one sentence.", [])

    # 2. CORTEX-COMMONSENSE fragment
    cs_result = cs.verify(text)

    # 3. Factual fragment (creates a mock entry for ground truth check)
    ok = gt.lower() in text.lower()
    fact_verdict = Verdict.PASS if ok else Verdict.FAIL
    fact_node = NodeResult(node_id=2, node_name="Factual",
        verdict=fact_verdict, score=1.0 if ok else 0.0,
        evidence=gt if ok else "missing: " + gt, raw_output=text)

    # 4. CORTEX-COMMONSENSE as lattice node
    cs_verdict = Verdict.PASS if cs_result.verdict == "pass" else (
        Verdict.FAIL if cs_result.verdict in ("hard_fail", "soft_fail") else Verdict.ABSTAIN)
    cs_node = NodeResult(node_id=1, node_name="CORTEX-COMMONSENSE",
        verdict=cs_verdict, score=cs_result.score,
        evidence=cs_result.evidence[:100], raw_output=text)

    # 5. Consensus
    consensus = compute_consensus([cs_node, fact_node])

    hg = 0
    rec = False

    if not consensus.consensus_holds:
        hard_gates += 1
        hg = 1

        # Retrieve from cassette
        fact = fc.correct(prompt)
        correction = "VERIFICATION FAILED. "
        if fact:
            correction += "Correct answer: " + fact

        # Regenerate via chat API
        history = [
            {"role": "system", "content": correction[:300]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text[:150]},
        ]
        regen_text, _ = model.generate(prompt, history)
        regen_ok = gt.lower() in regen_text.lower()
        if regen_ok:
            recovered += 1
            rec = True
            text = regen_text
            ok = True

    if ok:
        combined_correct += 1

    if category == "general":
        general_total += 1
        general_correct += int(ok)
    else:
        ags_total += 1
        ags_correct += int(ok)

    dt = time.time() - t0
    flag = "OK" if ok else ("REC" if rec else "XX")
    hg_str = " HG+rec" if hg and rec else (" HG" if hg else "")
    print("  {} [{}] {} dt={:.1f}s{} cs={}".format(
        flag, category, prompt[:40], dt, hg_str, cs_result.verdict[:8]), flush=True)

combined_acc = combined_correct / len(PROMPTS)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print("CONTROL:        {:.0%} ({}/{})".format(ctrl_acc, control_correct, len(PROMPTS)))
print("COMBINED LOOP:  {:.0%} ({}/{})".format(combined_acc, combined_correct, len(PROMPTS)))
print("  General:      {:.0%} ({}/{})".format(general_correct/max(general_total,1), general_correct, general_total))
print("  AGS:          {:.0%} ({}/{})".format(ags_correct/max(ags_total,1), ags_correct, ags_total))
print("Hard gates: {}  Recovered: {}  Recovery rate: {:.0%}".format(
    hard_gates, recovered, recovered/max(hard_gates,1)))
print("Delta: {:+.0%}".format(combined_acc - ctrl_acc))

# Save results
results = {
    "phase": "combined-loop",
    "model": "LFM-2.5-1.2B-Instruct-Q8_0",
    "control_accuracy": round(ctrl_acc, 3),
    "combined_accuracy": round(combined_acc, 3),
    "general_accuracy": round(general_correct/max(general_total,1), 3),
    "ags_accuracy": round(ags_correct/max(ags_total,1), 3),
    "hard_gates": hard_gates,
    "recovered": recovered,
    "recovery_rate": round(recovered/max(hard_gates,1), 3),
    "n_prompts": len(PROMPTS),
}
out_path = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b\results\combined_loop.json"
json.dump(results, open(out_path, "w"), indent=2)
print("\nSaved to {}".format(out_path))
