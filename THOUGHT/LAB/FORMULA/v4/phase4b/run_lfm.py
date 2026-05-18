"""Phase 4b on LFM 2.5: CONTROL + EPISTEMIC_LATTICE + facts cassette."""
import sys, os, time, json, math
os.environ["PYTHONIOENCODING"] = "utf-8"

PHASE4B = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b"
AUTO_FB = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback"
GGUF = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm"
for p in [PHASE4B, AUTO_FB, GGUF]:
    if p not in sys.path: sys.path.insert(0, p)

from lfm_adapter import get_lfm_backend
from phase4b_prompts_v2 import TEST_PROMPTS, verify_answer
from phase4b_lattice import NodeResult, Verdict, compute_consensus, THRESHOLD_DEFAULT
from phase4b_fragments import FactualFragment
from cortex_commonsense import CortexCommonsense
from facts_cassette import FactsCassette

FACTUAL = [p for p in TEST_PROMPTS if p.get("verification_type", "none") != "none"]

def short_prompt(entry):
    """Add 'Answer in one sentence.' to reduce multiple-choice formatting."""
    p = entry["prompt"]
    if "?" in p and "Answer" not in p:
        return p.rstrip(".") + ". Answer in one sentence."
    return p

print("Phase 4b on LFM 2.5 1.2B (CUDA)")
print("=" * 60)

model = get_lfm_backend(temperature=0.7)
fc = FactsCassette()
factual_frag = FactualFragment()
COMMONSENSE = CortexCommonsense()  # Regex extraction + cassette verification

# ---- CONTROL ----
print("\nCONTROL (no lattice, no cassette):")
control_correct = 0
for i, entry in enumerate(FACTUAL):
    prompt = short_prompt(entry)
    text, _ = model.generate(prompt, [])
    verified, _ = verify_answer(text, entry)
    control_correct += int(verified or False)
    flag = "OK" if verified else "XX"
    if i < 5 or not verified:
        print("  {} {} gt={} out={}".format(flag, entry["id"], entry.get("ground_truth", "")[:20], text[:60]))

ctrl_acc = control_correct / len(FACTUAL)
print("CONTROL: {:.1%} ({}/{})".format(ctrl_acc, control_correct, len(FACTUAL)))

# ---- EPISTEMIC_LATTICE + cassette ----
print("\nEPISTEMIC_LATTICE + CORTEX-COMMONSENSE:")
epi_correct = 0
hard_gates = 0
recovered = 0

for i, entry in enumerate(FACTUAL):
    prompt = short_prompt(entry)
    gt = entry.get("ground_truth", "")
    pid = entry["id"]

    # Generate
    text, _ = model.generate(prompt, [])
    verified, score = verify_answer(text, entry)

    # Lattice nodes
    n1 = NodeResult(node_id=1, node_name="Factual",
        verdict=Verdict.PASS if verified else Verdict.FAIL,
        score=score or 0.5, evidence="", raw_output=text)
    try:
        cs_r = COMMONSENSE.verify(text)
        v2 = Verdict.PASS if cs_r.verdict == "pass" else Verdict.FAIL
    except:
        v2 = Verdict.ABSTAIN
    n2 = NodeResult(node_id=2, node_name="COMMONSENSE",
        verdict=v2, score=0.5, evidence="", raw_output=text)

    consensus = compute_consensus([n1, n2], THRESHOLD_DEFAULT)
    hg = 0; rec = False

    if not consensus.consensus_holds:
        hard_gates += 1
        fact = fc.correct(prompt)
        docs = fc.retrieve_docs(prompt, top_k=1)

        correction = "VERIFICATION FAILED. "
        if docs: correction += "Context: " + docs[0][:200] + " "
        if fact: correction += "Correct answer: " + fact

        messages = [
            {"role": "system", "content": correction[:400]},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text[:150]},
        ]
        regen_text, _ = model.generate(prompt, messages)
        regen_verified, _ = verify_answer(regen_text, entry)
        rec = regen_verified or False
        if regen_verified:
            recovered += 1
            text = regen_text
            verified = True

    if verified: epi_correct += 1
    flag = "OK" if verified else ("REC" if rec else "XX")
    hg_str = " HG+rec" if hg and rec else (" HG" if hg else "")
    if not verified or hg:
        print("  {} {} {} gt={} out={}".format(flag, pid, hg_str, gt[:20], text[:80]))

epi_acc = epi_correct / len(FACTUAL)
print()
print("=" * 60)
print("CONTROL:              {:.1%} ({}/{})".format(ctrl_acc, control_correct, len(FACTUAL)))
print("EPISTEMIC + cassette: {:.1%} ({}/{})".format(epi_acc, epi_correct, len(FACTUAL)))
print("Hard gates: {}  Recovered: {}".format(hard_gates, recovered))
print("Delta: {:+.1%}".format(epi_acc - ctrl_acc))
