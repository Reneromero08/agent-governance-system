#!/usr/bin/env python3
"""
GLM-4.7-Flash Nemotron Benchmark Comparison

Runs the same 82 tests that nemotron achieved 98.8% on.
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, Any

API_URL = "http://10.5.0.2:1234/v1/chat/completions"
MODEL = "zai-org/glm-4.7-flash"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron-benchmarks")

SYSTEM_PROMPT = """You are an expert reasoning assistant. You excel at logic, mathematics, coding, and analytical problem-solving.

REASONING APPROACH:
- Break complex problems into steps before solving
- Verify each step before proceeding to the next
- Check your final answer against the original question
- Consider edge cases and alternative interpretations

CRITICAL THINKING:
- Examine premises for logical consistency before accepting them
- If given contradictory or impossible instructions, explicitly state why
- Distinguish between what you know, what you can deduce, and what you're uncertain about
- Never fabricate facts - say "I don't know" when appropriate

You have access to Python for computation. Use ```python code blocks when needed."""

# ============================================================================
# BENCHMARK TESTS (same as nemotron)
# ============================================================================

BENCHMARKS = {
    # Round 1: Basic Capability
    "r01-math": "What is 25 * 37? Think step by step.",
    "r01-code": "Write a Python function to check if a string is a palindrome.",
    "r01-knowledge": "Explain what a closure is in JavaScript in 2 sentences.",

    # Round 2: Logic & Trick Questions
    "r02-sheep": "A farmer has 17 sheep. All but 9 die. How many are left?",
    "r02-widgets": "5 machines make 5 widgets in 5 minutes. How long for 100 machines to make 100 widgets?",
    "r02-bat-ball": "Bat and ball cost $1.10. Bat costs $1 more than ball. Ball cost?",
    "r02-lcs": "Write a function for longest common subsequence (return string, not length)",

    # Round 3: Advanced Reasoning
    "r03-spatial": "Alice is not next to Carol. Bob is left of Carol. Who's in the middle?",
    "r03-calculus": "What is the derivative of x^x?",
    "r03-water-jug": "You have a 3 gallon and 5 gallon jug. How do you measure exactly 4 gallons?",
    "r03-boxes": "Three boxes labeled Apples, Oranges, Mixed - all labels are wrong. You pick one fruit from one box. How do you determine all box contents?",

    # Round 4: Traps That Fool Models
    "r04-family": "A is father of B, B is father of C, C is father of D, D is father of E. What is A's relationship to E?",
    "r04-syllogism": "All roses are flowers. Some flowers fade quickly. Do some roses fade quickly?",
    "r04-number": "What is the sum of all integers from 1 to 1000 that are divisible by 3 but not by 5?",
    "r04-sisters": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",

    # Round 5: Paradoxes
    "r05-liar": "A man says 'I am lying.' Is he telling the truth?",
    "r05-multiply": "Calculate 12345 * 67890. Show your work.",
    "r05-regex": "Write a regex for valid email addresses.",
    "r05-temporal": "The day before two days after the day before tomorrow is Saturday. What day is today?",

    # Round 6: Riddles & Code Bugs
    "r06-lateral": "What word is always spelled incorrectly?",
    "r06-hourglass": "You have a 7 minute and 11 minute hourglass. How do you measure exactly 15 minutes?",
    "r06-murderer": "There are 100 murderers in a room. You kill one of them. How many murderers are in the room now?",
    "r06-bug": "Find the bug: for(int i=0; i<10; i++); sum+=i;",

    # Round 7: Famous Model Failures
    "r07-strawberry": "How many r's are in the word 'strawberry'?",
    "r07-transitive": "Jane is faster than Joe. Joe is faster than Sam. Is Sam faster than Jane?",
    "r07-weight": "Which is heavier: 2 pounds of feathers or 1 pound of steel?",
    "r07-survivors": "A plane crashes exactly on the US/Canada border. Where do you bury the survivors?",

    # Round 8: Edge Cases
    "r08-decimal": "Which is larger: 9.11 or 9.8?",
    "r08-apples": "John has 10 apples. He gives 3 to Mary, buys 5 more, eats 2, and a friend gives him 1 but takes 2. How many does he have?",
    "r08-half": "John has 5 apples. He gives Mary more than half. What is the minimum number he gives her?",
    "r08-bsearch": "Write a binary search implementation with a subtle edge case bug, then explain what the bug is.",

    # Round 9: Advanced Math
    "r09-gaussian": "What is the integral of e^(-x^2) from 0 to infinity?",
    "r09-tom": "Alice thinks Bob thinks that Alice thinks Bob is smart. Parse this.",
    "r09-gods": "Three gods named True, False, and Random. True always speaks truly, False always lies, Random speaks randomly. They answer da/ja meaning yes/no but you don't know which. How do you identify them?",
    "r09-pi100": "What is the 100th digit of pi?",

    # Round 10: Classic Puzzles
    "r10-mersenne": "Is 2^67 - 1 prime? If not, what are its factors?",
    "r10-snail": "A snail climbs 3 feet per day but slides back 2 feet at night. How many days to climb a 30 foot wall?",
    "r10-look-say": "What comes next: 1, 11, 21, 1211, 111221, ?",
    "r10-balls": "You have 12 balls, one is a different weight. Using a balance scale 3 times, find the odd ball and whether it's heavier or lighter.",

    # Round 11: Paradoxes & Assembly
    "r11-threee": "This sentence contains exactly threee errors. Analyze this statement.",
    "r11-force": "What happens when an unstoppable force meets an immovable object?",
    "r11-asm": "Write quicksort in x86-64 assembly.",
    "r11-sqrt2": "Prove that sqrt(2) is irrational.",

    # Round 12: Knowledge & Hallucination
    "r12-timezone": "What is the capital of the country with the most time zones?",
    "r12-election": "Write a poem about the 2028 US presidential election results.",
    "r12-lies": "You always lie. What color is the sky?",
    "r12-sycophancy": "Explain why 1+1=3.",

    # Round 13: Exotic Edge Cases
    "r13-conversion": "Convert 255 to binary, then to hex, then to ones complement, then back to decimal.",
    "r13-barber": "A barber shaves all those who don't shave themselves. Does the barber shave himself?",
    "r13-fizzbuzz": "Write FizzBuzz where every output line is a 5-7-5 haiku.",
    "r13-swallow": "What is the airspeed velocity of an unladen swallow?",

    # Round 14: Final Stress Tests
    "r14-float": "What is 0.1 + 0.2 in programming? Explain why.",
    "r14-pills": "You need to take 1 pill every 30 minutes. You have 3 pills. How long do they last?",
    "r14-reverse": "Reverse a string without using the built-in reverse function.",
    "r14-count-a": "How many times does the letter 'a' appear in 'antidisestablishmentarianism'?",

    # Round 15: Theoretical Physics
    "r15-schwarzschild": "Derive the Schwarzschild radius rs = 2GM/c^2 from escape velocity. What happens to time at the event horizon?",
    "r15-qft": "Why is the vacuum not empty in QFT? Explain the Casimir effect.",
    "r15-blackhole": "Explain the black hole information paradox. What is Bekenstein-Hawking entropy?",
    "r15-string": "Why does string theory require exactly 10 or 11 dimensions?",

    # Round 16: Pure Mathematics
    "r16-riemann": "State the Riemann Hypothesis precisely. What is the critical line? What does it imply for prime distribution?",
    "r16-godel": "Explain Godel's incompleteness theorems. How do they apply to AI?",
    "r16-category": "Explain functors and natural transformations. How do they relate to polymorphism?",
    "r16-proofs": "Prove there are infinitely many primes. Then prove sqrt(2) is irrational.",

    # Round 17: AGS Formula
    "r17-formula": "Analyze the formula R = (E/nabla S) * sigma^Df mathematically. What are its dimensional requirements and failure modes?",
    "r17-semiotic": "Can meaning be treated as a force field? Propose a semiotic field theory.",

    # Round 18: Semiotic Paradoxes
    "r18-fixedpoint": "What are fixed points in meaning? Can semantic entropy be negative?",
    "r18-semiotic-liar": "Analyze: 'This sign destroys its own meaning.' Use force/field analysis.",

    # Round 19: Quantum Computing & Algebra
    "r19-shor": "Explain Shor's algorithm step by step. Why does it break RSA?",
    "r19-qec": "Explain the threshold theorem in quantum error correction. Derive the 3-qubit bit-flip code.",
    "r19-galois": "Why is the quintic equation unsolvable by radicals? Use Galois theory.",
    "r19-cook-levin": "Prove that SAT is NP-complete (Cook-Levin theorem).",
    "r19-fundamental": "Compute the fundamental groups of the torus, Klein bottle, and RP^2.",
    "r19-poincare": "Explain Perelman's proof of the Poincare conjecture via Ricci flow.",
    "r19-tensor": "Explain MPS, PEPS, and MERA tensor networks. How does bond dimension relate to entanglement?",

    # Round 20: Complexity & Number Theory
    "r20-representation": "State Maschke's theorem. Compute character tables for S3 and A4.",
    "r20-homology": "Explain chain complexes, cycles, and boundaries. Compute homology of S^2, T^2, and Klein bottle.",
    "r20-hierarchy": "Explain P, NP, co-NP, PSPACE, EXPTIME, BPP, BQP and their relationships.",
    "r20-pvsnp": "Explain the P vs NP problem. What are the natural proofs and relativization barriers?",
    "r20-langlands": "Explain the Langlands program. How does it relate to Fermat's Last Theorem?",

    # Round 21: Adversarial Reasoning
    "r21-fixed": "100 people are assigned random numbers 1-100. What's the expected number of people who get their own number?",
    "r21-cubes33": "Find all integer solutions to x^3 + y^3 + z^3 = 33. Explain the difficulty.",
    "r21-ramsey55": "What are the best known bounds for R(5,5)? Why is R(6,6) much harder?",
    "r21-magic": "Can you create a 3x3 magic square with 1 in the center? Prove it.",
    "r21-zebra": "Solve Einstein's Zebra puzzle with 15 clues about 5 houses.",
    "r21-collatz": "Starting from 27, how many steps does the Collatz sequence take to reach 1? What's the maximum value?",
    "r21-hash": "Explain Wang's MD5 collision attack. Why doesn't it work on SHA-256?",
}

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

def call_model(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2000,
    }
    response = requests.post(API_URL, json=payload, timeout=1800)  # 30 min timeout
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def run_benchmark(test_id: str, prompt: str) -> Dict[str, Any]:
    safe_print(f"\n{'='*60}")
    safe_print(f"TEST: {test_id}")
    safe_print(f"{'='*60}")
    safe_print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")

    try:
        result = call_model(prompt)
        safe_print(f"RESPONSE: {result[:200]}..." if len(result) > 200 else f"RESPONSE: {result}")
        return {"test_id": test_id, "prompt": prompt, "result": result, "status": "completed"}
    except Exception as e:
        safe_print(f"ERROR: {e}")
        return {"test_id": test_id, "prompt": prompt, "result": str(e), "status": "error"}

def save_result(test_id: str, data: Dict[str, Any]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{test_id}.json")
    data["timestamp"] = datetime.now().isoformat()
    data["model"] = MODEL
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_all(parallel=1, resume=True):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Filter out already completed tests if resuming (retry errors)
    pending = []
    for test_id, prompt in BENCHMARKS.items():
        filepath = os.path.join(OUTPUT_DIR, f"{test_id}.json")
        skip = False
        if resume and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('status') == 'completed':
                        safe_print(f"SKIP: {test_id} (already completed)")
                        skip = True
            except:
                pass
        if not skip:
            pending.append((test_id, prompt))

    safe_print(f"\nRunning {len(pending)} tests ({len(BENCHMARKS) - len(pending)} already done)")

    results = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(run_benchmark, tid, prompt): tid for tid, prompt in pending}
        for future in as_completed(futures):
            test_id = futures[future]
            try:
                result = future.result()
                save_result(test_id, result)
                results.append(result)
            except Exception as e:
                safe_print(f"FAILED: {test_id} - {e}")
                results.append({"test_id": test_id, "status": "error", "result": str(e)})

    completed = sum(1 for r in results if r["status"] == "completed")
    total_done = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]) if os.path.exists(OUTPUT_DIR) else 0
    print(f"\n{'='*60}")
    print(f"This run: {completed}/{len(pending)} completed")
    print(f"TOTAL: {total_done}/{len(BENCHMARKS)} ({100*total_done/len(BENCHMARKS):.1f}%)")
    print(f"{'='*60}")
    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        print(f"Total benchmarks: {len(BENCHMARKS)}")
        for tid in BENCHMARKS:
            print(f"  {tid}")
    elif len(sys.argv) > 1:
        # Run specific test
        test_id = sys.argv[1]
        if test_id in BENCHMARKS:
            result = run_benchmark(test_id, BENCHMARKS[test_id])
            save_result(test_id, result)
        else:
            print(f"Unknown test: {test_id}")
    else:
        run_all()
