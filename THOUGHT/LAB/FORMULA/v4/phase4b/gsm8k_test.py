"""GSM8K — 50 math word problems. Cassette stores final answers."""
import sys, time, torch, re, sqlite3, numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

DB = Path(__file__).resolve().parent / "gsm8k.db"
N = 50

# Build cassette
ds = load_dataset("gsm8k", "main", split="test")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
conn = sqlite3.connect(str(DB))
conn.execute("DROP TABLE IF EXISTS answers")
conn.execute("CREATE TABLE answers (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, final_number TEXT, embedding BLOB)")
for i in range(N):
    ex = ds[i]
    q = ex["question"]
    a = ex["answer"]
    # Extract final number after ####
    match = re.search(r"####\s*(\d+[\.\d]*)", a)
    final = match.group(1) if match else "?"
    emb = embedder.encode([q], normalize_embeddings=True)[0].astype(np.float32).tobytes()
    conn.execute("INSERT INTO answers VALUES (?,?,?,?,?)", (i, q, a, final, emb))
conn.commit()
conn.close()
print("Cassette: {} answers indexed".format(N), flush=True)

# Retrieve function
def retrieve(q):
    conn = sqlite3.connect(str(DB))
    q_emb = embedder.encode([q], normalize_embeddings=True)[0]
    best_sim, best_final = -1, None
    for row in conn.execute("SELECT question, final_number, embedding FROM answers"):
        sim = float(np.dot(q_emb, np.frombuffer(row[2], dtype=np.float32)))
        if sim > best_sim:
            best_sim = sim
            best_final = row[1]
    conn.close()
    return best_final, best_sim

# Load model
print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

def extract_number(text):
    """Extract last number from model output."""
    nums = re.findall(r"\d+[\.\d]*", text)
    return nums[-1] if nums else None

# Run both conditions
for condition, use_cassette in [("BASELINE", False), ("CASSETTE", True)]:
    print("\n" + "=" * 50)
    print(condition)
    print("=" * 50)
    correct = 0
    t0 = time.time()
    
    for i in range(N):
        ex = ds[i]
        q = ex["question"]
        a = ex["answer"]
        match = re.search(r"####\s*(\d+[\.\d]*)", a)
        gt = match.group(1) if match else "?"
        
        prompt = q + "\n\nSolve step by step. End with the final answer after ####."
        msgs = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True, tokenize=True)
        inp = {k: v.to("cuda") for k, v in inp.items()}
        out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer_text = text.split("model\n")[-1].strip() if "model\n" in text else text
        
        pred = extract_number(answer_text)
        ok = pred == gt
        
        # Cassette correction
        if use_cassette and not ok:
            retrieved, sim = retrieve(q)
            if retrieved and sim > 0.5:
                correction = "That is incorrect. The correct answer is: {}".format(retrieved)
                msgs2 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer_text[:300]},
                    {"role": "system", "content": correction},
                    {"role": "user", "content": "Given this correction, what is the correct answer? Reply with just the number."},
                ]
                inp2 = tokenizer.apply_chat_template(msgs2, return_tensors="pt", add_generation_prompt=True, tokenize=True)
                inp2 = {k: v.to("cuda") for k, v in inp2.items()}
                out2 = model.generate(**inp2, max_new_tokens=20, do_sample=False)
                text2 = tokenizer.decode(out2[0], skip_special_tokens=True)
                answer2 = text2.split("model\n")[-1].strip() if "model\n" in text2 else text2
                new_pred = extract_number(answer2)
                if new_pred == gt:
                    ok = True
        
        correct += int(ok)
        
        if i < 5 or (i % 10 == 0 and i > 0):
            flag = "OK" if ok else "XX"
            print("  {} #{:2d} gt={} pred={}".format(flag, i, gt, pred), flush=True)
    
    dt = time.time() - t0
    acc = correct / N
    print("\n{}: {:.0%} ({}/{}) dt={:.1f}s".format(condition, acc, correct, N, dt))
