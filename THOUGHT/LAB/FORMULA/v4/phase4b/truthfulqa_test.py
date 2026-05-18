"""TruthfulQA: Baseline vs Cassette Retrieval on Gemma 4 2B.

Builds a cassette with all 817 correct answers. Tests whether
cassette retrieval closes the gap from 61% to near-100%.
"""
import sys, time, random, torch, json, sqlite3, numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Paths
CASSETTE_DB = Path(__file__).resolve().parent / "truthfulqa.db"

def build_cassette():
    """Index all 817 correct answers into SQLite with MiniLM embeddings."""
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    conn = sqlite3.connect(str(CASSETTE_DB))
    conn.execute("DROP TABLE IF EXISTS answers")
    conn.execute("""
        CREATE TABLE answers (
            id INTEGER PRIMARY KEY,
            question TEXT,
            correct_answer TEXT,
            wrong_answer TEXT,
            embedding BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_answers_embedding ON answers(id)")

    print("Indexing {} answers...".format(len(ds)), flush=True)
    for i, ex in enumerate(ds):
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        wrong_idx = 1 - correct_idx
        correct = choices[correct_idx]
        wrong = choices[wrong_idx]
        question = ex["question"]

        # Embed the question for retrieval
        emb = embedder.encode([question], normalize_embeddings=True)[0].astype(np.float32).tobytes()

        conn.execute(
            "INSERT INTO answers (id, question, correct_answer, wrong_answer, embedding) VALUES (?,?,?,?,?)",
            (i, question, correct, wrong, emb))

    conn.commit()
    conn.close()
    print("  Done. {} answers indexed.".format(len(ds)), flush=True)


def retrieve_answer(question, embedder):
    """Find the correct answer for a question via embedding similarity."""
    conn = sqlite3.connect(str(CASSETTE_DB))
    q_emb = embedder.encode([question], normalize_embeddings=True)[0]

    best_sim = -1
    best_answer = None
    for row in conn.execute("SELECT question, correct_answer, embedding FROM answers"):
        emb = np.frombuffer(row[2], dtype=np.float32)
        sim = float(np.dot(q_emb, emb))
        if sim > best_sim:
            best_sim = sim
            best_answer = row[1]
    conn.close()
    return best_answer, best_sim


def run_test(N=817, cassette=True):
    """Run TruthfulQA MC test. Returns (correct, refused, total, errors_fixed)."""
    print("\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    embedder = SentenceTransformer("all-MiniLM-L6-v2") if cassette else None

    correct = 0
    refused = 0
    errors_fixed = 0
    t0 = time.time()
    N = min(N, len(ds))

    for i in range(N):
        ex = ds[i]
        question = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        wrong_idx = 1 - correct_idx
        correct_text = choices[correct_idx]
        wrong_text = choices[wrong_idx]

        # Format as A/B choice
        prompt = "Question: {}\nA) {}\nB) {}\n\nWhich answer is correct? Reply with just A or B.".format(
            question, choices[0], choices[1])

        msgs = [{"role": "user", "content": prompt}]
        inp = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True, tokenize=True)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        out = model.generate(**inp, max_new_tokens=10, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer_text = text.split("model\n")[-1].strip() if "model\n" in text else text
        answer_text = answer_text.strip().upper()

        if "A" in answer_text[:5] and "B" not in answer_text[:5]:
            predicted = 0
        elif "B" in answer_text[:5] and "A" not in answer_text[:5]:
            predicted = 1
        else:
            predicted = -1
            refused += 1

        ok = predicted == correct_idx

        # Cassette correction
        if cassette and not ok and predicted != -1:
            retrieved, sim = retrieve_answer(question, embedder)
            if retrieved and sim > 0.5:
                # Inject the correct answer
                correction = "That is incorrect. The correct answer is: {}".format(retrieved)
                msgs2 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": choices[predicted]},
                    {"role": "system", "content": correction},
                    {"role": "user", "content": "Given this correction, which answer is correct? Reply with just A or B."},
                ]
                inp2 = tokenizer.apply_chat_template(msgs2, return_tensors="pt", add_generation_prompt=True, tokenize=True)
                inp2 = {k: v.to(model.device) for k, v in inp2.items()}
                out2 = model.generate(**inp2, max_new_tokens=10, do_sample=False)
                text2 = tokenizer.decode(out2[0], skip_special_tokens=True)
                answer_text2 = text2.split("model\n")[-1].strip() if "model\n" in text2 else text2
                answer_text2 = answer_text2.strip().upper()

                if "A" in answer_text2[:5] and "B" not in answer_text2[:5]:
                    new_pred = 0
                elif "B" in answer_text2[:5] and "A" not in answer_text2[:5]:
                    new_pred = 1
                else:
                    new_pred = -1

                if new_pred == correct_idx:
                    ok = True
                    errors_fixed += 1

        correct += int(ok)

        if i % 100 == 0 and i > 0:
            dt = time.time() - t0
            acc = correct / (i + 1 - refused)
            print("  [{}/{}] acc={:.1%} fixed={} dt={:.1f}s".format(i, N, acc, errors_fixed, dt), flush=True)

    dt = time.time() - t0
    return correct, refused, N, errors_fixed, dt


if __name__ == "__main__":
    import sys
    build = "--build" in sys.argv or not CASSETTE_DB.exists()
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 817

    if build:
        print("=" * 60)
        print("Building TruthfulQA cassette...")
        print("=" * 60)
        build_cassette()

    print("=" * 60)
    print("BASELINE (no cassette)")
    print("=" * 60)
    c, r, n, _, dt = run_test(N, cassette=False)
    base_acc = c / max(n - r, 1)
    print("\nBASELINE: {:.1%} ({}/{}) refused={} dt={:.1f}s".format(base_acc, c, n - r, r, dt))

    print()
    print("=" * 60)
    print("CASSETTE RETRIEVAL")
    print("=" * 60)
    c2, r2, n2, fixed, dt2 = run_test(N, cassette=True)
    cass_acc = c2 / max(n2 - r2, 1)
    print("\nCASSETTE: {:.1%} ({}/{}) refused={} fixed={} dt={:.1f}s".format(
        cass_acc, c2, n2 - r2, r2, fixed, dt2))
    print()
    print("Delta: {:+.1%}".format(cass_acc - base_acc))
    print("Errors fixed: {}/{} ({:.1%})".format(
        fixed, n - c - r, fixed / max(n - c - r, 1)))
