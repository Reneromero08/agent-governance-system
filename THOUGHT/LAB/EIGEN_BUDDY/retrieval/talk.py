"""Feral Talk — query the knowledge DB through Core geodesics.

Pipeline:
  query -> Core navigates vector space -> finds geodesically relevant text -> returns it.
  Cosine for initial ranking, Core for geodesic refinement of the query vector.
"""
import torch, numpy as np, time, sys, sqlite3, math, random
torch.manual_seed(42)

sys.path.insert(0, r'THOUGHT/LAB/FERAL_RESIDENT')
from native_eigen_reasoner import NativeEigenReasoner
from sentence_transformers import SentenceTransformer

DB = r'THOUGHT/LAB/FERAL_RESIDENT/data/db/feral_eternal.db'
D_emb = 384; D = D_emb // 2

print("Loading Core...")
eigen = NativeEigenReasoner(d=64, heads=4, layers=4, cycles=4)
eigen._init_core()

print("Loading Feral DB knowledge...")
conn = sqlite3.connect(DB)
rows = conn.execute("""
    SELECT v.vec_blob, m.text, v.sequence_id
    FROM vectors v JOIN memories m ON m.text IS NOT NULL
    WHERE v.sequence_id IS NOT NULL
    LIMIT 5000
""").fetchall()

# Fallback: memories directly (different join)
if len(rows) == 0:
    rows = conn.execute("""
        SELECT v.vec_blob, v.vector_id, v.sequence_id
        FROM vectors v WHERE v.sequence_id IS NOT NULL LIMIT 3000
    """).fetchall()
    # Load text separately
    mems = conn.execute("SELECT memory_id, text FROM memories LIMIT 5000").fetchall()
    mem_dict = {m[0]: m[1] for m in mems}
    knowledge = []
    for blob, vid, paper in rows:
        text = mem_dict.get(vid, '')[:300]
        if text:
            vec = np.frombuffer(blob, dtype=np.float32)
            knowledge.append({'vector': vec, 'text': text, 'paper': paper})
else:
    knowledge = []
    for blob, text, paper in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        knowledge.append({'vector': vec, 'text': text[:300], 'paper': paper})

conn.close()

# Also try direct memories table
if len(knowledge) == 0:
    print("Trying memories table directly...")
    conn = sqlite3.connect(DB)
    mems = conn.execute("SELECT text FROM memories WHERE text NOT LIKE '%error%' LIMIT 2000").fetchall()
    knowledge = []
    for (text,) in mems:
        if len(text) > 50:
            vec = eigen.embed.encode(text[:500])
            knowledge.append({'vector': vec, 'text': text[:300], 'paper': 'memory'})
    conn.close()

print(f"Loaded {len(knowledge)} entries")

if len(knowledge) == 0:
    print("No knowledge loaded — building from WikiText...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sents = []
    for ex in ds:
        text = str(ex["text"]).strip()
        if text:
            for s in text.replace('\n', ' ').split('. '):
                if 20 <= len(s) <= 300: sents.append(s)
        if len(sents) >= 1000: break
    print(f"Encoding {len(sents)} sentences...")
    knowledge = []
    for s in sents:
        knowledge.append({'vector': eigen.embed.encode(s), 'text': s, 'paper': 'wikitext'})

# ---- Navigate and talk ----
def talk(query, k=3):
    qv = eigen.embed.encode(query)
    
    # Step 1: Cosine ranking (fast, no Core)
    scored = []
    for entry in knowledge:
        sim = float(np.dot(qv, entry['vector']) /
                   (np.linalg.norm(qv) * np.linalg.norm(entry['vector']) + 1e-8))
        if sim > 0.10:  # broader threshold for diverse queries
            scored.append((sim, entry))
    scored.sort(key=lambda x: -x[0])
    candidates = scored[:12]
    
    if not candidates:
        return [], '', ''

    # Step 1: top by cosine
    top_cosine_text = candidates[0][1]['text'][:80]
    
    # Step 2: Core geodesic refinement
    seq_vecs = [qv] + [c[1]['vector'] for c in candidates]
    seq = np.stack(seq_vecs)
    
    z = torch.complex(torch.tensor(seq[:, :D], dtype=torch.float32).unsqueeze(0),
                      torch.tensor(seq[:, D:], dtype=torch.float32).unsqueeze(0))
    zp = torch.complex(eigen._in_r(z.real) - eigen._in_i(z.imag),
                       eigen._in_r(z.imag) + eigen._in_i(z.real))
    
    with torch.no_grad():
        for cycle in range(eigen.cycles):
            z_out, _ = eigen._core(zp + zp)
            zp = z_out
    
    refined = z_out[0, 0]
    pr = eigen._out_r(refined.real) + eigen._out_i(refined.imag)
    pi = eigen._out_r(refined.imag) - eigen._out_i(refined.real)
    nav_vec = torch.cat([pr, pi], dim=-1).detach().numpy()
    
    # Step 3: Re-rank by geodesic distance
    results = []
    for entry in knowledge:
        sim = float(np.dot(nav_vec, entry['vector']) /
                   (np.linalg.norm(nav_vec) * np.linalg.norm(entry['vector']) + 1e-8))
        results.append((sim, entry['text'], entry['paper']))
    results.sort(key=lambda x: -x[0])
    
    # Show what the Core changed
    top_cosine = candidates[0][1]['text'][:80] if candidates else 'none'
    top_geodesic = results[0][1][:80] if results else 'none'
    
    return results[:k], top_cosine, top_geodesic

# ---- Test ----
print("\n" + "=" * 60)
print("FERAL TALK — Core navigates, cosine measures, DB responds")
print("=" * 60)

queries = [
    "How do neural networks compress information?",
    "What is the transformer attention mechanism?",
    "What is the chemical formula of water?",
    "How does training work?",
]

for q in queries:
    t0 = time.time()
    results, cos_top, geo_top = talk(q, k=2)
    print(f"\nQ: {q}")
    print(f"  time: {time.time()-t0:.1f}s")
    if results:
        print(f"  cosine best: {cos_top}")
        print(f"  geodesic best: {geo_top}")
        for i, (score, text, paper) in enumerate(results):
            print(f"  [{i+1}] {score:.3f} ({paper}): {text[:150]}")
    else:
        print(f"  (no matches in DB)")
