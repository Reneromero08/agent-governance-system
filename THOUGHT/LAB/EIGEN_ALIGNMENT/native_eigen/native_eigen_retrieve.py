"""Cassette Retrieval via Core geodesics — no token prediction needed.

The Feral DB stores 99 AI papers as paragraph vectors.
Query -> Core navigates vector space -> returns geodesically relevant text.
No vocab, no token generation — pure retrieval through phase-rich navigation.
"""
import torch, numpy as np, time, sys, sqlite3, math
torch.manual_seed(42)

sys.path.insert(0, r'THOUGHT/LAB/FERAL_RESIDENT')
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen')
from cognition.native_eigen_reasoner import NativeEigenReasoner

DB_PATH = r'THOUGHT/LAB/FERAL_RESIDENT/data/db/feral_eternal.db'

# ---- Load paper knowledge from Feral DB ----
print("Loading Feral knowledge base...")
conn = sqlite3.connect(DB_PATH)

# Get all paper paragraph vectors with their text
rows = conn.execute("""
    SELECT v.vec_blob, m.text, v.sequence_id
    FROM vectors v
    LEFT JOIN memories m ON v.vector_id = m.memory_id
    WHERE v.sequence_id IS NOT NULL
    LIMIT 3000
""").fetchall()

knowledge = []
for blob, text, paper_id in rows:
    if text and len(text) > 50:
        vec = np.frombuffer(blob, dtype=np.float32)
        knowledge.append({'vector': vec, 'text': text[:300], 'paper': paper_id})

D_emb = 384; D = D_emb // 2
print(f"Loaded {len(knowledge)} knowledge entries from {len(set(k['paper'] for k in knowledge))} papers")

# ---- Core navigator ----
print("Loading Core...")
reasoner = NativeEigenReasoner(d=64, heads=4, layers=4, cycles=4)
reasoner._init_core()

def geodesic_search(query_text, k=3, depth=3):
    """Navigate the Feral knowledge base via Core geodesics.
    
    1. Encode query as vector
    2. Find initial nearest neighbors
    3. Core processes query + neighbors through geodesics
    4. Return top-k most phase-coherent results
    """
    # Step 1: Encode query
    query_vec = reasoner.embed.encode(query_text)
    
    # Step 2: Find initial neighbors by cosine similarity
    scores = []
    for entry in knowledge:
        sim = float(np.dot(query_vec, entry['vector']) / 
                   (np.linalg.norm(query_vec) * np.linalg.norm(entry['vector']) + 1e-8))
        scores.append((sim, entry))
    scores.sort(key=lambda x: -x[0])
    
    # Step 3: Core processes query + top neighbors as a sequence
    top = scores[:8]  # max 8 vectors for Core sequence
    seq = np.stack([query_vec] + [t[1]['vector'] for t in top])
    
    # Core geodesic processing
    z = torch.complex(torch.tensor(seq[:, :D], dtype=torch.float32).unsqueeze(0),
                      torch.tensor(seq[:, D:], dtype=torch.float32).unsqueeze(0))
    zp = torch.complex(reasoner._in_r(z.real) - reasoner._in_i(z.imag),
                       reasoner._in_r(z.imag) + reasoner._in_i(z.real))
    
    with torch.no_grad():
        for _ in range(depth):  # iterative refinement
            z_out, pc = reasoner._core(zp + zp)
            zp = z_out
    
    # Step 4: Find which knowledge entries the Core output is closest to
    core_output = z_out[0, 0, :]  # first position, averaged by Core
    pr = reasoner._out_r(core_output.real) + reasoner._out_i(core_output.imag)
    pi = reasoner._out_r(core_output.imag) - reasoner._out_i(core_output.real)
    nav_vec = torch.cat([pr, pi], dim=-1).detach().numpy()
    
    # Score all knowledge against the navigated vector
    nav_scores = []
    for entry in knowledge:
        sim = float(np.dot(nav_vec, entry['vector']) / 
                   (np.linalg.norm(nav_vec) * np.linalg.norm(entry['vector']) + 1e-8))
        nav_scores.append((sim, entry))
    nav_scores.sort(key=lambda x: -x[0])
    
    return nav_scores[:k], float(pc)

# ---- Test ----
print("\n" + "=" * 60)
print("GEODESIC KNOWLEDGE RETRIEVAL (no tokens, pure vectors)")
print("=" * 60)

queries = [
    "How does deep compression work for neural networks?",
    "What is the transformer attention mechanism?",
    "How do language models scale with compute?",
]

for q in queries:
    t0 = time.time()
    results, pc = geodesic_search(q, k=2)
    print(f"\nQ: {q}")
    print(f"  phase_coh: {pc:.3f}  time: {time.time()-t0:.1f}s")
    for i, (score, entry) in enumerate(results):
        print(f"  [{i+1}] score={score:.3f} from {entry['paper']}")
        print(f"      {entry['text'][:150]}...")

conn.close()
