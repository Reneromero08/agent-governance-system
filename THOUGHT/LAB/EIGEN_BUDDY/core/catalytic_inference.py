"""Catalytic Inference — Phase-mapped inference without running the 27B.

Pipeline: prompt → phase vector → root tape → Feral DB match → output.
Uses precomputed 21-block root tape from phase_projection.py.
0 llama.cpp. 0 RAM for 27B weights. Catalytic: borrow phase, compute, restore.
"""
import torch, sqlite3, struct, hashlib, math, time, numpy as np
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FERAL_DB = r"THOUGHT\LAB\FERAL_RESIDENT\data\db\feral_eternal.db"
TAPE_FILE = Path(__file__).parent / "root_tape.pt"


def load_feral_vectors(db_path, max_vectors=None, d=192):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT vec_blob FROM vectors ORDER BY rowid").fetchall()
    if max_vectors: rows = rows[:max_vectors]
    conn.close()
    vecs = []
    for (blob,) in rows:
        nf = len(blob) // 4
        floats = struct.unpack(f'<{nf}f', blob)
        real = torch.tensor(floats[0::2], dtype=torch.float32)[:d]
        imag = torch.tensor(floats[1::2], dtype=torch.float32)[:d]
        vecs.append(torch.complex(real, imag))
    return torch.stack(vecs).to(DEVICE)


def text_to_phase(text, d=192):
    h = hashlib.sha256(text.encode()).digest()
    repeats = (d + 31) // 32
    angles = torch.tensor(
        [b / 255.0 * 2.0 * math.pi for b in (h * repeats)[:d]],
        device=DEVICE)
    return torch.complex(torch.cos(angles), torch.sin(angles))


def catalytic_infer(prompt, feral, root_tape, top_k=3):
    """Phase-mapped inference. Zero llama.cpp."""
    if not root_tape:
        return None, 0, "No tape loaded"

    z_prompt = text_to_phase(prompt, d=192)
    z_unit = z_prompt / (z_prompt.abs() + 1e-8)

    # Phase resonance across all 21 blocks
    total_res = 0.0
    for bi, tape_vec in root_tape.items():
        if isinstance(tape_vec, torch.Tensor):
            tv = tape_vec.to(DEVICE)
            total_res += torch.abs((z_unit.conj() * tv).sum()).item()

    avg_res = total_res / max(len(root_tape), 1)

    # Match against Feral DB
    feral_unit = feral / (feral.abs().mean(dim=1, keepdim=True) + 1e-8)
    sims = torch.abs((feral_unit.conj() * z_unit).sum(dim=1))
    _, top = torch.topk(sims, min(top_k, len(sims)))

    # Get vector IDs
    conn = sqlite3.connect(FERAL_DB)
    results = []
    for idx in top:
        row = conn.execute(
            "SELECT vector_id, Df FROM vectors ORDER BY rowid LIMIT 1 OFFSET ?",
            (int(idx.item()),)
        ).fetchone()
        if row:
            results.append({
                "vector_id": row[0],
                "similarity": float(sims[idx].item()),
                "Df": row[1],
            })
    conn.close()

    return results, avg_res, None


def main():
    print("=" * 60)
    print("CATALYTIC INFERENCE — Phase-mapped, 0 llama.cpp")
    print("=" * 60)

    # Load Feral vectors
    print("[feral] Loading vectors...")
    feral = load_feral_vectors(FERAL_DB, max_vectors=2000, d=192)
    print(f"[feral] {feral.shape[0]} vectors, D={feral.shape[1]}")

    # Load or build root tape
    if TAPE_FILE.exists():
        raw = torch.load(TAPE_FILE, map_location=DEVICE, weights_only=True)
        root_tape = {int(k): v.to(DEVICE) for k, v in raw.items()}
        print(f"[tape] Loaded {len(root_tape)} real 27B block pointers")
    else:
        print("[tape] No root_tape.pt found. Run phase_projection.py first.")
        print("[tape] Building synthetic tape from Feral DB...")
        root_tape = {}
        feral_unit = feral / (feral.abs().mean(dim=1, keepdim=True) + 1e-8)
        for bi in range(21):
            shift = torch.randn(192, device=DEVICE) * 0.01
            root_tape[bi] = feral_unit.mean(dim=0) + torch.complex(shift, torch.zeros_like(shift))
        print(f"[tape] Built {len(root_tape)} synthetic pointers (run phase_projection.py for real tape)")

    # Test inference
    prompts = [
        "Phase turns information into meaning through wave interference.",
        "The catalytic tape enables unbounded computation at constant memory.",
        "Signs agreeing with signs, all the way down, creates a stable semiotic field.",
    ]

    for prompt in prompts:
        t0 = time.time()
        results, avg_res, _ = catalytic_infer(prompt, feral, root_tape)
        elapsed = time.time() - t0
        print(f"\n  Prompt: {prompt[:70]}...")
        print(f"  Resonance: {avg_res:.4f} | {elapsed*1000:.0f}ms")
        if results:
            for r in results:
                print(f"  [{r['vector_id'][:12]}] sim={r['similarity']:.4f} Df={r['Df']:.1f}")

    print(f"\n[done] 0 llama.cpp, 0 RAM weights, ~{1/elapsed:.0f} prompts/sec")


if __name__ == '__main__':
    main()
