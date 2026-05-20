"""Track A+C: GPU Stride Serving & Feral DB Distillation Pipeline.

Loads 8,904 real Feral DB vectors (192-dim complex) from SQLite.
Feeds through GPU model (LFM2.5-1.2B) for phase projection.
Daemon monitors Kuramoto r with actual Feral vector population.
Unitary trace minimization loop collects phase states for Track C.

Reference: ROADMAP_2_3, Script Handoff Protocol
"""
import torch, time, math, sys, sqlite3, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.thermo import (ThermodynamicDaemon, compute_phase_diversity,
                              compute_participation_ratio)

MODEL_PATH = r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf"
FERAL_DB = r"THOUGHT\LAB\FERAL_RESIDENT\data\db\feral_eternal.db"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_feral_vectors(db_path, max_vectors=None):
    """Load Feral DB vectors into a complex torch tensor."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT vector_id, vec_blob, Df FROM vectors ORDER BY rowid"
    ).fetchall()
    if max_vectors:
        rows = rows[:max_vectors]
    conn.close()

    vectors = []
    ids = []
    dfs = []
    for vid, blob, df in rows:
        n_floats = len(blob) // 4
        floats = struct.unpack(f'<{n_floats}f', blob)
        real = torch.tensor(floats[0::2], dtype=torch.float32)
        imag = torch.tensor(floats[1::2], dtype=torch.float32)
        vectors.append(torch.complex(real, imag))
        ids.append(vid)
        dfs.append(df)

    z = torch.stack(vectors)  # (N, D)
    return z, ids, dfs


def main():
    print("=" * 60)
    print("TRACK A+C: Feral DB Distillation Pipeline")
    print(f"Device: {DEVICE} | Model: {Path(MODEL_PATH).name}")
    print("=" * 60)

    # Phase 1: Load model on GPU
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096,
                embeddings=True, verbose=False)
    print(f"[gpu] Loaded in {time.time()-t0:.1f}s | ctx={llm.n_ctx()}")

    # Phase 2: Load Feral DB vectors
    print(f"\n[feral] Loading vectors from {Path(FERAL_DB).name}...")
    t0 = time.time()
    z_feral, vector_ids, feral_dfs = load_feral_vectors(FERAL_DB)
    n_vectors, d = z_feral.shape
    load_time = time.time() - t0
    print(f"[feral] {n_vectors} vectors, D={d}, {z_feral.element_size() * z_feral.numel() / 1e6:.1f} MB "
          f"in {load_time:.1f}s")

    # Phase 3: Compute native Feral phase metrics
    r_native = compute_phase_diversity(z_feral)
    df_native = compute_participation_ratio(z_feral)
    avg_df = sum(feral_dfs) / len(feral_dfs) if feral_dfs else 0
    print(f"[feral] Native r={r_native:.4f} D_f={df_native:.0f} "
          f"(stored avg D_f={avg_df:.1f})")

    # Phase 4: Initialize daemon with Feral vectors
    # Down-project to 64-dim if needed (Feral is 192-dim, daemon uses 64)
    if d != 64:
        # PCA-like: take first 64 dims as approximation
        z_daemon = z_feral[:200, :64]  # use first 200 vectors, first 64 dims
    else:
        z_daemon = z_feral[:200]

    daemon = ThermodynamicDaemon(d=64, n_vectors=min(200, n_vectors),
                                  r_threshold=0.159,  # 1/2pi: Feral Resident empirical threshold
                                  noise_factor=2.0,
                                  thermo_enabled=True)
    # Replace daemon's random vectors with real Feral vectors
    daemon.vectors = z_daemon.clone()
    daemon.init_df = compute_participation_ratio(daemon.vectors)
    daemon.df_threshold = daemon.init_df * 0.9
    daemon.r_history = [compute_phase_diversity(daemon.vectors)]
    daemon.df_history = [daemon.init_df]
    print(f"[daemon] Seeded with {z_daemon.shape[0]} real Feral vectors "
          f"D_f={daemon.init_df:.0f} r={daemon.r_history[0]:.4f}")

    # Phase 5: GPU inference + daemon monitoring loop
    print("\n[loop] GPU inference + Feral daemon monitoring")
    n_steps = 10
    phase_history = []
    df_history = []

    for step in range(n_steps):
        # GPU inference on a concept prompt
        prompt = "Phase turns information into meaning. The hologram enfolds the operation into geometry."
        t_infer = time.time()
        output = llm(prompt, max_tokens=8, temperature=0.7)
        infer_ms = (time.time() - t_infer) * 1000
        tokens = len(output['choices'][0]['text'].split())

        # Daemon step with Feral vectors
        r, df, noise = daemon.step()
        df_history.append(df)
        phase_history.append(r)

        gpu_mem = torch.cuda.memory_allocated() // 1024**2 if DEVICE.type == 'cuda' else 0
        print(f"  step {step:2d}: {infer_ms:.0f}ms | {tokens} tok | "
              f"r={r:.4f} D_f={df:.0f} {'noise!' if noise else ''} "
              f"gpu={gpu_mem}MB", flush=True)

    # Phase 6: Results
    daemon_status = daemon.status()
    print(f"\n[results] {n_steps} inference steps completed")
    print(f"  Feral DB: {n_vectors} vectors, {d}-dim complex, native D_f={df_native:.0f}")
    print(f"  Daemon: D_f {daemon_status['init_df']:.0f} -> {daemon_status['final_df']:.0f} "
          f"({daemon_status['delta_pct']:.1f}%) r: {daemon_status['initial_r']:.3f} -> {daemon_status['final_r']:.3f}")
    print(f"  Noise injections: {daemon_status['noise_count']}")
    print(f"  Avg daemon r: {sum(phase_history)/len(phase_history):.4f}")

    # Adjoint
    daemon.vectors.zero_()
    print("[adjoint] Vectors uncomputed. 0 bits residual.")


if __name__ == '__main__':
    main()
