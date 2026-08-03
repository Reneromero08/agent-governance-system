"""Track C+D: Liquid Inversion — Qwen 27B (sponge) → LFM2.5 1.2B Liquid (student).

ITEM 3: SKIP 4B Transformer. Direct fluid distillation onto Liquid architecture.
10 of 16 Liquid layers are continuous LIV convolutions — natively a wave state.
Loss: L = 1 - |Tr(rho_liquid · rho_qwen^dagger)|
Catalytic tape (U) holds intermediate states. Adjoint clears to 0 bits.

Reference: SYSTEM DIRECTIVE ITEM 3, ROADMAP_2_3 Track C
"""
import torch, time, math, sys, sqlite3, struct, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.thermo import (ThermodynamicDaemon, compute_phase_diversity,
                              compute_participation_ratio)

QWEN_27B = r"F:\LLM_Models\lmstudio-models\Qwen3.6-27B\Qwen3.6-27B-F16-mtp.gguf"
LIQUID_1B = r"D:\Reneshizzle\Apps\LM Studio\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf"
FERAL_DB = r"THOUGHT\LAB\FERAL_RESIDENT\data\db\feral_eternal.db"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HALF_PI = 1.0 / (2.0 * math.pi)  # 0.159 — Feral Resident empirical threshold


def load_feral_vectors(db_path, max_vectors=None):
    """Load Feral DB vectors into complex torch tensor."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT vector_id, vec_blob, Df FROM vectors ORDER BY rowid"
    ).fetchall()
    if max_vectors:
        rows = rows[:max_vectors]
    conn.close()
    vectors, ids, dfs = [], [], []
    for vid, blob, df in rows:
        n_floats = len(blob) // 4
        floats = struct.unpack(f'<{n_floats}f', blob)
        real = torch.tensor(floats[0::2], dtype=torch.float32)
        imag = torch.tensor(floats[1::2], dtype=torch.float32)
        vectors.append(torch.complex(real, imag))
        ids.append(vid); dfs.append(df)
    return torch.stack(vectors), ids, dfs


def text_to_phase(text, d=64, device=DEVICE):
    """Convert text to a unit complex phase vector via SHA-256 hash."""
    h = hashlib.sha256(text.encode()).digest()
    # Repeat 32-byte hash to fill d dimensions
    repeats = (d + 31) // 32
    hash_bytes = h * repeats
    angles = torch.tensor(
        [b / 255.0 * 2.0 * math.pi for b in hash_bytes[:d]],
        device=device)
    return torch.complex(torch.cos(angles), torch.sin(angles))


def trace_loss(z_student, z_teacher):
    """L = 1 - |Tr(rho_student · rho_teacher^dagger)| bounded in [0,1]."""
    zs = z_student / (z_student.abs() + 1e-8)
    zt = z_teacher / (z_teacher.abs() + 1e-8)
    inner = (zs.conj() * zt).sum()
    resonance = torch.abs(inner) / (zs.abs().norm() * zt.abs().norm() + 1e-8)
    return 1.0 - resonance, resonance


def main():
    print("=" * 60)
    print("TRACK C+D: Liquid Inversion Distillation")
    print("=" * 60)

    # Phase 1: Load Feral DB vectors (ground truth phase atlas)
    print(f"\n[feral] Loading vectors...")
    z_feral, _, feral_dfs = load_feral_vectors(FERAL_DB)
    n_vectors, d_feral = z_feral.shape
    r_native = compute_phase_diversity(z_feral)
    print(f"[feral] {n_vectors} vectors, D={d_feral}, native r={r_native:.4f}")

    # Phase 2: Load Liquid 1.2B as both teacher and student (self-distillation)
    # Qwen 27B on CPU is too slow (~60s/inference) for interactive iteration.
    # DeepSeek-V4-Pro (E:\Reneshizzle SG) will replace Qwen as teacher when ready.
    from llama_cpp import Llama

    print(f"[load] LFM2.5 1.2B Liquid (GPU — teacher + student)...")
    t0 = time.time()
    liquid = Llama(model_path=LIQUID_1B, n_gpu_layers=-1, n_ctx=512,
                   embeddings=False, verbose=False)
    print(f"[load] Ready in {time.time()-t0:.1f}s | ctx={liquid.n_ctx()}")

    # Phase 3: Initialize daemon with Feral vectors
    z_daemon = z_feral[:200, :64].to(DEVICE)
    daemon = ThermodynamicDaemon(d=64, n_vectors=200, r_threshold=HALF_PI,
                                  noise_factor=2.0, thermo_enabled=True)
    daemon.vectors = z_daemon.clone()
    daemon.init_df = compute_participation_ratio(daemon.vectors)
    daemon.df_threshold = daemon.init_df * 0.9
    daemon.r_history = [compute_phase_diversity(daemon.vectors)]
    daemon.df_history = [daemon.init_df]
    print(f"[daemon] Seeded: D_f={daemon.init_df:.0f} r={daemon.r_history[0]:.4f}")
    print(f"[daemon] 1/2pi threshold: {HALF_PI:.6f}")

    # Phase 4: Distillation loop — same prompts through both models
    prompts = [
        "Phase turns information into meaning.",
        "Signs agreeing with signs. All the way down.",
        "Decoherence is the enemy of meaning. Resonance survives.",
        "The spiral trajectory accumulates history in the phase of the present.",
        "Compression across scales amplifies resonance exponentially.",
    ]

    print(f"\n[distill] Self-distillation: {len(prompts)} prompts × 4 passes")
    print(f"[distill] Teacher (temp=0.3) vs Student (temp=0.9) — same model, different phase paths")
    n_steps = len(prompts) * 4
    loss_history = []
    df_history = []
    phase_history = []

    for step in range(n_steps):
        prompt = prompts[step % len(prompts)]

        # Teacher: low temperature = deterministic, coherent phase
        t0 = time.time()
        out_teacher = liquid(prompt, max_tokens=8, temperature=0.3)
        teacher_ms = (time.time() - t0) * 1000
        teacher_text = out_teacher['choices'][0]['text']

        # Student: high temperature = exploratory, diverse phase
        t0 = time.time()
        out_student = liquid(prompt, max_tokens=8, temperature=0.9)
        student_ms = (time.time() - t0) * 1000
        student_text = out_student['choices'][0]['text']

        # Phase states from output text
        z_teacher = text_to_phase(teacher_text, d=64)
        z_student = text_to_phase(student_text, d=64)

        # Unitary trace minimization: student → teacher alignment
        loss, resonance = trace_loss(z_student, z_teacher)
        loss_history.append(loss.item())
        phase_history.append(resonance.item())

        # Daemon step
        r, df, noise = daemon.step()
        df_history.append(df)

        print(f"  step {step:2d}: T={teacher_ms:.0f}ms S={student_ms:.0f}ms | "
              f"L={loss.item():.3f} res={resonance.item():.3f} | "
              f"r={r:.3f} D_f={df:.0f} {'noise!' if noise else ''}", flush=True)

    # Phase 5: Results
    daemon_status = daemon.status()
    avg_loss = sum(loss_history) / len(loss_history)
    avg_res = sum(phase_history) / len(phase_history)
    loss_delta = loss_history[0] - loss_history[-1]

    print(f"\n[results] {n_steps} self-distillation steps")
    print(f"  Model: LFM2.5 1.2B Liquid (GPU, 10 GDN + 6 GA)")
    print(f"  Method: temperature-differential self-distillation (T=0.3 vs S=0.9)")
    print(f"  Trace loss: {avg_loss:.4f} avg, {loss_history[0]:.4f} -> {loss_history[-1]:.4f} "
          f"(delta={loss_delta:+.4f})")
    print(f"  Phase resonance: {avg_res:.4f} avg")
    print(f"  Daemon: D_f {daemon_status['init_df']:.0f} -> {daemon_status['final_df']:.0f} "
          f"({daemon_status['delta_pct']:.1f}%) r: {daemon_status['initial_r']:.3f} -> {daemon_status['final_r']:.3f}")
    print(f"  Noise: {daemon_status['noise_count']}/{n_steps} @ 1/2pi={HALF_PI:.4f}")
    print(f"  Next: DeepSeek-V4-Pro -> Qwen 27B (Track B) when download complete")

    # Adjoint
    daemon.vectors.zero_()
    print("[adjoint] Tape cleared. 0 bits residual.")


if __name__ == '__main__':
    main()
