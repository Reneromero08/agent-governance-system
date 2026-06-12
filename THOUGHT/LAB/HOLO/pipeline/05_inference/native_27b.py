"""
Holographic Knowledge Distillation: Qwen 27B (.holo) -> Qwen 0.5B
================================================================
Teacher: qwen_27b_catalytic_k256.holo (3.7 GB, rank-256 eigenvectors)
         NEVER loaded as a model. Only SVD eigen-directions.

Student: Qwen 0.5B safetensors (trainable, ~1 GB)

Method: Subspace Alignment Distillation
  For each student layer, the hidden states are projected through the
  corresponding 27B layer's U_k eigen-directions (o_proj, down_proj).
  The reconstruction error through the 27B subspace measures alignment.

  L_total = L_CE(next-token) + lambda * L_subspace_alignment

Student: 24 layers, hidden_dim=896, rank=128
Teacher: 64 layers, hidden_dim=5120, rank=256
"""
import os, math, re, torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(r'd:\CCC 2.0\AI\agent-governance-system')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HD = 896     # Qwen 0.5B hidden dim
T_HD = 5120  # Qwen 27B hidden dim
K_T = 256    # 27B holo rank
N_S = 24     # student layers
N_T = 64     # teacher layers

MODEL_DIR = str(REPO / "THOUGHT" / "LAB" / "CAT_CAS" / "3_physics_complexity" / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b")
HOLO_27B = str(REPO / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "cybernetic_truth" / "qwen_27b_catalytic_k256.holo")

def s_to_t(s_layer):
    return int(round(s_layer * N_T / N_S))

def load_teacher_subspaces(path):
    print("[*] Loading 27B .holo teacher eigenvectors...", flush=True)
    d = torch.load(path, map_location="cpu", weights_only=False)
    registry = {}
    for k, v in d.items():
        m = re.match(r'model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.weight\.U', k)
        if not m:
            continue
        t_layer, kind, name = int(m.group(1)), m.group(2), m.group(3)
        if kind == 'self_attn' and name == 'o_proj':
            registry.setdefault(t_layer, {})['o'] = v.float()
        elif kind == 'mlp' and name == 'down_proj':
            registry.setdefault(t_layer, {})['d'] = v.float()
    print(f"     {len(registry)} teacher layers with U_k subspaces", flush=True)
    del d
    return registry

def build_subspace_map(teacher_reg):
    smap = {}
    for s in range(N_S):
        t = s_to_t(s)
        Us = []
        if t in teacher_reg:
            if 'o' in teacher_reg[t]:
                Us.append(('o', teacher_reg[t]['o']))
            if 'd' in teacher_reg[t]:
                Us.append(('d', teacher_reg[t]['d']))
        if Us:
            smap[s] = Us
    return smap

def compute_subspace_loss(hidden, subspace_map, proj_up):
    loss = torch.tensor(0.0, device=DEVICE)
    n = 0
    for s_layer, u_list in subspace_map.items():
        h_s = hidden[s_layer + 1].float()
        h_p = proj_up(h_s)
        for tag, U_t in u_list:
            U_d = U_t.to(h_p.device)
            h_e = torch.matmul(h_p, U_d)
            h_r = torch.matmul(h_e, U_d.T)
            loss = loss + F.mse_loss(h_p, h_r)
            n += 1
    return loss / n if n > 0 else loss

def main():
    print(f"[*] Device: {DEVICE}", flush=True)

    # --- Load teacher subspaces from .holo (NEVER instantiates 27B) ---
    teacher_reg = load_teacher_subspaces(HOLO_27B)
    subspace_map = build_subspace_map(teacher_reg)
    n_mapped = sum(len(v) for v in subspace_map.values())
    print(f"     {n_mapped} subspace targets across {len(subspace_map)} student layers", flush=True)
    del teacher_reg

    # --- Load 0.5B student (trainable) ---
    print("[*] Loading 0.5B student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, device_map=DEVICE, torch_dtype=torch.bfloat16
    )
    student.train()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Shared projection: student hidden (896) -> teacher dim (5120) ---
    proj_up = nn.Linear(HD, T_HD, bias=False).to(DEVICE, dtype=torch.float32)
    nn.init.normal_(proj_up.weight, std=0.02)

    # --- Tokenize real text corpus (instant, no generation) ---
    seeds = [
        "The holographic principle states that information about a volume of space is encoded on its boundary. This has profound implications for our understanding of quantum gravity and the nature of reality itself. In the context of neural networks, holographic compression suggests that the essential structure of a large model can be preserved in a lower-dimensional subspace, much like a hologram stores a 3D image on a 2D surface.",
        "Knowledge distillation is a technique where a smaller student model learns to mimic a larger teacher model. The student is trained to match the teacher's output logits, hidden states, or both. This transfers the teacher's learned representations to the student, enabling the student to achieve performance close to the teacher while being much more efficient at inference time.",
        "The eigendecomposition of a neural network's weight matrices reveals the principal directions along which information flows through the network. By projecting activations onto these principal components, we can understand which features are most important for the network's computations and identify the intrinsic dimensionality of the representations learned by each layer.",
        "Quantum computing leverages superposition and entanglement to perform computations that would be intractable for classical computers. While full-scale quantum computers remain elusive, quantum-inspired classical algorithms—such as those based on tensor networks, amplitude amplification, and the quantum Fourier transform—have demonstrated advantages in machine learning and optimization tasks.",
        "In the cybernetic truth framework, resonance R = Tr(rho C) measures the alignment between a model's current state density matrix and a reference truth vector. This provides a continuous metric of factual correctness that can be used to dynamically modulate the model's sampling temperature, steering generation toward more truthful outputs when uncertainty is high.",
    ]
    corpus = tokenizer(" ".join(seeds), return_tensors="pt")['input_ids'][0].tolist()
    seq_len = 64
    n_steps = 500
    lmbda = 0.1
    opt = torch.optim.AdamW(list(student.parameters()) + list(proj_up.parameters()), lr=5e-6)
    print(f"[*] Training {n_steps} steps, seq_len={seq_len}, lambda={lmbda}, corpus={len(corpus)} tokens", flush=True)

    for step in range(n_steps):
        start = torch.randint(0, len(corpus) - seq_len - 1, (1,)).item()
        input_ids = torch.tensor([corpus[start:start + seq_len]], device=DEVICE)
        labels = torch.tensor([corpus[start + 1:start + seq_len + 1]], device=DEVICE)

        out = student(input_ids, labels=labels, output_hidden_states=True)
        loss_ce = out.loss
        loss_ss = compute_subspace_loss(out.hidden_states, subspace_map, proj_up)
        loss = loss_ce.float() + lmbda * loss_ss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if step % 25 == 0:
            print(f"  [Step {step:4d}] CE: {loss_ce.item():.4f} | SS: {loss_ss.item():.6f} | Total: {loss.item():.4f}", flush=True)

    # --- Test generation ---
    print("\n[*] Testing generation:", flush=True)
    student.eval()
    prompt = "The fundamental nature of consciousness is"
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)['input_ids']
    with torch.no_grad():
        for _ in range(40):
            out = student(ids)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {tokenizer.decode(ids[0].tolist())}", flush=True)

    # --- Save distilled student ---
    save_path = str(REPO / "THOUGHT" / "LAB" / "HOLO" / "4_holographic_brain" / "student_distilled.pt")
    torch.save(student.state_dict(), save_path)
    print(f"\n[+] Distilled student saved to {save_path}", flush=True)

if __name__ == "__main__":
    main()
