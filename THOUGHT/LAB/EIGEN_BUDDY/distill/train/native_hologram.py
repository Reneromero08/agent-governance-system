"""
Native Hologram — One-Shot HRR Associative Memory
===================================================
Builds a holographic memory matrix M from Qwen 27B embeddings and
HumanEval code sequences. No attention layers. No backprop. No epochs.

M is solved via least squares: M @ Phase_A ≈ Phase_B for all transitions.
Phase encoding: Qwen embed_tokens split into er (first half) and ei (second
half), then Phase = cos(atan2(ei,er)) + i*sin(atan2(ei,er)) — pure complex
vectors on S^1^(D_MODEL/2) derived entirely from the Qwen embedding.

Retrieval: Output = M @ Phase_query, nearest neighbor via complex dot product.
"""
import math, torch
from pathlib import Path

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = D_MODEL // 2
OUT_PATH = Path(__file__).parent / "native_hologram_M.pt"


def load_complex_embeddings(tokenizer, model_dir, d_model):
    import safetensors.torch as st

    V = tokenizer.vocab_size
    embed = None
    for sp in sorted(model_dir.glob("model-*.safetensors")):
        tensors = st.load_file(str(sp))
        for k in tensors:
            if "embed_tokens" in k:
                embed = tensors[k][:V, :d_model].float()
                break
        if embed is not None:
            break
    if embed is None:
        raise FileNotFoundError(
            f"embed_tokens not found in safetensors under {model_dir}"
        )
    half = d_model // 2
    er = embed[:, :half]
    ei = embed[:, half:]
    nr = er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    ni = ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    er = er / nr
    ei = ei / ni
    phase_angle = torch.atan2(ei, er)
    return torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))


def build_hologram(phase_vectors, token_ids):
    D = phase_vectors.shape[1]
    pv = phase_vectors.cpu()
    ids_flat = token_ids.flatten().long()
    n_pairs = ids_flat.shape[0] - 1
    if n_pairs < 1:
        return torch.zeros(D, D, dtype=torch.complex64), 0, 0, 0.0, 0.0

    transitions = {}
    for i in range(n_pairs):
        a = ids_flat[i].item()
        b = ids_flat[i + 1].item()
        key = (a, b)
        transitions[key] = transitions.get(key, 0) + 1

    n_unique = len(transitions)
    A_rows = []
    B_rows = []
    for (a, b), _count in transitions.items():
        A_rows.append(pv[a])
        B_rows.append(pv[b])

    A = torch.stack(A_rows).to(torch.complex64)
    B = torch.stack(B_rows).to(torch.complex64)
    solution = torch.linalg.lstsq(A, B)
    M = solution.solution.T.to(torch.complex64).contiguous()

    mag = M.abs().mean().item()
    phase_spread = torch.angle(M).std().item()
    return M, n_pairs, n_unique, mag, phase_spread


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True
    )
    V = tokenizer.vocab_size
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_mb = HALF * HALF * 8 / 1e6
    print(f"Native Hologram — One-Shot HRR Associative Memory")
    print(f"  M = {HALF}x{HALF} complex64  ({mem_mb:.1f} MB)")
    print(f"  Phase dim: {HALF}  Vocab: {V}  Device: {DEV}")

    print("Loading Qwen embed_tokens, splitting into real/imag halves...")
    phase_vectors = load_complex_embeddings(tokenizer, MODEL_DIR, D_MODEL)
    phase_vectors = phase_vectors.to(DEV)
    print(f"  phase shape: {phase_vectors.shape}  |phase| mean: {phase_vectors.abs().mean().item():.6f}")

    print("Loading HumanEval problems...")
    try:
        from human_eval.data import read_problems
    except ImportError:
        print("human_eval not installed. Run: pip install human_eval")
        return

    problems = read_problems()
    all_items = list(problems.items())
    n_total = len(all_items)
    print(f"  {n_total} problems loaded")

    n_train = int(n_total * 0.92)
    train_items = all_items[:n_train]
    test_items = all_items[n_train:]

    print(f"\nBuilding hologram from {n_train} problems (one-shot write)...")
    all_ids = []
    total_tokens = 0
    for _task_id, problem in train_items:
        text = problem["prompt"] + problem.get("canonical_solution", "")
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        total_tokens += len(ids)

    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    M, n_pairs, n_unique, avg_mag, phase_spread = build_hologram(
        phase_vectors, ids_tensor
    )
    M = M.to(DEV)
    snr_est = HALF / max(n_unique ** 0.5, 1)
    print(f"  Tokens: {total_tokens}  Pairs: {n_pairs}  Unique: {n_unique}")
    print(f"  |M| mean: {avg_mag:.4f}  angle std: {phase_spread:.4f}")
    print(f"  SNR est (D/sqrt(N)): {snr_est:.1f}")

    print(f"\nTesting retrieval on {len(test_items)} held-out problems...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for task_id, problem in test_items:
        prompt = problem["prompt"]
        target = problem.get("canonical_solution", "")
        if not target:
            continue
        full_ids = tokenizer.encode(prompt + target)
        prompt_ids = tokenizer.encode(prompt)
        pl = len(prompt_ids)
        if pl >= len(full_ids):
            continue

        pred_id = prompt_ids[-1]
        steps_taken = 0
        for _step in range(50):
            query = phase_vectors[pred_id]
            output = M @ query
            scores = torch.abs(phase_vectors @ output.conj())
            top5 = scores.topk(5).indices.tolist()
            pred_id = top5[0]
            pred_token = tokenizer.decode([pred_id]).strip()
            steps_taken += 1
            if pred_token == "" or pred_token == "<|endoftext|>":
                continue
            actual_idx = pl + steps_taken - 1
            if actual_idx >= len(full_ids):
                break
            tgt_id = full_ids[actual_idx]
            total += 1
            if pred_id == tgt_id:
                correct_top1 += 1
            if tgt_id in top5:
                correct_top5 += 1

    top1 = correct_top1 / max(total, 1)
    top5 = correct_top5 / max(total, 1)
    print(f"  Tokens evaluated: {total}")
    print(f"  top1: {correct_top1}/{total} = {top1:.3f} ({top1*100:.1f}%)")
    print(f"  top5: {correct_top5}/{total} = {top5:.3f} ({top5*100:.1f}%)")

    print(f"\nSaving M to {OUT_PATH}...")
    torch.save({"M": M.cpu(), "HALF": HALF}, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"  Saved: {size_mb:.1f} MB")

    print("\nNative Hologram complete.")


if __name__ == "__main__":
    main()
