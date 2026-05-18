"""Q51: Does embedding space have intrinsic complex structure? Berry phase test."""
import sys, json, time, ctypes
import numpy as np
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

so = "THOUGHT/LAB/EIGEN_ALIGNMENT/qgt_lib/build_minimal/lib/libdiffgeo.so"
lib = ctypes.CDLL(so)

class ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]

class DiffgeoEngine(ctypes.Structure):
    pass

lib.bridge_engine_create.restype = ctypes.POINTER(DiffgeoEngine)
lib.bridge_compute_berry_phase.argtypes = [
    ctypes.POINTER(DiffgeoEngine),
    ctypes.POINTER(ctypes.POINTER(ComplexFloat)),
    ctypes.c_size_t, ctypes.c_size_t,
]
lib.bridge_compute_berry_phase.restype = ComplexFloat

def berry_phase_c(states):
    engine = lib.bridge_engine_create()
    n_states = len(states)
    dim = states[0].shape[0]
    state_ptrs = (ctypes.POINTER(ComplexFloat) * n_states)()
    state_arrays = []
    for s_idx in range(n_states):
        arr = (ComplexFloat * dim)()
        for i in range(dim):
            arr[i].real = float(states[s_idx][i].real)
            arr[i].imag = float(states[s_idx][i].imag)
        state_ptrs[s_idx] = arr
        state_arrays.append(arr)
    result = lib.bridge_compute_berry_phase(engine, state_ptrs, n_states, dim)
    return complex(result.real, result.imag)

# Analogy loops
LOOPS = {
    "king->man->woman->queen": ["king", "man", "woman", "queen", "king"],
    "paris->france->berlin->germany": ["paris", "france", "berlin", "germany", "paris"],
    "walk->walking->run->running": ["walk", "walking", "run", "running", "walk"],
    "good->better->bad->worse": ["good", "better", "bad", "worse", "good"],
    "cat->kitten->dog->puppy": ["cat", "kitten", "dog", "puppy", "cat"],
    "doctor->hospital->teacher->school": ["doctor", "hospital", "teacher", "school", "doctor"],
    "hot->cold->fast->slow": ["hot", "cold", "fast", "slow", "hot"],
    "buy->sell->give->take": ["buy", "sell", "give", "take", "buy"],
}

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    D = model.get_sentence_embedding_dimension()
    K = 96

    print(f"\n{'='*64}")
    print(f"{name} (K={K})")
    print(f"{'='*64}")
    print(f"{'Loop':<40s} {'|Berry|':>10s} {'arg(Berry)':>12s} {'Re':>10s} {'Im':>10s}")

    for loop_name, words in LOOPS.items():
        embs = model.encode(words, normalize_embeddings=True)
        # PCA projection
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        proj = centered @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        norms[norms == 0] = 1
        proj = proj / norms

        # Complexify via Hilbert
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
        z = z / (zn + 1e-12)

        bp = berry_phase_c([z[i] for i in range(len(words))])
        mag = abs(bp)
        arg = np.angle(bp)

        # Is the phase significant? Test: random permuted loop as null
        np.random.seed(42)
        rand_idx = np.random.permutation(len(words))
        z_rand = [z[i] for i in rand_idx]
        bp_rand = berry_phase_c(z_rand)
        mag_rand = abs(bp_rand)

        sig = "***" if mag > 3 * mag_rand else "   "
        print(f"{loop_name:<40s} {mag:10.4f} {arg:12.4f} {bp.real:10.4f} {bp.imag:10.4f} {sig}")

    # Control: 20 random word loops
    print(f"\n  Random word loops (N=5):")
    np.random.seed(99)
    WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river",
        "tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish",
        "horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood",
        "mother","father","child","friend","king","queen","love","hate","truth","life",
        "death","time","space","power","peace","war","hope","fear","joy","pain","dream",
        "thought","book","door","house","road","food","money","stone","gold"]
    rand_mags = []
    for trial in range(20):
        words = list(np.random.choice(WORDS_POOL, 5, replace=False)) + [np.random.choice(WORDS_POOL)]
        embs = model.encode(words, normalize_embeddings=True)
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
        proj = centered @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
        z = z / (zn + 1e-12)
        bp = berry_phase_c([z[i] for i in range(len(words))])
        rand_mags.append(abs(bp))
    rand_mags = np.array(rand_mags)
    print(f"  Mean |Berry|: {rand_mags.mean():.4f} +/- {rand_mags.std():.4f}")

    # Control: Real embeddings (no complexification) — should be ~0
    print(f"\n  Real (no complexification) — expected |Berry| ~ 0:")
    words = ["king", "man", "woman", "queen", "king"]
    embs = model.encode(words, normalize_embeddings=True)
    z_real = embs.astype(np.complex128)
    bp_real = berry_phase_c([z_real[i] for i in range(len(words))])
    print(f"  |Berry| = {abs(bp_real):.4f}  (should be ~0 for real vectors)")

print(f"\n{'='*64}")
print("Q51 VERDICT:")
print("  If analogy loops have |Berry phase| >> random loops: C5 satisfied (intrinsic complex)")
print("  If |Berry phase| ~ random loops: C5 violated (complex structure is extrinsic)")
