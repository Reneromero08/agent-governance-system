"""Q36: R connects to Bohm's implicate/explicate order.

Bohm: explicate = unfolded observable world. Implicate = enfolded hidden structure.
Mapping: magnitude = explicate (shared across models, r=+0.28).
         phase = implicate (model-specific, enfolded, r=+0.08).

Tests:
  1. Implicate carries model fingerprint — phase patterns are model-unique
  2. Hilbert = enfolding operation — doubles effective dimensionality (Df)
  3. Born rule = unfolding — projects implicate structure into explicate observables
  4. Explicate and implicate are COMPLEMENTARY within a model, not correlated
"""
import numpy as np, math
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from scipy.signal import hilbert

print("Loading models...")
minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')

words = ['dog','cat','bird','fish','horse','cow','pig','bear','lion',
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'red','blue','green','yellow','black','white','brown','gray',
    'paris','london','tokyo','berlin','moscow','rome','madrid',
    'hot','cold','big','small','good','bad','love','hate','life','death',
    'light','dark','day','night','man','woman','up','down','high','low',
    'fast','slow','happy','sad','young','old','war','peace',
    'mother','father','child','friend','enemy','teacher','leader',
    'water','fire','earth','air','sun','moon','star','sky','mountain','river',
    'house','car','book','tree','flower','food','music','art','science','god',
    'destroy','create','give','take','think','know','see','hear','speak','walk']

ml = minilm.encode(words, show_progress_bar=False)
mp = mpnet.encode(words, show_progress_bar=False)

def hilbert_cpx(embeds):
    N, D = embeds.shape
    cpx = np.zeros((N, D), dtype=np.complex128)
    for d in range(D): cpx[:, d] = hilbert(embeds[:, d])
    return cpx

ml_cpx = hilbert_cpx(ml)
mp_cpx = hilbert_cpx(mp)

# Explicate: magnitude (real part after Hilbert — the original real embeddings)
ml_ex = np.abs(ml_cpx)
mp_ex = np.abs(mp_cpx)

# Implicate: phase (imaginary argument after Hilbert)
ml_im = np.angle(ml_cpx)
mp_im = np.angle(mp_cpx)

# Distance matrices
def dist_matrix(emb, is_phase=False):
    N = emb.shape[0]
    if is_phase:
        D = np.zeros((N, N))
        for i in range(N):
            diff = emb[i] - emb
            D[i] = np.mean(np.abs(np.sin(diff)), axis=1)
    else:
        n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        D = 1.0 - n @ n.T
    return D

D_ml_ex = dist_matrix(ml_ex); D_mp_ex = dist_matrix(mp_ex)
D_ml_im = dist_matrix(ml_im, True); D_mp_im = dist_matrix(mp_im, True)

tri = np.triu_indices(len(words), k=1)

print("\n" + "=" * 60)
print("Q36: BOHM IMPLICATE/EXPLICATE ORDER")
print("=" * 60)

# Test 1: Explicate is shared across models
r_ex = spearmanr(D_ml_ex[tri], D_mp_ex[tri])[0]
print(f"\n  TEST 1: Explicate (magnitude) shared across models")
print(f"    MiniLM-MPNet magnitude correlation: r = {r_ex:+.4f}")
print(f"    {'EXPLICATE IS SHARED' if r_ex > 0.2 else 'Explicate not shared'}")

# Test 2: Implicate is model-specific
r_im = spearmanr(D_ml_im[tri], D_mp_im[tri])[0]
print(f"\n  TEST 2: Implicate (phase) is model-specific")
print(f"    MiniLM-MPNet phase correlation: r = {r_im:+.4f}")
print(f"    {'IMPLICATE IS MODEL-SPECIFIC' if abs(r_im) < 0.15 else 'Implicate partially shared'}")

# Test 3: Implicate and explicate are COMPLEMENTARY (orthogonal within a model)
r_ml_exim = spearmanr(D_ml_ex[tri], D_ml_im[tri])[0]
r_mp_exim = spearmanr(D_mp_ex[tri], D_mp_im[tri])[0]
print(f"\n  TEST 3: Explicate and implicate are COMPLEMENTARY")
print(f"    MiniLM: ex-im correlation r = {r_ml_exim:+.4f}")
print(f"    MPNet:  ex-im correlation r = {r_mp_exim:+.4f}")
if abs(r_ml_exim) < 0.1 and abs(r_mp_exim) < 0.1:
    print(f"    COMPLEMENTARY — explicate and implicate are orthogonal")
elif abs(r_ml_exim) < 0.2:
    print(f"    Mostly complementary — weak correlation")
else:
    print(f"    Significant correlation — not fully complementary")

# Test 4: Hilbert doubles Df (enfolding)
def participation_ratio(emb, complex_input=False):
    if complex_input:
        G = np.array([[np.vdot(emb[i], emb[j]) for j in range(len(emb))] for i in range(len(emb))])
    else:
        n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        G = n @ n.T
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = np.maximum(ev, 0)
    ev = ev / np.sum(ev)
    return np.sum(ev) ** 2 / max(np.sum(ev ** 2), 1e-8)

pr_ml_real = participation_ratio(ml)
pr_ml_cpx = participation_ratio(ml_cpx, True)
pr_mp_real = participation_ratio(mp)
pr_mp_cpx = participation_ratio(mp_cpx, True)

print(f"\n  TEST 4: Hilbert transform = enfolding (doubles Df)")
print(f"    MiniLM: real Df={pr_ml_real:.1f} -> complex Df={pr_ml_cpx:.1f} ({pr_ml_cpx/pr_ml_real:.1f}x)")
print(f"    MPNet:  real Df={pr_mp_real:.1f} -> complex Df={pr_mp_cpx:.1f} ({pr_mp_cpx/pr_mp_real:.1f}x)")
print(f"    {'HILBERT = ENFOLDING' if pr_ml_cpx/pr_ml_real > 1.5 else 'Weak enfolding'}")

# Test 5: Phase fingerprint uniqueness (on shared dimension count)
ml_phase_by_dim = np.std(np.sin(ml_im), axis=0)
mp_phase_by_dim = np.std(np.sin(mp_im), axis=0)
K = 50
ml_top = np.sort(ml_phase_by_dim)[-K:]
mp_top = np.sort(mp_phase_by_dim)[-K:]
r_fingerprint = spearmanr(ml_top, mp_top)[0] if len(ml_top) == len(mp_top) else 0
print(f"\n  TEST 5: Phase fingerprint uniqueness (top-{K} dims)")
print(f"    Per-dim phase diversity correlation: r = {r_fingerprint:+.4f}")
print(f"    {'PHASE IS MODEL FINGERPRINT' if abs(r_fingerprint) < 0.3 else 'Phase shares structure across models'}")

# Verdict
print(f"\n{'='*60}")
print(f"Q36 VERDICT")
print(f"{'='*60}")
criteria = [r_ex > 0.2, abs(r_im) < 0.15, abs(r_ml_exim) < 0.1, pr_ml_cpx/pr_ml_real > 1.5]
passed = sum(criteria)
print(f"  Explicate shared:      {'PASS' if criteria[0] else 'FAIL'} (r={r_ex:+.3f})")
print(f"  Implicate model-specific: {'PASS' if criteria[1] else 'FAIL'} (r={r_im:+.3f})")
print(f"  Complementary:         {'PASS' if criteria[2] else 'FAIL'} (r={r_ml_exim:+.3f})")
print(f"  Hilbert = enfolding:   {'PASS' if criteria[3] else 'FAIL'} ({pr_ml_cpx/pr_ml_real:.1f}x)")

if passed >= 3:
    print(f"\n  BOHM MAPPING CONFIRMED — magnitude=explicate, phase=implicate")
    print(f"  The Living Formula describes the enfolding/unfolding of meaning")
elif passed >= 2:
    print(f"\n  PARTIAL Bohm mapping — some criteria hold")
else:
    print(f"\n  Bohm mapping not supported")
