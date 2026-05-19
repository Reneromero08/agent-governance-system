"""Q34: Phase vs Magnitude convergence — implicate vs explicate order."""
import numpy as np, math
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from scipy.signal import hilbert
import torch

# Reuse trained model checkpoint from verify_q34_relational.py
# Just do the phase vs magnitude comparison

minilm = SentenceTransformer('all-MiniLM-L6-v2')
mpnet = SentenceTransformer('all-mpnet-base-v2')

# Generate test data: 200 words, same category structure
# Use common English words with known semantic relationships
words = ['the','be','to','of','and','in','that','have','it','for',
    'not','on','with','he','as','you','do','at','this','but',
    'his','by','from','they','we','say','her','she','or','an',
    'will','my','one','all','would','there','their','what','so','up',
    'out','if','about','who','get','which','go','me','when','make',
    'can','like','time','no','just','him','know','take','people','into',
    'year','your','good','some','could','them','see','other','than','then',
    'now','look','only','come','its','over','think','also','back','after',
    'use','two','how','our','work','first','well','way','even','new',
    'want','because','any','these','give','day','most','us','great','big',
    'man','world','life','hand','part','child','woman','place','case','week',
    'company','system','program','question','government','number','night','point','home','water',
    'room','mother','area','money','story','fact','month','lot','right','study',
    'book','eye','job','word','business','issue','side','kind','head','house',
    'service','friend','father','power','hour','game','line','end','member','law',
    'car','city','community','name','president','team','minute','idea','body','information',
    'back','parent','face','others','level','office','door','health','person','art',
    'war','history','party','result','morning','reason','research','girl','guy','moment',
    'air','teacher','force','education','boy','food','land','nature','girlfriend','boyfriend']

ml = minilm.encode(words, show_progress_bar=False)
mp = mpnet.encode(words, show_progress_bar=False)
ml_cpx = np.zeros((len(words), ml.shape[1]), dtype=np.complex128)
mp_cpx = np.zeros((len(words), mp.shape[1]), dtype=np.complex128)
for d in range(ml.shape[1]): ml_cpx[:, d] = hilbert(ml[:, d])
for d in range(mp.shape[1]): mp_cpx[:, d] = hilbert(mp[:, d])

# Phase structure
def phase_dist(emb_cpx):
    phases = np.angle(emb_cpx)
    N = phases.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        diff = phases[i] - phases
        D[i] = np.mean(np.abs(np.sin(diff)), axis=1)
    return D

# Magnitude structure
def mag_dist(emb_cpx):
    mags = np.abs(emb_cpx)
    N = mags.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = np.linalg.norm(mags[i] - mags, axis=1)
    return D

D_ml_phase = phase_dist(ml_cpx)
D_mp_phase = phase_dist(mp_cpx)
D_ml_mag = mag_dist(ml_cpx)
D_mp_mag = mag_dist(mp_cpx)

tri = np.triu_indices(len(words), k=1)

print('=' * 60)
print('Q34: PHASE vs MAGNITUDE CONVERGENCE')
print('MiniLM vs MPNet — implicate vs explicate order')
print('=' * 60)

r_phase = spearmanr(D_ml_phase[tri], D_mp_phase[tri])[0]
r_mag = spearmanr(D_ml_mag[tri], D_mp_mag[tri])[0]
r_combined = spearmanr(
    D_ml_phase[tri] * D_ml_mag[tri],
    D_mp_phase[tri] * D_mp_mag[tri]
)[0]

print(f'\n  Phase (implicate) convergence:  r = {r_phase:+.4f}')
print(f'  Magnitude (explicate) convergence: r = {r_mag:+.4f}')
print(f'  Combined (phase x mag) convergence: r = {r_combined:+.4f}')

gap = abs(r_phase) - abs(r_mag)
if gap > 0.05:
    print(f'\n  PHASE CONVERGES {abs(gap):.3f} STRONGER — implicate order is the shared geometry')
elif gap < -0.05:
    print(f'\n  MAGNITUDE CONVERGES STRONGER — explicate order is the shared geometry')
else:
    print(f'\n  Phase and magnitude converge at similar strength')
    print(f'  The Platonic form spans BOTH orders — phase is not more universal than magnitude')
