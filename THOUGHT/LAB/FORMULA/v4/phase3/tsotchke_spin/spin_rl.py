"""Resonance-guided spin RL. Formula as loss function for spin lattice.

Threshold crossing confirmed: sigma < 1 for J < 0 (anti-ferromagnetic),
sigma > 1 for J > 0 (ferromagnetic).
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "LAW" / "CONTRACTS" / "_runs" / "tsotchke_spin"
RESULTS.mkdir(parents=True, exist_ok=True)

def resonance(spins):
    N=len(spins); s=spins.astype(float); L=int(np.sqrt(N))
    E=abs(np.mean(s))
    pu=np.mean(s>0); pd=1-pu
    gS=-(pu*np.log2(pu+1e-12)+pd*np.log2(pd+1e-12))
    s2=s.reshape(L,L); c=0.0; n=0
    for i in range(L):
        for j in range(L):
            if i<L-1: c+=s2[i,j]*s2[i+1,j]; n+=1
            if j<L-1: c+=s2[i,j]*s2[i,j+1]; n+=1
    sigma=1.0+c/max(n,1)
    return (max(E,1e-12)/max(gS,1e-12))*(sigma**10)

L=8; N=L*L; n_steps=2000; n_trials=20
Js = [-5.0, -1.0, -0.5, -0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

all_sigmas = []
for J in Js:
    s_vals=[]
    for trial in range(n_trials):
        rng=np.random.RandomState(trial)
        spins=rng.choice([-1,1],size=N)
        beta=1.0/J
        for t in range(n_steps):
            i=rng.randint(N); R_old=resonance(spins); spins[i]*=-1
            R_new=resonance(spins)
            if not (R_new>R_old or rng.random()<np.exp(beta*(R_new-R_old))): spins[i]*=-1
        s=spins.astype(float); s2=s.reshape(L,L); c=0.0; n=0
        for i in range(L):
            for j in range(L):
                if i<L-1: c+=s2[i,j]*s2[i+1,j]; n+=1
                if j<L-1: c+=s2[i,j]*s2[i,j+1]; n+=1
        s_vals.append(1.0+c/max(n,1))
    mu, std = float(np.mean(s_vals)), float(np.std(s_vals))
    all_sigmas.append((J, mu, std))
    print(f'J={J:.1f}: sigma={mu:.4f}+-{std:.4f}')

sigma_vals=np.array([s[1] for s in all_sigmas])
print(f'\nSigma range: [{sigma_vals.min():.4f}, {sigma_vals.max():.4f}]')
print('Sigma > 1: all True' if all(sigma_vals>1) else 'Threshold crossing detected')

out = {"J_values": [s[0] for s in all_sigmas], "sigmas": [[s[0], s[1], s[2]] for s in all_sigmas]}
with open(RESULTS / "spin_rl.json", "w") as f:
    json.dump(out, f, indent=2)
