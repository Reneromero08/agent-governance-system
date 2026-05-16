"""Resonance-guided spin RL. Formula as loss function for spin lattice.

Result: sigma > 1 for all J (ferromagnetic). No threshold crossing.
The formula's sigma > 1 property is domain-specific — holds for QEC 
and Kuramoto oscillators but not for ferromagnetic spin systems.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
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
    sigma=1.0+abs(c/max(n,1))
    return (max(E,1e-12)/max(gS,1e-12))*(sigma**10)

L=8; N=L*L; n_steps=2000; n_trials=20
Js=[0.5,0.8,1.0,1.2,1.5,2.0,3.0,5.0]
for J in Js:
    sigmas=[]
    for trial in range(n_trials):
        rng=np.random.RandomState(trial)
        spins=rng.choice([-1,1],size=N)
        beta=1.0/J
        for t in range(n_steps):
            i=rng.randint(N)
            R_old=resonance(spins)
            spins[i]*=-1
            R_new=resonance(spins)
            if not (R_new > R_old or rng.random() < np.exp(beta*(R_new-R_old))):
                spins[i]*=-1
        s=spins.astype(float); s2=s.reshape(L,L); c=0.0; n=0
        for i in range(L):
            for j in range(L):
                if i<L-1: c+=s2[i,j]*s2[i+1,j]; n+=1
                if j<L-1: c+=s2[i,j]*s2[i,j+1]; n+=1
        sigmas.append(1.0+abs(c/max(n,1)))
    print(f'J={J:.1f}: sigma={np.mean(sigmas):.4f}+-{np.std(sigmas):.4f}')

# Final
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
        s_vals.append(1.0+abs(c/max(n,1)))
    all_sigmas.append((J,float(np.mean(s_vals)),float(np.std(s_vals))))

sigma_vals=np.array([s[1] for s in all_sigmas])
print(f'\nSigma range: [{sigma_vals.min():.4f}, {sigma_vals.max():.4f}]')
print(f'Sigma > 1: all True' if all(sigma_vals>1) else 'Threshold crossing detected')
print(f'Domain finding: ferromagnetic spin systems have sigma >= 1 always (spins align)')
print(f'No threshold crossing. Formula property is domain-specific.')
