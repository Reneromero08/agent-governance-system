"""Q28: Attractor structure — import architecture cleanly, run attractor tests."""
import sys, math, random
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Import architecture from native_eigen.py without running main
sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ne", "THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen/native_eigen.py")
ne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ne)

NativeEigen = ne.NativeEigen
load = ne.load
torch.manual_seed(42); random.seed(42)

def phase_coh_from_weights(re_w, im_w):
    z = re_w + 1j*im_w
    z = z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    z = z/np.abs(z+1e-15)
    n = min(200, len(z))
    np.random.seed(0)
    idx = np.random.choice(len(z), n, replace=False)
    zi = z[idx]
    H = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(i,n):
            v=np.conj(zi[i]).dot(zi[j]); H[i,j]=v; H[j,i]=np.conj(v)
    ev = np.linalg.eigvalsh(H); ev=np.maximum(ev,1e-15); ev/=ev.sum()
    return 1.0-(-np.sum(ev*np.log(ev+1e-15)))/math.log(n)

D = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {D}")
data, V = load(N=2000)

# TEST 1: Basin convergence
print("\n=== ATTRACTOR TEST 1: Basin convergence (5 seeds) ===")
final_pcs = []
for seed in range(5):
    torch.manual_seed(seed); random.seed(seed)
    model = NativeEigen(V=V, d=2, L=2).to(D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    for ep in range(5):
        for i in range(0, len(data), 16):
            b = data[i:i+16]
            if not b: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
            loss = F.cross_entropy(model(x).view(-1,V), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    rw = model.emb.re.weight.detach().cpu().numpy()
    iw = model.emb.im.weight.detach().cpu().numpy()
    pc = phase_coh_from_weights(rw, iw)
    final_pcs.append(pc)
    print(f"  seed={seed}: {pc:.4f}")

fp = np.array(final_pcs)
cv = fp.std()/fp.mean()
print(f"  Mean={fp.mean():.4f} Std={fp.std():.4f} CV={cv:.2%}")
print(f"  {'CONVERGED (CV<2%)' if cv<0.02 else 'SPREAD'}")

# TEST 2: Perturbation recovery
print("\n=== ATTRACTOR TEST 2: Perturbation recovery ===")
torch.manual_seed(0)
model = NativeEigen(V=V, d=2, L=2).to(D)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()
for ep in range(5):
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        if not b: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x).view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

rw = model.emb.re.weight.detach().cpu().numpy()
iw = model.emb.im.weight.detach().cpu().numpy()
baseline = phase_coh_from_weights(rw, iw)
print(f"  Baseline: {baseline:.4f}")

for sigma in [0.01, 0.05, 0.1]:
    m2 = NativeEigen(V=V, d=2, L=2).to(D)
    m2.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})
    with torch.no_grad():
        for p in m2.parameters(): p.add_(torch.randn_like(p)*sigma)
    rw2 = m2.emb.re.weight.detach().cpu().numpy()
    iw2 = m2.emb.im.weight.detach().cpu().numpy()
    perturbed = phase_coh_from_weights(rw2, iw2)

    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01)
    m2.train()
    recovery = []
    for ep in range(3):
        for i in range(0, len(data), 16):
            b = data[i:i+16]
            if not b: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
            loss = F.cross_entropy(m2(x).view(-1,V), y.view(-1))
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0); opt2.step()
        rw2 = m2.emb.re.weight.detach().cpu().numpy()
        iw2 = m2.emb.im.weight.detach().cpu().numpy()
        recovery.append(phase_coh_from_weights(rw2, iw2))

    final = recovery[-1]
    pct = (final-perturbed)/(baseline-perturbed+1e-10)*100
    print(f"  sigma={sigma:.2f}: {baseline:.4f}->{perturbed:.4f}->{final:.4f} ({pct:.0f}% return)")

# TEST 3: Exponential convergence
print("\n=== ATTRACTOR TEST 3: Convergence rate ===")
torch.manual_seed(99)
model = NativeEigen(V=V, d=2, L=2).to(D)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
model.train()
pcs = []; steps = []; gs = 0
interval = max(1, len(data)//40)
for ep in range(5):
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        if not b: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x).view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        gs += 1
        if gs % interval == 0:
            rw = model.emb.re.weight.detach().cpu().numpy()
            iw = model.emb.im.weight.detach().cpu().numpy()
            pcs.append(phase_coh_from_weights(rw, iw)); steps.append(gs)

pcs=np.array(pcs); steps=np.array(steps)
def exp_decay(t,a,b,c): return a+b*np.exp(-c*t)
try:
    popt,_ = curve_fit(exp_decay, steps, pcs, p0=[pcs[-1],pcs[0]-pcs[-1],0.001], maxfev=5000)
    pred = exp_decay(steps, *popt)
    r2 = 1-np.sum((pcs-pred)**2)/np.sum((pcs-pcs.mean())**2)
    hl = math.log(2)/popt[2]
    print(f"  R^2={r2:.4f} half-life={hl:.0f} steps")
    print(f"  {'EXPONENTIAL CONVERGENCE (attractor)' if r2>0.9 else 'NOT EXPONENTIAL'}")
except Exception as e:
    print(f"  Fit failed: {e}")

print(f"\n{'='*60}")
print(f"Basins converge: {'YES' if cv<0.02 else 'NO'} | Recovery: {pct:.0f}% | Exponential: {r2:.4f}")
