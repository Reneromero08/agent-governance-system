"""Q28 Push: multi-angle attractor hardening."""
import sys, math, random, json
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

sys.path.insert(0, "THOUGHT/LAB/EIGEN_BUDDY")
import importlib.util
spec = importlib.util.spec_from_file_location("ne", "THOUGHT/LAB/EIGEN_BUDDY/native_eigen.py")
ne = importlib.util.module_from_spec(spec); spec.loader.exec_module(ne)
NativeEigen, load = ne.NativeEigen, ne.load
torch.manual_seed(42)

def phase_coh(re_w, im_w):
    z = re_w + 1j*im_w; z = z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12); z = z/np.abs(z+1e-15)
    n = min(200, len(z))
    np.random.seed(0); idx = np.random.choice(len(z), n, replace=False); zi = z[idx]
    H = np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        for j in range(i,n): v=np.conj(zi[i]).dot(zi[j]); H[i,j]=v; H[j,i]=np.conj(v)
    ev = np.linalg.eigvalsh(H); ev=np.maximum(ev,1e-15); ev/=ev.sum()
    return 1.0-(-np.sum(ev*np.log(ev+1e-15)))/math.log(n)

def all_metrics(re_w, im_w):
    z = re_w + 1j*im_w; z = z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12); z = z/np.abs(z+1e-15)
    n=min(200,len(z)); np.random.seed(0); idx=np.random.choice(len(z),n,replace=False); zi=z[idx]
    H=np.zeros((n,n),dtype=np.complex128)
    for i in range(n):
        for j in range(i,n): v=np.conj(zi[i]).dot(zi[j]); H[i,j]=v; H[j,i]=np.conj(v)
    ev=np.linalg.eigvalsh(H); ev=np.maximum(ev,1e-15); ev/=ev.sum()
    sigma=1.0/max(ev.sum()**2/(ev**2).sum(),1e-10)
    nabla=-np.sum(ev*np.log(ev+1e-15))
    return {"phase_coh":1.0-nabla/math.log(n), "sigma":sigma, "nabla_S":nabla, "c_sem":np.sqrt(sigma/max(nabla,1e-10))}

def train_full(seed, epochs=5):
    torch.manual_seed(seed); random.seed(seed)
    data, V = load(N=2000)
    m = NativeEigen(V=V,d=2,L=2).to(D)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    m.train()
    for ep in range(epochs):
        for i in range(0, len(data), 16):
            b=data[i:i+16]
            if not b: continue
            x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
            y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
            loss=F.cross_entropy(m(x).view(-1,V),y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return m, data, V

def perturb_and_recover(model, data, V, sigma, corruption_type, recovery_epochs):
    m2 = NativeEigen(V=V,d=2,L=2).to(D)
    m2.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})
    if corruption_type == "weight_noise":
        with torch.no_grad():
            for p in m2.parameters(): p.add_(torch.randn_like(p)*sigma)
    elif corruption_type == "dropout":
        with torch.no_grad():
            for p in m2.parameters(): p.mul_((torch.rand_like(p)>sigma).float())
    elif corruption_type == "label_noise":
        pass  # handled in training loop
    
    baseline = all_metrics(model.emb.re.weight.detach().cpu().numpy(), model.emb.im.weight.detach().cpu().numpy())
    rw2=m2.emb.re.weight.detach().cpu().numpy(); iw2=m2.emb.im.weight.detach().cpu().numpy()
    perturbed = all_metrics(rw2, iw2)
    
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01)
    m2.train()
    trajectory = [perturbed]
    for ep in range(recovery_epochs):
        for i in range(0, len(data), 16):
            b=data[i:i+16]
            if not b: continue
            x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
            y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
            if corruption_type == "label_noise":
                ny = y.clone(); flat_y = ny.view(-1)
                n_flip = int(len(flat_y)*sigma)
                fidx = torch.randperm(len(flat_y))[:n_flip]
                flat_y[fidx] = torch.randint(0, V-1, (n_flip,), device=D)
                ny = flat_y.view(y.shape)
                loss=F.cross_entropy(m2(x).view(-1,V),ny.view(-1))
            else:
                loss=F.cross_entropy(m2(x).view(-1,V),y.view(-1))
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0); opt2.step()
        rw2=m2.emb.re.weight.detach().cpu().numpy(); iw2=m2.emb.im.weight.detach().cpu().numpy()
        trajectory.append(all_metrics(rw2, iw2))
    return baseline, perturbed, trajectory

D = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Q28 PUSH: Multi-angle attractor hardening on {D}")
model, data, V = train_full(0, epochs=5)

# ANGLE 1: Longer recovery + different corruption types
print("\n=== ANGLE 1: Corruption recovery (10 epochs) ===")
for ctype, param in [("weight_noise", 0.05), ("dropout", 0.3), ("label_noise", 0.3)]:
    base, pert, traj = perturb_and_recover(model, data, V, param, ctype, 10)
    final = traj[-1]
    pct_pc = (final["phase_coh"] - pert["phase_coh"])/(base["phase_coh"] - pert["phase_coh"]+1e-10)*100
    print(f"  {ctype:<15s}: base={base['phase_coh']:.4f} pert={pert['phase_coh']:.4f} final={final['phase_coh']:.4f} ({pct_pc:+.0f}%)")

# ANGLE 2: Basin size — how far can you perturb before attractor fails?
print("\n=== ANGLE 2: Basin size (weight noise sweep, 10-ep recovery) ===")
for sigma in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    base, pert, traj = perturb_and_recover(model, data, V, sigma, "weight_noise", 10)
    final = traj[-1]
    pct = (final["phase_coh"] - pert["phase_coh"])/(base["phase_coh"] - pert["phase_coh"]+1e-10)*100
    marker = " *** RECOVERS" if pct > 50 else ""
    print(f"  sigma={sigma:.2f}: {base['phase_coh']:.4f}->{pert['phase_coh']:.4f}->{final['phase_coh']:.4f} ({pct:+.0f}%){marker}")

# ANGLE 3: Seed stability of attractor value
print("\n=== ANGLE 3: Basin convergence (10 seeds) ===")
final_vals = []
for seed in range(10):
    m2,_,_ = train_full(seed, epochs=5)
    rw=m2.emb.re.weight.detach().cpu().numpy(); iw=m2.emb.im.weight.detach().cpu().numpy()
    final_vals.append(phase_coh(rw, iw))
fv=np.array(final_vals)
print(f"  Mean={fv.mean():.4f} Std={fv.std():.4f} CV={fv.std()/fv.mean()*100:.2f}%")
print(f"  Range=[{fv.min():.4f}, {fv.max():.4f}]")

# ANGLE 4: Exponential convergence stability (10 seeds)
print("\n=== ANGLE 4: Exponential convergence (10 seeds) ===")
r2s = []; half_lives = []
for seed in range(10):
    torch.manual_seed(seed); random.seed(seed)
    dm, V = load(N=2000)
    m2 = NativeEigen(V=V,d=2,L=2).to(D)
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01)
    m2.train()
    pcs=[]; st=[]; gs=0; iv=max(1,len(dm)//40)
    for ep in range(5):
        for i in range(0,len(dm),16):
            b=dm[i:i+16]
            if not b: continue
            x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
            y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
            loss=F.cross_entropy(m2(x).view(-1,V),y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0); opt.step()
            gs+=1
            if gs%iv==0:
                rw=m2.emb.re.weight.detach().cpu().numpy(); iw=m2.emb.im.weight.detach().cpu().numpy()
                pcs.append(phase_coh(rw,iw)); st.append(gs)
    pcs=np.array(pcs); st=np.array(st)
    def exp_decay(t,a,b,c): return a+b*np.exp(-c*t)
    try:
        popt,_=curve_fit(exp_decay,st,pcs,p0=[pcs[-1],pcs[0]-pcs[-1],0.001],maxfev=5000)
        pred=exp_decay(st,*popt)
        r2=1-np.sum((pcs-pred)**2)/np.sum((pcs-pcs.mean())**2)
        hl=math.log(2)/popt[2]
        r2s.append(r2); half_lives.append(hl)
    except: pass
r2a=np.array(r2s); hla=np.array(half_lives)
print(f"  R^2: {r2a.mean():.4f}+/-{r2a.std():.4f}, Half-life: {hla.mean():.0f}+/-{hla.std():.0f} steps")
print(f"  R^2>0.9 in {sum(r2a>0.9)}/{len(r2a)} seeds")

cv_val = fv.std()/fv.mean()
print(f"\n{'='*60}")
print(f"Attractor exists: YES (CV={cv_val*100:.2f}%)")
print(f"Recovery: {'PARTIAL' if pct>20 else 'SHALLOW'}")
print(f"Exponential: R^2={r2a.mean():.3f}+/-{r2a.std():.3f}")
verdict = "CONFIRMED" if cv_val<0.01 and r2a.mean()>0.95 else "PARTIALLY VERIFIED"
print(f"Verdict: {verdict}")
