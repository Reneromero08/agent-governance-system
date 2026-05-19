"""Q28: Dropout recovery test — focused."""
import sys,math,random,torch,torch.nn as nn,torch.nn.functional as F
import numpy as np
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen")
import importlib.util
spec=importlib.util.spec_from_file_location("ne","THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen/native_eigen.py")
ne=importlib.util.module_from_spec(spec);spec.loader.exec_module(ne)
NativeEigen,load=ne.NativeEigen,ne.load
D="cuda" if torch.cuda.is_available() else "cpu";torch.manual_seed(0)

def pc(rw,iw):
    z=rw+1j*iw;z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12);z=z/np.abs(z+1e-15)
    n=min(200,len(z));np.random.seed(0);idx=np.random.choice(len(z),n,replace=False);zi=z[idx]
    H=np.zeros((n,n),dtype=np.complex128)
    for i in range(n):
        for j in range(i,n):v=np.conj(zi[i]).dot(zi[j]);H[i,j]=v;H[j,i]=np.conj(v)
    ev=np.linalg.eigvalsh(H);ev=np.maximum(ev,1e-15);ev/=ev.sum()
    return 1.0-(-np.sum(ev*np.log(ev+1e-15)))/math.log(n)

data,V=load(N=2000)
m=NativeEigen(V=V,d=2,L=2).to(D);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.01)
m.train()
for ep in range(5):
    for i in range(0,len(data),16):
        b=data[i:i+16]
        if not b:continue
        x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
        y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
        loss=F.cross_entropy(m(x).view(-1,V),y.view(-1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
rw=m.emb.re.weight.detach().cpu().numpy();iw=m.emb.im.weight.detach().cpu().numpy()
base=pc(rw,iw)
print(f"Baseline: {base:.4f}")

for sigma in [0.1,0.2,0.3,0.5]:
    m2=NativeEigen(V=V,d=2,L=2).to(D);m2.load_state_dict({k:v.clone() for k,v in m.state_dict().items()})
    with torch.no_grad():
        for p in m2.parameters():p.mul_((torch.rand_like(p)>sigma).float())
    rw2=m2.emb.re.weight.detach().cpu().numpy();iw2=m2.emb.im.weight.detach().cpu().numpy()
    pert=pc(rw2,iw2)
    opt2=torch.optim.AdamW(m2.parameters(),lr=1e-3,weight_decay=0.01);m2.train()
    for ep in range(10):
        for i in range(0,len(data),16):
            b=data[i:i+16]
            if not b:continue
            x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
            y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
            loss=F.cross_entropy(m2(x).view(-1,V),y.view(-1))
            opt2.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0);opt2.step()
    rwf=m2.emb.re.weight.detach().cpu().numpy();iwf=m2.emb.im.weight.detach().cpu().numpy()
    final=pc(rwf,iwf)
    pct=(final-pert)/(base-pert+1e-10)*100
    print(f"  drop={sigma:.1f}: {base:.4f} -> {pert:.4f} -> {final:.4f} ({pct:+.0f}pct)")

print(f"\nDropout creates BETTER attractor: {'YES' if final>base else 'NO'}")
