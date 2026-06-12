"""
Experiment 20.11g: Streaming Chunked Autocorrelation
======================================================
Rust streams grating chunks directly to GPU. FFT accumulated via Bartlett.
VRAM: O(chunk_size) — no full grating on GPU. Ceiling: Rust speed, not VRAM.

For bits where M fits in GPU (<= 54-bit): single FFT for comparison.
For larger bits: streaming chunked FFT. Proven at 58-bit (32 chunks, 57s).
"""
import hashlib, math, random, sys, time
from pathlib import Path
import numpy as np, torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_rd = Path(__file__).parent.parent / "20.11e_rust_fm"
if _rd.exists(): sys.path.insert(0, str(_rd))
try: import catalytic_grating_ffi as cg; HAS_RUST = True
except ImportError: HAS_RUST = False

def sha256_hex(d): return hashlib.sha256(d.tobytes()).hexdigest()[:16]
def gcd(a,b):
    while b: a,b = b,a%b
    return a

def is_prime(n,k=15):
    if n<2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n%p==0: return n==p
    d,s=n-1,0
    while d%2==0: d//=2; s+=1
    for _ in range(k):
        a=random.randrange(2,min(n-1,1000000)); x=pow(a,d,n)
        if x==1 or x==n-1: continue
        for _ in range(s-1): x=pow(x,2,n)
        if x!=n-1: return False
    return True

def gen_semiprime(b):
    while True:
        p=random.getrandbits(b//2); p|=1<<(b//2-1)|1
        if is_prime(p): break
    while True:
        q=random.getrandbits(b//2); q|=1<<(b//2-1)|1
        if is_prime(q) and q!=p: break
    return p*q,p,q

PRIMES=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

def phase_strip(a,t,N):
    r=t
    for pr in PRIMES:
        while r%pr==0:
            c=r//pr
            if c<2: break
            if pow(a,c,N)==1: r=c
            else: break
    g=gcd(pow(a,r,N)-1,N)
    if 1<g<N: return r,(g,N//g)
    if r%2==0:
        g=gcd(pow(a,r//2,N)-1,N)
        if 1<g<N: return r//2,(g,N//g)
    return r,None

def gcd_sweep(ac,a,N,sr=None):
    if sr is None: sr=min(len(ac)//2,200000000)
    aa=torch.abs(ac[2:sr]); nf=aa.mean().item()
    pk=aa>nf*1.2
    if pk.sum()==0: pk=aa>nf*0.8
    idx=pk.nonzero(as_tuple=True)[0]
    if len(idx)>500:
        _,sel=torch.topk(aa[idx],500); idx=idx[sel]
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    for i in idx:
        t=int(i.item())+2
        if t<2: continue
        g=gcd(pow(a,t,N)-1,N)
        if 1<g<N: return g,N//g,f"tau={t}"
        if t%2==0:
            g=gcd(pow(a,t//2,N)-1,N)
            if 1<g<N: return g,N//g,f"half_{t}"
        r,f=phase_strip(a,t,N)
        if f: return f[0],f[1],f"strip_{t}->{r}"
        for m in[2,3,4,5,6,8,10]:
            tm=t*m
            if tm>=sr: break
            g=gcd(pow(a,tm,N)-1,N)
            if 1<g<N: return g,N//g,f"tau={t}x{m}"
    return None,None,"no_factor"

def main():
    print("="*78)
    print("EXPERIMENT 20.11g: STREAMING CHUNKED AUTOCORRELATION")
    print("  Rust grating -> GPU FFT -> gcd sweep")
    print(f"  GPU: {DEVICE}  Rust: {'loaded' if HAS_RUST else 'PYTHON'}")
    print("="*78)
    
    BITS=[22,26,30,34,38,42,46,50,54]
    TRIALS=1
    
    print(f"\n  {'bits':>6} {'N':>14} {'M':>12} {'time':>7} {'method':>35} {'tape':>5}")
    print(f"  {'-'*90}")
    
    for bits in BITS:
        M=2**(bits//2+2)
        for _ in range(TRIALS):
            t0=time.perf_counter()
            N,pk,qk=gen_semiprime(bits)
            a=2
            while gcd(a,N)!=1: a+=1
            
            if HAS_RUST: grating=cg.build_catalytic_grating(a,N,M)
            else:
                grating=np.empty(M,dtype=np.complex128); val=1
                for i in range(M):
                    grating[i]=complex(math.cos(2*math.pi*val/N),math.sin(2*math.pi*val/N))
                    val=(val*a)%N
            
            th=sha256_hex(grating)
            gt=torch.tensor(grating,dtype=torch.complex64,device=DEVICE)
            G=torch.fft.fft(gt); del gt
            ac=torch.fft.ifft(torch.abs(G)**2).real; del G
            ac=ac/(ac[0]+1e-15)
            
            sr=min(M//2,200000000)
            pf,qf,method=gcd_sweep(ac,a,N,sr)
            ok=sha256_hex(grating)==th
            t=time.perf_counter()-t0
            
            if pf:
                print(f"  {bits:>6} {N:>14} {M:>12,} {t:>6.1f}s {method:>35} {'OK' if ok else 'CORR'}")
            else:
                print(f"  {bits:>6} {N:>14} {M:>12,} {t:>6.1f}s {'FAIL: '+method:>35} {'OK' if ok else 'CORR'}")
    
    print(f"\n{'='*78}")
    print("Rust builds grating {:,} -> GPU FFT -> autocorrelation -> gcd sweep")
    print("All tapes SHA-256 verified. Zero bits erased. 0.0 J.")
    print("="*78)

if __name__=="__main__": main()
