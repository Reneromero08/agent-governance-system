"""
20.11k: SATURATED — Max Out GPU, CPU, RAM
============================================
Rust builds full grating in parallel (rayon). GPU FFT on largest
chunk that fits in VRAM. Multi-base strategy. Resource monitoring.

Strategy by bit size:
  <= 56-bit: Full Rust grating (parallel, fits in CPU RAM)
  >  56-bit: Rust chunked streaming
  GPU FFT: largest single chunk that fits in VRAM, else Bartlett streaming
"""
import hashlib, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch
import psutil

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_rd=Path(__file__).parent.parent/"20.11e_rust_fm"
if _rd.exists(): sys.path.insert(0,str(_rd))
try: import catalytic_grating_ffi as cg; HAS_RUST=True
except ImportError: HAS_RUST=False

PROC=psutil.Process()

def resources():
    cpu=psutil.cpu_percent()
    mem=psutil.virtual_memory()
    try: gpu=torch.cuda.utilization() if DEVICE.type=="cuda" else 0
    except: gpu=0
    return f"CPU:{cpu:>3}% RAM:{mem.percent:>3}% GPU:{gpu:>3}%"

def gcd(a,b):
    while b: a,b=b,a%b
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

def gpu_autocorr(grating):
    """Single-shot GPU FFT autocorrelation. Grating must fit in GPU VRAM."""
    M=len(grating)
    gt=torch.tensor(grating,dtype=torch.complex64,device=DEVICE)
    G=torch.fft.fft(gt); del gt
    ac=torch.fft.ifft(torch.abs(G)**2).real; del G
    ac=ac/(ac[0]+1e-15)
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    return ac

def gpu_chunked_autocorr(grating,chunk_size):
    """Chunked Bartlett FFT for large gratings. Grating is CPU numpy array."""
    M=len(grating); num=(M+chunk_size-1)//chunk_size
    accumulated=None
    for ci in range(num):
        s=ci*chunk_size; e=min(s+chunk_size,M)
        chunk=grating[s:e]
        ct=torch.tensor(chunk,dtype=torch.complex64,device=DEVICE)
        G=torch.fft.fft(ct); pc=torch.abs(G)**2; del ct,G
        if accumulated is None: accumulated=pc
        else:
            if pc.shape[0]<accumulated.shape[0]:
                p=torch.zeros_like(accumulated); p[:pc.shape[0]]=pc; accumulated+=p
            else: accumulated+=pc
        del pc
    accumulated/=num
    ac=torch.fft.ifft(accumulated).real; del accumulated
    ac=ac/(ac[0]+1e-15)
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    return ac

def gcd_sweep(ac,a,N,sr):
    aa=torch.abs(ac[2:sr]); nf=aa.mean().item()
    pk=aa>nf*1.2
    if pk.sum()==0: pk=aa>nf*0.8
    idx=pk.nonzero(as_tuple=True)[0]
    if len(idx)>500: _,sel=torch.topk(aa[idx],500); idx=idx[sel]
    for i in idx:
        t=int(i.item())+2
        if t<2: continue
        g=gcd(pow(a,t,N)-1,N)
        if 1<g<N: return g,N//g,f"tau={t}"
        if t%2==0:
            g=gcd(pow(a,t//2,N)-1,N)
            if 1<g<N: return g,N//g,f"half_{t}"
        for m in[2,3,4,5,6,8,10]:
            tm=t*m
            if tm>=sr: break
            g=gcd(pow(a,tm,N)-1,N)
            if 1<g<N: return g,N//g,f"tau={t}x{m}"
    return None,None,"no_factor"

BASES=[2,3,5,7,11,13,17,19,23,29]
GPU_CHUNK=200*1024*1024  # 200M complex64 = 1.6 GB, fits with FFT workspace

def main():
    print("="*78)
    print("20.11k: SATURATED — Max Out GPU, CPU, RAM")
    print(f"  Rust: {'parallel rayon' if HAS_RUST else 'Python fallback'}")
    print(f"  GPU: {DEVICE}, GPU chunk: {GPU_CHUNK//1e6:.0f}M")
    print(f"  Bases: {BASES}  |  {resources()}")
    print("="*78)
    
    BITS=[50,52,54,56,58]
    
    for bits in BITS:
        M=2**(bits//2+2)
        grating_mb=M*8/1e6  # complex64 = 8 bytes
        N,pk,qk=gen_semiprime(bits)
        print(f"\n  [{bits}-bit] N={N}  M={M:,}  grating={grating_mb:.0f}MB  {resources()}")
        
        factored=False
        for ai,a in enumerate(BASES):
            if gcd(a,N)!=1: continue
            t0=time.perf_counter()
            
            try:
                # Build grating in Rust (full, parallel)
                if HAS_RUST:
                    grating=cg.build_catalytic_grating(a,N,M)
                else:
                    grating=np.empty(M,dtype=np.complex128); val=1
                    for i in range(M):
                        grating[i]=complex(math.cos(2*math.pi*val/N),math.sin(2*math.pi*val/N))
                        val=(val*a)%N
                build_t=time.perf_counter()-t0
                
                # GPU FFT: single-shot if fits, else chunked
                if M<=GPU_CHUNK:
                    ac=gpu_autocorr(grating)
                else:
                    ac=gpu_chunked_autocorr(grating,GPU_CHUNK)
                fft_t=time.perf_counter()-t0-build_t
                
                # gcd sweep
                sr=min(M//2,200000000)
                pf,qf,method=gcd_sweep(ac,a,N,sr)
                elapsed=time.perf_counter()-t0
                
                if pf and qf and pf*qf==N:
                    chunks=(M+GPU_CHUNK-1)//GPU_CHUNK
                    print(f"  base={a:>3} {method:>30} build={build_t:.1f}s fft={fft_t:.1f}s total={elapsed:.1f}s {resources()}")
                    factored=True
                    break
                else:
                    print(f"  base={a:>3} no_factor ({elapsed:.1f}s)",end="")
                    if ai<3: print(f" {resources()}")
                    else: print()
            except Exception as e:
                print(f"  base={a:>3} ERROR: {e} {resources()}")
                break
        
        if not factored:
            print(f"  ALL BASES EXHAUSTED {resources()}")
    
    print(f"\n{'='*78}")
    print(f"  WALL: {resources()}")
    print(f"  RAM: {psutil.virtual_memory().total/1e9:.1f} GB total")
    print(f"  The wall is where Rust OOMs or GPU FFT exceeds VRAM")
    print("="*78)

if __name__=="__main__":
    main()
