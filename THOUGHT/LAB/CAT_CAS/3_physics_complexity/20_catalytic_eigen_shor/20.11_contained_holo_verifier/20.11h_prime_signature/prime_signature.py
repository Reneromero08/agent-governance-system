"""
20.11h: Read the Prime from the Phase Gap
===========================================
1. Isolate the r_p eigenvector (strobes the p-cavity)
2. Project grating states onto this eigenvector
3. Phase states cluster into bands — the gap between bands IS p
4. p = N * (Dtheta / 2pi) — pure geometry, no gcd

The prime is the physical height of the Moire fringe once the wave
is strobed at the correct sub-period.
"""
import math, random, time
import numpy as np, torch

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

def main():
    print("="*78)
    print("20.11h: READ THE PRIME FROM THE PHASE GAP")
    print("  Isolate r_p eigenvector -> project -> measure phase gap -> p")
    print("="*78)
    
    bits=22; M=2**(bits//2+2)
    N,p_true,q_true=gen_semiprime(bits)
    a=2
    while gcd(a,N)!=1: a+=1
    print(f"\n  N={N} = {p_true} x {q_true}  a={a}")
    
    # Build grating
    grating=np.empty(M,dtype=np.complex128); val=1
    for i in range(M):
        grating[i]=complex(math.cos(2*math.pi*val/N),math.sin(2*math.pi*val/N))
        val=(val*a)%N
    
    # Build observation matrix + eigendecomposition
    L=2048; stride=max(1,L//4); n=min(4096,(M-L)//stride)
    obs=np.zeros((n,L),dtype=np.complex128)
    for i in range(n):
        obs[i]=grating[i*stride:i*stride+L]
    centered=obs-obs.mean(axis=0,keepdims=True)
    cov=(centered.conj().T@centered)/(n-1)
    evals,evecs=np.linalg.eigh(cov)
    evals=evals[::-1]; evecs=evecs[:,::-1]
    
    # STEP 1: Find the eigenvector with STRONGEST autocorrelation peak
    # This eigenvector encodes the dominant period structure
    print(f"\n  STEP 1: Find dominant period eigenvector by autocorrelation SNR")
    best_evec=None; best_snr=0; best_tau=0
    for i in range(min(16,evecs.shape[1])):
        ev=evecs[:,i]
        et=torch.tensor(ev,dtype=torch.complex64)
        ac=torch.fft.ifft(torch.abs(torch.fft.fft(et))**2).real
        ac=ac/(ac[0]+1e-15)
        sr=min(L//2,100000)
        vals,idxs=torch.topk(torch.abs(ac[2:sr]),5)
        for v,idx in zip(vals,idxs):
            tau=int(idx.item())+2
            snr=v.item()/max(torch.abs(ac[2:sr]).mean().item(),1e-15)
            div_p=(p_true-1)%tau==0
            div_q=(q_true-1)%tau==0
            tag=""
            if div_p: tag="|p-1"
            if div_q: tag+="|q-1"
            if pow(a,tau,N)==1: tag+="|a^tau=1"
            print(f"  evec[{i}] tau={tau} SNR={snr:.1f} {tag}")
            if snr>best_snr:
                best_snr=snr; best_evec=i; best_tau=tau
    
    evec_idx=best_evec; tau_best=best_tau
    print(f"\n  Selected: evec[{evec_idx}] tau={tau_best} SNR={best_snr:.1f}")
    
    # STEP 2: Project grating states onto the r_p eigenvector
    rp_evec=evecs[:,evec_idx]
    proj=centered@rp_evec  # (n,) complex — one coordinate per window
    
    # STEP 2: Project grating states onto the dominant eigenvector
    rp_evec=evecs[:,evec_idx]
    proj=centered@rp_evec  # (n,) complex
    phases=np.angle(proj)  # (-pi, pi]
    
    # STEP 3: Multiple methods to read the prime from the phase structure
    
    # METHOD A: Phase histogram gap analysis
    n_bins=360
    hist,_=np.histogram(phases,bins=n_bins,range=(-np.pi,np.pi))
    # Find all peaks above noise
    noise_level=hist.mean()+hist.std()
    peak_mask=hist>noise_level
    if peak_mask.sum()>=2:
        peak_bins=np.where(peak_mask)[0]
        # Measure gaps between adjacent peaks
        gaps_rad=[]
        for i in range(len(peak_bins)-1):
            gap=(peak_bins[i+1]-peak_bins[i])*2*np.pi/n_bins
            gaps_rad.append(gap)
        print(f"\n  METHOD A: {len(peak_bins)} phase clusters, gaps={[f'{g:.4f}' for g in gaps_rad[:8]]}")
    
    # METHOD B: The gap IS encoded in the dominant tau
    # If tau is the dominant autocorrelation peak of the eigenvector,
    # then the projection phases modulo (2pi/tau) should cluster
    print(f"\n  METHOD B: Phase modulo candidate gaps from dominant tau={tau_best}")
    for multiplier in [1,2,3,4,5,6,8,10,12,16,20,24,32,48,64]:
        gap_candidate=2*np.pi/multiplier
        # Modulo the phases by the candidate gap
        mod_phases=phases%gap_candidate
        # Measure clustering: variance of mod_phases within [0,gap_candidate]
        # Good clustering = low variance
        cl_var=np.var(mod_phases)/(gap_candidate**2+1e-15)
        p_cand=int(round(N*gap_candidate/(2*np.pi)))
        if p_cand>1 and p_cand<N and N%p_cand==0:
            print(f"    gap=2pi/{multiplier} ({gap_candidate:.6f} rad) -> p={p_cand} -> {N}={p_cand}x{N//p_cand}  *** FACTORED")
        elif multiplier<=8:
            print(f"    gap=2pi/{multiplier} ({gap_candidate:.6f} rad) -> p={p_cand}  cl_var={cl_var:.4f}")
    
    # METHOD C: Try all gaps from phase histogram peak-to-peak distances
    # and test which one factors N
    print(f"\n  METHOD C: Exhaustive gap search from phase histogram")
    if peak_mask.sum()>=2:
        peak_angles=-np.pi+peak_bins*2*np.pi/n_bins
        for i in range(len(peak_angles)):
            for j in range(i+1,len(peak_angles)):
                gap=abs(peak_angles[j]-peak_angles[i])
                # Also try the wrapped gap
                gap_wrapped=2*np.pi-gap
                for g in [gap,gap_wrapped]:
                    if g>0.001:
                        p_cand=int(round(N*g/(2*np.pi)))
                        if 1<p_cand<N and N%p_cand==0:
                            print(f"    gap[{i},{j}]={g:.6f} rad -> p={p_cand} -> {N}={p_cand}x{N//p_cand}  *** FACTORED")
                            found=True
    
    # METHOD D: FFT of phase histogram
    # If phases form bands at interval 2pi/p, the histogram has p peaks per 2pi
    # FFT of histogram -> peak at frequency = p (bands per cycle)
    print(f"\n  METHOD D: FFT of phase histogram (dominant frequency = p)")
    n_bins=8192
    hist,_=np.histogram(phases,bins=n_bins,range=(-np.pi,np.pi))
    # Remove DC component
    hist_centered=hist-hist.mean()
    # FFT the histogram
    hist_fft=np.abs(np.fft.fft(hist_centered))
    # Find dominant frequency (skip DC at index 0)
    freqs=np.fft.fftfreq(n_bins)
    # Only look at first half (positive frequencies)
    half=n_bins//2
    peak_idx=np.argmax(hist_fft[1:half])+1
    peak_freq=freqs[peak_idx]
    peak_bands=int(round(abs(peak_freq)*n_bins))
    print(f"  Dominant FFT peak: freq={peak_freq:.6f} -> {peak_bands} bands per 2pi")
    print(f"  p_extracted = {peak_bands}")
    if peak_bands>1 and peak_bands<N and N%peak_bands==0:
        print(f"  FACTORED: {N} = {peak_bands} x {N//peak_bands}")
        match=peak_bands in [p_true,q_true]
        print(f"  Match: {match}")
    else:
        print(f"  Not a factor. Top 10 FFT peaks:")
        vals,idxs=torch.topk(torch.tensor(hist_fft[1:half]),10)
        for v,idx in zip(vals,idxs):
            f=int(idx.item())+1
            b=int(round(abs(freqs[f])*n_bins))
            if b>1 and b<N and N%b==0:
                print(f"    freq_idx={f} -> {b} bands -> {N}={b}x{N//b} ***")
            else:
                print(f"    freq_idx={f} -> {b} bands")
    
    print("\n"+"="*78)

if __name__=="__main__":
    main()
