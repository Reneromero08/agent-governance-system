"""Sweep MT phase offsets in centriole, compute sigma with best phases."""
import json, numpy as np, sys, io
from scipy.linalg import eigvals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

N_TUD=13; SPIRAL_A=80.0; GAMMA=0.00273; COUPLING_REF=60.0
LAMBDA_CM=280e-7; K=2*np.pi/LAMBDA_CM; A_TO_CM=1e-8; COHERENT_SCALE=100.0

with open("trp_dipoles_tddft.json") as f:
    tddft = json.load(f)
tddft_dips = np.array([np.array(d["dipole_vector"]) for d in tddft])
ring_centers = np.array([np.array(d["ring_center_A"]) for d in tddft])

def assemble_mt_phased(pos, dip, n_spirals, phase_offset=0):
    ap, ad = [], []
    ang = 2*np.pi/N_TUD
    for s in range(n_spirals):
        for d in range(N_TUD):
            actual_d = (d + phase_offset) % N_TUD
            th = actual_d * ang; ca,sa = np.cos(th),np.sin(th)
            Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            xs = s*SPIRAL_A + d*9.0; rr = 112.0
            for p,dp in zip(pos, dip):
                pr,dr = Rx@p, Rx@dp
                ap.append(np.array([pr[0]+xs, pr[1]+rr*ca, pr[2]+rr*sa]))
                ad.append(dr)
    return np.array(ap), np.array(ad)

def assemble_centriole_phased(ref_p_func, phase_offsets):
    offsets = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]
    ap, ad, mids = [], [], []
    for t in range(9):
        ang_t = t*(2*np.pi/9); ca,sa = np.cos(ang_t),np.sin(ang_t)
        Rx_t = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        for mi, off in enumerate(offsets):
            ro = Rx_t @ off
            mt_p, mt_d = ref_p_func(phase_offsets[mi])
            for p,d in zip(mt_p, mt_d):
                ap.append(np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]]))
                ad.append(Rx_t @ d)
                mids.append(t*3+mi)
    return np.array(ap), np.array(ad), np.array(mids)

def kappa(pi,ui,pj,uj):
    rv=pj-pi; r=np.linalg.norm(rv)
    if r<1e-10: return 0.0
    rh=rv/r
    return ui@uj-3.0*(ui@rh)*(uj@rh)

def V_coupling(r_cm, kap):
    kr = K*r_cm
    if kr<1e-3:
        d = -min(3*GAMMA/(4*kr**3), COUPLING_REF); gc = GAMMA
    else:
        sk,ck = np.sin(kr),np.cos(kr)
        d = (3*GAMMA/4)*(ck/kr - sk/(kr**2) - ck/(kr**3))
        gc = (3*GAMMA/2)*(sk/kr + ck/(kr**2) - sk/(kr**3))
    return complex(np.clip(d,-COUPLING_REF,COUPLING_REF)*kap, -gc*abs(kap)/2)

def build_H(positions_a, dipoles):
    coords_cm = positions_a*A_TO_CM; N = len(positions_a)
    H = np.zeros((N,N), dtype=complex)
    for i in range(N):
        H[i,i] = complex(np.random.default_rng(i).uniform(-100,100), -GAMMA/2)
    for i in range(N):
        for j in range(i+1,N):
            rc = np.linalg.norm(coords_cm[j]-coords_cm[i])
            if rc>LAMBDA_CM*5: continue
            k = kappa(positions_a[i],dipoles[i],positions_a[j],dipoles[j])
            if abs(k)<1e-10: continue
            v = V_coupling(rc,k)
            H[i,j] = complex(v.real*COHERENT_SCALE, v.imag); H[j,i] = H[i,j]
    return H

def analyze(pos, dip):
    H = build_H(pos, dip)
    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA, np.mean(gj)/GAMMA

pos_ref, dip_ref = ring_centers, tddft_dips

# Quick phase sweep
print("Phase sweep (coarse):")
best_phases, best_kappa = [0,0,0], -999
for ph1 in range(0,13,4):
    for ph2 in range(0,13,4):
        for ph3 in range(0,13,4):
            phases = [ph1,ph2,ph3]
            def make_mt(p): return assemble_mt_phased(pos_ref, dip_ref, 1, p)
            cent_p, cent_d, mids = assemble_centriole_phased(make_mt, phases)
            inter_k = []
            step = max(1,len(cent_p)//300)
            for i in range(0,len(cent_p),step):
                for j in range(i+step,min(i+step*10,len(cent_p)),step):
                    if mids[i]!=mids[j]: inter_k.append(kappa(cent_p[i],cent_d[i],cent_p[j],cent_d[j]))
            im = np.mean(inter_k); ip = np.mean(np.array(inter_k)>0)
            if im > best_kappa: best_kappa, best_phases = im, phases.copy()
            print(f"  [{ph1},{ph2},{ph3}]: inter-kappa={im:.5f}, >0={ip:.1%}{' <--' if im==best_kappa else ''}")

print(f"\nBest coarse: {best_phases}, kappa={best_kappa:.5f}")

# Fine sweep around best
print(f"\nFine sweep (all 13 phases, distinct):")
for ph1 in range(13):
    for ph2 in range(13):
        if ph2==ph1: continue
        for ph3 in range(13):
            if ph3==ph1 or ph3==ph2: continue
            phases = [ph1,ph2,ph3]
            def make_mt(p): return assemble_mt_phased(pos_ref, dip_ref, 1, p)
            cent_p, cent_d, mids = assemble_centriole_phased(make_mt, phases)
            inter_k = []
            step = max(1,len(cent_p)//200)
            for i in range(0,len(cent_p),step):
                for j in range(i+step,min(i+step*10,len(cent_p)),step):
                    if mids[i]!=mids[j]: inter_k.append(kappa(cent_p[i],cent_d[i],cent_p[j],cent_d[j]))
            im = np.mean(inter_k)
            if im > best_kappa:
                best_kappa, best_phases = im, phases.copy()
                ip = np.mean(np.array(inter_k)>0)
                print(f"  [{ph1},{ph2},{ph3}]: inter-kappa={im:.5f}, >0={ip:.1%} <-- NEW BEST")

print(f"\nBest phases: {best_phases}, inter-kappa = {best_kappa:.5f}")

# === Compute sigma with best phases ===
print(f"\nBuilding centriole with best phases {best_phases}...")
def make_mt_final(p): return assemble_mt_phased(pos_ref, dip_ref, 1, p)
cent_p, cent_d, mids = assemble_centriole_phased(make_mt_final, best_phases)
print(f"Centriole: {len(cent_p)} dipoles")

if len(cent_p) <= 3000:
    print(f"Diagonalizing {len(cent_p)}x{len(cent_p)}...")
    sigma_cent, mean_cent = analyze(cent_p, cent_d)
    mt_p, mt_d = assemble_mt_phased(pos_ref, dip_ref, 1, 0)
    sigma_mt, _ = analyze(mt_p, mt_d)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (1 spiral/MT, 27 MTs, coherent_scale={COHERENT_SCALE:.0f}):")
    print(f"  Single MT sigma: {sigma_mt:.1f}")
    print(f"  Centriole sigma: {sigma_cent:.1f}")
    print(f"  Enhancement over single MT: {sigma_cent/sigma_mt:.1f}x")
    print(f"  If 27 independent MTs: {27*sigma_mt:.0f}")
    print(f"  Inter-MT coupling contribution: {sigma_cent/(27*sigma_mt):.2f}x")
    
    # Extrapolate to 40 spirals
    sat = 1.5  # saturation factor from 1 to 40 spirals
    pred_40 = sigma_cent * sat
    print(f"\n  Extrapolated to 40 spirals (sat={sat}x): {pred_40:.0f}")
    print(f"  Paper centriole target: 4000")
    print(f"  Ratio: {pred_40/4000:.2f}x")
    if 0.5 < pred_40/4000 < 2.0:
        print(f"  *** BLIND PREDICTION IN RANGE ***")
    else:
        print(f"  Gap: {pred_40/4000:.2f}x")
else:
    print(f"Too large")
