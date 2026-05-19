"""Test MT orientations facing neighbors for constructive inter-MT coupling."""
import json, numpy as np
from scipy.linalg import eigvals

GAMMA = 0.00273; LAMBDA_CM = 280e-7; K_WAVE = 2*np.pi/LAMBDA_CM; A_TO_CM = 1e-8

with open("celardo_dipoles.json") as f:
    celardo = json.load(f)
pos1 = np.array([[d[0], d[1], d[2]] for d in celardo])
dip1 = np.array([[d[3], d[4], d[5]] for d in celardo])

def V_correct(r_cm, ui, uj, r_hat):
    kr = K_WAVE * r_cm
    k1 = ui@uj - (ui@r_hat)*(uj@r_hat)
    k3 = ui@uj - 3.0*(ui@r_hat)*(uj@r_hat)
    if kr < 1e-3:
        Omega = np.clip((3*GAMMA/4)*k3/(kr**3), -200, 200)
        Upsilon = (3*GAMMA/2)*k1
    else:
        coskr, sinkr = np.cos(kr), np.sin(kr)
        Omega = -(3*GAMMA/4)*k1*coskr/kr + (3*GAMMA/4)*k3*(sinkr/kr**2 + coskr/kr**3)
        Upsilon = (3*GAMMA/2)*k1*sinkr/kr + (3*GAMMA/2)*k3*(coskr/kr**2 - sinkr/kr**3)
    return complex(Omega, -Upsilon/2)

def analyze(pos_a, dip_a):
    coords_cm = pos_a * A_TO_CM; N = len(pos_a)
    H = np.zeros((N,N), dtype=complex)
    for i in range(N): H[i,i] = complex(0, -GAMMA/2)
    for i in range(N):
        for j in range(i+1,N):
            r_vec = coords_cm[j] - coords_cm[i]
            r = np.linalg.norm(r_vec)
            if r > LAMBDA_CM * 5: continue
            V = V_correct(r, dip_a[i], dip_a[j], r_vec/r)
            H[i,j] = V; H[j,i] = V
    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA

mt_off = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]

# Single MT
ap1, ad1 = [], []
for p,d in zip(pos1, dip1):
    ap1.append(p.copy()); ad1.append(d)
s1 = analyze(np.array(ap1), np.array(ad1))
print(f"Single MT (1sp): {s1:.1f}")

# MT1 faces MT2 (60 deg), MT3 faces MT2 (-60 deg = 300 deg)
print("\nSweeping MT2 orientation (MT1=60, MT3=300):")
for ang2 in range(0, 360, 30):
    all_p, all_d = [], []
    for t in range(1):  # 1 triplet
        ang_t = 0; ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
        Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
        for mi, (off, ang) in enumerate(zip(mt_off, [60, ang2, 300])):
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            Rx_mt = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            ro = Rx_t @ off
            for p,d in zip(pos1, dip1):
                pr, dr = Rx_mt @ p, Rx_mt @ d
                all_p.append(np.array([pr[0]+ro[0], pr[1]+ro[1], pr[2]+ro[2]]))
                all_d.append(Rx_t @ dr)
    all_p, all_d = np.array(all_p), np.array(all_d)
    s = analyze(all_p, all_d)
    marker = " <--" if s > 1.5*s1 else ""
    print(f"  MT2={ang2:>3} deg: sigma={s:.1f}, vs 3x single={3*s1:.0f}{marker}")

# Full centriole with best orientation
print("\nFull centriole (9 triplets) with MTs facing neighbors:")
all_p, all_d = [], []
for t in range(9):
    ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
    for mi, (off, ang) in enumerate(zip(mt_off, [60, 0, 300])):
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        Rx_mt = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        ro = Rx_t @ off
        for p,d in zip(pos1, dip1):
            pr, dr = Rx_mt @ p, Rx_mt @ d
            all_p.append(np.array([pr[0]+ro[0], pr[1]+ro[1], pr[2]+ro[2]]))
            all_d.append(Rx_t @ dr)
all_p, all_d = np.array(all_p), np.array(all_d)
sc = analyze(all_p, all_d)
print(f"  N={len(all_p)}, sigma={sc:.1f}, per-MT={sc/27:.1f}")
print(f"  Paper centriole (40sp): 4000")
print(f"  Ratio: {sc/4000:.3f}x")
