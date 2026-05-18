"""Correct Hamiltonian using paper's eq S3 with k1 AND k3 geometric factors."""
import json, numpy as np
from scipy.linalg import eigvals

GAMMA = 0.00273
LAMBDA_CM = 280e-7; K_WAVE = 2*np.pi/LAMBDA_CM; A_TO_CM = 1e-8
# No coherent_scale -- paper's eq S3 gives the correct coupling directly

with open("celardo_dipoles.json") as f:
    celardo = json.load(f)
pos1 = np.array([[d[0], d[1], d[2]] for d in celardo])
dip1 = np.array([[d[3], d[4], d[5]] for d in celardo])

def V_correct(r_cm, ui, uj, r_hat):
    """Correct coupling from paper eq S3 with k1 and k3."""
    kr = K_WAVE * r_cm
    # Geometric factors
    k1 = ui@uj - (ui@r_hat)*(uj@r_hat)
    k3 = ui@uj - 3.0*(ui@r_hat)*(uj@r_hat)

    # Handle kr -> 0 limit carefully
    if kr < 1e-3:
        # Near-field limits:
        # cos(kr)/kr -> 1/kr (dominates)
        # sin(kr)/kr^2 -> 1/kr  
        # cos(kr)/kr^3 -> 1/kr^3 (dominates real part)
        # sin(kr)/kr -> 1
        # For kr << 1: Omega ~ (3*gamma/4)*k3/kr^3
        #              Upsilon ~ (3*gamma/2)*k1
        Omega = (3*GAMMA/4) * k3 / (kr**3)
        Upsilon = (3*GAMMA/2) * k1
        # Cap Omega at reasonable value
        Omega = np.clip(Omega, -200, 200)
    else:
        coskr, sinkr = np.cos(kr), np.sin(kr)
        # Omega (real/coherent part)
        Omega = -(3*GAMMA/4)*k1*coskr/kr + (3*GAMMA/4)*k3*(sinkr/kr**2 + coskr/kr**3)
        # Upsilon (imaginary/dissipative part)
        Upsilon = (3*GAMMA/2)*k1*sinkr/kr + (3*GAMMA/2)*k3*(coskr/kr**2 - sinkr/kr**3)
    
    return complex(Omega, -Upsilon/2)  # H_mn = Omega - i*Upsilon/2


def analyze(pos_a, dip_a, disorder=True):
    coords_cm = pos_a * A_TO_CM; N = len(pos_a)
    H = np.zeros((N, N), dtype=complex)
    rng = np.random.default_rng(42)
    for i in range(N):
        e = rng.uniform(-100, 100) if disorder else 0.0
        H[i, i] = complex(e, -GAMMA/2)

    for i in range(N):
        for j in range(i+1, N):
            r_vec = coords_cm[j] - coords_cm[i]
            r = np.linalg.norm(r_vec)
            if r > LAMBDA_CM * 5: continue
            r_hat = r_vec / r
            V = V_correct(r, dip_a[i], dip_a[j], r_hat)
            H[i, j] = V
            H[j, i] = V

    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA, np.mean(gj)/GAMMA


# ---- Test single MT ----
print("CORRECT HAMILTONIAN (eq S3, k1 + k3)")
print("=" * 60)

for ns in [1, 3, 5]:
    ap, ad = [], []
    for s in range(ns):
        xs = s * 80.0
        for p, d in zip(pos1, dip1):
            ap.append(np.array([p[0]+xs, p[1], p[2]]))
            ad.append(d)
    ap, ad = np.array(ap), np.array(ad)
    s_d, m_d = analyze(ap, ad, True)
    s_c, m_c = analyze(ap, ad, False)
    print(f"  MT {ns} spiral(s), N={len(ap)}: disorder={s_d:.1f}, clean={s_c:.1f}, mean(G/g)={m_d:.3f}")

print(f"  Paper target: sigma=35 for MT")

# ---- 2 MTs (centriole triplet adjacent) ----
print(f"\n2-MT test (adjacent in triplet):")
mt_offsets = [np.array([0, 870, -225.167]), np.array([0, 1000, 0])]
for ns in [1, 3]:
    mt_p, mt_d = [], []
    for s in range(ns):
        xs = s * 80.0
        for p, d in zip(pos1, dip1):
            mt_p.append(np.array([p[0]+xs, p[1], p[2]]))
            mt_d.append(d)
    mt_p, mt_d = np.array(mt_p), np.array(mt_d)
    ap, ad = [], []
    for off in mt_offsets:
        for p, d in zip(mt_p, mt_d):
            ap.append(np.array([p[0]+off[0], p[1]+off[1], p[2]+off[2]]))
            ad.append(d)
    ap, ad = np.array(ap), np.array(ad)
    s_d, m_d = analyze(ap, ad, True)
    print(f"  {ns} spiral(s)/MT, N={len(ap)}: sigma={s_d:.1f}")

# ---- Centriole (27 MTs) ----
print(f"\nCentriole (27 MTs, 1 spiral each):")
mt_offsets_full = [
    np.array([0, 870, -225.167]),
    np.array([0, 1000, 0]),
    np.array([0, 1130, 225.167]),
]
all_p, all_d = [], []
for t in range(9):
    ang_t = t * (2*np.pi/9); ca, sa = np.cos(ang_t), np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    for off in mt_offsets_full:
        ro = Rx_t @ off
        for p, d in zip(pos1, dip1):
            all_p.append(np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]]))
            all_d.append(Rx_t @ d)
all_p, all_d = np.array(all_p), np.array(all_d)
Nc = len(all_p)
print(f"  N = {Nc}")
if Nc <= 3000:
    sc, mc = analyze(all_p, all_d, True)
    print(f"  sigma = {sc:.1f}")
    print(f"  Single MT (1 spiral) sigma = {s_d:.0f}")
    print(f"  Enhancement: {sc/(s_d if ns==1 else 25):.1f}x over single MT")
    print(f"  If 27 independent: {27*25:.0f}")
    print(f"  Paper centriole (40 spirals): 4000")
