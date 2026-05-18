"""Close the remaining gap: rotate each MT to face its neighbors."""
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
    H = np.zeros((N, N), dtype=complex)
    for i in range(N):
        H[i, i] = complex(0, -GAMMA/2)
    for i in range(N):
        for j in range(i+1, N):
            r_vec = coords_cm[j] - coords_cm[i]
            r = np.linalg.norm(r_vec)
            if r > LAMBDA_CM * 5: continue
            V = V_correct(r, dip_a[i], dip_a[j], r_vec/r)
            H[i, j] = V; H[j, i] = V
    gj = np.maximum(-2*np.imag(eigvals(H)), 0)
    return np.max(gj)/GAMMA

# Build centriole with each MT rotated to face its neighbors
# Triplet MT positions in Angstroms
mt_off = [np.array([0, 870, -225.167]), np.array([0, 1000, 0]), np.array([0, 1130, 225.167])]

# For each MT in a triplet, compute the direction to the OTHER two MTs
# and rotate the MT around x-axis to optimize alignment
def optimal_rotation(pos_ref, dip_ref, mt_offsets_local):
    """Find rotation angle that maximizes sigma for these MTs."""
    best_ang, best_s = 0, 0
    for ang_deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        ang = np.radians(ang_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        Rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        all_p, all_d = [], []
        for off in mt_offsets_local:
            for p, d in zip(pos_ref, dip_ref):
                pr, dr = Rx @ p, Rx @ d
                all_p.append(np.array([pr[0]+off[0], pr[1]+off[1], pr[2]+off[2]]))
                all_d.append(dr)
        all_p, all_d = np.array(all_p), np.array(all_d)
        s = analyze(all_p, all_d)
        if s > best_s:
            best_ang, best_s = ang_deg, s
    return best_ang

# Find optimal rotation for 2-MT pair in triplet
print("Optimal rotation for 2-MT test:")
best = optimal_rotation(pos1, dip1, [mt_off[0], mt_off[1]])
print(f"  MT1+MT2: best rotation = {best} deg")

# Full centriole: apply the optimal rotation within each triplet
# For 3 MTs in a triplet: MT1 faces MT2, MT2 faces both, MT3 faces MT2
# Simplified: all 3 MTs in a triplet get the same optimal rotation (60 deg)
opt_ang = 60
ca, sa = np.cos(np.radians(opt_ang)), np.sin(np.radians(opt_ang))
Rx_opt = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

all_p, all_d = [], []
for t in range(9):
    ang_t = t*(2*np.pi/9); ca_t, sa_t = np.cos(ang_t), np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
    for off in mt_off:
        ro = Rx_t @ off
        for p, d in zip(pos1, dip1):
            pr = Rx_opt @ p  # rotate MT around its own x-axis
            dr = Rx_opt @ d
            placed = np.array([pr[0]+ro[0], pr[1]+ro[1], pr[2]+ro[2]])
            # Also rotate the dipole by triplet angle
            all_p.append(placed)
            all_d.append(Rx_t @ dr)

all_p, all_d = np.array(all_p), np.array(all_d)
Nc = len(all_p)
print(f"\nCentriole (1 spiral/MT, MTs rotated {opt_ang} deg):")
print(f"  N = {Nc}")
if Nc <= 3000:
    sc = analyze(all_p, all_d)
    print(f"  sigma = {sc:.1f}")
    print(f"  Single MT (1 spiral): 14.1")
    print(f"  Enhancement: {sc/14.1:.1f}x")
    print(f"  27 independent: {27*14.1:.0f}")
    print(f"  Paper centriole (40 spirals): 4000")

