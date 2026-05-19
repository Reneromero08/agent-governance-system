# Test: orient each MT in triplet to face its neighbors
# MT1 at (0,870,-225): faces MT2 at (0,1000,0). Direction = (0,130,225) -> angle ~60 deg in yz
# MT2 at (0,1000,0): faces MT1 (angle ~240 deg) and MT3 (angle ~60 deg)
# MT3 at (0,1130,225): faces MT2 at (0,1000,0). Direction = (0,-130,-225) -> angle ~240 deg

print("\nOrienting MTs to face neighbors within triplet:")
# Angles to rotate each MT around x-axis so protofilaments face the adjacent MT
# MT1: face toward MT2 -> rotate by 60 deg (point y-axis away from x toward MT2)
# MT2: symmetric -> 0 deg (faces both equally)  
# MT3: face toward MT2 -> rotate by -60 deg (300 deg) 

for ang1 in [0, 60]:
 for ang2 in [0, 60, 300]:
  for ang3 in [0, 300, 60]:
    all_p, all_d = [], []
    for t in range(3):  # 3 triplets for speed
        ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
        Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
        for mi, (off, ang) in enumerate(zip(mt_off, [ang1, ang2, ang3])):
            ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            Rx_mt = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            ro = Rx_t @ off
            for p,d in zip(pos1, dip1):
                pr, dr = Rx_mt @ p, Rx_mt @ d
                all_p.append(np.array([pr[0]+ro[0], pr[1]+ro[1], pr[2]+ro[2]]))
                all_d.append(Rx_t @ dr)
    all_p, all_d = np.array(all_p), np.array(all_d)
    s = analyze(all_p, all_d)
    print(f"  [{ang1},{ang2},{ang3}] 3 triplets (9 MTs): sigma={s:.1f}")

# Best orientation for full centriole
best_angs = [60, 0, 300]
print(f"\nFull centriole with best orientations {best_angs}:")
all_p, all_d = [], []
for t in range(9):
    ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
    for mi, (off, ang) in enumerate(zip(mt_off, best_angs)):
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        Rx_mt = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        ro = Rx_t @ off
        for p,d in zip(pos1, dip1):
            pr, dr = Rx_mt @ p, Rx_mt @ d
            all_p.append(np.array([pr[0]+ro[0], pr[1]+ro[1], pr[2]+ro[2]]))
            all_d.append(Rx_t @ dr)
all_p, all_d = np.array(all_p), np.array(all_d)
sc = analyze(all_p, all_d)
print(f"  N={len(all_p)}, sigma={sc:.1f}")
print(f"  Single MT (1sp): 14.1")
print(f"  Paper centriole (40sp): 4000")