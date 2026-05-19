"""Spiral up: push 1-triplet centriole to max N, see per-chr trend."""
exec(open("final_test.py").read().split("# Centriole")[0])

mt_off = [np.array([0,870,-225.167]), np.array([0,1000,0]), np.array([0,1130,225.167])]

print("Spiraling: 1-triplet centriole vs single MT")
print(f"{'sp':>3} {'1-MT sigma':>12} {'1-MT/chr':>10} {'3-MT sigma':>12} {'3-MT/chr':>10} {'ratio':>8}")
print("-"*58)

results = {}
for ns in [1,2,3,4,5,6,7,8]:
    sp,sd = assemble_mt(trp_pos, trp_dip, ns)
    sm = analyze(sp,sd); sm_chr = sm/len(sp)
    
    ap,ad = [], []
    for off in mt_off:
        mp,md = assemble_mt(trp_pos, trp_dip, ns)
        for p,d in zip(mp,md):
            ap.append(np.array([p[0]+off[0], p[1]+off[1], p[2]+off[2]]))
            ad.append(d)
    ap,ad = np.array(ap), np.array(ad)
    N = len(ap)
    if N <= 2800:
        sc = analyze(ap,ad); sc_chr = sc/N
        results[ns] = (sm, sm_chr, sc, sc_chr)
        print(f"{ns:>3} {sm:>12.1f} {sm_chr:>10.4f} {sc:>12.1f} {sc_chr:>10.4f} {sc/sm:>8.2f}x")
    else:
        print(f"{ns:>3} {sm:>12.1f} {sm_chr:>10.4f} {'-- too large':>12} {'--':>10}")

# Full centriole at 1 spiral
print("\nFull centriole (9 triplets, 1sp):")
ap,ad = [], []
for t in range(9):
    ang_t = t*(2*np.pi/9); ca_t,sa_t = np.cos(ang_t),np.sin(ang_t)
    Rx_t = np.array([[1,0,0],[0,ca_t,-sa_t],[0,sa_t,ca_t]])
    for off in mt_off:
        ro = Rx_t @ off
        mp,md = assemble_mt(trp_pos, trp_dip, 1)
        for p,d in zip(mp,md):
            ap.append(np.array([p[0]+ro[0], p[1]+ro[1], p[2]+ro[2]]))
            ad.append(Rx_t @ d)
ap,ad = np.array(ap), np.array(ad)
sc9 = analyze(ap,ad)
print(f"  sigma={sc9:.1f}, per-chr={sc9/len(ap):.4f}")

# Extrapolate: if 1-triplet ratio holds for 9 triplets
if results:
    best_ns = max(results.keys())
    sm, sm_chr, sc, sc_chr = results[best_ns]
    ratio_1trip = sc / sm
    print(f"\n  At {best_ns}sp: 1-triplet ratio = {ratio_1trip:.2f}x")
    print(f"  If full centriole follows same ratio: {11.5 * ratio_1trip:.0f}")
    print(f"  Full centriole actual (1sp): {sc9:.0f}")
    print(f"  Full/1-triplet at 1sp: {sc9/sc if 1 in results else 0:.1f}x")
    print(f"  Expected 9/3 = 3x if linear")
