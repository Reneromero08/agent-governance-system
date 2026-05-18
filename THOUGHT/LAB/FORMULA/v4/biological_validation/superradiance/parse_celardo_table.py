"""Parse Celardo et al. (2019) Table A1 dipole data from PDF."""
import fitz, re, json

doc = fitz.open("/tmp/celardo2019.pdf")
all_nums = []

for pg in [15, 16]:
    text = doc[pg].get_text()
    # Extract all lines that look like numbers
    for line in text.split("\n"):
        line = line.strip()
        # Match full numbers: e.g., "-2.378", "103.218", "-0.701"
        m = re.match(r'^[−-]?\d+\.\d+$', line)
        if m:
            val = float(line.replace("\u2212", "-"))
            all_nums.append(val)

print(f"Found {len(all_nums)} numeric lines")

# The data format: each Trp entry = 3 position numbers + 6 dipole-component parts
# Position numbers: typically >1 in magnitude  
# Dipole component high parts: typically <1, have 3 decimal places
# Dipole component low parts: typically integer-like continuation digits (e.g., "14", "99")

# But wait - in the PDF extraction, both position and dipole-hi parts look similar (floating point).
# The distinction is magnitude. Positions are 10-150, dipole-hi are -1 to 1.
# But we lost the dipole-lo parts because they don't have decimal points!

# Let me re-extract INCLUDING integer-only numbers (the continuation digits)
all_tokens = []
for pg in [15, 16]:
    text = doc[pg].get_text()
    for line in text.split("\n"):
        line = line.strip()
        parts = line.split()
        for p in parts:
            # Match any numeric token: integer or decimal
            if re.match(r'^[−-]?\d+\.?\d*$', p):
                try:
                    val = float(p.replace("\u2212", "-"))
                    all_tokens.append(val)
                except:
                    pass

print(f"Found {len(all_tokens)} total numeric tokens")

# Now scan: look for position triples (3 numbers with magnitude > 1)
# followed by 6 numbers (3 dipole pairs: hi1 lo1 hi2 lo2 hi3 lo3)
dipoles = []
i = 0
while i < len(all_tokens) - 8:
    x, y, z = all_tokens[i], all_tokens[i+1], all_tokens[i+2]
    # Position check
    if abs(x) > 0.5 and abs(y) > 0.5 and abs(z) > 0.5:
        # Next 6 tokens are dipole pairs
        mx_hi, mx_lo = all_tokens[i+3], all_tokens[i+4]
        my_hi, my_lo = all_tokens[i+5], all_tokens[i+6]
        mz_hi, mz_lo = all_tokens[i+7], all_tokens[i+8]
        
        # Reconstruct full dipole component:
        # mx_hi is like -0.701 (has 3 decimal places of precision)
        # mx_lo is like 14 (two more digits)
        # Full value: sign(mx_hi) * (abs(mx_hi) + mx_lo/100000)
        sign_x = 1 if mx_hi >= 0 else -1
        sign_y = 1 if my_hi >= 0 else -1
        sign_z = 1 if mz_hi >= 0 else -1
        
        mx = mx_hi + sign_x * abs(mx_lo) / 100000
        my = my_hi + sign_y * abs(my_lo) / 100000
        mz = mz_hi + sign_z * abs(mz_lo) / 100000
        
        mag = (mx*mx + my*my + mz*mz)**0.5
        if 0.8 < mag < 1.3:
            dipoles.append((x, y, z, mx/mag, my/mag, mz/mag))
            i += 9
        else:
            i += 1
    else:
        i += 1

print(f"Parsed {len(dipoles)} Trp dipoles from Table A1 (expecting 104)")
for i in range(min(5, len(dipoles))):
    x,y,z,mx,my,mz = dipoles[i]
    mag = (mx*mx + my*my + mz*mz)**0.5
    print(f"  Trp {i}: pos=({x:.3f},{y:.3f},{z:.3f}) dir=({mx:.6f},{my:.6f},{mz:.6f}) |m|={mag:.3f}")

if len(dipoles) > 0:
    out = "/mnt/d/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/FORMULA/v4/biological_validation/superradiance/celardo_dipoles.json"
    json.dump(dipoles, open(out, "w"))
    print(f"Saved {len(dipoles)} dipoles to celardo_dipoles.json")
