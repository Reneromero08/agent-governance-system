"""Parse Celardo Table A1 from arXiv PDF (each number on own line)."""
import fitz, re, json

doc = fitz.open("/tmp/celardo_arxiv.pdf")

# Collect ALL numeric lines from Table A1 pages
all_nums = []
in_table = False
for pg_idx in range(22, 30):
    text = doc[pg_idx].get_text()
    for line in text.split("\n"):
        line = line.strip()
        if "Table A1" in line:
            in_table = True
            continue
        if not in_table:
            continue
        # Stop at "Appendix" or known end markers
        if line.startswith("Appendix") or "References" in line or "Acknowledg" in line:
            in_table = False
            continue
        # Match a single decimal number
        m = re.match(r'^[−-]?\d+\.\d+$', line)
        if m:
            all_nums.append(float(line.replace("\u2212", "-")))

print(f"Found {len(all_nums)} numeric values in Table A1")

# Group into sets of 6: x, y, z, mx, my, mz
dipoles = []
for i in range(0, len(all_nums) - 5, 6):
    x, y, z, mx, my, mz = all_nums[i:i+6]
    if abs(x) < 200 and abs(y) < 200 and abs(z) < 200:
        mag = (mx*mx + my*my + mz*mz)**0.5
        if 0.9 < mag < 1.1:
            dipoles.append((x, y, z, mx, my, mz))

print(f"Extracted {len(dipoles)} valid dipoles (expecting 104)")
for i in range(min(5, len(dipoles))):
    x,y,z,mx,my,mz = dipoles[i]
    print(f"  [{i}]: ({x:.3f},{y:.3f},{z:.3f}) -> ({mx:.5f},{my:.5f},{mz:.5f})")

out = "/mnt/d/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/FORMULA/v4/biological_validation/superradiance/celardo_dipoles.json"
json.dump(dipoles, open(out, "w"))
print(f"\nSaved {len(dipoles)} dipoles")
