import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "CAPABILITY/TESTBENCH", "--tb=no", "-v"],
    capture_output=True,
    text=True,
    cwd=r"d:\CCC 2.0\AI\agent-governance-system"
)

failures = []
for line in result.stdout.split('\n'):
    if 'FAILED' in line:
        failures.append(line.strip())

with open(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\failures.txt", "w") as f:
    f.write('\n'.join(failures))

print(f"Found {len(failures)} failures")
for f in failures:
    print(f)
