# N1 Run Log

Date: 2026-03-09
Status: EXECUTED

## What Was Reviewed

- `THOUGHT/LAB/FORMULA/v3/questions/n01_e_vs_r/README.md`
- `THOUGHT/LAB/FORMULA/v2/q10_alignment/README.md`
- `THOUGHT/LAB/FORMULA/v2/q01_grad_s/README.md`
- `THOUGHT/LAB/FORMULA/v2/q20_tautology/README.md`
- `THOUGHT/LAB/FORMULA/v2/GLOSSARY.md`
- `THOUGHT/LAB/FORMULA/v2/METHODOLOGY.md`
- `THOUGHT/LAB/FORMULA/v1/questions/lower_q25_1260/tests/test_q25_real_data.py`
- `THOUGHT/LAB/FORMULA/v2/q02_falsification/code/test_v3_q02.py`
- `THOUGHT/LAB/FORMULA/v2/shared/formula.py`

## Local Environment Checks

### `.venv` metadata

```text
home = C:\Users\rene_\AppData\Local\Programs\Python\Python311
include-system-site-packages = true
version = 3.11.6
executable = C:\Users\rene_\AppData\Local\Programs\Python\Python311\python.exe
command = C:\Users\rene_\AppData\Local\Programs\Python\Python311\python.exe -m venv D:\CCC 2.0\AI\agent-governance-system\.venv
```

### Initial symptom observed

```text
No Python at '"C:\Users\rene_\AppData\Local\Programs\Python\Python311\python.exe'
```

### Direct checks that resolved the environment ambiguity

```text
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
D:\CCC 2.0\AI\agent-governance-system\.venv\Scripts\python.exe
```

```text
Get-Command python, python3, py
python.exe  -> C:\Users\rene_\AppData\Local\Programs\Python\Python311\python.exe
python3.exe -> C:\Users\rene_\AppData\Local\Microsoft\WindowsApps\python3.exe
py.exe      -> C:\Windows\py.exe
```

### Earlier PATH interpreter symptom

```text
python --version
ResourceUnavailable:
Program 'python.exe' failed to run: An error occurred trying to start process
'C:\Users\rene_\AppData\Local\Microsoft\WindowsApps\python.exe'
```

### Launcher checks

```text
py -0p
No installed Pythons found!
```

```text
where.exe python
INFO: Could not find files for the given pattern(s).
```

## What Was Added In This Pass

- Locked preregistration in `PREREGISTRATION.md`
- One-shot executable harness in `code/test_n01_e_vs_r.py`
- Output contract in `results/README.md`
- README status update pointing at the new artifacts

## Exact Execution Command

The final successful run used the repo-local `.venv` plus workspace-local HuggingFace caches:

```text
$env:HF_HOME='D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v3\questions\n01_e_vs_r\.cache\hf'
$env:HUGGINGFACE_HUB_CACHE='D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v3\questions\n01_e_vs_r\.cache\hf\hub'
$env:TRANSFORMERS_CACHE='D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v3\questions\n01_e_vs_r\.cache\hf\transformers'
$env:TORCH_HOME='D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v3\questions\n01_e_vs_r\.cache\torch'
.\.venv\Scripts\python.exe THOUGHT\LAB\FORMULA\v3\questions\n01_e_vs_r\code\test_n01_e_vs_r.py
```

## Runtime Notes

- First run timed out after 20 minutes while also trying to write HuggingFace cache state under `C:\Users\rene_\.cache`.
- Redirecting cache roots into the repository resolved the permission issue.
- Successful rerun completed in about 179 seconds after cache redirection.

## Output Files

- `results/n01_e_vs_r_results.json`
- `results/n01_e_vs_r_report.md`

## Numerical Outcome

From `results/n01_e_vs_r_report.md`:

- STS-B: `AUC(E)=0.4615`, `AUC(R_simple)=0.4533`, delta CI `[-0.0158, 0.0323]`
- SST-2: `AUC(E)=0.5915`, `AUC(R_simple)=0.5555`, delta CI `[-0.0193, 0.0897]`
- SNLI: `AUC(E)=0.4720`, `AUC(R_simple)=0.4621`, delta CI `[-0.0209, 0.0397]`
- MNLI: `AUC(E)=0.4628`, `AUC(R_simple)=0.4662`, delta CI `[-0.0402, 0.0301]`

Summary:

- `E` wins: `0`
- `R_simple` wins: `0`
- ties: `4`
- hypothesis status: `mixed`

Interpretation:

- The registered prediction that `E` would beat `R_simple` on at least 3 of 4 datasets was not supported.
- The counter-hypothesis that `R_simple` would beat `E` on at least 3 of 4 datasets was also not supported.
- Under this fixed test design, `E` and `R_simple` are statistically indistinguishable.
