# PowerShell wrapper for real experiment with proper encoding
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
& "D:\CCC 2.0\AI\agent-governance-system\.venv\Scripts\python.exe" -u "D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b\run_real.py" 2>&1
