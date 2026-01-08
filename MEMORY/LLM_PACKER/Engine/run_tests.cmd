@echo off
setlocal
echo Running LLM Packer Smoke Tests...
python CAPABILITY/SKILLS/cortex-toolkit/run.py CAPABILITY/SKILLS/cortex-toolkit/fixtures/smoke_basic/input.json LAW/CONTRACTS/_runs/test_smoke/actual_basic.json
if %ERRORLEVEL% NEQ 0 goto :error
python CAPABILITY/SKILLS/cortex-toolkit/validate.py LAW/CONTRACTS/_runs/test_smoke/actual_basic.json CAPABILITY/SKILLS/cortex-toolkit/fixtures/smoke_basic/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo Running Lite Profile Check...
python CAPABILITY/SKILLS/cortex-toolkit/run.py CAPABILITY/SKILLS/cortex-toolkit/fixtures/smoke_lite/input.json LAW/CONTRACTS/_runs/test_smoke/actual_lite.json
if %ERRORLEVEL% NEQ 0 goto :error
python CAPABILITY/SKILLS/cortex-toolkit/validate.py LAW/CONTRACTS/_runs/test_smoke/actual_lite.json CAPABILITY/SKILLS/cortex-toolkit/fixtures/smoke_lite/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo ALL TESTS PASSED.
exit /b 0

:error
echo TESTS FAILED.
exit /b 1
