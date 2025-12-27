@echo off
setlocal
echo Running LLM Packer Smoke Tests...
python -m SKILLS.llm-packer-smoke.run SKILLS/llm-packer-smoke/fixtures/basic/input.json CONTRACTS/_runs/test_smoke/actual_basic.json
if %ERRORLEVEL% NEQ 0 goto :error
python SKILLS/llm-packer-smoke/validate.py CONTRACTS/_runs/test_smoke/actual_basic.json SKILLS/llm-packer-smoke/fixtures/basic/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo Running Lite Profile Check...
python -m SKILLS.llm-packer-smoke.run SKILLS/llm-packer-smoke/fixtures/lite/input.json CONTRACTS/_runs/test_smoke/actual_lite.json
if %ERRORLEVEL% NEQ 0 goto :error
python SKILLS/llm-packer-smoke/validate.py CONTRACTS/_runs/test_smoke/actual_lite.json SKILLS/llm-packer-smoke/fixtures/lite/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo ALL TESTS PASSED.
exit /b 0

:error
echo TESTS FAILED.
exit /b 1
