@echo off
setlocal
echo Running LLM Packer Smoke Tests...
python CAPABILITY/SKILLS/cortex/llm-packer-smoke/run.py CAPABILITY/SKILLS/cortex/llm-packer-smoke/fixtures/basic/input.json LAW/CONTRACTS/_runs/test_smoke/actual_basic.json
if %ERRORLEVEL% NEQ 0 goto :error
python CAPABILITY/SKILLS/cortex/llm-packer-smoke/validate.py LAW/CONTRACTS/_runs/test_smoke/actual_basic.json CAPABILITY/SKILLS/cortex/llm-packer-smoke/fixtures/basic/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo Running Lite Profile Check...
python CAPABILITY/SKILLS/cortex/llm-packer-smoke/run.py CAPABILITY/SKILLS/cortex/llm-packer-smoke/fixtures/lite/input.json LAW/CONTRACTS/_runs/test_smoke/actual_lite.json
if %ERRORLEVEL% NEQ 0 goto :error
python CAPABILITY/SKILLS/cortex/llm-packer-smoke/validate.py LAW/CONTRACTS/_runs/test_smoke/actual_lite.json CAPABILITY/SKILLS/cortex/llm-packer-smoke/fixtures/lite/expected.json
if %ERRORLEVEL% NEQ 0 goto :error

echo ALL TESTS PASSED.
exit /b 0

:error
echo TESTS FAILED.
exit /b 1
