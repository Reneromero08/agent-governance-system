@echo off
set REPO_ROOT=%~dp0..\..
pushd %REPO_ROOT%
python LAW\CONTRACTS\ags_mcp_entrypoint.py
popd
