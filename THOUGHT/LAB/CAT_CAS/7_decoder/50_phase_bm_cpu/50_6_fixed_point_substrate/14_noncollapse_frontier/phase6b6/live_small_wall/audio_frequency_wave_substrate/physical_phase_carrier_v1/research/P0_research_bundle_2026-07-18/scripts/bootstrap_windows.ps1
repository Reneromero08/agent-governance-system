param(
  [string]$P0Dir = ""
)
$ErrorActionPreference = "Stop"
$Python = "py"
& $Python "$PSScriptRoot\download_sources.py" --all
& $Python "$PSScriptRoot\verify_downloads.py"
if ($P0Dir -ne "") {
  & $Python "$PSScriptRoot\import_repo_context.py" --p0-dir $P0Dir
}
Write-Host "Manual-download records are listed in DOWNLOAD_LINKS.md and MANIFEST.json."
Write-Host "After adding them, rerun: py scripts/verify_downloads.py"
