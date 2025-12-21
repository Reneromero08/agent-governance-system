param(
  [string]$OutDir = "",
  [ValidateSet("full","delta")]
  [string]$Mode = "full",
  [string]$Stamp = "",
  [switch]$Zip = $true,
  [switch]$Combined = $true
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  $packerParentDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
  return (Resolve-Path (Split-Path -Parent $packerParentDir)).Path
}

$repoRoot = Get-RepoRoot
$packer = Join-Path $PSScriptRoot "packer.py"

if (-not (Test-Path -LiteralPath $packer)) {
  throw "Missing Python packer at: $packer"
}

if ($Stamp -eq "") { $Stamp = (Get-Date).ToString("yyyy-MM-dd_HH-mm-ss") }
if ($OutDir -eq "") { $OutDir = "MEMORY/LLM-PACKER/_packs/llm-pack-$Stamp" }

$args = @(
  "python",
  $packer,
  "--mode", $Mode,
  "--out-dir", $OutDir
)

$args += @("--stamp", $Stamp)
if ($Zip) { $args += "--zip" }
if ($Combined) { $args += "--combined" }

Write-Host "Running: $($args -join ' ')"
& $args[0] $args[1..($args.Count - 1)]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
